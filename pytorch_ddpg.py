import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time
import matplotlib.pyplot as plt


#####################  hyper parameters  ####################
mode = 'test'
PRETRAIN = False
MAX_EPISODES = 10000
MAX_EP_STEPS = 1000
LR_A = 2e-4   # learning rate for actor
LR_C = 1e-3    # learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 1e-3      # soft replacement
MEMORY_CAPACITY = 1000000
MEMORY_WARMUP_SIZE = 1e4
#MEMORY_CAPACITY = 5000
BATCH_SIZE = 256

RENDER = False
ENV_NAME = 'LunarLanderContinuous-v2'
# ENV_NAME = 'MountainCarContinuous-v0'
#ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################

class OrnsteinUhlenbeckProcess(object):
    """ Ornstein-Uhlenbeck Noise (original code by @slowbull)
    """
    def __init__(self, theta=0.15, mu=0, sigma=1, x0=0, dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size

    def generate(self, step):
        sigma = max(0, self.sigma_step * step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x0 = x
        return x

class ANet(nn.Module):   # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 128)
        self.fc1.weight.data.normal_(0, 0.01) # initialization
        self.out = nn.Linear(128, a_dim)
        self.out.weight.data.normal_(0, 0.01) # initialization
    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = F.tanh(x)
        return x


class CNet(nn.Module):   # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.fcs = nn.Linear(s_dim+a_dim, 120)
        self.fcs.weight.data.normal_(0, 0.01)  # initialization
        self.out = nn.Linear(120, 1)
        self.out.weight.data.normal_(0, 0.01)  # initialization

    def forward(self, s, a):
        x = self.fcs(torch.cat((s, a), 1))
        net = F.relu(x)
        value = self.out(net)
        return value


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        #self.sess = tf.Session()
        self.Actor_eval = ANet(s_dim, a_dim)
        self.Actor_target = ANet(s_dim, a_dim)
        self.Critic_eval = CNet(s_dim, a_dim)
        self.Critic_target = CNet(s_dim, a_dim)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=LR_C)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        with torch.no_grad():
            action = self.Actor_eval(s)[0].detach()
        action = action.numpy()
        return action # ae（s）

    def learn(self, t):
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Actor_target.' + x + '.data.add_(TAU*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.Critic_target.' + x + '.data.add_(TAU*self.Critic_eval.' + x + '.data)')

        # soft target replacement
        # self.sess.run(self.soft_replace)  # 用ae、ce更新at，ct
        m = min(self.pointer, MEMORY_CAPACITY)
        indices = np.random.choice(m, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        with torch.enable_grad():
            a = self.Actor_eval(bs)
            q = self.Critic_eval(bs,a)  # loss=-q=-ce（s,ae（s））更新ae   ae（s）=a   ae（s_）=a_
            # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
            loss_a = -torch.mean(q)
            # print(q)
            # print(loss_a)
            self.atrain.zero_grad()
            loss_a.backward()
            self.atrain.step()

        with torch.enable_grad():
            a_ = self.Actor_target(bs_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
            q_ = self.Critic_target(bs_,a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
            q_target = br+GAMMA*q_  # q_target = 负的
            q_v = self.Critic_eval(bs,ba)
            # print(q_v)
            td_error = self.loss_td(q_target,q_v)
            # td_error=R + GAMMA * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
            # print(td_error)
            self.ctrain.zero_grad()
            td_error.backward()
            self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

###############################  training  ####################################
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)
s_dim = env.observation_space.shape[0]
print(env.action_space.shape)
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)


if mode == 'train':
    if PRETRAIN:
        ddpg.Actor_eval.load_state_dict(torch.load('saves/150_Amodel.pth'))
        ddpg.Critic_eval.load_state_dict(torch.load('saves/150_Cmodel.pth'))
    var = 3  # control exploration
    t1 = time.time()
    mean_test = collections.deque(maxlen=10)
    episodebuf = []
    trainreward_curve = []
    best_reward = -500
    random_rate = 1.0
    for i in range(MAX_EPISODES):
        s = env.reset()
        total_reward = 0
        treward = 0
        # noise = OrnsteinUhlenbeckProcess(size=a_dim)

        for j in range(MAX_EP_STEPS):
            if RENDER:
                env.render()
            # Add exploration noise
            a = ddpg.choose_action(s.astype('float32'))
            #### MountainCar ####
            # a = np.clip(a+noise.generate(MAX_EP_STEPS*i+MAX_EP_STEPS), -env.action_space.high, env.action_space.high)    # add randomness to action selection for exploration
            #### MountainCar ####

            ###### Lunarlander####
            a = np.clip(np.squeeze(a), -1.0, 1.0)
            a = a*(1.0-random_rate)+np.clip(np.random.normal(a, 0.5), -1.0, 1.0)*random_rate
            ###### Lunarlander####

            s_, r, done, info = env.step(a)
            total_reward += r

            treward += r
            #ddpg.store_transition(s, a, r / 10, s_)
            ddpg.store_transition(s, a, r, s_)
            if ddpg.pointer > MEMORY_WARMUP_SIZE:
                ddpg.learn(MAX_EP_STEPS*i+MAX_EP_STEPS)

            s = s_
            if(done):
                break;

        print('episode: {} , total_reward: {} , treward: {}'.format(i, round(total_reward, 3), round(treward, 3)))
        if ddpg.pointer > MEMORY_WARMUP_SIZE:
            episodebuf.append(i + 1)
            trainreward_curve.append(np.clip(total_reward, -200, 300))
            plt.clf()
            plt.plot(episodebuf, trainreward_curve)
            plt.xlabel('episode_train')
            plt.ylabel('trainReward')
            plt.pause(0.001)

        if i % 10 == 0 and ddpg.pointer > MEMORY_WARMUP_SIZE:
            for _ in range(10):
                state = env.reset()
                test_reward = 0
                for __ in range(MAX_EP_STEPS):
                    if _ == 0: env.render()
                    action = ddpg.choose_action(state)
                    next_state, reward, done, info = env.step(action)
                    test_reward += reward
                    state = next_state
                    if done:
                        env.close()
                        break
                mean_test.append(test_reward)
            print('episode: {} , mean_reward: {}'.format(i, round(np.mean(mean_test), 3)))
            # save model
            if np.mean(mean_test) > best_reward:
                random_rate *= 1
                best_reward = np.mean(mean_test)
                torch.save(ddpg.Actor_eval.state_dict(), 'saves/{}_Amodel.pth'.format(i))
                torch.save(ddpg.Critic_eval.state_dict(), 'saves/{}_Cmodel.pth'.format(i))

    print('Running time: ', time.time() - t1)
elif mode == 'test':
    ddpg.Actor_eval.load_state_dict(torch.load('saves/180_Amodel.pth'))
    ddpg.Critic_eval.load_state_dict(torch.load('saves/180_Cmodel.pth'))
    mean_reward = []
    for _ in range(100):
        s = env.reset()
        total_reward = 0
        while True:
            env.render()
            a = ddpg.choose_action(s)
            s_, r, d, info = env.step(a)
            total_reward += r
            s = s_
            if d: break
        mean_reward.append(total_reward)
        print('episode {} , reward: {} '.format(_+1, total_reward))
    print('mean_reward: {}'.format(np.mean(mean_reward)))