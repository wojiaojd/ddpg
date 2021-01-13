import collections

import gym.spaces
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from itertools import count
import time

import os
import sys

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from Agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

random_seed = 2

env = gym.make('MountainCarContinuous-v0')
env.seed(random_seed)

# size of each action
action_size = env.action_space.shape[0]
print('Size of each action:', action_size)

# examine the state space
state_size = env.observation_space.shape[0]
print('Size of state:', state_size)

action_low = env.action_space.low
print('Action low:', action_low)

action_high = env.action_space.high
print('Action high: ', action_high)


agent = Agent(state_size=state_size, action_size=action_size, random_seed=random_seed)

def save_model():
    print("Model Save...")
    torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')


def ddpg(n_episodes=100000, max_t=1500, print_every=1, save_every=20):
    scores_deque = deque(maxlen=print_every)
    scores = []
    mean_test = collections.deque(maxlen=10)
    best_reward = 0
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()           #重置环境，获取初始状态
        agent.reset()                 # noise.reset
        score = 0
        timestep = time.time()
        for t in range(max_t):
            action = agent.act(state) #预测动作
            next_state, reward, done, _ = env.step(action)  #执行动作，获得下一状态、奖励、结束标志
            agent.step(state, action, reward, next_state, done, t)  #智能体学习
            score += reward           #累积回报
            state = next_state        #后续输入状态
            if done:
                break

        scores_deque.append(score)
        scores.append(score)
        score_average = np.mean(scores_deque)

        if i_episode % save_every == 0:
            save_model()

        if i_episode % print_every == 0:
            print('\rEpisode {}, Average Score: {:.2f}, Max: {:.2f}, Min: {:.2f}, Time: {:.2f}' \
                  .format(i_episode, score_average, np.max(scores), np.min(scores), time.time() - timestep), end="\n")

        if np.mean(scores_deque) >= 90.0:
            save_model()
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, score_average))
            break
        if i_episode % 10 == 0:
            state = env.reset()
            test_reward = 0
            for k in range(1500):
                env.render()
                action = agent.act(state, True)
                next_state, reward, done, info = env.step(action)
                test_reward += reward
                state = next_state
                if done:
                    env.close()
                    break
            print('episode: {} , test_reward: {}'.format(i_episode, round(test_reward, 3)))
            mean_test.append(test_reward)
            if np.mean(mean_test) > best_reward:
                best_reward = np.mean(mean_test)
    return scores


scores = ddpg()
"""
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores) + 1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
"""
"""
agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth', map_location=torch.device('cpu')))
agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth', map_location=torch.device('cpu')))

for _ in range(5):
    state = env.reset()
    for t in range(1200):
        action = agent.act(state, add_noise=False)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break

env.close()
"""
