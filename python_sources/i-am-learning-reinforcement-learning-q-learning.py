# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gym
import random
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
env = gym.make('Taxi-v2').env
env.render() # show
# env.reset() # reset env and random initial state
print('State space: ', env.observation_space)
print('Action space: ', env.action_space)

# %%
# taxi row, taxi column, passenger index, destination
state = env.encode(3,1,2,3)
print('State number: ', state)

env.s = state
env.render()

env.P[331]

# %%
env.reset()
total_reward_list = []
for j in range(5):
    time_step = 0
    total_reward = 0
    list_visualize =[]
    
    while True:
        time_step += 1
        # choose action
        action = env.action_space.sample()
        # perform action and get reward
        state, reward, done, info = env.step(action)
        # total reward
        total_reward += reward
        # visualize
        list_visualize.append({
            'frame': env.render(mode = 'ansi'),
            'state': state,
            'action': action,
            'reward': reward,
            'total_reward': total_reward
        })
        
        #env.render()
        
        if done:
            total_reward_list.append(total_reward)
            break
print(total_reward_list)   
# %%
import time

for i, frame in enumerate(list_visualize):
    print(frame['frame'])
    print('time_step: ', i + 1)
    print('state: ',frame['state'])
    print('action: ',frame['action'])
    print('reward: ',frame['reward'])
    print('total_reward: ',frame['total_reward'])
    #time.sleep(3)

# q-learnin taxi project
# https://gym.openai.com/envs/Taxi-v2/
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py

env = gym.make('Taxi-v2').env

# Q table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameter
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Plotting Matrix
reward_list = []
dropouts_list = []

episode_number = 10000

for i in range(1,episode_number):
    
    # initialize enviroment
    state = env.reset()
    reward_count = 0
    dropouts = 0
    while True:
        
        # Exploit vs explore to find action
        # %10 explore, %90 exploit
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        # Action process and take reward / observation
        next_state, reward, done, _ = env.step(action)
        
        # Q learning funtion
        old_value = q_table[state, action] # old_value
        next_max = np.max(q_table[next_state]) # nex_max
        next_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max) # q-learnin algorithm
        
        # Q table update
        q_table[state, action] = next_value
        
        # update state
        state = next_state
        
        #find wrong dropouts
        if reward == -10:
            dropouts += 1
            
        if done:
            break
        
        reward_count += reward
    
    if i % 10 == 0:
        dropouts_list.append(dropouts)
        reward_list.append(reward_count)
        print('Episode: {}, reward: {}, wrong dropout: {}'.format(i, reward_count, dropouts))
        
fig, axs = plt.subplots(1,2)
axs[0].plot(reward_list)
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Reward')

axs[1].plot(dropouts_list)
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Dropouts')

axs[0].grid(True)
axs[1].grid(True)

plt.show()

# print(q_table)

# q learning frozen lake project
# https://gym.openai.com/envs/FrozenLake-v0/
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
env = gym.make('FrozenLake-v0').env

from gym.envs.registration import register
#register(
#    id='FrozenLakeNotSlippery-v0',
#    entry_point='gym.envs.toy_text:FrozenLakeEnv',
#    kwargs={'map_name' : '4x4', 'is_slippery': False},
#    max_episode_steps=100,
#    reward_threshold=0.78, # optimum = .8196
#)
# Q table
q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameter
alpha = 0.8
gamma = 0.95
epsilon = 0.1

# Plotting Matrix
reward_list = []

episode_number = 75000

for i in range(1,episode_number):
    
    # initialize enviroment
    state = env.reset()
    reward_count = 0
    while True:
        
        # Exploit vs explore to find action
        # %10 explore, %90 exploit
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        
        # Action process and take reward / observation
        next_state, reward, done, _ = env.step(action)
        # Q learning funtion
        old_value = q_table[state, action] # old_value
        next_max = np.max(q_table[next_state]) # nex_max
        next_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max) # q-learnin algorithm
        
        # Q table update
        q_table[state, action] = next_value
        
        # update state
        state = next_state

        if done:
            break
        
        reward_count += reward
    
    if i % 10 == 0:
        reward_list.append(reward_count)
        print('Episode: {}, reward: {}'.format(i, reward_count))
        
plt.plot(reward_list)
plt.show()