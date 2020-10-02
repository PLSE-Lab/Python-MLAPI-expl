#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import random
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **1. Preliminary Work**

# In[ ]:


# import gym library
# for more information please visit 
# - https://gym.openai.com/envs/Taxi-v3/
# - https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py

import gym


# In[ ]:


# create taxi environment
env = gym.make('Taxi-v2').env


# In[ ]:


# taxi environment has been created, to test this:
env


# In[ ]:


# to see this environment use: .render():
env.render()


# **2. Explanations**

# In[ ]:


env.reset() # reset environment and return random initial state


# Passenger locations:
#     - 0: R(ed)
#     - 1: G(reen)
#     - 2: Y(ellow)
#     - 3: B(lue)
#     - 4: in taxi
#     
# Destinations: 
#     - 0: R(ed)
#     - 1: G(reen)
#     - 2: Y(ellow)
#     - 3: B(lue)
#     
#  Rendering:
#     - blue: passenger
#     - magenta: destination
#     - yellow: empty taxi
#     - green: full taxi
#     - other letters (R, G, Y and B): locations for passengers and destinations
#     
# state space is represented by:
#      - (taxi_row, taxi_col, passenger_location, destination)
# 
# Environment is 5x5 matrix
# 
# There are 5x5x5x4 = 500 random states in total

# ![](https://storage.googleapis.com/lds-media/images/Reinforcement_Learning_Taxi_Env.width-1200.png)

# In[ ]:


print('State space: ', env.observation_space) # will show us all possible states


# In[ ]:


print('Action space: ', env.action_space) # will show us all possible actions


# In[ ]:


# taxi row, taxi columnn, passenger location, destination respectively
state = env.encode(3,1,2,3) # return state value from 500 state space options
print('state number: ',state)


# In[ ]:


# lets see this location, we expect that taxi is at location 3x1, 
# passenger is at 2 (namely at Y) and destination is 3 (namely B)
env.s = state
env.render()


# Actions:
#     There are 6 discrete deterministic actions:
#     - 0: move south
#     - 1: move north
#     - 2: move east 
#     - 3: move west 
#     - 4: pickup passenger
#     - 5: dropoff passenger
#     
# 

# In[ ]:


env.P[331]


# **3. Let taxi to start its journey in the environment **

# In[ ]:


env.reset() # reset first

time_step = 0
total_reward = 0
list_visualize = []

# this while loop is only one episopde
while True:
    
    time_step +=1
    
    # choose action
    action = env.action_space.sample() # take random sample from action space {0,1,2,3,4,5}
    
    # perform action and get reward
    state, reward, done, _ = env.step(action) # here state = next_state
    
    # total reward
    total_reward += reward
    
    # visualize
    list_visualize.append({'frame': env, 
                       'state': state,
                       'action': action,
                       'reward': reward,
                       'Total_Reward': total_reward
                      })
    # visualize all steps
    if time_step %100 == 0:
        env.render()
    
    if done: 
        break
    


# When the taxi is yellow it means it is empty and when it becomes green it means that taxi has picked-up the passenger

# In[ ]:


print('number of iterations: ', time_step)
print('Total reward: ', total_reward)


# In[ ]:


# to see slowly how our taxi moves in the environment: 
'''
import time  

for c, value in enumerate(list_visualize):
    print(value["frame"].render())
    print('Time_step: ', c + 1)
    print('Action: ', value["action"])
    print('State: ', value["state"])
    print('Reward: ', value["reward"])
    print('Total_reward: ', value["Total_Reward"])
    time.sleep(1)
'''


# In[ ]:


# lets make 3 episodes (3 while loops)

for i in range(3):
    
    env.reset()
    new_time_step = 0
    
    while True:
    
        new_time_step +=1

        # choose action
        action = env.action_space.sample() # take random sample from action space {0,1,2,3,4,5}

        # perform action and get reward
        state, reward, done, _ = env.step(action) # here state = next_state

        # total reward
        total_reward += reward

        # visualize
        list_visualize.append({'frame': env, 
                           'state': state,
                           'action': action,
                           'reward': reward,
                           'Total_Reward': total_reward
                          })

        if done: 
            break
    print('number of iterations: ', new_time_step)
    print('Total reward: ', total_reward)
    print('-'*40)


# **4. Q LEARNING**

# Q learning algorithm and function
# 
# https://www.freecodecamp.org/news/an-introduction-to-q-learning-reinforcement-learning-14ac0b4493cc/

# ![](https://cdn-media-1.freecodecamp.org/images/TnN7ys7VGKoDszzv3WDnr5H8txOj3KKQ0G8o)

# In[ ]:


# Q learning template

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('Taxi-v2').env

# Q table

q_table = np.zeros([env.observation_space.n, env.action_space.n]) # zeros(states, actions) and use .n to make it integer

# hyperparameters: alpha, gamma, epsilon

alpha = 0.1
gamma = 0.9
epsilon = 0.1

# plotting metrix

reward_list = []
dropouts_list = []

episode_number = 1000 # number of trainings 

for i in range(1, episode_number):
    
    # initialize environment
    
    state = env.reset() # For each episode, reset our environment and it returns new starting state 
    
    reward_count = 0
    dropouts = 0
    
    while True:
        
        # exploit OR explore in order to choose action (using The Epsilon-Greedy Algorithm)
        # epsilon = 0.1 means 10% explore and 90% exploit
    
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state]) # let state = 4, so action = argument where 
                                               # q_table[4] has max value
                                               # q_table is 500x5 matrix
                                               # q_table[4] = [0, 12, 32, 2, 5]
                                               # action = in third column corresponding to 32 let say south
        
        # action process and take reward / observation
        next_state, reward, done, _ = env.step(action) # .step(action) performs action and returns 4 parameteres 
                                                       # which are the next state, reward, false or true, end probaility
        
        # Q learning funtion update
        
        # Q(s,a)
        old_q_value = q_table[state,action]
        
        # max Q`(s`, a`)
        next_q_max = np.max(q_table[next_state])
        
        # find new Q value using Q funtion
        next_q_value = (1 - alpha) * old_q_value + alpha * (reward + gamma * next_q_max)
        
        # Q table update
        q_table[state,action] = next_q_value
        
        # update state
        state = next_state
        
        # find wrong drop-outs, no need for this actually, we will use it for visualization purposes 
        if reward == -10:
            dropouts += 1
        
        # find total reward
        reward_count += reward
        
        if done:
            break
    if i%10 == 0:
        dropouts_list.append(dropouts)
        reward_list.append(reward_count)
        print('Episode: {}, reward: {}, wrong dropouts: {}'.format(i, reward_count, dropouts))


# **5. Visualization of wrong dropouts and rewards**

# In[ ]:


fig, (axs1,axs2) = plt.subplots(1,2, figsize=(12, 6)) # create in 1 line 2 plots

axs1.plot(reward_list)
axs1.set_xlabel('episode*10')
axs1.set_ylabel('reward')
axs1.grid(True)

axs2.plot(dropouts_list)
axs2.set_xlabel('episode*10')
axs2.set_ylabel('wrong dropouts')
axs2.grid(True)

plt.show()


# **6. Lets investigate Q table**

# In[ ]:


# now we have a good Q table that can help me
q_table


# In[ ]:


env.render() # our environment right now


# **lets remember again**
# 
# Passenger locations:
#     - 0: R(ed)
#     - 1: G(reen)
#     - 2: Y(ellow)
#     - 3: B(lue)
#     - 4: in taxi
#     
# Destinations: 
#     - 0: R(ed)
#     - 1: G(reen)
#     - 2: Y(ellow)
#     - 3: B(lue)

# In[ ]:


at_this_state = env.encode(3,0,2,3) # taxi is at location 3x0, passenger is at location 2 and destination is 3
env.s = at_this_state
env.render()


# what kind of action is expected from the taxi at this moment?
# 
# answer: When look at the environment above, taxi is yellow which means it is empty. So, it should go to **south** first in order to pick-up the passeenger

# In[ ]:


q_table[at_this_state]


# - we see that max value of Q table at this state is at column 0
# - q_table[at_this_state, 0] should be performed, namely we anticipate to take the action 0
# 
# lets remember actions again 
# 
# Actions:
#     There are 6 discrete deterministic actions:
#     - 0: move south
#     - 1: move north
#     - 2: move east 
#     - 3: move west 
#     - 4: pickup passenger
#     - 5: dropoff passenger
#     

# We see that most probably taxi will take the correct action :)

# **Another example**

# In[ ]:


# let taxi be at 1x4, passenger in taxi (taxi will be green) (4), destination G (1) 
another_state = env.encode(1,4,4,1)
env.s = another_state
env.render()


# we anticipate that taxi will go towards **north**
# 
# lets see what kind of action it will most probably take

# In[ ]:


q_table[another_state]


# we see that max value is at column 1 (take action 1) meaning move north 

# # these codes below are not related with this kernel
# 
# from collections import deque
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# 
# # at this point, go to setting on the right menu to enable internet option ON
# !pip install --upgrade pip
# 
# !pip install pygame
