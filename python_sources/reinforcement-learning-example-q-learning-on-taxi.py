#!/usr/bin/env python
# coding: utf-8

# # Classical Reinforcement Learning Example with Q-Learning Algorithm & Taxi Enviroment (GYM TOOLKIT)

# You can check & learn q-learning algorithm [here](https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56)

# ![1_foZSvhano3gHO5476pYV6Q.png](attachment:1_foZSvhano3gHO5476pYV6Q.png)

# ## Taxi enviroment 
#  **There are 4 locations (labeled by different letters) and your job is to pick up the passenger at one location and drop him off in another. You receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes. There is also a 10 point penalty for illegal pick-up and drop-off actions.**

# ![0_9UeNpmTbV3NZjWUk.png](attachment:0_9UeNpmTbV3NZjWUk.png)

# **First of all, import necessary libraries. We are going to use "numpy" to use math operations, "gym" library to reach taxi enviroment, "matplotlib" to make graphics and visualize, "random" to create random numbers.**

# In[ ]:


import gym
import numpy as np
import matplotlib.pyplot as plt
import random


# **In this section, we define the enviroment and q table. The rows of q table are observation spaces and the columns of q table are actions.
# There are 5x5 grids, 5 different customer, 4 different pickup-dropoff locations. So obs space is 500.**

# In[ ]:


env = gym.make("Taxi-v3").env

# Q table 500 sample(observation space = 5*5*5*4 = 500) - 6 action (left,right, up, down, pickup, dropout)
q_table = np.zeros([env.observation_space.n,env.action_space.n])


# **epsilon is explore rate**
# 
# **alpha = learning rate**
# 
# **gamma = discount facto**
# 
# ** reward list is defined to keep rewards.**
# ** droput list is defined to keep wrong dropouts to see how algorithm works**

# In[ ]:



alpha = 0.1
gamma = 0.9 
epsilon = 0.1
# plotting metric
reward_list = []
dropout_list = []


# **I choose 10000 as episode number. It's like epoch in Machine learning. Every episode is a wrong dropout or dropof which means a completed test. As you see, as long as episode number (test number) increasing, the wrong dropouts decreases and it limits at 0. The prizes are negative just because of time penalty in each timestep.**

# In[ ]:


episode_number = 10000

for i in range(1,episode_number):
    
    # init enviroment
    state = env.reset()
    
    reward_count = 0
    dropouts = 0
    
    while True:
        
        # exploit vs explore to find action epsilon 0.1 => %10 explore %90 explotit
        if random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            
        # action process and take reward / take observation
        next_state, reward, done, _ = env.step(action)
        
        # q learning funct
        
        old_value = q_table[state, action]  #old value
        next_max = np.max(q_table[next_state]) #next max
        
        next_value = (1-alpha)*old_value + alpha*(reward + gamma*next_max)
        # q table update
        q_table[state,action] = next_value
        
        # update state
        state = next_state
        
        # find wrong dropout 
        if reward == -10:
            dropouts += 1
            
        if done:
            break
        
        reward_count  += reward
    if i%10 == 0:
        
        dropout_list.append(dropouts)
        reward_list.append(reward_count)
        print("Episode: {}, reward {}, wrong dropout {}".format(i, reward_count,dropouts))


# **In this section, we are going to visualize the reward and wrong dropouts. As you see, the wrong dropouts decreases and converges at zero. Reward also converges ar -1 and zero which is expected result just because of time penalty.**

# In[ ]:


fig, axs = plt.subplots(1,2)

axs[0].plot(reward_list)
axs[0].set_xlabel("episode")
axs[0].set_ylabel("reward")

axs[1].plot(dropout_list)
axs[1].set_xlabel("episode")
axs[1].set_ylabel("wrong dropout")

axs[0].grid(True)
axs[1].grid(True)
plt.show()


# **Let's try an example. We will use "encode" method and see action that algorithm make.
# env.encode(taxi row, taxi col, passenger index, destination)**

# In[ ]:


env.s = env.encode(0,0,3,4)
env.render()


# **As you see, algorithm make correct desicion as "dropoff" customer.**
