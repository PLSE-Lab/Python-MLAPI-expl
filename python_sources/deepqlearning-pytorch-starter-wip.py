#!/usr/bin/env python
# coding: utf-8

# This is still work in progress , still trying to resolve the error , any help would be appreciated. 
# 
# Current Issue: Reward coming as none in the training loop.
# 
# **Resolved the reward issues** - taking help from - https://www.kaggle.com/phunghieu/connectx-with-deep-q-learning
# 
# Added -
#         if done:
#             if reward == 1: # Won
#                 reward = 20
#             elif reward == 0: # Lost
#                 reward = -20
#             else: # Draw
#                 reward = 10
#         else:
#             reward = -0.05

# In[ ]:


get_ipython().system("pip install 'kaggle-environments>=0.1.6' > /dev/null 2>&1")


# In[ ]:


import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from random import choice
from tqdm.notebook import tqdm
from kaggle_environments import evaluate, make
import math


# In[ ]:


class ConnectX(gym.Env):
    def __init__(self, switch_prob=0.5):
        self.env = make('connectx', debug=True)
        self.pair = [None, 'random']
        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob
        
        # Define required gym fields (examples):
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(config.columns * config.rows)

    def switch_trainer(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)

    def step(self, action):
        return self.trainer.step(action)
    
    def reset(self):
        if random.uniform(0, 1) < self.switch_prob:
            self.switch_trainer()
        return self.trainer.reset()
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)


# In[ ]:


env = ConnectX()


# In[ ]:


import torch
from torch import optim
import torch.nn as nn
# if gpu is to be used
use_cuda = torch.cuda.is_available()

device = torch.device("cuda:0" if use_cuda else "cpu")


# In[ ]:


seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)


# In[ ]:


###### PARAMS ######
learning_rate = 0.01
num_episodes = 2000
gamma = 0.85

egreedy = 0.9
egreedy_final = 0.02
egreedy_decay = 500

####################


# In[ ]:


env.observation_space.n , env.action_space.n


# In[ ]:


number_of_inputs = env.observation_space.n + 1
number_of_outputs = env.action_space.n
hidden_layer = 64
number_of_inputs , number_of_outputs , hidden_layer


# In[ ]:


def calculate_epsilon(steps_done):
    epsilon = egreedy_final + (egreedy - egreedy_final) *               math.exp(-1. * steps_done / egreedy_decay )
    return epsilon


# In[ ]:


class NeauralNetwork(nn.Module):
    def __init__(self):
        super(NeauralNetwork , self).__init__()
        self.l1 = nn.Linear(number_of_inputs,hidden_layer)
        self.l2 = nn.Linear(hidden_layer,number_of_outputs)
        self.activation = nn.ReLU()
        
    def forward(self , x):
        output = self.activation(self.l1(x))
        output = self.l2(output)
        return output


# In[ ]:


class QAgent(object):
    def __init__(self):
        self.nn = NeauralNetwork()
        self.loss_func = nn.MSELoss()
        self.optimizer = optim.Adam(params=self.nn.parameters() , lr=learning_rate)
        
    def select_action(self,state,epsilon):
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > epsilon:      
            
            with torch.no_grad():
                
                state = torch.Tensor(state).to(device)
                action_from_nn = self.nn(state)
                action = torch.max(action_from_nn,0)[1]
                action = action.item()        
        else:
            action = env.action_space.sample()
        
        return action
    
    def optimize(self, state, action, new_state, reward, done):
        state = torch.Tensor(state).to(device)
        new_state = torch.Tensor(new_state).to(device)
        #print(reward)
        #print(action)
        reward = torch.Tensor([reward]).to(device)
        
        if done:
            target_value = reward
        else:
            new_state_values = self.nn(new_state).detach()
            max_new_state_values = torch.max(new_state_values)
            target_value = reward + gamma * max_new_state_values
        #print(action)
        #print(self.nn(state))
        predicted_value = self.nn(state)[action]
        
        loss = self.loss_func(predicted_value , target_value)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


# In[ ]:



qnet_agent = QAgent()

steps_total = []

frames_total = 0 

for i_episode in range(num_episodes):
    print('No of episode:',i_episode)
    state1 = env.reset()
    state = state1.board[:]
    state.append(state1.mark)
    state = np.array(state, dtype=np.float32)
    step = 0
    #for step in range(100):
    while True:
        
        step += 1
        frames_total += 1
        
        epsilon = calculate_epsilon(frames_total)
        
        #action = env.action_space.sample()
        action = qnet_agent.select_action(state, epsilon)
        
        new_state1, reward, done, info = env.step(action)
        
        if done:
            if reward == 1: # Won
                reward = 20
            elif reward == 0: # Lost
                reward = -20
            else: # Draw
                reward = 10
        else:
            reward = -0.05
        
        new_state = new_state1.board[:]
        new_state.append(new_state1.mark)
        new_state1 = np.array(new_state, dtype=np.float32)

        qnet_agent.optimize(state, action, new_state, reward, done )
        
        state = new_state
        
        if done:
            steps_total.append(step)
            print("Episode finished after %i steps" % step )
            break


# In[ ]:


print("Average reward: %.2f" % (sum(steps_total)/num_episodes))
print("Average reward (last 100 episodes): %.2f" % (sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green', width=5)
plt.show()

