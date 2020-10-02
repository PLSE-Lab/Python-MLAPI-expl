#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system("pip install 'kaggle-environments>=0.1.6'")


# In[ ]:


import os
import re
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import copy
import base64
from tqdm.notebook import tqdm
from collections import defaultdict, namedtuple, deque
from IPython.core.display import HTML

# sys.path.append('../input/connectx/kaggle-environments-0.1.4')
from kaggle_environments import evaluate, make, utils

import gym
from gym.spaces import Discrete

import torch
from torch import optim
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ## Create strategy

# In[ ]:


class ConnectX(gym.Env):
    
    def __init__(self, user, switch_prob=0.5):
        
        self.env = make('connectx', debug=True)
        self.pair = [user, None]
        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob
        
        config = self.env.configuration
        self.action_space = Discrete(config.columns)
        self.observation = Discrete(config.columns*config.rows)
        
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
    
    def run(self, user_list):
        return self.env.run(user_list)


# ## Test random game

# In[ ]:


env = ConnectX('negamax')
env.run(['random', "random"])
HTML(env.render(mode="ipython", width=700, height=600, header=False))


# ## Create model

# In[ ]:


class DeepQLearning(Module):
    
    def __init__(self):
        super(DeepQLearning, self).__init__()
        
        env = ConnectX('negamax')
        input_dim = env.observation.n
        output_dim = env.action_space.n
        
        self.fc = nn.Sequential(nn.Linear(input_dim, 32),
                                nn.ReLU(inplace=True),
                                nn.Linear(32, 16),
                                nn.ReLU(inplace=True),
                                nn.Linear(16, 16),
                                nn.ReLU(inplace=True),
                                nn.Linear(16, output_dim))
        
        
    def forward(self, x):
        out = self.fc(x)
        return out


# ## Stratify agent

# In[ ]:


class ExperienceBuffer:
    
    def __init__(self, capacity):
        self.experiences = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        self.memory = deque(maxlen=capacity)
    
    def __len__(self):
        return len(self.memory)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)
        
    def sample(self, batch_size=32):           
        experiences = random.sample(self.memory, batch_size)
        
        states = torch.from_numpy(np.vstack([(e.state)[:42] for e in experiences if e.state is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e.action is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward if e.reward is not None else 0.5 for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([(e.next_state)[:42] for e in experiences if e.next_state is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([1 if e.done else 0 for e in experiences if e.done is not None])).int().to(device)
        
        return (states,actions,rewards,next_states,dones)
    
    
    
class Agent(object):
    
    def __init__(self, alpha=1e-2, gamma=0.6, batch_size=64, update_step=1):
        
        self.env = ConnectX('negamax')
        self.model_train = DeepQLearning()
        self.model_target = DeepQLearning()
        self.model_target.load_state_dict(self.model_train.state_dict())
           
        self.creation = nn.MSELoss()
        self.optimizer = optim.Adam(self.model_train.parameters(), lr=alpha)
        self.alpha = alpha
        self.gamma = gamma
        self.batch_size = batch_size
        self.t_step = 0
        self.update_step = update_step
        self.memory = ExperienceBuffer(10000000)
        
    def step(self, state, action, reward, next_state, done):        
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step+1) % self.update_step
        if self.t_step == 0 and (len(self.memory) > self.batch_size):
            experience = self.memory.sample()
            self.learn(experience)
                
    def act(self, state, epsilon=0):
        
        if random.uniform(0, 1) > epsilon:
            max_reward = -100
            action_valid = deque()
#             state_check = np.array(state[:42]).reshape(-1, self.env.action_space.n)
            for index in range(self.env.action_space.n):
                if state[index] == 0:
                    if reward_calculate(state, index) > max_reward:
                        action_valid = [index]
                        max_reward = reward_calculate(state, index)
                    elif reward_calculate(state, index) == max_reward:
                        action_valid.append(index)
            action = random.choice(action_valid)
        else:
            action = random.choice([c for c in range(self.env.action_space.n) if state[c] == 0])
        return action
        
    def learn(self, experiences):
        self.model_train.train()
        self.model_target.eval()
                
        states, actions, rewards, next_states, dones = experiences
        with torch.no_grad():
            next_state_value = self.model_target(next_states).detach()
            max_next_state_value = next_state_value.max(1)[0].unsqueeze(1)
            q_target = rewards + self.gamma*max_next_state_value*(1-dones)
        
        q_expected = self.model_train(states).gather(1, actions)
        loss = self.creation(q_target, q_expected)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.model_target.load_state_dict(self.model_train.state_dict())


# ## Rewards

# In[ ]:


def array_to_string(array):
        return ''.join([str(i) for i in array])

def check_reward(string, string_opn, mark):
    if 4*str(mark) in string:
        reward = 30
    else:
        if 4*str(3-mark) in string_opn:
            reward = 20
        else:
            if ('0'+str(mark)*3) in string or (str(mark)*3+'0') in string or             (str(mark)*2+'0'+str(mark)) in string or (str(mark)+'0'+str(mark)*2) in string:
                reward = 10
            else:
                if ('0'+str(mark)*3) in string_opn or (str(mark)*3+'0') in string_opn or                 (str(mark)*2+'0'+str(mark)) in string_opn or (str(mark)+'0'+str(mark)*2) in string_opn:
                    reward = 0
                else:
                    if ('00'+str(mark)*2) in string or (str(mark)*2+'00') in string:
                        reward = -10
                    else:
                        reward = -20

    return reward

def reward_calculate(board, action):
    mark = board[42:][0]
    state = np.array(board[:42]).reshape(6, -1)
    state_opn = state.copy()
    assert action in range(7)
    assert np.count_nonzero(state[:, action]) < 6
    
    for row in range(6)[::-1]:
        if state[row, action] == 0:
            valid_row = row
            state[row, action] = mark
            break  

    state_opn[valid_row, action] = 3-mark

    #check vertical
    vertical = state[:, action]
    vertical_self = array_to_string(vertical)
    vertical_opn = vertical_self[:valid_row] + str(3-mark) + vertical_self[valid_row+1:]
    vertical_reward = check_reward(vertical_self, vertical_opn, mark)

    #check horizontal
    horizontal = state[valid_row, :]
    horizontal_self = array_to_string(horizontal)
    horizontal_opn = horizontal_self[:action] + str(3-mark) + horizontal_self[action+1:]
    horizontal_reward = check_reward(horizontal_self, horizontal_opn, mark)

    #check left diagonal
    if action - valid_row >= 0:
        left_self = array_to_string(state[:, action-valid_row:].diagonal(0))
        if len(left_self) < 4:
            left_reward = -30
        else:
            left_opn = array_to_string(state_opn[:, action-valid_row:].diagonal(0))
            left_reward = check_reward(left_self, left_opn, mark)
    else:
        left_self = array_to_string(state[valid_row-action:,:].diagonal(0))
        if len(left_self) < 4:
            left_reward = -30
        else:
            left_opn = array_to_string(state_opn[:, action-valid_row+1:].diagonal(0))
            left_reward = check_reward(left_self, left_opn, mark)

    #check right diagonal        
    if action + valid_row < 7:
        right_self = array_to_string(state[:, :action+valid_row+1][:, ::-1].diagonal(0))
        if len(right_self) < 4:
            right_reward = -30
        else:
            right_opn = array_to_string(state_opn[:, :action+valid_row+1][:, ::-1].diagonal(0))
            right_reward = check_reward(right_self, right_opn, mark)
    else:
        right_self = array_to_string(state[action+valid_row-6:, :][:, ::-1].diagonal(0))
        if len(right_self) < 4:
            right_reward = -30
        else:
            right_opn = array_to_string(state_opn[action+valid_row-6:, :][:, ::-1].diagonal(0))
            right_reward = check_reward(right_self, right_opn, mark)
    
    reward = max(vertical_reward, horizontal_reward, left_reward, right_reward)
    
    # check if next move opponent won
    if reward < 20:
        if (state[:, action] == 0).any():
            valid_row -= 1
            state_opt_next = state.copy()
            state_opt_next[valid_row, action] = 3-mark

            #check horizontal
            horizontal = array_to_string(state_opt_next[valid_row, :])
            if 4*str(3-mark) in horizontal:
                return -50

            #check left diagonal
            if action - valid_row >= 0:
                left_self = array_to_string(state_opt_next[:, action-valid_row:].diagonal(0))
            else:
                left_self = array_to_string(state_opt_next[valid_row-action:,:].diagonal(0))            
            if 4*str(3-mark) in left_self:
                return -50

            #check right diagonal
            if action + valid_row < 7:
                right_self = array_to_string(state_opt_next[:, :action+valid_row+1][:, ::-1].diagonal(0))
            else:
                right_self = array_to_string(state_opt_next[action+valid_row-6:, :][:, ::-1].diagonal(0))
            if 4*str(3-mark) in right_self:
                return -50
        
    return reward


# ## Train model

# In[ ]:


def dqn(user_list, episodes_list, eps_start=0.999,
        eps_end=0.01, eps_decay=0.995, gamma=0.6):
    
    scores = deque()
    
    env = ConnectX('random')
    agent = Agent()
    
    for user, episodes in zip(user_list, episodes_list):
        env = ConnectX(user)
        epsilon = eps_start
        
        for episode in tqdm(range(episodes)):
            step = 0
            observation = env.reset()
            state = observation.board[:]
            
            done = False
            rewards = 0
            while not done:
                    
                state.append(observation.mark)
                step += 1
                action = agent.act(state, epsilon)
                if isinstance(action, torch.Tensor):
                    action = action.item()
                
                if action == -1:
                    break
                    
                reward = reward_calculate(state, action)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.board[:]
                
                rewards += reward
                agent.step(state, action, reward, next_state, done)
                
                state = next_state
                
            scores.append(rewards)
            epsilon = max(epsilon*eps_decay, eps_end)

            if episode % 5 == 0:
                torch.save(agent.model_train.state_dict(),'model.pth') 
                    
    env.close()
    return list(scores)

random_number = 100000
negamax_number = 10000
scores = dqn(['random', 'negamax'], [random_number, negamax_number])


# In[ ]:


def display_average(scores, title):
    
    episode_list = deque()
    for episode in range(len(scores)):
        mean_score = np.mean(list(scores)[:episode+1])
        episode_list.append(mean_score)
        
    plt.figure(figsize=(15, 10))
    plt.plot(np.arange(len(scores)), episode_list)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title(title, size=25, color='b')
    plt.show()
    
display_average(scores[:random_number], 'random')


# In[ ]:


display_average(scores[random_number:], 'negamax')


# ## Load weight

# In[ ]:


def string_convert(paras):
    list_ = []
    for para in paras:
        para = re.sub('\s+', ', ', str(para))
        para = para.replace('[, ', '[').replace(', ]', ']')
        list_.append(para)

    string = ','.join(list_)
    string = 'np.array([' + string + '])'
    return string


# In[ ]:


model_path = '../working/model.pth'
 
param = torch.load(model_path)
key = [*param.keys()]

list_para = []
for index in range(len(key)//2):
    weight = key[2*index]
    bias = key[2*index+1]
    
    list_para.append('w_' + str(index) + ' = ' + string_convert(param[weight].cpu().numpy()))
    list_para.append('b_' + str(index) + ' = ' + string_convert(param[bias].cpu().numpy()))

string_para = '\n    '.join(list_para)


# ## Save agent

# In[ ]:


my_agent = '''import random
import numpy as np

def act(observation, configuration):
    
    
    state = observation.board[:]
    state = np.array(state, dtype=np.float32)
'''

my_agent = my_agent + '    ' + string_para + '''
    
    action_values = np.maximum(np.matmul(w_0, state) + b_0, 0)
    action_values = np.maximum(np.matmul(w_1, action_values) + b_1, 0)
    action_values = np.maximum(np.matmul(w_2, action_values) + b_2, 0)
    action_values = np.matmul(w_3, action_values) + b_3
    
    value_max = -1e10
    for index, value in enumerate(action_values):        
        if value > value_max and observation.board[index] == 0:
            value_max = value
            action = index
        
    return action
'''


with open('submission.py', 'w') as f:
    f.write(my_agent)

out = sys.stdout
submission = utils.read_file('/kaggle/working/submission.py')
agent = utils.get_last_callable(submission)
sys.stdout = out

env = make('connectx', debug=True)
env.run(['random', agent])
print(env.render())
print('Success!' if env.state[0].status == env.state[1].status == 'DONE' else 'Failed...')


# ## Visualize game

# In[ ]:


env.run([agent, 'negamax'])
HTML(env.render(mode="ipython", width=700, height=600, header=False))


# In[ ]:


env.run(['negamax', agent])
HTML(env.render(mode="ipython", width=700, height=600, header=False))

