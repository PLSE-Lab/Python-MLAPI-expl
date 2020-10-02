#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning with Dueling DDQN

# This notebook was forked and edited from Phung's notebook: https://www.kaggle.com/phunghieu/connectx-with-deep-q-learning
# 
# Here we improve the DQN by adapting it as according to Dueling DQN: https://arxiv.org/abs/1511.06581 and Double DQN: https://arxiv.org/abs/1509.06461
# 
# We also implement a new opponent for training, namely dumbamax, which is a more stochastic version of negamax. 

# # Table of Contents <a class="anchor" id="ToC"></a>
# 1. [Install libraries](#install_libraries)
# 1. [Import libraries](#import_libraries)
# 1. [Define useful classes](#define_useful_classes)
# 1. [Define helper-functions](#define_helper_functions)
# 1. [Configure hyper-parameters](#configure_hyper_parameters)
# 1. [Train the agent](#train_the_agent)
# 1. [Save weights](#save_weights)
# 1. [Evaluate the agent](#evaluate_the_agent)

# # Install libraries <a class="anchor" id="install_libraries"></a>
# [Back to Table of Contents](#ToC)

# In[ ]:


get_ipython().system("pip install 'kaggle-environments>=0.1.6' > /dev/null 2>&1")


# # Import libraries <a class="anchor" id="import_libraries"></a>
# [Back to Table of Contents](#ToC)

# In[ ]:


import os
from collections import deque
import copy
import random
import time
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pickle
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from kaggle_environments import evaluate, make
from kaggle_environments.envs.connectx import connectx
#from stable_baselines.common.vec_env import SubprocVecEnv


# # Define useful classes <a class="anchor" id="define_useful_classes"></a>
# 
# [Back to Table of Contents](#ToC)

# In[ ]:


'''
   Code for the environment:
'''

def process_ob(ob):
    board = np.array(ob['board'], dtype=np.float32)
    if ob['mark'] == 2:
        board[board==2] = -1
        board[board==1] = 2
        board[board==-1] = 1
    return board


def dumbamax(dumbometer, decay=0.99):
    state = {
        'dumbometer': dumbometer,
        'decay': decay }

    def play(obs, config):
        state['dumbometer'] *= state['decay']

        if random.random() < state['dumbometer']:
            return random.choice([c for c in range(config.columns) if obs.board[c] == 0])
        else:
            return connectx.negamax_agent(obs, config)
    return play


class ConnectX(gym.Env):
    def __init__(self, switch_prob=0.5, opponent='negamax', process=True):
        self.env = make('connectx', debug=True)
        self.pair = [None, opponent]
        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob
        self.process = process

        # Define required gym fields (examples):
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(config.rows * config.columns)

    def switch_trainer(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)

    def step(self, action):
        ob, rew, done, info = self.trainer.step(action)
        if self.process:
            ob = process_ob(ob)
        return ob, rew, done, info

    def reset(self):
        if random.uniform(0, 1) < self.switch_prob:
            self.switch_trainer()
        ob = self.trainer.reset()
        if self.process:
            return process_ob(ob)
        return ob

    def render(self, **kwargs):
        return self.env.render(**kwargs)


def vectorized(num_envs, **kwargs):
    env_fns = [lambda: ConnectX(**kwargs) for i in range(num_envs)]
    return SubprocVecEnv(env_fns)


'''
   Code for the model:
'''
class Memory:
    def __init__(self, maxlen, observation_dim, action_dim, n_envs=1):
        self.obs = torch.empty((maxlen, n_envs, observation_dim), dtype=torch.float32)
        self.new_obs = torch.empty((maxlen, n_envs, observation_dim), dtype=torch.float32)
        self.actions = torch.empty((maxlen, n_envs, 1), dtype=int)
        self.rewards = torch.empty((maxlen, n_envs, 1), dtype=torch.float32)
        self.dones = torch.empty((maxlen, n_envs, 1), dtype=int)
        self.curlen = 0
        self.maxlen = maxlen
        self.t = 0

    def __len__(self):
        return self.curlen

    def save(self, obs, act, rew, done, new_obs):
        self.obs[self.t] = torch.tensor(obs, dtype=torch.float32).squeeze()
        self.new_obs[self.t] = torch.tensor(new_obs, dtype=torch.float32).squeeze()
        self.actions[self.t] = torch.tensor(act, dtype=torch.float32).reshape((-1, 1))
        self.rewards[self.t] = torch.tensor(rew, dtype=torch.float32).reshape((-1, 1))
        self.dones[self.t] = torch.tensor(done, dtype=torch.float32).reshape((-1, 1))
        self.t = (self.t + 1) % self.maxlen
        self.curlen = max(self.curlen, self.t)

    def sample(self, batch_size):
        idx = random.sample(range(self.curlen), batch_size)
        return self.obs[idx].reshape((-1, self.obs.shape[2])),                self.actions[idx].reshape((-1, 1)),                self.rewards[idx].reshape((-1, 1)),                self.dones[idx].reshape((-1, 1)),                self.new_obs[idx].reshape((-1, self.obs.shape[2]))
    
class Base(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1=512, hidden2=512):
        super(Base, self).__init__()
        self.val1 = nn.Linear(input_dim, hidden1)
        self.val2 = nn.Linear(hidden1, hidden2)
        self.val3 = nn.Linear(hidden2, 1)

        self.adv1 = nn.Linear(input_dim, hidden1)
        self.adv2 = nn.Linear(hidden1, hidden2)
        self.adv3 = nn.Linear(hidden2, output_dim)

    def forward(self, x):
        # Dueling
        val = F.relu(self.val1(x))
        val = F.relu(self.val2(val))
        val = self.val3(val)

        adv = F.relu(self.adv1(x))
        adv = F.relu(self.adv2(adv))
        adv = self.adv3(adv)
        return val + adv - adv.mean()


'''
Dueling DDQN
'''
class DQN:

    def __init__(self, observation_space, action_space, params, test=True, n_envs=1):
        self.observation_dim = observation_space.n
        self.action_dim = action_space.n
        self.n_envs = n_envs

        self.timestep = 0
        self.test = test
        self.params = params
        self.device = params['device']
        self.eps = params['epsilon']
        self.lr = params['lr']

        self.memory = Memory(params['max_experiences'],
                             self.observation_dim,
                             self.action_dim,
                             n_envs=n_envs)

        self.qnetwork_local = Base(self.observation_dim,
                                   self.action_dim,
                                   hidden1=params['hidden1'],
                                   hidden2=params['hidden2']).to(self.device)
        self.qnetwork_target = copy.deepcopy(self.qnetwork_local)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), params['lr'])
        lmbda = lambda e: params['lr_decay']
        #self.scheduler = optim.lr_scheduler.MultiplicativeLR(self.optimizer, lmbda)

    def act(self, observations):
        if len(observations.shape) == 1:
            observations = observations.reshape((1, -1))

        n_obs = len(observations)
        actions = np.zeros(n_obs, dtype=int)

        X = torch.tensor(observations)
        X = X.to(self.device).to(torch.float32)
        mask = observations[:, :self.action_dim] == 0

        for i in range(n_obs):
            valid_actions = np.where(mask[i])[0]
            if self.eps > random.random() and not self.test:
                actions[i] = random.sample(valid_actions.tolist(), 1)[0]
            else:
                self.qnetwork_local.eval()
                with torch.no_grad():
                    values = self.qnetwork_local(X[i]).detach()
                    valid_values = values[valid_actions]
                    idx = torch.argmax(valid_values)
                    actions[i] = valid_actions[idx]
        if self.n_envs > 1:
            return actions.tolist()
        else:
            return int(actions[0])


    def step(self, ob, action, reward, done, new_ob):
        self.memory.save(ob, action, reward, done, new_ob)

        for _ in np.where(done)[0]:
            eps = self.eps * self.params['decay']
            self.eps = max(self.params['min_epsilon'], eps)
            print(' ' * 100, f'eps: {eps}  lr: {self.lr}', end='\r')

        self.timestep = (self.timestep + 1) % self.params['update_every']
        if self.timestep == 0:
            if len(self.memory) > self.params['batch_size']:
                self.update()
                
    def update(self):
        batch = self.memory.sample(self.params['batch_size'])
        obs, actions, rewards, dones, new_obs = batch
        obs     = obs.to(self.device)
        new_obs = new_obs.to(self.device)
        dones   = dones.to(self.device)
        rewards = rewards.to(self.device)
        brange = torch.arange(len(obs))

        self.qnetwork_local.train()
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(obs)[brange, actions]

        # Double
        gamma = self.params['gamma']
        actions = self.qnetwork_local(new_obs).max(1)[1]
        Q_targets_next = self.qnetwork_target(new_obs)[brange, actions]
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #self.scheduler.step()

        # ---- update target network ---- #
        self.soft_update()

    def soft_update(self):
        target_model = self.qnetwork_target
        local_model = self.qnetwork_local
        tau = self.params['tau']
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
            
    def save(self, filename):
        if filename is None:
            return
        torch.save(self.qnetwork_local.state_dict(), filename + ".local")
        torch.save(self.qnetwork_target.state_dict(), filename + ".target")

    def load(self, filename):
        if os.path.exists(filename + ".local"):
            self.qnetwork_local.load_state_dict(torch.load(filename + ".local"))
        if os.path.exists(filename + ".target"):
            self.qnetwork_target.load_state_dict(torch.load(filename + ".target"))


# # Define helper-functions <a class="anchor" id="define_helper_functions"></a>
# [Back to Table of Contents](#ToC)

# In[ ]:


def change_reward(r):
    if r == 1:             # Won
        return 20
    elif r == 0:           # Lost
        return -20
    else:                  # Draw / Invalid Action / ...
        return 0


def play_game(env, agent):
    rewards = 0
    done = False
    ob = env.reset()

    while not done:
        action = agent.act(ob)

        # Take action
        new_ob, reward, done, _ = env.step(action)

        # Apply new rules
        if done:
            reward = change_reward(reward)
        else:
            reward = 0
        rewards += reward

        # Adding experience into buffer
        agent.step(ob, action, reward, done, new_ob)

        ob = new_ob

    return rewards

def train(episodes, agent, env, end_episode_callback=None):
    all_total_rewards = np.empty(episodes)
    all_avg_rewards = np.empty(episodes) # Last 500 steps

    for e in range(episodes):
        total_reward = play_game(env, agent)
        all_total_rewards[e] = total_reward
        avg_reward = all_total_rewards[max(0, e - 500):(e + 1)].mean()
        all_avg_rewards[e] = avg_reward
        print(f'Episode {e+1}/{episodes}: AVG Reward (500 eps). {avg_reward}', end='\r')

 #       if (e+1) % SAVE_EVERY == 0:
 #           agent.save(SAVE_PATH)

        if end_episode_callback is not None:
            end_episode_callback(e+1, agent, env)

    print('')
    return avg_reward


def evaluate(agent, env, episodes=10):
    results = 0
    for e in range(episodes):
        ob = env.reset()
        done = False
        while not done:
            ob, reward, done, _ = env.step(agent.act(ob))
        results += reward
        print(f'Episodes ran {e+1}/{episodes}  -  Current score {results}/{e+1}', end='\r')
    print('')
    return results / episodes


# # Configure hyper-parameters <a class="anchor" id="configure_hyper_parameters"></a>
# [Back to Table of Contents](#ToC)

# In[ ]:


device = torch.device('cuda')
#device = torch.device('cpu')
SAVE_PATH = './models/'
LOAD_PATH = './models/'
EPISODES_VS_RANDOM = 1000
EPISODES_VS_DUMBAMAX = 20000
TEST_VS_RANDOM = 100
TEST_VS_DUMBAMAX = 100
TEST_VS_NEGAMAX = 2

PARAMS = {
    'device': device,
    'gamma': 0.99,
    'max_experiences': 5000,
    'batch_size': 2 ** 6,
    'lr': 0.000347,
    'lr_decay': 1 - 1.25e-6,
    'tau': 0.072,
    'decay': 0.9997,
    'epsilon': 0.99,
    'min_epsilon': 0.02,
    'update_every': 5,
    'hidden1': 2 ** 11,
    'hidden2': 2 ** 12 }


# # Train the agent <a class="anchor" id="train_the_agent"></a>
# [Back to Table of Contents](#ToC)

# In[ ]:


def run(params=None, load=False, test_only=False):
    env = ConnectX(opponent='random')
    n_envs = 1
    #n_envs = 50
    #env = vectorized(n_envs, opponent='random')

    agent = DQN(env.observation_space, env.action_space, params, n_envs=n_envs)
    agent.test = False
    if load:
        agent.load(SAVE_PATH)

    if not test_only:
        print('Playing againts RANDOM')
        random_score = train(EPISODES_VS_RANDOM, agent, env)
        print(f'Reward (AVG. last 500 | VS Random): {random_score}')

        print('Playing againts DUMBAMAX')
        opponent = dumbamax(0.3, decay=1.)
        env = ConnectX(opponent=opponent)
        dumbamax_score = train(EPISODES_VS_DUMBAMAX, agent, env)

    agent.test = True

    print('Testing againts RANDOM')
    env = ConnectX(opponent='random', switch_prob=0.5)
    random_score = evaluate(agent, env, TEST_VS_RANDOM)
    print(f'Win percentage (VS random): {random_score}')

    print('Testing againts DUMBAMAX 0.25')
    opponent = dumbamax(0.25, decay=1)
    env = ConnectX(opponent=opponent, switch_prob=0.5)
    dumbamax_score = evaluate(agent, env, TEST_VS_DUMBAMAX)
    print(f'Win percentage (VS dumbamax): {dumbamax_score}')

    print('Testing againts DUMBAMAX 0.1')
    opponent = dumbamax(0.1, decay=1)
    env = ConnectX(opponent=opponent, switch_prob=0.5)
    dumbamax_score = evaluate(agent, env, TEST_VS_DUMBAMAX)
    print(f'Win percentage (VS dumbamax): {dumbamax_score}')

    print('Testing againts NEGAMAX')
    env = ConnectX(opponent='negamax', switch_prob=0)
    negamax_score = evaluate(agent, env, 5)
    print(f'Win percentage (VS negamax): {negamax_score}')

    return agent


# In[ ]:


# Train and Evaluate
agent = run(load=True, test_only=False, params=PARAMS)

