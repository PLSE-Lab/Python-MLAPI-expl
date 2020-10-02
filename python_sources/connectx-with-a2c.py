#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


get_ipython().system("pip install 'kaggle-environments>=0.1.6' > /dev/null")
get_ipython().system('pip install git+https://github.com/openai/baselines > /dev/null')


# In[ ]:


import numpy as np
import base64
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm_notebook as tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim

import gym
from gym import spaces
from gym.spaces.box import Box

from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from kaggle_environments import evaluate, make
from kaggle_environments.envs.connectx.connectx import is_win


# # Set Parameters

# In[ ]:


NUM_PROCESSES = 16
NUM_ADVANCED_STEP = 5
GAMMA = 0.99

TOTAL_MOVES = 3e5
NUM_UPDATES = int(TOTAL_MOVES / NUM_ADVANCED_STEP / NUM_PROCESSES)


# In[ ]:


# A2C loss
value_loss_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5

# RMSprop
lr = 7e-4
eps = 1e-5
alpha = 0.99


# # Define Classes and Functions

# ## ConnectX Environment

# In[ ]:


class ConnectX(gym.Env):
    def __init__(self, switch_prob=0.5, opponent='random'):
        self.env = make('connectx')
        self.pair = [None, opponent]
        self.trainer = self.env.train(self.pair)
        self.switch_prob = switch_prob
        
        self.rows = self.env.configuration.rows
        self.columns = self.env.configuration.columns
        self.action_space = spaces.Discrete(self.columns)
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(3, self.rows, self.columns), dtype=np.uint8)
    def switch_trainer(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)
    
    def observation(self, observation):
        obs = observation.board
        if observation.mark == 2:
            obs = [3 - x if x != 0 else 0 for x in obs]
        
        obs = np.array(obs).reshape(self.rows, self.columns)
        obs = np.eye(3)[obs].transpose(2, 0, 1)
        return obs

    def step(self, action):
        obs, reward, done, info = self.trainer.step(int(action))
        
        if reward == 1: # Won
            reward = 1
        elif reward == 0: # Lost
            reward = -1
        else:
            reward = 0
            
        return self.observation(obs), reward, done, info
    
    def reset(self):
        if np.random.random() < self.switch_prob:
            self.switch_trainer()
        obs = self.trainer.reset()
        return self.observation(obs)


# ## Opponent

# In[ ]:


def weighted_random(obs, config):
    from kaggle_environments.envs.connectx.connectx import is_win
    from random import choices
    from scipy.stats import norm
    
    columns = [c for c in range(config.columns) if obs.board[c] == 0]
    for mark in [obs.mark, 3 - obs.mark]:
        for column in columns:
            if is_win(obs.board, column, mark, config, False):
                return column

    return choices(columns, weights=norm.pdf(columns, 3, 1))[0]


# In[ ]:


def make_env():
    def _thunk():
        env = ConnectX(opponent=weighted_random)
        return env

    return _thunk


# ## Memory

# In[ ]:


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape):
        self.observations = torch.zeros(
            num_steps + 1, num_processes, *obs_shape).cuda()
        self.masks = torch.ones(num_steps + 1, num_processes, 1).cuda()
        self.rewards = torch.zeros(num_steps, num_processes, 1).cuda()
        self.actions = torch.zeros(
            num_steps, num_processes, 1).long().cuda()

        self.returns = torch.zeros(num_steps + 1, num_processes, 1).cuda()
        self.index = 0

    def insert(self, current_obs, action, reward, mask):
        self.observations[self.index + 1].copy_(current_obs)
        self.masks[self.index + 1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step + 1] *                 GAMMA * self.masks[ad_step + 1] + self.rewards[ad_step]


# ## Model

# In[ ]:


def init(module, gain):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, n_out):
        super(Net, self).__init__()

        def init_(module): return init(
            module, gain=nn.init.calculate_gain('relu'))

        self.conv = nn.Sequential(
            init_(nn.Conv2d(3, 24, kernel_size=3, padding=1)),
            nn.ReLU(),
            init_(nn.Conv2d(24, 48, kernel_size=3)),
            nn.ReLU(),
            init_(nn.Conv2d(48, 48, kernel_size=3)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(48 * 2 * 3, 80)),
            nn.ReLU()
        )

        def init_(module): return init(module, gain=1.0)
        # Critic
        self.critic = init_(nn.Linear(80, 1))

        def init_(module): return init(module, gain=0.01)
        # Actor
        self.actor = init_(nn.Linear(80, n_out))

        self.train()

    def forward(self, x):
        conv_output = self.conv(x)
        critic_output = self.critic(conv_output)
        actor_output = self.actor(conv_output)

        return critic_output, actor_output

    def act(self, x):
        value, actor_output = self(x)  

        for i in range(x.size(0)):
            for j in range(x.size(-1)):
                if x[i][0][0][j] != 1:
                    actor_output[i][j] = -1e7
        
        probs = F.softmax(actor_output, dim=1)
        action = probs.multinomial(num_samples=1)

        return action

    def get_value(self, x):
        value, actor_output = self(x)
        return value

    def evaluate_actions(self, x, actions):
        value, actor_output = self(x)
        
        log_probs = F.log_softmax(actor_output, dim=1)
        action_log_probs = log_probs.gather(1, actions)

        probs = F.softmax(actor_output, dim=1)
        dist_entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, dist_entropy


# ## Brain

# In[ ]:


class Brain(object):
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic
        self.optimizer = optim.RMSprop(
            actor_critic.parameters(), lr=lr, eps=eps, alpha=alpha)

    def update(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, 1))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_gain = (advantages.detach() * action_log_probs).mean()

        total_loss = (value_loss * value_loss_coef -
                      action_gain - dist_entropy * entropy_coef)

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        self.optimizer.step()


# # Training

# In[ ]:


class Trainer(object):
    def __init__(self, config):
        self.config = config
    
    def checkmate(self, board, mark):    
        columns = [c for c in range(self.config.columns) if board[c] == 0]
        for mark in [mark, 3 - mark]:
            for column in columns:
                if is_win(board, column, mark, self.config, False):
                    return column

    def train(self):
        seed_num = 1
        torch.manual_seed(seed_num)
        torch.cuda.manual_seed(seed_num)

        torch.set_num_threads(seed_num)
        envs = [make_env() for i in range(NUM_PROCESSES)]
        envs = SubprocVecEnv(envs)

        n_out = envs.action_space.n
        actor_critic = Net(n_out).cuda()
        global_brain = Brain(actor_critic)

        obs_shape = envs.observation_space.shape
        rollouts = RolloutStorage(
            NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)
        episode_rewards = torch.zeros([NUM_PROCESSES, 1])
        final_rewards = torch.zeros([NUM_PROCESSES, 1])

        obs = envs.reset()
        obs = torch.from_numpy(obs).float()
        rollouts.observations[0].copy_(obs)
        
        complete_count = 0

        for i in tqdm(range(NUM_UPDATES)):
            for step in range(NUM_ADVANCED_STEP):
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])

                cpu_actions = action.squeeze(1).cpu().numpy()

                for j in range(NUM_PROCESSES):	
                    board = rollouts.observations[step][j].cpu().numpy().argmax(axis=0).reshape(-1)	
                    forced_action = self.checkmate(board, 1)
                    if forced_action is not None:
                        cpu_actions[j] = forced_action

                obs, reward, done, info = envs.step(cpu_actions)

                reward = np.expand_dims(np.stack(reward), 1)
                reward = torch.from_numpy(reward).float()
                episode_rewards += reward

                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])

                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards

                episode_rewards *= masks

                masks = masks.cuda()

                obs = torch.from_numpy(obs).float()
                rollouts.insert(obs, action.data, reward, masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.observations[-1]).detach()

            rollouts.compute_returns(next_value)

            global_brain.update(rollouts)
            rollouts.after_update()

            if i % 125 == 0:
                print("finished moves {}, mean/median reward {:.2f}/{:.2f}, min/max reward {:.2f}/{:.2f}".
                      format(i*NUM_PROCESSES*NUM_ADVANCED_STEP,
                             final_rewards.mean(),
                             final_rewards.median(),
                             final_rewards.min(),
                             final_rewards.max()))
            
            complete_count = complete_count + 1 if final_rewards.mean() >= 0.75 else 0	            	
            if complete_count == 10:	
                print("finished training")	
                break

        torch.save(global_brain.actor_critic.state_dict(), 'weight.pth')


# In[ ]:


config = make("connectx").configuration
trainer = Trainer(config)
trainer.train()


# # Create an Agent and Write Submission File

# In[ ]:


get_ipython().run_cell_magic('writefile', 'submission.py', '\nimport numpy as np\nimport io\nimport base64\nimport torch\nfrom torch import nn\nimport torch.nn.functional as F\nfrom kaggle_environments.envs.connectx.connectx import is_win\n\nclass Flatten(nn.Module):\n    def forward(self, x):\n        return x.view(x.size(0), -1)\n\nclass Net(nn.Module):\n    def __init__(self, n_out):\n        super(Net, self).__init__()\n        self.conv = nn.Sequential(\n            nn.Conv2d(3, 24, kernel_size=3, padding=1),\n            nn.ReLU(),\n            nn.Conv2d(24, 48, kernel_size=3),\n            nn.ReLU(),\n            nn.Conv2d(48, 48, kernel_size=3),\n            nn.ReLU(),\n            Flatten(),\n            nn.Linear(48 * 2 * 3, 80),\n            nn.ReLU()\n        )\n        self.critic = nn.Linear(80, 1)\n        self.actor = nn.Linear(80, n_out)\n\n    def forward(self, x):\n        conv_output = self.conv(x)\n        critic_output = self.critic(conv_output)\n        actor_output = self.actor(conv_output)\n\n        return critic_output, actor_output\n\n    def act(self, x):\n        value, actor_output = self(x)  \n\n        for i in range(x.size(0)):\n            for j in range(x.size(-1)):\n                if x[i][0][0][j] != 1:\n                    actor_output[i][j] = -1e7\n        \n        probs = F.softmax(actor_output, dim=1)\n        action = probs.multinomial(num_samples=1)\n\n        return action')


# In[ ]:


with open('weight.pth', 'rb') as f:
    raw_bytes = f.read()
    encoded_weights = base64.encodebytes(raw_bytes)

template = f"""
actor_critic = Net({config.columns})
decoded = base64.b64decode({encoded_weights})
buffer = io.BytesIO(decoded)
actor_critic.load_state_dict(torch.load(buffer, map_location='cpu'))
"""

with open('submission.py', 'a') as f:
    f.write(template)


# In[ ]:


get_ipython().run_cell_magic('writefile', '-a submission.py', '\ndef my_agent(obs, config):    \n    board = obs.board\n    columns = [c for c in range(config.columns) if board[c] == 0]\n    for mark in [obs.mark, 3 - obs.mark]:\n        for column in columns:\n            if is_win(board, column, mark, config, False):\n                return column\n    \n    if obs.mark == 2:\n        board = [3 - x if x != 0 else 0 for x in board]\n    board = np.array(board).reshape(config.rows, config.columns)\n    board = np.eye(3)[board].transpose(2, 0, 1)\n    board = torch.from_numpy(board).view([1, 3, config.rows, config.columns]).float()\n    \n    with torch.no_grad():\n        action = actor_critic.act(board)\n    action = action.item()\n\n    return action')


# In[ ]:


get_ipython().run_line_magic('run', 'submission.py')


# # Test the Agent

# In[ ]:


env = make("connectx", debug=True)
env.reset()
env.run([my_agent, weighted_random])
env.render(mode="ipython", width=500, height=450)


# # Evaluate the Agent

# In[ ]:


def mean_reward(rewards):
    return sum(r[0] for r in rewards) / sum(r[0] + r[1] for r in rewards)

# Run multiple episodes to estimate its performance.
print("My Agent vs Weighted Random Agent:", mean_reward(evaluate("connectx", [my_agent, weighted_random], num_episodes=100)))
print("Weighted Random Agent vs My Agent:", mean_reward(evaluate("connectx", [weighted_random, my_agent], num_episodes=100)))
print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))
print("Negamax Agent vs My Agent:", mean_reward(evaluate("connectx", ["negamax", my_agent], num_episodes=10)))


# In[ ]:




