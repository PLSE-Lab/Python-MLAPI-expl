#!/usr/bin/env python
# coding: utf-8

# # Double Q-Learning Implementation
# > NOTE : This kernel uses [Hieu Phung's Kernel](https://www.kaggle.com/phunghieu/connectx-with-q-learning) as a base
# 
# # Introduction
# [Connect Four](https://en.wikipedia.org/wiki/Connect_Four) is a game where two players alternate turns dropping colored discs into a vertical grid. Each player uses a different color (usually red or yellow), and the objective of the game is to be the first player to get four discs in a row.

# <a class="anchor" id="ToC"></a>
# # Table of Contents
# * [The Competition "ConnectX"](#competion_connectx)
# * [Reinforcement learning](#reinforcement_learning)
# * [Q-Learning](#q_learning)
# * [Double Q-Learning](#double_q_learning)
# * [Implementation](#implementation)
#     * [Define useful classes](#useful_classes)
#     * [Create ConnectX environment](#create_environment)
#     * [Configure hyper-parameters](#hyper_parameters)
#     * [Train the agent](#train)
#     * [Analyze results](#results)
#     * [Create an agent](#create_agent)
#     * [Evaluate the agent](#evaluation_agent)
# * [Conclusion](#conclusion)
# * [Credits](#credits)

# <a class="anchor" id="competition_connectx"></a>
# # The Competition "ConnectX"
# [Kaggle](https://www.kaggle.com/) annouce a beta-version of a brand-new type of Machine Learning competition called Simulations. In Simulation Compoetitions, you'll compete against a set of rules, rather than against an evaluation metric.
# 
# Instead of submitting a CSV file, or a Kaggle Notebook, you will submit a Python .py file. You'll also notice that the leaderboard is not based on how accurate your model is but rather how well you've performed against other users.

# <a class="anchor" id="reinforcement_learning"></a>
# # Reinforcement learning
# Reinforcement learning involves an agent, a set of states ***S***, and a set ***A*** of actions per state. By performing an action, the agent transitions from state to state. Executing an action in a specific state provides the agent with a reward (a numerical score).
# 
# The goal of the agent is to maximize its total reward. It does this by adding the maximum reward attainable from future states to the reward fro achieving its current state, effectively influencing the current action by the potention future reward.
# 
# In Q-Learning, before learning begins, a Q-table is initialized to a possibly arbitrary fixed value. 

# <a class="anchor" id="q_learning"></a>
# # Q-Learning
# Before learning begins, ***Q*** is initialized to a possibly arbitrary fixed value.
# ***Q*** is commonly called ***Q-table***, It is a table with the rewards for each states and actions.
# 
# ![Q-table](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Q-Learning_Matrix_Initialized_and_After_Training.png/440px-Q-Learning_Matrix_Initialized_and_After_Training.png)
# 
# For a game like Tic-Tac-Toe it is possible to store all the values because the game has less than 6000 states. The game "Connect 4" is far more complex than Tic-Tac-Toe because it has more than $10^{14}$. Initialize ***Q-table*** is almost impossible in a kernel, the approach that dynamically adding newly discovered states into an object of QTable class.
# 
# The Q-learning algorithm follow a function that easy to understand.
# ![Q-function](https://wikimedia.org/api/rest_v1/media/math/render/svg/678cb558a9d59c33ef4810c9618baf34a9577686)

# <a class="anchor" id="double_q_learning"></a>
# # Double Q-Learning
# In this competition, Q-Learning is already use and presented. If I would add my contribution in this competion I choose to implement Double Q-Learning algorithm.
# In a noisy environments Q-Learning can sometimes overestimate the action values, slowing the learning. A variant called Double Q-learning was proposed to correct this.
# 
# Two sperate value functions are trained in a mutually symmetric fashion using separate experiences, $Q^{A}$ and $Q^{B}$. The double Q-Learning update step is then as follow:
# 
# ![QA](https://wikimedia.org/api/rest_v1/media/math/render/svg/4941acabf5144d1b3e9c271606011abdc0df444d)
# ![QB](https://wikimedia.org/api/rest_v1/media/math/render/svg/3e37476013126ddd4afdba69ef7b03767f4c4b75)
# 
# Now the estimated value of the discounted future is evaluated using a diffrent policy, which solves the overestimation issue.

# In[ ]:


# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# ConnectX environment was defined in v0.1.6
get_ipython().system('pip install kaggle-environments')


# In[ ]:


import numpy as np
import gym
import random
import matplotlib.pyplot as plt
from random import choice
from tqdm.notebook import tqdm
from kaggle_environments import evaluate, make, utils
import pickle


# <a class="anchor" id="implementation"></a>
# # Implementation
# <a class="anchor" id="useful_classes"></a>
# # Define Useful Classes

# In[ ]:


class ConnectX(gym.Env):
    def __init__(self):
        self.env = make("connectx", debug=True)
        self.pair = [None, 'negamax']
        self.trainer = self.env.train(self.pair)
        config = self.env.configuration
        self.action_space = gym.spaces.Discrete(config.columns)
        self.observation_space = gym.spaces.Discrete(config.columns * config.rows)
        
    def switch_trainer(self):
        self.pair = self.pair[::-1]
        self.trainer = self.env.train(self.pair)
        
    def step(self, action):
        return self.trainer.step(action)
    
    def reset(self):
        return self.trainer.reset()
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
    
class QTable:
    def __init__(self, action_space):
        self.table = dict()
        self.action_space = action_space
        
    def add_item(self, state_key):
        self.table[state_key] = list(np.zeros(self.action_space.n))
        
    def set_table(self, table):
        self.table = table

    def __call__(self, state):
        board = state.board[:]
        board.append(state.mark)
        state_key = np.array(board).astype(str)
        state_key = hex(int(''.join(state_key), 3))[2:]
        if state_key not in self.table.keys():
            self.add_item(state_key)
        return self.table[state_key]


# <a class="anchor" id="create_environment"></a>
# # Create ConnectX environment

# In[ ]:


env = ConnectX()


# <a class="anchor" id="hyper_parameters"></a>
# # Define hyper-parameters

# In[ ]:


LEARNING_RATE = 0.3
DISCOUNT_RATE = 0.7
EPISODES = 10000

ALPHA_DECAY_STEP = 1000
ALPHA_DECAY_RATE = 0.9


# <a class="anchor" id="train"></a>
# # Train the agent

# In[ ]:


q_table_1 = QTable(env.action_space)
q_table_2 = QTable(env.action_space)


# In[ ]:


dql_all_epochs = []
dql_all_total_rewards = []
dql_all_avg_rewards = []
dql_all_qtable_rows = []


# In[ ]:


def maxAction(Q1, Q2, state):
    values = []
    for j in range(env.action_space.n):
        if state.board[j] == 0:
            values.append(Q1(state)[j] + Q2(state)[j])
        else:
            values.append(-1e7)
    action = np.argmax(values)
    return int(action)


# In[ ]:


for i in tqdm(range(EPISODES)):
    state = env.reset()
    epochs, total_rewards = 0, 0
    done = False
    
    while not done:
        action = choice([c for c in range(env.action_space.n) if state.board[c] == 0])
            
        next_state, reward, done, info = env.step(action)
        
        if done:
            if reward == 1: # Won
                reward = 20
            elif reward == 0: # Lost
                reward = -20
            else: # Draw
                reward = 10
        else:
            reward = -0.05
        
        rand = random.uniform(0, 1)
        if rand <= 0.5:
            new_value = maxAction(q_table_1, q_table_1, state)
            q_table_1(state)[action] = q_table_1(state)[action] + LEARNING_RATE * (reward + DISCOUNT_RATE * q_table_2(next_state)[new_value] - q_table_1(state)[action])
        elif rand > 0.5:
            new_value = maxAction(q_table_2, q_table_2, state)
            q_table_2(state)[action] = q_table_2(state)[action] + LEARNING_RATE * (reward + DISCOUNT_RATE * q_table_1(next_state)[new_value] - q_table_2(state)[action])
    
        state = next_state
        epochs += 1
        total_rewards += reward
        
    dql_all_epochs.append(epochs)
    dql_all_total_rewards.append(total_rewards)
    avg_rewards = np.mean(dql_all_total_rewards[max(0, i-100):(i+1)])
    dql_all_avg_rewards.append(avg_rewards)
    dql_all_qtable_rows.append(len(q_table_1.table))
    if (i+1) % ALPHA_DECAY_STEP == 0:
        LEARNING_RATE *= ALPHA_DECAY_RATE


# <a class="anchor" id="results"></a>
# # Analyze results

# In[ ]:


print(len(q_table_1.table))
print(len(q_table_2.table))


# In[ ]:


plt.plot(dql_all_avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Avg rewards (100)')
plt.show()


# In[ ]:


plt.plot(dql_all_qtable_rows)
plt.xlabel('Episode')
plt.ylabel('Explored states')
plt.show()


# <a class="anchor" id="create_agent"></a>
# # Create the agent

# In[ ]:


tmp_dict_q_table_1 = q_table_1.table.copy()
tmp_dict_q_table_2 = q_table_2.table.copy()

dict_q_table = dict()
for i in tmp_dict_q_table_1:
    dict_q_table[i] = tmp_dict_q_table_1[i]
for i in tmp_dict_q_table_2:
    if i in dict_q_table.keys():
        for j in range(env.action_space.n):
            dict_q_table[i][j] += tmp_dict_q_table_2[i][j]
    else:
        dict_q_table[i] = tmp_dict_q_table_2[i]
        
for k in dict_q_table:
    if np.count_nonzero(dict_q_table[k]) > 0:
        dict_q_table[k] = int(np.argmax(dict_q_table[k]))
    else:
        dict_q_table[k] = -1


# In[ ]:


import pickle
import zlib
import base64 as b64

def serializeAndCompress(value, verbose=True):
    serializedValue = pickle.dumps(value)
    if verbose:
        print('Lenght of serialized object:', len(serializedValue))
    c_data =  zlib.compress(serializedValue, 9)
    if verbose:
        print('Lenght of compressed and serialized object:', len(c_data))
    return b64.b64encode(c_data)


# In[ ]:


serialized_q_table = serializeAndCompress(dict_q_table)


# In[ ]:


my_agent = '''def my_agent(observation, configuration):
    from random import choice
    import pickle
    import zlib
    import base64 as b64
    
    serialized_q_table = ''' \
    + str(serialized_q_table) \
    + '''

    d_data_byte = b64.b64decode(serialized_q_table)
    data_byte = zlib.decompress(d_data_byte)
    q_table = pickle.loads(data_byte)

    board = observation.board[:]
    board.append(observation.mark)
    state_key = list(map(str, board))
    state_key = hex(int(''.join(state_key), 3))[2:]

    if state_key not in q_table.keys():
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

    action = q_table[state_key]
    
    if action == -1:
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

    if observation.board[action] != 0:
        return choice([c for c in range(configuration.columns) if observation.board[c] == 0])

    return action
    '''


# In[ ]:


with open('submission.py', 'w') as f:
    f.write(my_agent)


# <a class="anchor" id="evaluation_agent"></a>
# # Evaluate your Agent

# In[ ]:


from submission import my_agent


# In[ ]:


def mean_reward(rewards):
    return sum(r[0] for r in rewards) / float(len(rewards))

# Run multiple episodes to estimate its performance.
print("My Agent vs Random Agent:", mean_reward(evaluate("connectx", [my_agent, "random"], num_episodes=10)))
print("My Agent vs Negamax Agent:", mean_reward(evaluate("connectx", [my_agent, "negamax"], num_episodes=10)))


# <a class="anchor" id="validate"></a>
# # Validate submissions

# In[ ]:


env = make("connectx", debug=True)
env.run([my_agent, my_agent])
print("Success!" if env.state[0].status == env.state[1].status == "DONE" else "Failed...")


# <a class="anchor" id="conclusion"></a>
# # Conclusion
# For this competition, the double Q-Learning method is probably not the best because it would be necessary to be able to submit a final file larger than 1MB, and train the agent on more than 10000 episodes, usually double Q-Learning need more episodes to converge
# 
# According to the leaderboard, Double Q-Learning can manage the Deep Q-Learning (DQN) algorithm. We could consider implementing a double Deep Q-Learning (Double DQN).

# <a class="anchor" id="credits"></a>
# # Credits
# * [Adam - ConnectX Getting Started](https://www.kaggle.com/ajeffries/connectx-getting-started)
# * [Hieu Phung - ConnectX with Q-Learning](https://www.kaggle.com/phunghieu/connectx-with-q-learning)
# * [Sentdex - Reinforcement learning playlist video](https://www.youtube.com/playlist?list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7)
# * [Rubik's code - Introduction to Double Q-Learning](https://rubikscode.net/2020/01/13/introduction-to-double-q-learning/)
# * [Hado van Haselt - Double Q-Learning](https://papers.nips.cc/paper/3964-double-q-learning.pdf)
# * [Wikipedia - Q-Learning](https://en.wikipedia.org/wiki/Q-learning)
# 
