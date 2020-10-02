#!/usr/bin/env python
# coding: utf-8

# Since AlphaStar made the news for beating an elite player in StarCraft II, reinforcement learning has piqued my interest. I have read alot of articles and tutorials about the subject and it seems to me that they go quickly to deep q-learning without explaining really what q-learning is. In my experience, the articles I read will talk about the cartpole environnement without code and then jump straight up to deep learning with keras-rl.
# 
# When trying to implement simple q-learning myself, I was pretty lost. In a real example, what really is a Q-table?
# 
# In this notebook, I will try my best to make q-learning tangible.
# 
# Disclaimer 1 : Part of this code comes from an example of Siraj Raval from some time ago.
# 
# Disclaimer 2 : I am no expert in Python nor reinforcement learning so if you see ways to improve myself, do not hesitate to comment my notebook!

# # Install kaggle-environments

# In[ ]:


# 1. Enable Internet in the Kernel (Settings side pane)

# 2. Curl cache may need purged if v0.1.6 cannot be found (uncomment if needed). 
# !curl -X PURGE https://pypi.org/simple/kaggle-environments

# ConnectX environment was defined in v0.1.6
get_ipython().system("pip install 'kaggle-environments>=0.1.6'")


# # Create ConnectX Environment

# In[ ]:


from kaggle_environments import evaluate, make, utils

env = make("connectx", debug=True)


# # Fill-up your Q-Table
# 
# So, what is q-learning anyway? It is a technique that, given a state of the environnement, either chooses the best actions amongst the possible actions or take a random action for exploration.
# 
# For this example, I make a q-learning model with a toy example of ConnectX. There will only by 3 rows, 4 columns and the winner only has to put 3 in a row to win.

# In[ ]:


import random
import numpy as np
from matplotlib import pyplot as plt

env.configuration.rows = 3
env.configuration.columns = 4
env.configuration.inarow = 3
#env.specification


# For Q-learning, you need a Q-table. The Q-table is a matrix that maps every state possible to every actions possible.
# There is 4 actions, so the matrix will have 4 columns. For the rows, it is a little bit more complicated. the board that I designed has 3 rows and 4 columns so 12 squares. Every square can either have 0, 1 or 2 in it. 0 for nothing, 1 for a token of player 1 and 3 for a token of player 2. So the number of rows is the number of possible combinations : 3**(3*4) = 531441.
# 
# If you really wanted to optimize this, you would see that the board is the same when you flip it, so it reduces by half the possibilities for an even number of columns. Also, most combinations do not exists. You can't have 1,0,1 for column. But that will not be covered here.
# 
# For a toy example, the matrix is 500,000 * 4 = 2,000,000 elements.

# In[ ]:


# rows is state
# columns are actions
q_table = np.zeros([3**(env.configuration.rows * env.configuration.columns),env.configuration.columns])


# In q-learning, there is 3 hyperparameters : alpha, gamma and epsilon. I find that a decaying epsilon works best, so I will start it at 1 and decay it to 0.1. This means that my agent will take completely random actions in the beginning and at some point only take a random action 1 time out of 10.

# In[ ]:


# Hyperparameters
alpha = 0.9    # Learning rate
gamma = 0.1    # Discount factor (0 = only this step matters)
# epsilon is determined each step

all_epochs = []
all_penalties = []

NB_STEPS = 10000
NB_STEPS_RANDOM = 1000
EPSILON_0 = 1
EPSILON_END = 0.1
NB_STEPS_EPSILON = 1000
trainer = env.train([None, "random"])
episodes = 0
progression = []
step = 0
random.seed(10)
for _ in range(NB_STEPS):
    observation = trainer.reset()
  
    # Init Vars
    total_reward = 0
    done = False
    episode_step = 0  
    while not done:
        # This part computes the epsilon that will be used later on
        if step < NB_STEPS_RANDOM:
            epsilon = 1
        elif step < NB_STEPS_RANDOM + NB_STEPS_EPSILON:
            epsilon = EPSILON_0 - (EPSILON_0-EPSILON_END) * ((step-NB_STEPS_RANDOM) / NB_STEPS_EPSILON)
        else:
            epsilon = EPSILON_END
        
        # Sometimes, you cannot play a move because the column is full, so you take actions among the possible columns
        possible_action = observation.board[0:env.configuration.columns]
        possible_action = [i for i in range(len(possible_action)) if possible_action[i] == 0]
        # This next line converts the state to an integer using ternary converter.
        # For example, if the state is 0,1,2; the row will be 1*3 + 2 = 5. I will look the fifth row and take the action that yields the max value
        obs_value = int("".join(str(i) for i in observation.board),3)
        if random.uniform(0, 1) < epsilon:
            # Check the action space and choose a random action
            action = random.choice(possible_action)
        else:
            # Check the learned values
            action = possible_action[int(np.argmax(q_table[obs_value][possible_action],axis=0))]
        
        next_observation, reward, done, _ = trainer.step(action)
        # 1 for a win, 0 for a draw and -1 for a lose.
        if done :
            if reward == None:
                reward = -1
        if reward == 0.5:
            reward = 0
        elif reward == 0:
            reward = -1        
          
        # Old Q-table value
        old_value = q_table[obs_value, action]
        next_max = np.max(q_table[obs_value])

        # Update the new value
        # Bellman equation !
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[obs_value, action] = new_value

        observation = next_observation
        
        total_reward += reward
        step += 1               
        episode_step += 1

        if done:
            progression.append(total_reward)
            if episodes % 100 == 0:
                print(episodes, epsilon, sum(progression[-101:-1]),sum(sum(q_table)))
            episodes += 1   

# Lets look the performance of the agent for 100 games interval
chunks = 100
list_progression = [progression[i:i + chunks] for i in range(0, len(progression), chunks)]
list_progression = [sum(i)/len(i) for i in list_progression]
plt.plot(list_progression)
plt.show()


# The performance quickly peeks at 85 reward in a 100 games span. When the agent lose, it's -1, so this is probably around 93% win.

# # Evaluate your Agent
# This next chunk of code looks at the performance without having to take a random action once in a while.

# In[ ]:


for _ in range(1000):
    observation = trainer.reset()
  
    # Init Vars
    total_reward = 0
    done = False
    episode_step = 0  
    while not done:
        
        possible_action = observation.board[0:env.configuration.columns]
        possible_action = [i for i in range(len(possible_action)) if possible_action[i] == 0]

        # Check the learned values
        obs_value = int("".join(str(i) for i in observation.board),3)
        action = possible_action[int(np.argmax(q_table[obs_value][possible_action],axis=0))]
        
        next_observation, reward, done, _ = trainer.step(action)
        if done :
            if reward == None:
                reward = -1
        
        if reward == 0.5:
            reward = 0
        elif reward == 0:
            reward = -1        

        observation = next_observation
        
        total_reward += reward
        step += 1               
        episode_step += 1

        if done:
            progression.append(total_reward)
            if episodes % 10 == 0:
                print(episodes, epsilon, sum(progression[-11:-1]),sum(sum(q_table)))
            episodes += 1


# You can see with the second element in the list that the total reward for 10 games span is 10 most of the times, which means 10 win in 10 games. It is against a random agent, but still. It means the agent learned.

# I hoped you have learned something with this notebook. It certainly helped me on my journey of understanding reinforcement learning.
# 
# With a simple toy example, we have a q-table with 2,000,000 elements. With the real game, we would have 3**(6*7) which gives a huuge number. This is why we need deep q-learning, because q-learning does not scale well.
