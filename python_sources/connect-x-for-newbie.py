#!/usr/bin/env python
# coding: utf-8

# # This notebook for Newbie in Connect X

# # And for someone taking the course: "Intro to Game AI and Reinforcement Learning"

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

get_ipython().system('pip install kaggle_environments==0.1.6')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # The game environment
# There is the connectx environment for this game. 
# 
# There are also two agents created for you to play with as 'random' and 'negamax'.

# In[ ]:


from kaggle_environments import make, evaluate

# Create the game environment
# Set debug=True to see the errors if your agent refuses to run
env = make("connectx", debug=True)

# List of available default agents
print(list(env.agents))


# **You can make two agents plays against each other as**

# In[ ]:


# Two random agents play one game round
env.run(["random", "random"])

# Show the game
env.render(mode="ipython")


# # Defining Agents
# We need to create our own AGENT to participate in the competition.
# 
# Your agent should be implemented as a Python function that accepts two arguments: **obs** and **config**. It returns an integer with the selected column, where indexing starts at zero. So, the returned value is one of 0-6, inclusive.
# 
# 
# **obs** contains two pieces of information:
# 
# * obs.board - the game board (a Python list with one item for each grid location)
# * obs.mark - the piece assigned to the agent (either 1 or 2)
# 
# 
# obs.board is a Python list that shows the locations of the discs, where the first row appears first, followed by the second row, and so on. We use 1 to track player 1's discs, and 2 to track player 2's discs. 
# 
# For instance, for a game board: obs.board can be [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 1, 2, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 2, 1, 2, 0, 2, 0]
# 
# 
# config contains three pieces of information:
# 
# * config.columns - number of columns in the game board (7 for Connect Four)
# * config.rows - number of rows in the game board (6 for Connect Four)
# 
# 
# config.inarow - number of pieces a player needs to get in a row in order to win (4 for Connect Four)

# There are three simple agents from https://www.kaggle.com/alexisbcook/play-the-game

# In[ ]:


# Selects random valid column
def agent_random(obs, config):
    #an agent function should be fully encapsulated (no external dependencies)
    #then we need to import all librabries inside out agent
    import random
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return random.choice(valid_moves)

# Selects middle column
def agent_middle(obs, config):
    return config.columns//2

# Selects leftmost valid column
def agent_leftmost(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return valid_moves[0]


# Let agents play game against each other,we use the same env.run() method as before

# In[ ]:


# Agents play one game round
env.run([agent_random,agent_random])

# Show the game
env.render(mode="ipython")


# # Evaluating agents
# The outcome of a single game is usually not enough information to figure out how well our agents are likely to perform. To get a better idea, we'll calculate the win percentages for each agent, averaged over multiple games. For fairness, each agent goes first half of the time.
# 
# To do this, we'll use the get_win_percentages() function (defined in a hidden code cell). To view the details of this function, click on the "Code" button below.

# In[ ]:


# Link: https://www.kaggle.com/alexisbcook/play-the-game
def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time          
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time      
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    
    #these codes are not work because of the change in kaggle environment
    #you can fix this by downgrading the kaggle_environments
    #!pip install kaggle_environments==0.1.6 use this before any other code.
    
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,0])/len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([0,1])/len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0.5]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0.5, None]))
    print("Number of Draws (in {} game rounds):".format(n_rounds), outcomes.count([0.5, 0.5]))
    
    #print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
    #print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
    #print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    #print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
    #print("Number of Draws (in {} game rounds):".format(n_rounds), outcomes.count([0, 0]))


# Which agent do you think performs better against the random agent: the agent that always plays in the middle (agent_middle), or the agent that chooses the leftmost valid column (agent_leftmost)? Let's find out!

# In[ ]:


get_win_percentages(agent1=agent_random, agent2=agent_random)


# # Continue
# 
# Now you got the basic of Connect X. 
# 
# You should take the course **Intro to Game AI and Reinforcement Learning**
# 
# https://www.kaggle.com/learn/intro-to-game-ai-and-reinforcement-learning
# 
# 1. Here is my notebook with revised exercise for the first lesson.
# 
# https://www.kaggle.com/nlebang/corrected-exercise-play-the-game
# 
# 2. The revised notebook for the second lesson
# 
# https://www.kaggle.com/nlebang/revised-exercise-one-step-lookahead
# 
# 3. The revised notebook for the third lesson
# 
# https://www.kaggle.com/nlebang/revised-exercise-n-step-lookahead
# 
# 
# I will update more notesbooks for newbies to get started with Game AI & Reinforcement Learning here.
