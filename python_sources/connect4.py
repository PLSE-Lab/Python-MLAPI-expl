#!/usr/bin/env python
# coding: utf-8

# Many of you will be familar with a game called connect 4, the aim of this game is simple, you need to get 4 of your pieces in a row before the opponent gets 4 in a row. 
# 
# The aim of this challenge will be to create an AI to play the game.
# 
# <img src="https://i.imgur.com/40B1MGc.png" >

# For this we will be using kaggle kernels, these are online notebooks which are free to run

# # Getting Started

# 1. You'll need to create a kaggle account [here](https://www.kaggle.com/), and will need to verify this account with your phone number
# 
# 2. Duplicate this notebook by pressing the "copy and edit button" - you can create your own notebooks [here](https://www.kaggle.com/notebooks/welcome)

# # Kaggle environment

# In[ ]:


from kaggle_environments import make, evaluate

# Create the game environment
# Set debug=True to see the errors if your agent refuses to run
env = make("connectx", debug=True)

# List of available default agents, each agent is essentially an AI which can play the aim
print(list(env.agents))


# In[ ]:


# Two random agents(which select a random non-full column) play one game round
env.run(["random", "random"])

# Show the game
env.render(mode="ipython")


# To play against the you have created you can set the opponent to None

# In[ ]:


env.play(["random", None], width=500, height=450)


# You can use the player above to view the game in detail: every move is captured and can be replayed.
# 
# 

# # Defining agents

# Agents are the things you create which interactive with the connect4 board.
# 
# Your agent should be implemented as a Python function that accepts two arguments: obs and config. It returns an integer with the selected column, where indexing starts at zero. So, the returned value is one of 0-6, inclusive.
# 
# 

# ## config
# 
# config contains three pieces of information:
# 
# config.columns - number of columns in the game board (7 for Connect Four)
# 
# config.rows - number of rows in the game board (6 for Connect Four)
# 
# config.inarow - number of pieces a player needs to get in a row in order to win (4 for Connect Four)
# 
# ## obs
# 
# obs.board - the game board (a Python list with one item for each grid location)
# 
# obs.mark - the piece assigned to the agent (either 1 or 2)

# For instance, for this game board:
# 
# <img src="https://i.imgur.com/kSYx4Nx.png" width="20%">
#  
#     
# obs.board would be 
# 
# [0, 0, 0, 0, 0, 0, 0,
# 
#  0, 0, 1, 1, 0, 0, 0,
#  
#  0, 0, 2, 2, 0, 0, 0,
#  
#  0, 2, 1, 2, 0, 0, 0,
#  
#  0, 1, 1, 1, 0, 0, 0,
#  
#  0, 2, 1, 2, 0, 2, 0].

# # Examples

# An agent which always selects the middle column, whether it's valid or not! Note that if any agent selects an invalid move, it loses the game. An example of an invalid move would be where the agent picks a column which is already full

# In[ ]:


# Selects middle column
def agent_middle(obs, config):
    return config.columns//2


# An agent which behaves identically to the "random" agent above.

# In[ ]:


import random

def agent_random(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return random.choice(valid_moves)


# An agent which selects the leftmost valid column.

# In[ ]:


# Selects leftmost valid column
def agent_leftmost(obs, config):
    valid_moves = [col for col in range(config.columns) if obs.board[col] == 0]
    return valid_moves[0]


# # Evaluating agents

# In[ ]:


# Agents play one game round
env.run([agent_leftmost, agent_random])

# Show the game
env.render(mode="ipython")


# The outcome of a single game is usually not enough information to figure out how well our agents are likely to perform. To get a better idea, we'll calculate the win percentages for each agent, averaged over multiple games. For fairness, each agent goes first half of the time.
# 
# To do this, we'll use the get_win_percentages() function (defined in a hidden code cell). To view the details of this function, click on the "Code" button below.

# In[ ]:


import numpy as np

# To learn more about the evaluate() function, check out the documentation here: (insert link here)
def get_win_percentages(agent1, agent2, n_rounds=100):
    # Use default Connect Four setup
    config = {'rows': 6, 'columns': 7, 'inarow': 4}
    # Agent 1 goes first (roughly) half the time          
    outcomes = evaluate("connectx", [agent1, agent2], config, [], n_rounds//2)
    # Agent 2 goes first (roughly) half the time      
    outcomes += [[b,a] for [a,b] in evaluate("connectx", [agent2, agent1], config, [], n_rounds-n_rounds//2)]
    print("Agent 1 Win Percentage:", np.round(outcomes.count([1,-1])/len(outcomes), 2))
    print("Agent 2 Win Percentage:", np.round(outcomes.count([-1,1])/len(outcomes), 2))
    print("Number of Invalid Plays by Agent 1:", outcomes.count([None, 0]))
    print("Number of Invalid Plays by Agent 2:", outcomes.count([0, None]))
    print("Number of Draws (in {} game rounds):".format(n_rounds), outcomes.count([0, 0]))


# Which agent do you think performs better against the random agent: the agent that always plays in the middle (agent_middle), or the agent that chooses the leftmost valid column (agent_leftmost)? Let's find out!

# In[ ]:


get_win_percentages(agent1=agent_random, agent2=agent_random)


# In[ ]:


get_win_percentages(agent1=agent_middle, agent2=agent_random)


# In[ ]:


get_win_percentages(agent1=agent_leftmost, agent2=agent_random)


# In[ ]:




