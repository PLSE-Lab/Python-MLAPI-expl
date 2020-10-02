#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning Chess
# Reinforcement Learning Chess is a series of notebooks where I implement Reinforcement Learning algorithms to develop a chess AI. I start of with simpler versions (environments) that can be tackled with simple methods and gradually expand on those concepts untill I have a full-flegded chess AI.

# [Notebook 1: Policy Iteration](https://www.kaggle.com/arjanso/reinforcement-learning-chess-1-policy-iteration)  
# [Notebook 2: Model-free learning](https://www.kaggle.com/arjanso/reinforcement-learning-chess-2-model-free-methods)  
# [Notebook 3: Q-Learning](https://www.kaggle.com/arjanso/reinforcement-learning-chess-3-q-networks)  
# [Notebook 4: Policy Gradients](https://www.kaggle.com/arjanso/reinforcement-learning-chess-4-policy-gradients)  

# ### Notebook 5: Monte Carlo tree search (MCTS)
# The aim of this notebook is to build a chess AI that can plan moves and defeat a simple opponent in a regular game of chess. My approach can be summarized as follows:
# - Instead of Q-learning (learning action values) I use "V-learning" (learning state-values).
#     - An advantage is that the Neural Network can learn with fewer parameters since it doesn't need to learn a seperate value for each action. In my Q-learning and Policy Gradient notebook, the output vector has a size > 4000. Now the size is only 1.
# - The V-network is updated using Temporal Difference (TD) Learning, like explained in Notebook 1. 
#     - This option is the simplest to code.  Other options are TD-lambda and Monte Carlo Learning.
# - The Architecture of the V-network is quite arbitrary and can probably be optimized. I used a combination of small and large convolutions, combined with convolutions that range a full file or rank (1-by-8 or 8-by-1).
# - Moves are planned using Monte Carlo Tree Search. This involves simulating playouts. 
#     - Monte Carlo Tree Search greatly improves performance on games like chess and go because it helps the agent to plan ahead.
# - These playouts are truncated after N steps and bootstrapped. 
#     - This reduces the variance of the simulation outcomes and gives performance gains, since the simulation doesn't require a full playout. 
# - For this version, the opponent of the RL-agent is a myopic player, that always chooses the move that results in the most material on the board or a checkmate.

# ![chess_gif](https://images.chesscomfiles.com/uploads/game-gifs/90px/green/neo/0/cc/0/0/b3cwS2tzM05pcTlxYnEhVGRrVENrQ1hIQzQ_MzQ1N0Z3RjNWNTY4MHFrVlVtdVVnaGcyTTY_TUZnbzFUYXFXR2ZIWUlxR0tDSHlaUj84.gif)

# **Import**

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().system('pip install python-chess  # Python-Chess is the Python Chess Package that handles the chess environment')
get_ipython().system('pip install --upgrade git+https://github.com/arjangroen/RLC.git  # RLC is the Reinforcement Learning package')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import os
import inspect
import matplotlib.pyplot as plt


# In[ ]:


from keras.models import load_model


# In[ ]:


import os
os.listdir('../input')


# In[ ]:


from RLC.real_chess import agent, environment, learn, tree
import chess
from chess.pgn import Game


opponent = agent.GreedyAgent()
env = environment.Board(opponent, FEN=None)
player = agent.Agent(lr=0.0005,network='big')
learner = learn.TD_search(env, player,gamma=0.9,search_time=0.9)
node = tree.Node(learner.env.board, gamma=learner.gamma)
player.model.summary()


# ### Launching the network

# In[ ]:


n_iters = 10000  # maximum number of iterations
timelimit = 25000 # maximum time for learning
network_replacement_interval = 10  # For the stability of the nearal network updates, the network is not continuously replaced


# In[ ]:


learner.learn(iters=n_iters,timelimit_seconds=timelimit,c=network_replacement_interval) 


# ### Learning performance 

# In[ ]:


reward_smooth = pd.DataFrame(learner.reward_trace)
reward_smooth.rolling(window=1000,min_periods=1000).mean().plot(figsize=(16,9),title='average reward over the last 1000 steps')
plt.show()


# In[ ]:


reward_smooth = pd.DataFrame(learner.piece_balance_trace)
reward_smooth.rolling(window=50,min_periods=50).mean().plot(figsize=(16,9),title='average piece balance over the last 50 episodes')
plt.show()


# ### Final performance with large searchtime and more greedy behavior

# In[ ]:


learner.env.reset()
learner.search_time = 60
learner.temperature = 1/3


# In[ ]:


learner.play_game(n_iters,maxiter=128)


# In[ ]:


pgn = Game.from_board(learner.env.board)
with open("rlc_pgn","w") as log:
    log.write(str(pgn))


# In[ ]:


learner.agent.model.save('RLC_model.h5')


# # References
# **Reinforcement Learning: An Introduction**  
# Richard S. Sutton and Andrew G. Barto  
# 1st Edition  
# MIT Press, march 1998  
#   
# **RL Course by David Silver: Lecture playlist**  
# https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
