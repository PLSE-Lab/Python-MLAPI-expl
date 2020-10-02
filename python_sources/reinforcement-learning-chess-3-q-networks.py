#!/usr/bin/env python
# coding: utf-8

# ## Reinforcement Learning Chess
# Reinforcement Learning Chess is a series of notebooks where I implement Reinforcement Learning algorithms to develop a chess AI. I start of with simpler versions (environments) that can be tackled with simple methods and gradually expand on those concepts untill I have a full-flegded chess AI.

# [Notebook 1: Policy Iteration](https://www.kaggle.com/arjanso/reinforcement-learning-chess-1-policy-iteration)  
# [Notebook 2: Model-free learning](https://www.kaggle.com/arjanso/reinforcement-learning-chess-2-model-free-methods)  
# [Notebook 4: Policy Gradients](https://www.kaggle.com/arjanso/reinforcement-learning-chess-4-policy-gradients)

# ### Notebook III: Q-networks
# In this notebook I implement an simplified version of chess named capture chess. In this environment the agent (playing white) is rewarded for capturing pieces (not for checkmate).  After running this notebook, you end up with an agent that can capture pieces against a random oponnent as demonstrated in the gif below. The main difference between this notebook and the previous one is that I use Q-networks as an alternative to Q-tables. Q-tables are nice and straightforward, but can only contain a limited amount of action values. Chess has state space complexity of 10<sup>47</sup>. Needless to say, this is too much information to put in a Q-table. This is where supervised learning comes in. A Q-network can represent a generalized mapping from state to action values.
# 
# ![](https://images.chesscomfiles.com/uploads/game-gifs/90px/green/neo/0/cc/0/0/aXFZUWpyN1Brc1BPbHQwS211WEhudkh6cXohMGFPMExPUTJNUTY4MDY1OTI1NFpSND8yOT85M1Y5MTA3MUxLQ3RDUkpDSjcwTE0wN293V0d6Rzc2cHhWTXJ6NlhzQVg0dUM0WGNNWDU,.gif)
# 
# 

# #### Import and Install

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import inspect


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


get_ipython().system('pip install python-chess  # Python-Chess is the Python Chess Package that handles the chess environment')
get_ipython().system('pip install --upgrade git+https://github.com/arjangroen/RLC.git  # RLC is the Reinforcement Learning package')


# In[ ]:


import chess
from chess.pgn import Game
import RLC
from RLC.capture_chess.environment import Board
from RLC.capture_chess.learn import Q_learning
from RLC.capture_chess.agent import Agent


# ### The environment: Capture Chess
# In this notebook we'll upgrade our environment to one that behaves more like real chess. It is mostly based on the Board object from python-chess.
# Some modifications are made to make it easier for the algorithm to converge:
# * There is a maximum of 25 moves, after that the environment resets
# * Our Agent only plays white
# * The Black player is part of the environment and returns random moves
# * The reward structure is not based on winning/losing/drawing but on capturing black pieces:
#     - pawn capture: +1
#     - knight capture: +3
#     - bishop capture: +3
#     - rook capture: +5
#     - queen capture: +9
# * Our state is represent by an 8x8x8 array
#     - Plane 0 represents pawns
#     - Plane 1 represents rooks
#     - Plane 2 represents knights
#     - Plane 3 represents bishops
#     - Plane 4 represents queens
#     - Plane 5 represents kings
#     - Plane 6 represents 1/fullmove number (needed for markov property)
#     - Plane 7 represents can-claim-draw
# * White pieces have the value 1, black pieces are minus 1
#        
# 

# #### Board representation of python-chess:

# In[ ]:


board = Board()
board.board


# #### Numerical representation of the pawns (layer 0)
# Change the index of the first dimension to see the other pieces

# In[ ]:


board.layer_board[0,::-1,:].astype(int)


# ### The Agent
# * The agent is no longer a single piece, it's a chess player
# * Its action space consist of 64x64=4096 actions:
#     * There are 8x8 = 64 piece from where a piece can be picked up
#     * And another 64 pieces from where a piece can be dropped. 
# * Of course, only certain actions are legal. Which actions are legal in a certain state is part of the environment (in RL, anything outside the control of the agent is considered part of the environment). We can use the python-chess package to select legal moves. (It seems that AlphaZero uses a similar approach https://ai.stackexchange.com/questions/7979/why-does-the-policy-network-in-alphazero-work)

# #### Implementation

# In[ ]:


board = Board()
agent = Agent(network='conv',gamma=0.1,lr=0.07)
R = Q_learning(agent,board)
R.agent.fix_model()
R.agent.model.summary()


# In[ ]:


print(inspect.getsource(agent.network_update))


# #### Q learning with a Q-network
# **Theory**
# - The Q-network is usually either a linear regression or a (deep) neural network. 
# - The input of the network is the state (S) and the output is the predicted action value of each Action (in our case, 4096 values). 
# - The idea is similar to learning with Q-tables. We update our Q value in the direction of the discounted reward + the max successor state action value
# - I used prioritized experience replay to de-correlate the updates. If you want to now more about it, check the link in the references
# > - I used fixed-Q targets to stabilize the learning process. 

# #### Implementation
# - I built two networks, A linear one and a convolutional one
# - The linear model maps the state (8,8,8) to the actions (64,64), resulting in over 32k trainable weights! This is highly inefficient because there is no parameter sharing, but it will work.
# - The convolutional model uses 2 1x1 convulutions and takes the outer product of the resulting arrays. This results in only 18 trainable weights! 
#     - Advantage: More parameter sharing -> faster convergence
#     - Disadvantage: Information gets lost -> lower performance
# - For a real chess AI we need bigger neural networks. But now the neural network only has to learn to capture valuable pieces.

# In[ ]:


print(inspect.getsource(R.play_game))


# #### Demo

# In[ ]:


pgn = R.learn(iters=750)


# In[ ]:


reward_smooth = pd.DataFrame(R.reward_trace)
reward_smooth.rolling(window=125,min_periods=0).mean().plot(figsize=(16,9),title='average performance over the last 125 steps')


# The PGN file is exported to the output folder. You can analyse is by pasting it on the [chess.com analysis board](https://www.chess.com/analysis)

# In[ ]:


with open("final_game.pgn","w") as log:
    log.write(str(pgn))


# ## Learned action values analysis
# So what has the network learned? The code below checks the action values of capturing every black piece for every white piece. 
# - We expect that the action values for capturing black pieces is similar to the (Reinfeld) rewards we put in our environment. 
# - Of course the action values also depend on the risk of re-capture by black and the opportunity for consecutive capture. 

# In[ ]:


board.reset()
bl = board.layer_board
bl[6,:,:] = 1/10  # Assume we are in move 10
av = R.agent.get_action_values(np.expand_dims(bl,axis=0))

av = av.reshape((64,64))

p = board.board.piece_at(20)#.symbol()


white_pieces = ['P','N','B','R','Q','K']
black_piece = ['_','p','n','b','r','q','k']

df = pd.DataFrame(np.zeros((6,7)))

df.index = white_pieces
df.columns = black_piece

for from_square in range(16):
    for to_square in range(30,64):
        from_piece = board.board.piece_at(from_square).symbol()
        to_piece = board.board.piece_at(to_square)
        if to_piece:
            to_piece = to_piece.symbol()
        else:
            to_piece = '_'
        df.loc[from_piece,to_piece] = av[from_square,to_square]
        
        


# * ### Learned action values for capturing black (lower case) with white (upper case) pieces.
# Underscore represents capturing an empty square

# In[ ]:


df[['_','p','n','b','r','q']]


# ## References
# Reinforcement Learning: An Introduction  
# > Richard S. Sutton and Andrew G. Barto  
# > 1st Edition  
# > MIT Press, march 1998  
# 
# RL Course by David Silver: Lecture playlist  
# > https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ  
# 
# Experience Replay  
# > https://datascience.stackexchange.com/questions/20535/what-is-experience-replay-and-what-are-its-benefits
