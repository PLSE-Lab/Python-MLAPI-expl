#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning Chess 
# Reinforcement Learning Chess is a series of notebooks where I implement Reinforcement Learning algorithms to develop a chess AI. I start of with simpler versions (environments) that can be tackled with simple methods and gradually expand on those concepts untill I have a full-flegded chess AI. 
# 
# [**Notebook 1: Policy Iteration**](https://www.kaggle.com/arjanso/reinforcement-learning-chess-1-policy-iteration)  
# [**Notebook 3: Q-networks**](https://www.kaggle.com/arjanso/reinforcement-learning-chess-3-q-networks)  
# [**Notebook 4: Policy Gradients**](https://www.kaggle.com/arjanso/reinforcement-learning-chess-4-policy-gradients)  
# [**Notebook 5: Monte Carlo Tree Search**](https://www.kaggle.com/arjanso/reinforcement-learning-chess-5-tree-search)  

# # Notebook II: Model-free control
# In this notebook I use the same move-chess environment as in notebook 1. In this notebook I mentioned that policy evaluation calculates the state value by backing up the successor state values and the transition probabilities to those states. The problem is that these probabilities are usually unknown in real-world problems. Luckily there are control techniques that can work in these unknown environments. These techniques don't leverage any prior knowledge about the environment's dynamics, they are model-free.

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import inspect


# In[ ]:


get_ipython().system('pip install --upgrade git+https://github.com/arjangroen/RLC.git  # RLC is the Reinforcement Learning package')


# In[ ]:


from RLC.move_chess.environment import Board
from RLC.move_chess.agent import Piece
from RLC.move_chess.learn import Reinforce


# ### The environment
# - The state space is a 8 by 8 grid
# - The starting state S is the top-left square (0,0)
# - The terminal state F is square (5,7). 
# - Every move from state to state gives a reward of minus 1
# - Naturally the best policy for this evironment is to move from S to F in the lowest amount of moves possible.

# In[ ]:


env = Board()
env.render()
env.visual_board


# ### The agent
# - The agent is a chess Piece (king, queen, rook, knight or bishop)
# - The agent has a behavior policy determining what the agent does in what state

# In[ ]:


p = Piece(piece='king')


# ### Reinforce
# - The reinforce object contains the algorithms for solving move chess
# - The agent and the environment are attributes of the Reinforce object

# In[ ]:


r = Reinforce(p,env)


# # 2.1 Monte Carlo Control

# **Theory**  
# The basic intuition is:
# * We do not know the environment, so we sample an episode from beginning to end by running our current policy
# * We try to estimate the action-values rather than the state values. This is because we are working model-free so just knowning state values won't help us select the best actions. 
# * The value of a state-action value is defined as the future returns from the first visit of that state-action
# * Based on this we can improve our policy and repeat the process untill the algorithm converges
# 
# ![](http://incompleteideas.net/book/first/ebook/pseudotmp5.png)

# **Implementation**

# In[ ]:


print(inspect.getsource(r.monte_carlo_learning))


# **Demo**  
# We do 100 iterations of monte carlo learning while maintaining a high exploration rate of 0.5:

# In[ ]:


for k in range(100):
    eps = 0.5
    r.monte_carlo_learning(epsilon=eps)


# In[ ]:


r.visualize_policy()


# Best action value for each state:

# In[ ]:


r.agent.action_function.max(axis=2).astype(int)


# # 2.2 Temporal Difference Learning 

# **Theory**
# * Like Policy Iteration, we can back up state-action values from the successor state action without waiting for the episode to end. 
# * We update our state-action value in the direction of the successor state action value.
# * The algorithm is called SARSA: State-Action-Reward-State-Action.
# * Epsilon is gradually lowered (the GLIE property)

# **Implementation**

# In[ ]:


print(inspect.getsource(r.sarsa_td))


# **Demonstration**

# In[ ]:


p = Piece(piece='king')
env = Board()
r = Reinforce(p,env)
r.sarsa_td(n_episodes=10000,alpha=0.2,gamma=0.9)


# In[ ]:


r.visualize_policy()


# # 2.3 TD-lambda
# **Theory**  
# In Monte Carlo we do a full-depth backup while in Temporal Difference Learning we de a 1-step backup. You could also choose a depth in-between: backup by n steps. But what value to choose for n?
# * TD lambda uses all n-steps and discounts them with factor lambda
# * This is called lambda-returns
# * TD-lambda uses an eligibility-trace to keep track of the previously encountered states
# * This way action-values can be updated in retrospect

# **Implementation**

# In[ ]:


print(inspect.getsource(r.sarsa_lambda))


# **Demonstration**

# In[ ]:


p = Piece(piece='king')
env = Board()
r = Reinforce(p,env)
r.sarsa_lambda(n_episodes=10000,alpha=0.2,gamma=0.9)


# In[ ]:


r.visualize_policy()


# # 2.4 Q-learning

# **Theory**
# * In SARSA/TD0, we back-up our action values with the succesor action value
# * In SARSA-max/Q learning, we back-up using the maximum action value. 

# **Implementation**

# In[ ]:


print(inspect.getsource(r.sarsa_lambda))


# **Demonstration**

# In[ ]:


p = Piece(piece='king')
env = Board()
r = Reinforce(p,env)
r.q_learning(n_episodes=1000,alpha=0.2,gamma=0.9)


# In[ ]:


r.visualize_policy()


# In[ ]:


r.agent.action_function.max(axis=2).round().astype(int)


# # References
# 1. Reinforcement Learning: An Introduction  
#    Richard S. Sutton and Andrew G. Barto  
#    1st Edition  
#    MIT Press, march 1998
# 2. RL Course by David Silver: Lecture playlist  
#    https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ

# In[ ]:




