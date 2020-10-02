#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning Chess
# Hi there! If you're interested in learning about reinforcement learning, you are in the right place. As we all know the best way to learn about a topic is to build something and make a kernel about it. My plan is to make a series of notebooks where I work may way towards a full-fledged chess AI named RLC (Reinforcement Learning Chess). 
# 
# Tackling chess is a big challenge, mainly because of its huge state-space. Therefore I start with simpler forms of chess and solve these problems with elementary RL-techniques. Gradually I will expand this untill we end up in a chess AI that can play actual games of chess somewhat intelligibly. The forms of chess I want to cover in my notebooks are:  
# 
# #### 1. Move Chess 
# - Goal: Learn to find the shortest path between 2 squares on a chess board  
# - Motivation: Move Chess has a small statespace, which allows us to tackle this with simple RL algorithms.
# - Concepts: Dynamic Programming, Policy Evaluation, Policy Improvement, Policy Iteration, Value Iteration, Synchronous & Asynchronous back-ups, Monte Carlo (MC) Prediction, MC Control, Temporal Difference (TD) Learning, TD control, TD-lambda, SARSA(-max)
# 
# #### 2. Capture Chess
# - Goal: Capture as many pieces from the opponent within n fullmoves
# - Motivation: Piece captures happen more frequently than win-lose-draw events. This give the algorithm more information to learn from.
# - Concepts: Q-learning, value function approximation, experience replay, fixed-q-targets, policy gradients, REINFORCE, actor-critic
# 
# 
# #### 3. Real Chess (a.k.a. chess)
# - Goal: Play chess competitively against a human beginner
# - Motivation: A RL chess AI
# - Concepts: Monte Carlo Tree Search
# 
# #### Other notebooks
# [**Notebook 2: Model free learning**](https://www.kaggle.com/arjanso/reinforcement-learning-chess-2-model-free-methods)  
# [**Notebook 3: Q-networks**](https://www.kaggle.com/arjanso/reinforcement-learning-chess-3-q-networks)  
# [**Notebook 4: Policy Gradients**](https://www.kaggle.com/arjanso/reinforcement-learning-chess-4-policy-gradients)  
# [**Notebook 5: Monte Carlo Tree Search**](https://www.kaggle.com/arjanso/reinforcement-learning-chess-5-tree-search)
# 
# 
# In my notebooks, I will describe and reference the Reinforcement Learning theory but I will not fully explain it. For that there are resources available that do a match better job at explaining RL than I could. For that my advice would be to check out David Silver's (Deepmind) lectures that are available on Youtube and the book Introduction to Reinforcement Learning by Sutton and Barto referenced below.

# # Notebook I: Solving Move Chess

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import inspect


# In[ ]:


get_ipython().system('pip install python-chess  # Python-Chess is the Python Chess Package that handles the chess environment')
get_ipython().system('pip install --upgrade git+https://github.com/arjangroen/RLC.git  # RLC is the Reinforcement Learning package')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


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


# # 1.1 State Evaluation

# **Theory**
# 
# If we want our agent to optimize its rewards, we want its policy to guide behavior towards the states with the highest value. This value can be estimated using bootstrapping:
# * A state (s) is as valuable (V) as the successor state (s') plus the reward (R) for going from s to s'. 
# * Since there can be mulitple actions (a) and multiple successor states they are summed and weighted by their probability (pi). 
# * In a non-deterministic environment, a given action could result in multiple successor states. We don't have to take this into account for this problem because move chess is a deterministic game.
# * Successor state values are discounted with discount factor (gamma) that varies between 0 and 1.  
# * This gives us the following formula:  
# ![](http://incompleteideas.net/book/ebook/numeqnarraytmp7-2-2.png)  
# 
# Note that:
# * The successor state value is also en estimate. 
# * Evaluating a state is bootstrapping because you are making an estimate based on another estimate
# * In the code you'll see a synchronous parameter that will be explained later in the policy evaluation section
# 
# 

# **Python Implementation**

# In[ ]:


print(inspect.getsource(r.evaluate_state))


# **Demonstration**
# * The initial value function assigns value 0 to each state
# * The initial policy gives an equal probability to each action
# * We evaluate state (0,0)

# In[ ]:


r.agent.value_function.astype(int)


# In[ ]:


state = (0,0)
r.agent.value_function[0,0] = r.evaluate_state(state,gamma=1)


# In[ ]:


r.agent.value_function.astype(int)


# # 1.2 Policy Evaluation
# * Policy evaluation is the act of doe state evaluation for each state in the statespace
# * As you can see in my implementatin I simply iterate over all state and update the value function
# * This is the algorithm provided by Sutton and Barto:  
# ![](http://incompleteideas.net/book/ebook/pseudotmp0.png)
# 

# In[ ]:


print(inspect.getsource(r.evaluate_policy))


# In[ ]:


r.evaluate_policy(gamma=1)


# We end up with the following value of -1 for all states except the terminal state. 

# In[ ]:


r.agent.value_function.astype(int)


# We can iterate this until the value function is stable:

# **Demonstration**

# In[ ]:


eps=0.1
k_max = 1000
value_delta_max = 0
gamma = 1
synchronous=True
value_delta_max = 0
for k in range(k_max):
    r.evaluate_policy(gamma=gamma,synchronous=synchronous)
    value_delta = np.max(np.abs(r.agent.value_function_prev - r.agent.value_function))
    value_delta_max = value_delta
    if value_delta_max < eps:
        print('converged at iter',k)
        break


# This value function below shows the expected discounted future reward from state (0,0) = -185

# In[ ]:


r.agent.value_function.astype(int)


# # Policy Improvement

# Now that we know what the values of the states are, we want to improve our Policy so that we the behavior is guided towards the state with the highest value. Policy Improvement is simply the act of making the policy greedy with respect to the value function.
# * In my implementation, we do this by setting the value of the action that leads to the most valuable state to 1 (while the rest remains 0)

# In[ ]:


print(inspect.getsource(r.improve_policy))


# In[ ]:


r.improve_policy()
r.visualize_policy()


# * Please note that my visual can print only 1 arrow per square, but there may be multiple optimal actions.

# # 1.3 Policy Iteration  
# **Theory**  
# We can now find the optimal policy by doing policy evaluation and policy improvement untill the policy is stable:
# ![](http://www.incompleteideas.net/book/first/ebook/pseudotmp1.png)

# **Python implementation**

# In[ ]:


print(inspect.getsource(r.policy_iteration))


# **Demonstration**

# In[ ]:


r.policy_iteration()


# # 1.4 Asynchronous Policy Iteration
# 

# **Theory**  
# With policy evaluation, we bootstrap: we make an estimate based on another estimate. So which estimate do we take? We have to options:
# 1. We bootstrap from the previous policy evaluation. This means each state value estimate update is based on the same iteration of policy evaluation. This is called synchronous policy iteration
# 2. We bootstrap from the freshest estimate. This means a estimate update can be based on the previous or the current value funtion, or a combination of the two. This is called asynchrronous policy iteration
# 
# The **Implementation** is the same as policy iteration, only we pass the argument sychronous=False

# **Demonstration**

# In[ ]:


agent = Piece(piece='king')
r = Reinforce(agent,env)


# In[ ]:


r.policy_iteration(gamma=1,synchronous=False)


# In[ ]:


r.agent.value_function.astype(int)


# # 1.5 Value Iteration

# ** Theory **  
# Value iteration is nothing more than a simple parameter modification to policy iteration. Remember that policy iteration consists of policy evaluation and policy improvement. The policy evaluation step does not necessarily have to be repeated until convergence before we improve our policy. Recall that the policy iteration above took over 400 iterations to converge. If we use ony 1 iteration instead we call it value iteration.

# **Demonstration**

# In[ ]:


agent = Piece(piece='rook')  # Let's pick a rook for a change.
r = Reinforce(agent,env)
r.policy_iteration(k=1,gamma=1)  # The only difference here is that we set k_max to 1.


# # That's all!
# In the next notebook I'll cover model-free methods such as Monte Carlo and Temporal Difference based methods. These methods help us when we don't know the transition probalities of a Markov Decision Process. 
# 
# I expect to have my second RLC notebook up and running around mid-june!
# Hope you enjoyed!

# # References

# 1. Reinforcement Learning: An Introduction  
#    Richard S. Sutton and Andrew G. Barto  
#    1st Edition  
#    MIT Press, march 1998
# 2. RL Course by David Silver: Lecture playlist  
#    https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ

# In[ ]:




