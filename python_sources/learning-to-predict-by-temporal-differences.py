#!/usr/bin/env python
# coding: utf-8

# ## Objective:
# 
# Learning temporal difference and making an implementation of random walk problem and solve it using the method of temporal differences.

# ## Random walk:
# 
# The random walk contains seven states - A, B, C, D, E, F, G. Every walk begins in state D and it will terminate in state A or G. A walk will generate a sequence of states. (ex: DCBCDEFEFG). In each step, the agent can make a move only to it's adjacent state.
# 
# We wish to estimate the probabilities of a walk ending in righmost state, G, given that it is in each of other states.
# 
# A walk's outcome is 0 if it ends in left most state (A) and 1 when it end's in right most state (G).
# 
# ![image.png](attachment:image.png)

# ## Some Q&A:
# **What is the problem addressed here?**
# 
# An agent has to learn to move towards state G. The The answproblem addressed is how can the agent assign a probability from 0-1 about the favourability of every state? You can also refer to figure 2.
# 
# **How is the problem addressed here?**
# 
# The problem is addressed by the method of temporal differences.
# 
# **Why temporal difference?**
# 
# There are **two kind of prediction problems** - single step prediction and multi step prediction.
# 
# An example: weather temperature prediction
# 
# In a single step problem, you have to predict temperature for friday. You look at the temperature on monday and give a prediction for friday. You make item-association pairs ( here temperature on monday and friday ). All information about correctness of prediction is revealed at once.
#     
# As a multi step problem, you have to predict temperature for friday. You look at temperature on monday, make a prediction for friday. Depending on temperatures on tuesday to thursday, you make changes to prediction for friday. On each sub sequent day, you make your prediction better.
# 
# Since this is a multi-step prediction problem, we use the method of temporal differences.
# 
# **When not supervised learning?**
# 
# Supervised learning procedures perform poorly in case of multi-step prediction problems.
# 
# **How TD works?**
# 
# TD learns from difference between temporally successive predictions. A finite amount of training data is presented over and over again until the learning process converges. We follow a weight updation rule which is
# 
# ![image.png](attachment:image.png)

# ## Play Area
# 
# **Generating a walk**

# In[ ]:


def play():
    state='D'
    seq='D'
    while state!='A' and state!='G':
        action=random.choice((-1,1))
        state=chr(ord(state)+action)
        seq+=state
    return (seq)


# **Temporal difference**
# 
# The method used below is **TD(1)**

# In[ ]:


def temporal_difference(walk,probs,alpha=0.005):
    gradient_sum=0
    for i in range(1,len(walk)):
        current_state=walk[i]
        prev_state=walk[i-1]
        gradient=X[prev_state]
        gradient_sum+=gradient
        probs[1:6]=probs[1:6]+alpha*(probs[ord(current_state)-65]-probs[ord(prev_state)-65])*gradient_sum
    return (probs)


# **Main code**

# In[ ]:


import random
import numpy as np

probs=np.array([0,0.5,0.5,0.5,0.5,0.5,1]) #We take an inital weight of 0 for state A, 0.5 for states B,C,D,E,F, 1 for state G.

X={'B':np.array([1,0,0,0,0]),'C':np.array([0,1,0,0,0]),'D':np.array([0,0,1,0,0]),
   'E':np.array([0,0,0,1,0]),'F':np.array([0,0,0,0,1])}

sequences=[]

for i in range(100):
    sequences.append(play())    
    
for walk in sequences:
    probs=temporal_difference(walk,probs)
    
probs=[round(i,2) for i in probs]
print ('Probability of winning from each state is ', probs)    


# ## Reference
# [Learning to predict by method of temporal differences](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
