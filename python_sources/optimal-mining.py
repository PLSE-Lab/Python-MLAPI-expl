#!/usr/bin/env python
# coding: utf-8

# We will work out how to compare going to various cells containing halite and determine which is the optimal one to go to.
# NOTE: update July 3, 2020:  This new notebook does the calculation including carried halite by ships:  https://www.kaggle.com/solverworld/optimal-mining-with-carried-halite
# 

# In[ ]:


import numpy as np       # matrices
import scipy.optimize    # optimization routines
import matplotlib.pyplot as plt


# ## PROBLEM 1
# If you mine a cell containing H halite for m turns, what is the total amount mined?
# 
# **ANSWER**
# Since 25% is mined at each step, 75% remains.  Thus $.75^m H$ remains after m steps mining, and $(1-0.75^m) H$ is mined by the ship.  Note: since the halite increases by 2% each square, this can be modified to account for that; it is a small adjustment and we will ignore it for the time being.
# $$ \text{Total Mined}=(1-.75^m) H $$
# 
# 
# ## PROBLEM 2 
# A ship can travel to a square in $n_1$ steps, mine the halite there (at 25% each step) and then return to a shipyard that is $n_2$ steps from the halite cell.  What is the correct number of steps to mine in order to maximize the overall halite per step?
# 
# 
# 
# 
# 

# **ANSWER**
# 
# The amount of halite mined per step R is therefore 
# $$R(n_1,n_2,m,H)=\frac{(1-.75^m) H}{n_1+n_2+m}$$

# In[ ]:


def R(n1,n2,m,H):
    return (1-.75**m)*H/(n1+n2+m)

H=500
n1=10
n2=10
r=[]
for m in range(14):
   r.append(R(n1,n2,m,H))

plt.title('For total travel = 20 steps')
plt.xlabel('steps mining')
plt.ylabel('halite per step')
plt.plot(r)


# We can use scipy.optimize to calculate the maximum point on this graph for various travel distances.  A couple of things to note: 
# 1. Only the total travel distance $n_1+n_2$ matters, not the individual distances.
# 1. The amount of halite in the cell does not affect the optimal number of steps, only the resulting average halite per step
# 
# Therefore, for each total number of travel steps, we can compute the optimal number of steps, using the scipy.optimize function:
# 

# In[ ]:


opt=[]
fig=plt.figure(1)
for travel in range(30):
    def h(mine):
        return -R(0,travel,mine,500)  # we put - because we want to maximize R
    res=scipy.optimize.minimize_scalar(h, bounds=(1,15),method='Bounded')
    opt.append(res.x)
plt.plot(opt)    
plt.xlabel('total travel steps')
plt.ylabel('mining steps')
plt.title('Optimal steps for mining by total travel')


# So you can see that as a cell is farther away, you want to spend more time mining it, because you invested more effort into getting there.
# Rather than run this optimization each time you want to determine the value of mining a cell, we can build a simple lookup function.  Since we can't mine for fractional steps, we just round off to the nearest integer:

# In[ ]:


def num_turns_to_mine(rt_travel):
  #given the number of steps round trip, compute how many turns we should plan on mining
  if rt_travel <= 1:
    return 2
  if rt_travel <= 2:
    return 3
  elif rt_travel <=4:
    return 4
  elif rt_travel <=7:
    return 5
  elif rt_travel <=12:
    return 6
  elif rt_travel <=19:
    return 7
  elif rt_travel <= 28:
    return 8
  else:
    return 9

ints=[]
for travel in range(30):
    ints.append(num_turns_to_mine(travel))
plt.plot(opt,label='exact')
plt.plot(ints, label='int approx')
plt.legend()


# Note that because of the wrap around nature of the game board, the farthest away a cell can be is 10 steps vertically and 10 steps horizontally for a total of 20 steps maximum (by the manhattan distance).

# 

# ## PROBLEM 3
# Compare various potential targets to see which cell is the most productive to mine, based on maximum halite mined per step.
# 
# **ANSWER**
# This is a straight forward application of the previous results.  For example:
# 
# 

# In[ ]:


def best_cell(data):
  #given a list of (travel, halite) tuples, determine the best one
  halite_per_turn=[]
  for t,h in data:
    halite_per_turn.append(R(0,t,num_turns_to_mine(t),h))
  mx=max(halite_per_turn)
  idx=halite_per_turn.index(mx)
  mine=num_turns_to_mine(data[idx][0])
  print('best cell is {} for {:6.1f} halite per step, mining {} steps'.format(idx,mx,mine))
data=[(5,200), (7,190), (10,300),(12,500)]                    
best_cell(data)

