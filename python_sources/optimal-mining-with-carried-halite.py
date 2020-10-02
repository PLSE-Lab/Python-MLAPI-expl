#!/usr/bin/env python
# coding: utf-8
A more advanced algorithm takes into account carried halite
# In[ ]:


import numpy as np       # matrices
import scipy.optimize    # optimization routines
import matplotlib.pyplot as plt
import math
import pprint


# ## PROBLEM 1
# If you mine a cell containing H halite for m turns, what is the total amount mined?
# 
# **ANSWER**
# Since 25% is mined at each step, 75% remains.  Thus $.75^m H$ remains after m steps mining, and $(1-0.75^m) H$ is mined by the ship.  Note: since the halite increases by 2% each square, this can be modified to account for that; it is a small adjustment and we will ignore it for the time being.
# $$ \text{Total Mined}=(1-.75^m) H $$
# 
# 
# ## PROBLEM 2 
# A ship can travel to a square in $n_1$ steps, mine the halite there (at 25% each step) and then return to a shipyard that is $n_2$ steps from the halite cell.  What is the correct number of steps to mine in order to maximize the overall halite per step?  NEW: Ship is carrying C halite.
# 
# 
# 
# 
# 

# **ANSWER**
# 
# The amount of halite mined per step R is therefore 
# $$R(n_1,n_2,m,H,C)=\frac{C+(1-.75^m) H}{n_1+n_2+m}$$

# In[ ]:


def R(n1,n2,m,H,C=0):
    return (C+(1-.75**m)*H)/(n1+n2+m)

H=500
C=650
n1=5
n2=0
r=[]
for C in [0,300,650]:
    r=[]
    for m in range(14):
       r.append(R(n1,n2,m,H,C=C))
    plt.plot(r,label='C/H={}'.format(C/H))
plt.title('For total travel = {} steps, H={}'.format(n1+n2,H))
plt.xlabel('steps mining')
plt.ylabel('halite per step')
plt.legend()


# We can use scipy.optimize to calculate the maximum point on this graph for various travel distances.  A couple of things to note: 
# 1. Only the total travel distance $n_1+n_2$ matters, not the individual distances.
# 1. The amount of halite in the cell does not affect the optimal number of steps, only the resulting average halite per step
# 
# Therefore, for each total number of travel steps, we can compute the optimal number of steps, using the scipy.optimize function:
# 

# In[ ]:


fig=plt.figure(1)
H=500
for CHratio in [0,.2,.45,.75,1.1,1.7,2.5,3.7]:
    opt=[]
    for travel in range(21):
        def h(mine):
            return -R(0,travel,mine,500,C=CHratio*H)  # we put - because we want to maximize R
        res=scipy.optimize.minimize_scalar(h, bounds=(0,15),method='Bounded')
        opt.append(res.x)
    plt.plot(opt,label='C/H={}'.format(CHratio))    
plt.xlabel('total travel steps')
plt.ylabel('mining steps')
plt.legend()
plt.title('Optimal steps for mining by total travel')


# In[ ]:


#Now do it with rounded off integers
fig=plt.figure(1)
H=500
#integer ch = 2.5 ln(C/H) + 5, so each 1.5x change gives increment of 1
chrange=11
maxsteps=21
matrix=np.zeros((chrange,maxsteps))  # turn into matrix when done
for ch in range(chrange):
    if ch==0:
        CHratio=0
    else:
        CHratio=math.exp((ch-5)/2.5)
    opt=[]
    for travel in range(maxsteps):
        def h(mine):
            return -R(0,travel,mine,500,C=CHratio*H)  # we put - because we want to maximize R
        res=scipy.optimize.minimize_scalar(h, bounds=(0,15),method='Bounded')
        opt.append(int(res.x+.5))
        matrix[ch,travel]=int(res.x+.5)
    plt.plot(opt,label='C/H={:6.2f}'.format(CHratio))    
plt.xlabel('total travel steps')
plt.ylabel('mining steps')
plt.legend()
plt.title('Optimal steps for mining by total travel')
#write out the numbers in a matrix for importing into agent
pprint.pprint(repr(np.array(matrix).astype(np.int)))


# So you can see that as a cell is farther away, you want to spend more time mining it, because you invested more effort into getting there.
# Rather than run this optimization each time you want to determine the value of mining a cell, we can build a simple lookup function.  Since we can't mine for fractional steps, we just round off to the nearest integer.  The matrix output above can be used by looking up in it matrix[].

# Note that because of the wrap around nature of the game board, the farthest away a cell can be is 10 steps vertically and 10 steps horizontally for a total of 20 steps maximum (by the manhattan distance).
