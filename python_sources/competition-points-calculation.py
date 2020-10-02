#!/usr/bin/env python
# coding: utf-8

# ![](http://)You can find the current formula [here.](http://www.kaggle.com/progression)

# In[ ]:


import numpy as np 


# In[ ]:


def calculate_point(n_teammates,rank,n_teams,t):
    point = (100000/(n_teammates)**0.5)*((rank)**-0.75)*np.log10(1+np.log10(n_teams))* np.exp(-1*t/500) #Point calculation as of 17-04-2019
    return point


# In[ ]:


n_teammates = 5 # number of teammates
rank = 10 # position landed on LB
n_teams = 8800 # number of teams 
t = 0 # t is the number of days elapsed since the point was awarded.


# Team of 5 at #10

# In[ ]:


calculate_point(n_teammates = n_teammates,rank = rank,n_teams = n_teams,t = t)


# Solo competitor at #29

# In[ ]:


calculate_point(n_teammates = 1,rank = 29,n_teams = n_teams,t = t)


# In[ ]:




