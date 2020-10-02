#!/usr/bin/env python
# coding: utf-8

# # SIMPLE DATA EXPLORATION 
# [A Santosh Kumar](https://github.com/santoshk17) - May 2018

# In[54]:


import os
import pandas as pd
 


# In[55]:


path = '../input/train_1/'
train_events = os.listdir(path)


# In[56]:


print('# File sizes')
for f in os.listdir('../input'):
    if 'zip' not in f:
        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')


# In[57]:


print('# File sizes')
count = 0
for f in os.listdir('../input/train_1/'):
    if 'zip' not in f:
        if (count < 20):
           print(f.ljust(30) + str(round(os.path.getsize('../input/train_1/' + f) / 1000000, 2)) + 'MB')
           count = count + 1


# In[58]:


len(train_events)


# 
# 
# Each event has four associated files
# 1. hits
# 2. cells
# 3. particles
# 4. truth
# 
# The files of a particular event has prefix, e.g. event000000010, i.e always **event **followed by 9 digits. 

# In[63]:


hits_df = pd.read_csv(path+'event000002560-hits.csv')
cells_df = pd.read_csv(path+'event000002560-cells.csv')
particles_df = pd.read_csv(path+'event000002560-particles.csv')
truth_df = pd.read_csv(path+'event000002560-truth.csv')


# In[64]:


hits_df.head(5)


# In[65]:


cells_df.head(5)


# In[66]:


truth_df.head(5)


# In[67]:


particles_df.head(5)


# In[ ]:





# In[ ]:




