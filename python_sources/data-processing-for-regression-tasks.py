#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This Kernel is simple data processing example of [League Of Legends high elo Ranked Games Data Set](https://www.kaggle.com/gyejr95/league-of-legends-challenger-ranked-games2020).
# 
# You can use this kernel to make a model to predict win/loss of game.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ### Load Data

# In[ ]:


dir = '/kaggle/input/league-of-legends-challenger-ranked-games2020/'
ranks = ['Challenger', 'GrandMaster', 'Master']

data = pd.DataFrame()

for rank in ranks:
    data_in = pd.read_csv(dir+rank+'_Ranked_Games.csv')
    data_in['Rank'] = rank
    print("Size of {}: {}".format(rank,data_in.shape))
    data = data.append(data_in, ignore_index=True)
    
print("Total size: {}".format(data.shape))


# In[ ]:


data.head()


# # Data Process for Regression Tasks
# Let's Process Data for Regression Task.

# # Remove duplicate column & gameId
# We don't need duplicate Columns like `bludWins` and `redWins` since there is no draw in LOL.   
# Let's remove all of these duplicate data by removing red team's data.
# 
# But we should careful for some column! There can be ***no baron slained*** games. So We should keep some columns like `FirstBaron`, `FirstDragon` and `FirstInhibitor` data.

# In[ ]:


data = data.drop('redWins', axis=1)
data = data.drop('redFirstBlood', axis=1)
data = data.drop('redFirstTower', axis=1)
data = data.drop('gameId', axis=1)


# In[ ]:


data.head()


# # Shuffle Data
# Currently, Data is aligned by Rank. So we should shuffle the data.

# In[ ]:


data = data.sample(frac=1).reset_index(drop=True)


# In[ ]:


data.head()


# # Devide data as X and Y (Label)

# In[ ]:


y_data = data['blueWins']
x_data = data.drop('blueWins', axis=1)
y_data.head()


# # Choose How to deal with Rank
# You can choose how to deal with column `Rank`
# 
# 1. Remove(Ignore) Rank Column
# 2. Keep Rank Column

# ## Remove Rank

# In[ ]:


# x_data = x_data.drop('Rank', axis=1)


# ## Keep Rank
# You should encode Rank by One-Hot.

# In[ ]:


x_data = pd.get_dummies(x_data)
x_data.head()


# # Devide data into Train, Test Sets
# We have 199925 data rows. Let's devide them into Train, Test.

# In[ ]:


x_train = x_data[:180000]
y_train = y_data[:180000]
print(x_train.shape)
print(y_train.shape)


# In[ ]:


x_test = x_data[180000:]
y_test = y_data[180000:]
print(x_test.shape)
print(y_test.shape)

