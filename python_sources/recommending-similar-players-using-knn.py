#!/usr/bin/env python
# coding: utf-8

# Here, we are going to analyze the player attributes from the FIFA 19 dataset provided and recommend similar players.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

ds = pd.read_csv('../input/data.csv')

print(ds.head())


# Selecting the attributes from the dataset.
# The attributes selected for analysis are:- 
# * Crossing
# * Finishing
# * HeadingAccuracy
# * ShortPassing
# * Volleys
# * Dribbling
# * Curve
# * FKAccuracy
# * LongPassing
# * BallControl
# * Acceleration
# * SprintSpeed
# * Agility
# * Reactions
# * Balance
# * ShotPower
# * Jumping
# * Stamina
# * Strength
# * LongShots
# * Aggression
# * Interceptions
# * Positioning
# * Vision
# * Penalties
# * Composure
# * Marking
# * StandingTackle
# * SlidingTackle
# * Work Rate
# * Skill Moves

# In[ ]:


#Data Preprocessing
attributes = ds.iloc[:, 54:83]
attributes['Skill Moves'] = ds['Skill Moves']
workrate = ds['Work Rate'].str.get_dummies(sep='/ ')
attributes = pd.concat([attributes, workrate], axis=1)
df = attributes
attributes = attributes.dropna()
df['Name'] = ds['Name']
df = df.dropna()
print(attributes.columns)


# We scale the attribute values since the values of feature for Work Rate is in binary compare to other features.

# In[ ]:


scaled = StandardScaler()
X = scaled.fit_transform(attributes)


# We fit the model with the scaled attributes

# In[ ]:


recommendations = NearestNeighbors(n_neighbors=6, algorithm='ball_tree').fit(X)


# Pass the player name from the dataset to the function and get 5 similar players as output

# In[ ]:


player_indices = recommendations.kneighbors(X)[1]

def get_index(x):
    return df[df['Name']==x].index.tolist()[0]

def recommend_me(player):
    print('Here are 5 players similar to', player, ':' '\n')
    index = get_index(player)
    for i in player_indices[index][1:]:
            print(df.iloc[i]['Name'], '\n')

recommend_me("L. Messi")

