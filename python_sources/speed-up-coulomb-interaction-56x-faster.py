#!/usr/bin/env python
# coding: utf-8

# This kernel is inspired by *rio114* kernel http://https://www.kaggle.com/rio114/coulomb-interaction
# > get distances from each atom belonging to the molecule and pickup 'num = 5' nearest regarding to each atom [H, C, N, O, F]
# 
# I try to utilize pandas to calculate those interesting features instead of looping per molecule.
# 
# It speed up 56x faster from **216.16 s ** to **3.84 s** for first 100 molecules

# ## Load the data

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime as dt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

import warnings
warnings.simplefilter('ignore')


# In[ ]:


FOLDER = '../input/'
OUTPUT = '../preprocessed/'
os.listdir(FOLDER)


# In[ ]:


df_structures = pd.read_csv(os.path.join(FOLDER, 'structures.csv'))


# ## Compute Distance

# I use self join instead of distance matrix to get the distance of atoms among molecule because I want to avoid looping per molecule
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', "df_distance = df_structures.merge(df_structures, how = 'left', on= 'molecule_name', suffixes = ('_0', '_1'))\n# remove same molecule\ndf_distance = df_distance.loc[df_distance['atom_index_0'] != df_distance['atom_index_1']]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_distance['distance'] = np.linalg.norm(df_distance[['x_0','y_0', 'z_0']].values - \n                                           df_distance[['x_1', 'y_1', 'z_1']].values, axis=1, ord = 2)")


# In[ ]:


df_distance.head(10)


# We can use ```df_distance``` to join to train and test dataset 

# ## Interaction Data Frame

# In[ ]:


def get_interaction_data_frame(df_distance, num_nearest = 5):
    time_start = dt.datetime.now()
    print("START", time_start)
    
    # get nearest 5 (num_nearest) by distances
    df_temp = df_distance.groupby(['molecule_name', 'atom_index_0', 'atom_1'])['distance'].nsmallest(num_nearest)
    
    # make it clean
    df_temp = pd.DataFrame(df_temp).reset_index()[['molecule_name', 'atom_index_0', 'atom_1', 'distance']]
    df_temp.columns = ['molecule_name', 'atom_index', 'atom', 'distance']
    
    time_nearest = dt.datetime.now()
    print("Time Nearest", time_nearest-time_start)
    
    # get rank by distance
    df_temp['distance_rank'] = df_temp.groupby(['molecule_name', 'atom_index', 'atom'])['distance'].rank(ascending = True, method = 'first').astype(int)
    
    time_rank = dt.datetime.now()
    print("Time Rank", time_rank-time_nearest)
    
    # pivot to get nearest distance by atom type 
    df_distance_nearest = pd.pivot_table(df_temp, index = ['molecule_name','atom_index'], columns= ['atom', 'distance_rank'], values= 'distance')
    
    time_pivot = dt.datetime.now()
    print("Time Pivot", time_pivot-time_rank)
    del df_temp
    
    columns_distance_nearest =  np.core.defchararray.add('distance_nearest_', 
                                          np.array(df_distance_nearest.columns.get_level_values('distance_rank')).astype(str) +  
                                          np.array(df_distance_nearest.columns.get_level_values('atom')) )
    df_distance_nearest.columns = columns_distance_nearest
    
    # 1 / r^2 to get the square inverse same with the previous kernel
    df_distance_sq_inv_farthest = 1 / (df_distance_nearest ** 2)
    
    columns_distance_sq_inv_farthest = [col.replace('distance_nearest', 'distance_sq_inv_farthest') for col in columns_distance_nearest]

    df_distance_sq_inv_farthest.columns = columns_distance_sq_inv_farthest
    time_inverse = dt.datetime.now()
    print("Time Inverse Calculation", time_inverse-time_pivot)
    
    df_interaction = pd.concat([df_distance_sq_inv_farthest, df_distance_nearest] , axis = 1)
    df_interaction.reset_index(inplace = True)
    
    time_concat = dt.datetime.now()
    print("Time Concat", time_concat-time_inverse)
    
    return df_interaction


# In[ ]:


first_100_molecules = df_structures['molecule_name'].unique()[:100]


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_interaction = get_interaction_data_frame(df_distance.loc[df_distance['molecule_name'].isin(first_100_molecules)])")


# Computing **N smallest object** is take a lot of time compare to others step. 
# 
# We need to tackle that compuation, because it still to long if we want to run for all molecule

# In[ ]:


df_interaction.head(20)


# The result is same with http://https://www.kaggle.com/rio114/coulomb-interaction 
# just need to join to train dataset

# Hope This is helps for our feature engineering
# 
# Cheers :D  
