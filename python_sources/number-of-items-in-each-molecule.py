#!/usr/bin/env python
# coding: utf-8

# Number of items of Hydrogen (H), Carbon(C) an Nitrogen(N) are playing role in deciding bond or scalr_coupling_constant for each molecule.
# I have tried here simple approach to find out count of each item in each molecule.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_original = pd.read_csv("../input/train.csv")
structures_original = pd.read_csv("../input/structures.csv")
test_original = pd.read_csv("../input/test.csv")


# In[ ]:


train_original.head()


# In[ ]:


structures_original.head()


# Lets see how many items are there in dsgdb9nsd_000015
# #

# In[ ]:


structures_original[structures_original['molecule_name'] == 'dsgdb9nsd_000015']


# As per above there are total 9 items in molecule.
# 
# Carbon - 2
# 
# Oxygen - 1
# 
# Hydrogen - 6

# Total items in each molecule can be calculated by simply grouping structures_original dataframe by molecule_name ans atom with count as a aggregate function

# In[ ]:


moleculeCount = structures_original.groupby(by=['molecule_name','atom'])[['atom']].count()
moleculeCount.rename(columns={'atom':'count'},inplace = True)
moleculeCount = moleculeCount.unstack(fill_value=0)
moleculeCount = moleculeCount['count'].reset_index()

moleculeCount.head()


# In[ ]:


moleculeCount[moleculeCount['molecule_name'] == 'dsgdb9nsd_000015']


# Merge structures_original and moleculeCount in single dataframe

# In[ ]:


structures = pd.DataFrame.merge(structures_original,moleculeCount
                               ,how='inner'
                               ,left_on = ['molecule_name'] 
                               ,right_on = ['molecule_name']
                              )

structures.head()


# Join structures dataframe with train and test data to include item counts in train and test data.
# 
# Also, I am using below kernel to calculate distance between 2 items in a molecule.
#  
# https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark
# 
# The Frobenius norm is given by [1]:
# ||A||F = [\sum{i,j} abs(a_{i,j})^2]^{1/2}

# In[ ]:


tmp_merge = pd.DataFrame.merge(train_original,structures
                               ,how='left'
                               ,left_on = ['molecule_name','atom_index_0'] 
                               ,right_on = ['molecule_name','atom_index']
                              )

tmp_merge = tmp_merge.merge(structures
                ,how='left'
                ,left_on = ['molecule_name','atom_index_1'] 
                ,right_on = ['molecule_name','atom_index']
               )

tmp_merge.drop(columns=['atom_index_x','atom_index_y','C_x','F_x','H_x','N_x','O_x'],inplace=True)
tmp_merge.columns = ['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type' , 'scalar_coupling_constant' , 
                      'atom_nm_0' , 'x_0' , 'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O']

train = tmp_merge[['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type'  , 'atom_nm_0' , 'x_0' ,
           'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O', 'scalar_coupling_constant']]
train.sort_values(by=['id','molecule_name'],inplace=True)
train.reset_index(inplace=True,drop=True)

tmp_merge = None

train['dist'] = np.linalg.norm(train[['x_0', 'y_0', 'z_0']].values - train[['x_1', 'y_1', 'z_1']].values, axis=1)
train.drop(columns=['x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1'],inplace=True)

train.head()


# In[ ]:


tmp_merge = pd.DataFrame.merge(test_original,structures
                               ,how='left'
                               ,left_on = ['molecule_name','atom_index_0'] 
                               ,right_on = ['molecule_name','atom_index']
                              )

tmp_merge = tmp_merge.merge(structures
                ,how='left'
                ,left_on = ['molecule_name','atom_index_1'] 
                ,right_on = ['molecule_name','atom_index']
               )

tmp_merge.drop(columns=['atom_index_x','atom_index_y','C_x','F_x','H_x','N_x','O_x'],inplace=True)
tmp_merge.columns = ['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type' , 
                      'atom_nm_0' , 'x_0' , 'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O']

test = tmp_merge[['id' , 'molecule_name' , 'atom_0' , 'atom_1' , 'type'  , 'atom_nm_0' , 'x_0' ,
           'y_0' , 'z_0' , 'atom_nm_1' , 'x_1' , 'y_1' , 'z_1','C','F','H','N','O']]
test.sort_values(by=['id','molecule_name'],inplace=True)
test.reset_index(inplace=True,drop=True)

tmp_merge = None

test['dist'] = np.linalg.norm(test[['x_0', 'y_0', 'z_0']].values - test[['x_1', 'y_1', 'z_1']].values, axis=1)
test.drop(columns=['x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1'],inplace=True)

test.head()

