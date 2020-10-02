#!/usr/bin/env python
# coding: utf-8

# # It's all wrong!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm_notebook

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


structures = pd.read_csv('../input/structures.csv')

structures


# In[ ]:


xyz = structures[['x','y','z']].values


# In[ ]:


ss = structures.groupby('molecule_name').size()
ss = ss.cumsum()
ss


# In[ ]:


ssx = np.zeros(len(ss) + 1, 'int')
ssx[1:] = ss
ssx


# In[ ]:


molecule_id = 20
print(ss.index[molecule_id])
start_molecule = ssx[molecule_id]
end_molecule = ssx[molecule_id+1]
xyz[start_molecule:end_molecule]


# We can compare with the information we get from the original pandas dataframe

# In[ ]:


structures_idx = structures.set_index('molecule_name')


# In[ ]:


structures_idx.loc['dsgdb9nsd_000022'][['x', 'y', 'z']].values


# Looks good.
# 
# We can now rewrite our function using our arrays.
# 

# In[ ]:


def get_fast_dist_matrix(xyz, ssx, molecule_id):
    start_molecule, end_molecule = ssx[molecule_id], ssx[molecule_id+1]
    locs = xyz[start_molecule:end_molecule]    
    num_atoms = end_molecule - start_molecule
    loc_tile = np.tile(locs.T, (num_atoms,1,1))
    dist_mat = np.sqrt((loc_tile - loc_tile.T)**2).sum(axis=1)
    return dist_mat


# Let's check we get the same result with both techniques.  We use methane, molecule_id=0

# In[ ]:


molecule_id = 0
molecule = ss.index[molecule_id]
print(molecule)
get_fast_dist_matrix(xyz, ssx, molecule_id)


# # Okay, that is the distance matrix for molecule_id==0. Let's check which molecule it is.

# In[ ]:


print(ss.index[molecule_id])


# # As expected, it is dsgdb9nsd_000001. But we know it should be very symmetric with the same distances. Let's calculate it using the structures file.

# In[ ]:


structures.loc[structures['molecule_name']=='dsgdb9nsd_000001']


# # Let's calculate the distances from the Carbon atom to the Hydrogens

# In[ ]:


carbon_x = -0.012698
carbon_y = 1.085804
carbon_z = 0.008001


# In[ ]:


def distance_from_carbon_to_xyz(xyz):
    return np.sqrt( (carbon_x - xyz[0])**2 + (carbon_y - xyz[1])**2 + (carbon_z - xyz[2])**2)


# In[ ]:


structures.loc[structures['molecule_name']=='dsgdb9nsd_000001', ['x','y','z']].apply(distance_from_carbon_to_xyz, 1)


# # We can see the distances are 0 and 1.09 pretty much. Let's check the distance matrix again.

# In[ ]:


molecule_id = 0
molecule = ss.index[molecule_id]
print(molecule)
get_fast_dist_matrix(xyz, ssx, molecule_id)


# # That distance is not there. Therefore the distance computation is wrong.
