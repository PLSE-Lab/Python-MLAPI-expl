#!/usr/bin/env python
# coding: utf-8

# # kNN features
# I'd like to share some of the features of my current kernel with you. 
# I'd be happy to read your comments or suggestions (it's still a py/pandas beginners code).

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

import os
import warnings  
print(os.listdir("../input"))


# # Structures data

# In[ ]:


structures = pd.read_csv('../input/structures.csv') 

# uncomment for debugging 
structures = structures.head(n=100)

structures.head(n=10)


# The nn_feateres() function extracts the atom names, distances and coordinates of k nearest neighbors. I'm using 4 neighbors.
# Because my primary skills are SQL and I'm fairly new to py and pandas I've tried to build the function to use it in a SQL-like "..nn_features() over(partition by molecule_name)" kind of manner. I therefor used pd.transform to pass the indexes of a molecule to the function and look up its atoms. If you know of a better or faster way please let me know.

# In[ ]:



def nn_features(l):
    ''' l: indexed pd.Series of a molecule '''
    
    # number of nearest neighbors +1
    k = 4+1
    
    # lookup coordinates of atoms in molecule 
    x=np.array(structures.loc[l.index,'x'])
    y=np.array(structures.loc[l.index,'y'])
    z=np.array(structures.loc[l.index,'z'])
    coord = np.append(np.append(x,y),z).reshape((l.size,3),order='F')
    
    # NN calculations
    nbrs = NearestNeighbors(n_neighbors=min(len(coord),k), algorithm='ball_tree').fit(coord)
    distances, indices = nbrs.kneighbors(coord)
    
    
    if indices.shape != (1,1):
        # PCA - not relevant for nn, but nice feature anyway
        pca = PCA(n_components=2)
        p=pca.fit_transform(coord)
        
        # NN id and NN distance
        atm = np.pad(indices[:,1:l.size],((0,0),(0, max(0, k-l.size))), 'constant', constant_values=(999, 999))
        dst = np.pad(distances[:,1:l.size], ((0,0),(0,max(0,k-l.size))), 'constant', constant_values=(0, 0))
        
        # LookUps for atom name and x,y,z, default value N/A or 0
        lu = np.append(np.array(structures.loc[l.index,'atom']),np.array('N/A'))
        lu_x = np.append(np.array(structures.loc[l.index,'x']),np.array(0))
        lu_y = np.append(np.array(structures.loc[l.index,'y']),np.array(0))
        lu_z = np.append(np.array(structures.loc[l.index,'z']),np.array(0))
        
        # for each nn look up coordinates and atom name 
        nn_x = np.take(lu_x, atm, mode = 'clip') 
        nn_y = np.take(lu_y, atm, mode = 'clip') 
        nn_z = np.take(lu_z, atm, mode = 'clip') 
        atm = np.take(lu, atm, mode = 'clip')
    else: 
        # in case the molecule contains only 1 atom (e.g. while debugging a small dataset)
        p = np.ones((1, 2))*(999)
        atm = np.ones((1, max(0, k-l.size)))*(999) 
        dst = np.ones((1, max(0, k-l.size)))*(999)
        nn_x = np.ones((1, max(0, k-l.size)))*(999)
        nn_y = np.ones((1, max(0, k-l.size)))*(999)
        nn_z = np.ones((1, max(0, k-l.size)))*(999)
    
    # put together atom names, distances, coordinates of nnearest neighbors and pca
    out = np.append(np.append(np.append(np.append(np.append(atm,dst,axis=1),nn_x, axis=1),nn_y, axis=1),nn_z, axis=1) ,p, axis=1)
    
    return [i for i in out]


# For the hole structures dataset it takes about 12 minutes to calculate the features of 4 nearest neighbors.

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nwarnings.filterwarnings('ignore')\n\nstructures['nearestn'] = structures.groupby('molecule_name')['x'].transform(nn_features)\n\nstructures.head(n=10)\n#11mi 12s")


# Split the list of nn features. (30 sec)

# In[ ]:


get_ipython().run_cell_magic('time', '', "\n# atom name of nn\nstructures['nn_1'] = structures['nearestn'].apply(lambda x: x[0])\nstructures['nn_2'] = structures['nearestn'].apply(lambda x: x[1])\nstructures['nn_3'] = structures['nearestn'].apply(lambda x: x[2])\nstructures['nn_4'] = structures['nearestn'].apply(lambda x: x[3])\n\n# eucledian distances to nn\nstructures['nn_1_dist'] = structures['nearestn'].apply(lambda x: x[4])\nstructures['nn_2_dist'] = structures['nearestn'].apply(lambda x: x[5])\nstructures['nn_3_dist'] = structures['nearestn'].apply(lambda x: x[6])\nstructures['nn_4_dist'] = structures['nearestn'].apply(lambda x: x[7])\n\n# x,y,z distances to nn\nstructures['nn_dx_1'] = structures['nearestn'].apply(lambda x: x[8])  - structures['x']\nstructures['nn_dx_2'] = structures['nearestn'].apply(lambda x: x[9])  - structures['x']\nstructures['nn_dx_3'] = structures['nearestn'].apply(lambda x: x[10])  - structures['x']\nstructures['nn_dx_4'] = structures['nearestn'].apply(lambda x: x[11])  - structures['x']\n\nstructures['nn_dy_1'] = structures['nearestn'].apply(lambda x: x[12])  - structures['y']\nstructures['nn_dy_2'] = structures['nearestn'].apply(lambda x: x[13])  - structures['y']\nstructures['nn_dy_3'] = structures['nearestn'].apply(lambda x: x[14])  - structures['y']\nstructures['nn_dy_4'] = structures['nearestn'].apply(lambda x: x[15])  - structures['y']\n\nstructures['nn_dz_1'] = structures['nearestn'].apply(lambda x: x[16])  - structures['z']\nstructures['nn_dz_2'] = structures['nearestn'].apply(lambda x: x[17])  - structures['z']\nstructures['nn_dz_3'] = structures['nearestn'].apply(lambda x: x[18])  - structures['z']\nstructures['nn_dz_4'] = structures['nearestn'].apply(lambda x: x[19])  - structures['z']\n\n# 2 dim pca\nstructures['pca_x'] = structures['nearestn'].apply(lambda x: x[20])\nstructures['pca_y'] = structures['nearestn'].apply(lambda x: x[21])\n\nstructures = structures.drop(columns='nearestn',axis=0)\nstructures.head(n=10)")


# In[ ]:




