#!/usr/bin/env python
# coding: utf-8

# # Data exploration 
# 
# In this notebook we will review the data contained in the NBV classification dataset

# In[ ]:


import numpy as np
from random import randint
import os
import matplotlib.pyplot as plt

import classification_utils as cnbv


# In[ ]:


# configure the dataset address

dataset_folder = '../input/nbv-classification/classification/classification/training/'
print(os.listdir('../input/nbv-classification/'))

file_lbl = 'dataset_pose.npy'


# ## The grids
# 
# The inputs are probabilistic grids represented by a single vector. The vector must be reshaped to 32 x 32 x 32 to make it sense as a 3D grid. The grids are not related with coordinates.

# In[ ]:


# address
file_vol = 'dataset_vol_classification_training.npy'

# load the inputs
path_input_vol = os.path.join(dataset_folder, file_vol)
dataset_vol = np.load(path_input_vol)

print("Input data size: \n",dataset_vol.shape)


# In[ ]:


#lets draw some grids

for i in range(3):
    cnbv.showGrid(dataset_vol[randint(0, len(dataset_vol))])


# ## The Labels

# In[ ]:


# The Labels

file_lbl = 'dataset_lbl_classification_training.npy'

path_input_lbl = os.path.join(dataset_folder, file_lbl)
dataset_lbl = np.load(path_input_lbl)
print("Labels data size: \n",dataset_lbl.shape)
classes = np.unique(dataset_lbl)
print("Available clases: n", classes)


# In[ ]:


# Read the pose that corresponds to a class.
# such poses are the vertices of a icosahedron

# This function converts a class to its corresponding pose
def getPositions(nbv_class, positions):
    return np.array(positions[nbv_class])


# In[ ]:


# Read the pose that corresponds to a class.
nbv_positions = np.genfromtxt('../input/nbv-classification/points_in_sphere.txt')

# This function converts a class to its corresponding pose
def getPositions(nbv_class, positions):
    return np.array(positions[nbv_class])


# In[ ]:


positions = getPositions(classes, nbv_positions)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(positions[:,1], positions[:,2], positions[:,2], color='red')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# In[ ]:


# draw the some grids with nbv examples
for i in range(3):
    idx = randint(0, len(dataset_vol))
    print(getPositions(dataset_lbl[idx], nbv_positions)[0])
    cnbv.showGrid(dataset_vol[idx], getPositions(dataset_lbl[idx], nbv_positions)[0])


# In[ ]:




