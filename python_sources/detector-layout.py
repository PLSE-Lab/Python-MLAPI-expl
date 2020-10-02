#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[3]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#from trackml.dataset import load_event, load_dataset


# In[5]:


hits = pd.read_csv('../input/train_1/event000001000-hits.csv')  
hits.head()


# Coordinates are in mm and momentum in MeV/c

# In[6]:


def display_detector_r_z_view():
    x_coord = hits['x'].values
    y_coord = hits['y'].values
    z_coord = hits['z'].values
    radius = np.sqrt(x_coord**2 + y_coord**2)
    
    font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 26,
        }
    plt.figure(figsize=(15,12))
    plt.scatter(z_coord, radius)
    plt.ylabel('r [mm]')
    plt.xlabel('z [mm]')
    plt.text(-2500, 900, '16', fontdict=font)
    plt.text(0,     900, '17', fontdict=font)
    plt.text(2500,  900, '18', fontdict=font)
    plt.text(-2500, 500, '12', fontdict=font)
    plt.text(0,     500, '13', fontdict=font)
    plt.text(2500,  500, '14', fontdict=font)
    plt.text(-1100, 100, '7', fontdict=font)
    plt.text(0,     110, '8', fontdict=font)
    plt.text(1000,  100, '9', fontdict=font)

    plt.show()


# In[8]:


# show the volume_id
display_detector_r_z_view()


# In[9]:


def plot_volume(vol_id, view='yx'):
    x_coord = hits[(hits['volume_id']==vol_id)]['x'].values
    print('length of x_coord: ', len(x_coord)) 
    y_coord = hits[(hits['volume_id']==vol_id)]['y'].values
    print('length of y_coord: ', len(y_coord)) 
    z_coord = hits[(hits['volume_id']==vol_id)]['z'].values
    print('length of y_coord: ', len(z_coord)) 
    if view == 'yx':
        plt.figure(figsize=(15,12))
        plt.scatter(x_coord, y_coord)
        plt.ylabel('y [mm]')
        plt.xlabel('x [mm]')
        plt.title('Volume_id='+str(vol_id))
        plt.show()
    if view == 'yz':
        plt.figure(figsize=(15,12))
        plt.scatter(z_coord, y_coord)
        plt.ylabel('y [mm]')
        plt.xlabel('z [mm]')
        plt.title('Volume_id='+str(vol_id))
        plt.show()
    


# In[10]:


def plot_volume_and_layer(vol_id, layer_id, view='yx'):
    x_coord = hits[(hits['volume_id']==vol_id) & (hits['layer_id']==layer_id)]['x'].values
    print('length of x_coord: ', len(x_coord)) 
    y_coord = hits[(hits['volume_id']==vol_id) & (hits['layer_id']==layer_id)]['y'].values
    print('length of y_coord: ', len(y_coord)) 
    z_coord = hits[(hits['volume_id']==vol_id) & (hits['layer_id']==layer_id)]['z'].values
    print('length of y_coord: ', len(z_coord)) 
    if view == 'yx':
        plt.figure(figsize=(15,12))
        plt.scatter(x_coord, y_coord)
        plt.ylabel('y [mm]')
        plt.xlabel('x [mm]')
        plt.title('Volume_id='+str(vol_id)+'_layer_'+str(layer_id))
        plt.show()
    if view == 'yz':
        plt.figure(figsize=(15,12))
        plt.scatter(z_coord, y_coord)
        plt.ylabel('y [mm]')
        plt.xlabel('z [mm]')
        plt.title('Volume_id='+str(vol_id)+'_layer_'+str(layer_id))
        plt.show()


# The two functions defined above allow for a bit more detailed view of the detector; **plot_volume**  allows you to select the volume id and you can displays two views: 
# transverse: "yx" and longitudinal: "yz"
# With **plot_volume_and_layer** you can include the layer_id in your choice. The display is also possible in two views.

# In[11]:


plot_volume(8, "yx")


# In[12]:


# The layer ids listed in this dataset are:
layers = hits['layer_id'].unique()
print('Layer ids: ', layers)


# The layers of all volumes are labelled from the inside out; volume 9 has its first layer (2) at z~600 mm, its last layer (14) at z~1500 mm Similar labelling is used for the 'barrel' detectors.

# In[13]:


plot_volume_and_layer(9, 2, 'yz')


# This figure is a bit confusing, it implies that the 'disc' detectors have four elements in each layer. Or this could be an artifact of the hit coordinate algorithm.

# From this first look at the data, it is clear that even though the hits appear to be listed by volume_id and layer_id, all memory of the tracks has been removed from the data set. As stated by the organizers.

# In[ ]:




