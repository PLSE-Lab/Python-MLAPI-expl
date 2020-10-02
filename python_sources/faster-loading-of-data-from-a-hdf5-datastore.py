#!/usr/bin/env python
# coding: utf-8

# Loading images is pretty slow, especially when you are reading 4 images per example. Here I attempt to create a HDF5 datastore for faster loading of data.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import h5py
from tqdm import tqdm
import cv2


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
channels = ['red', 'green', 'blue', 'yellow']
hdf_path = f'./train.hdf5'


# In[ ]:


def load_image(id):
    img = np.zeros((4, 512, 512), dtype=np.uint8)
    for c, ch in enumerate(channels):
        img[c, ...] = cv2.imread('../input/train/{}_{}.png'.format(id, ch), cv2.IMREAD_GRAYSCALE)
    return img


# In[ ]:


with h5py.File(hdf_path, mode='w') as train_hdf5:
    train_hdf5.create_dataset("train", (len(train_df), 4, 512, 512), np.uint8)
    for i, id in tqdm(enumerate(train_df['Id'][:100])):    #Remove the [:100] for full dataset
        img = load_image(id)
        train_hdf5['train'][i, ...] = img


# **Rough Benchmark**

# In[ ]:


randind = np.random.randint(0, len(train_df), 8)
randind = np.sort(randind)


# In[ ]:


train_hdf5 = h5py.File(hdf_path, "r")


# Loading from the HDF5 Datastore

# In[ ]:


get_ipython().run_cell_magic('timeit', '', '# with h5py.File(hdf_path, "r") as train_hdf5:       # Causes 20% slowdown :(\nbatch = train_hdf5[\'train\'][randind, ...]')


# In[ ]:


train_hdf5.close()


# In[ ]:


get_ipython().run_cell_magic('timeit', '', "batch = np.zeros((8, 4, 512, 512), dtype=np.uint8)\nfor i, ind in enumerate(randind):\n    batch[i, ...] = load_image(train_df['Id'][ind])")


# This is my first kernel, and the first time I'm experimenting with HDF5, so suggestions and feedback are welcome.
# 
# Can someone tell me what happens if I don't close an open datastore? Opening and closing per batch is slow, and I want to know if I will corrupt the data if I interrupt training without closing.
