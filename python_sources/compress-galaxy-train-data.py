#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr

from skimage.transform import resize
from skimage import io

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


ls ../input/galaxy-zoo-the-galaxy-challenge/44352/


# In[ ]:


DATA_DIR = '../input/galaxy-zoo-the-galaxy-challenge/44352/'


# In[ ]:


df = pd.read_csv(os.path.join(DATA_DIR, 'training_solutions_rev1.csv'))
df.head()


# In[ ]:


# Based on https://www.kaggle.com/helmehelmuto/keras-cnn
# but uses float32 to save memory/space

ORIG_SHAPE = (424,424)
CROP_SIZE = (256,256)
IMG_SHAPE = (64,64)

def get_image(path, x1,y1, shape, crop_size):
    x = plt.imread(path)
    x = x[x1:x1+crop_size[0], y1:y1+crop_size[1]]
    x = resize(x, shape)
    return x
    
def get_train_data(dataframe, shape=IMG_SHAPE, crop_size=CROP_SIZE):
    x1 = (ORIG_SHAPE[0]-CROP_SIZE[0])//2
    y1 = (ORIG_SHAPE[1]-CROP_SIZE[1])//2
   
    sel = dataframe.values
    ids = sel[:,0].astype(int).astype(str)
    y_batch = sel[:,1:].astype(np.float32)
    x_batch = []
    for i in tqdm(ids):
        x = get_image(DATA_DIR + 'images_training_rev1/'+i+'.jpg', x1, y1, shape=shape, crop_size=crop_size)
        x_batch.append(x.astype(np.float32))
    x_batch = np.array(x_batch)
    return x_batch, y_batch

get_ipython().run_line_magic('time', 'X_train, y_train = get_train_data(df)')


# In[ ]:


X_train.shape, y_train.shape


# In[ ]:


plt.pcolormesh(X_train[1,:,:,0])
plt.colorbar()


# In[ ]:


X_train.max()


# In[ ]:


ds = xr.Dataset({
    'image_train': (('sample', 'x', 'y', 'channel'), X_train),
    'label_train': (('sample', 'feature'), y_train)
}).astype(np.float32)


# In[ ]:


ds.nbytes / 1e9  # GB


# In[ ]:


get_ipython().run_line_magic('time', "ds.to_netcdf('galaxy_train.nc')")


# In[ ]:


ls -lh ./galaxy_train.nc


# In[ ]:




