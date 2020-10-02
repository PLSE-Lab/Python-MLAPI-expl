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


test_image_path = glob.glob(os.path.join(DATA_DIR, 'images_test_rev1/*.jpg'))
test_image_name = [os.path.basename(path) for path in test_image_path]
test_image_name.sort()
len(test_image_name)


# In[ ]:


test_image_name[:5]


# In[ ]:


GalaxyID = [name[:-4] for name in test_image_name]
GalaxyID[:5]


# In[ ]:


pd.Series(GalaxyID).to_csv('galaxy_id_test.csv', header=False)


# In[ ]:


ORIG_SHAPE = (424,424)
CROP_SIZE = (256,256)
IMG_SHAPE = (64,64)

def get_image(path, x1,y1, shape, crop_size):
    x = plt.imread(path)
    x = x[x1:x1+crop_size[0], y1:y1+crop_size[1]]
    x = resize(x, shape)
    return x
    
def get_test_images(shape=IMG_SHAPE, crop_size=CROP_SIZE):
    x1 = (ORIG_SHAPE[0]-CROP_SIZE[0])//2
    y1 = (ORIG_SHAPE[1]-CROP_SIZE[1])//2
   
    x_batch = []
    for image_name in tqdm(test_image_name):
        x = get_image(os.path.join(DATA_DIR, 'images_test_rev1', image_name), 
                      x1, y1, shape=shape, crop_size=crop_size)
        x_batch.append(x.astype(np.float32))
    x_batch = np.array(x_batch)
    return x_batch

get_ipython().run_line_magic('time', 'X_test = get_test_images()')


# In[ ]:


ds = xr.Dataset({
    'image_test': (('sample', 'x', 'y', 'channel'), X_test)
})


# In[ ]:


get_ipython().run_line_magic('time', "ds.to_netcdf('galaxy_test.nc')")


# In[ ]:


ls -lh galaxy_test.nc

