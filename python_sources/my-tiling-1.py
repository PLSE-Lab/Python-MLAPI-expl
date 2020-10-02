#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import cv2
import skimage.io
from tqdm.notebook import tqdm
import zipfile
import numpy as np
import pandas as pd


# In[4]:


get_user = os.environ.get('USER', 'KAGGLE')

if get_user == 'KAGGLE':
    my_env = 'KAGGLE'
elif get_user == 'jupyter':
    my_env = 'GCP'
elif get_user == 'user':
    my_env = 'LOCAL'
else:
    my_env = None
    
assert my_env is not None    

env_input_fn = {
    'KAGGLE': '../input/prostate-cancer-grade-assessment/',
    'LOCAL':  '../data/',
    'GCP':    '../../',
}

input_fn = env_input_fn[my_env]


# In[5]:


train_df = pd.read_csv(input_fn + 'train.csv')


# ### Tiling Code

# In[6]:


TRAIN = input_fn + 'train_images/'
MASKS = input_fn + 'train_label_masks/'
OUT_TRAIN = 'train.zip'
OUT_MASKS = 'masks.zip'
sz = 128
N = 16


# In[7]:


def tile(img, mask):
    result = []
    shape = img.shape
    pad0,pad1 = (sz - shape[0]%sz)%sz, (sz - shape[1]%sz)%sz
    img = np.pad(img,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=255)
    mask = np.pad(mask,[[pad0//2,pad0-pad0//2],[pad1//2,pad1-pad1//2],[0,0]],
                constant_values=0)
    img = img.reshape(img.shape[0]//sz,sz,img.shape[1]//sz,sz,3)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    mask = mask.reshape(mask.shape[0]//sz,sz,mask.shape[1]//sz,sz,3)
    mask = mask.transpose(0,2,1,3,4).reshape(-1,sz,sz,3)
    if len(img) < N:
        mask = np.pad(mask,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=0)
        img = np.pad(img,[[0,N-len(img)],[0,0],[0,0],[0,0]],constant_values=255)
    idxs = np.argsort(img.reshape(img.shape[0],-1).sum(-1))[:N]
    img = img[idxs]
    mask = mask[idxs]
    for i in range(len(img)):
        result.append({'img':img[i], 'mask':mask[i], 'idx':i})
    return result


# In[17]:


x_tot,x2_tot = [],[]
names = [name[:-10] for name in os.listdir(MASKS)]

img_fns = os.listdir(TRAIN)

img_fn = img_fns[0]
mask_fn = img_fn.split('.')[0] +'_mask.tiff'

img_fn, mask_fn


# In[22]:


img = skimage.io.MultiImage(TRAIN + img_fn)
mask = skimage.io.MultiImage(MASKS + mask_fn)


# ##### The problem section:
# 
# ```python
# -> 2486             raise ValueError("cannot decompress %s" % self.compression)
#    2487         if 'sample_format' in self.tags:
#    2488             tag = self.tags['sample_format']
# 
# ValueError: cannot decompress jpeg
# 
# ```

# In[35]:


img2 = img[-1]
mask2 = mask[-1]


# In[19]:


ret = tile(img, mask)


# In[ ]:




