#!/usr/bin/env python
# coding: utf-8

# This kernel is a little bit modified version of [this kernel](https://www.kaggle.com/manojprabhaakr/similar-duplicate-images-in-aptos-data).   
# Credit to [@ManojPrabhakar](https://www.kaggle.com/manojprabhaakr).  
# <br>
# You can download the duplicated list from **output**.
# 
# ---
# # <font color=dimgray> Import Package </font>

# In[ ]:


import pandas as pd 
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import matplotlib.pyplot as plt
import imagehash
import psutil

from PIL import Image
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display
import time

get_ipython().run_line_magic('matplotlib', 'inline')

plt.style.use('ggplot')
pd.set_option('display.max_columns', 300)
pd.set_option('display.max_rows', 100)


# ---
# # <font color=dimgray> Load CSV </font>

# In[ ]:


train = pd.read_csv("../input/train.csv")


# ---
# # <font color=dimgray> Get Info of train images </font>
# 
# Getting the path of the Image

# In[ ]:


df = train[['diagnosis']]
df['path'] = glob.glob('../input/train_images/*.png')
df.head()


# Calculating the Hash, Shape, Mode, Length and Ratio of each image

# In[ ]:


def getImageMetaData(file_path):
    with Image.open(file_path) as img:
        img_hash = imagehash.phash(img)
        return img.size, img.mode, str(img_hash), file_path

    
img_meta_l = Parallel(n_jobs=psutil.cpu_count(), verbose=1)(
    (delayed(getImageMetaData)(fp) for fp in glob.glob('../input/train_images/*.png'))
)
img_meta_df = pd.DataFrame(np.array(img_meta_l))
img_meta_df.columns = ['Size', 'Mode', 'Hash', 'path']


# In[ ]:


df = df.merge(img_meta_df, on='path', how='left')
df.head()


# In[ ]:


df.to_csv('./image_info.csv', index=False)


# In[ ]:


df_gb = df.groupby('Hash').count().reset_index()
df_gb.head()


# In[ ]:


df_gb_dup = df_gb.query('path > 1')
df_gb_dup


# In[ ]:


df_gb_dup['path'].value_counts()


# In[ ]:


dup_hash_l = df_gb_dup['Hash'].values


# In[ ]:


df_dup = df.loc[df['Hash'].isin(dup_hash_l)].sort_values('Hash')
df_dup.to_csv('./duplicated_info.csv')
df_dup.head(10)


# In[ ]:


samp_hash = df_dup['Hash'].sample(1).values[0]
print(samp_hash)


# In[ ]:


dups = df_dup.query('Hash == @samp_hash')['path'].values

fig, ax = plt.subplots(len(dups), 1, figsize=(7, 5 * len(dups)))
for i, d in enumerate(dups):
    ax[i].imshow(mpimg.imread(d))
    ax[i].grid(alpha=0.1)
fig.tight_layout();


# In[ ]:


dups = df_dup.query('Hash == "969a6b60246f3967"')['path'].values

fig, ax = plt.subplots(len(dups), 1, figsize=(7, 5 * len(dups)))
for i, d in enumerate(dups):
    ax[i].imshow(mpimg.imread(d))
    ax[i].grid(alpha=0.1)
fig.tight_layout();


# ---
# # <font color=dimgray> EOF </font>
