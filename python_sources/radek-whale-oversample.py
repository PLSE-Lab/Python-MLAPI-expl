#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from skimage.util import montage
import pandas as pd
from torch import optim
import re

from utils import *


# ## Prepare data

# In[ ]:


df = pd.read_csv('../input/train.csv')
df.head()


# In[ ]:


im_count = df[df.Id != 'new_whale'].Id.value_counts()
im_count.name = 'sighting_count'
df = df.join(im_count, on='Id')
val_fns = set(df.sample(frac=1)[(df.Id != 'new_whale') & (df.sighting_count > 1)].groupby('Id').first().Image)


# In[ ]:


# pd.to_pickle(val_fns, 'data/val_fns')
#val_fns = pd.read_pickle('data/val_fns')


# In[ ]:


fn2label = {row[1].Image: row[1].Id for row in df.iterrows()}


# In[ ]:


SZ = 224
BS = 64
NUM_WORKERS = 0
SEED=0


# In[ ]:


path2fn = lambda path: re.search('\w*\.jpg$', path).group(0)


# In[ ]:


df = df[df.Id != 'new_whale']


# In[ ]:


df.shape


# In[ ]:


df.sighting_count.max()


# In[ ]:


df_val = df[df.Image.isin(val_fns)]
df_train = df[~df.Image.isin(val_fns)]
df_train_with_val = df


# In[ ]:


df_val.shape, df_train.shape, df_train_with_val.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nres = None\nsample_to = 15\n\nfor grp in df_train.groupby('Id'):\n    n = grp[1].shape[0]\n    additional_rows = grp[1].sample(0 if sample_to < n  else sample_to - n, replace=True)\n    rows = pd.concat((grp[1], additional_rows))\n    \n    if res is None: res = rows\n    else: res = pd.concat((res, rows))")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nres_with_val = None\nsample_to = 15\n\nfor grp in df_train_with_val.groupby('Id'):\n    n = grp[1].shape[0]\n    additional_rows = grp[1].sample(0 if sample_to < n  else sample_to - n, replace=True)\n    rows = pd.concat((grp[1], additional_rows))\n    \n    if res_with_val is None: res_with_val = rows\n    else: res_with_val = pd.concat((res_with_val, rows))")


# In[ ]:


res.shape, res_with_val.shape


# Our training set increased 6-fold, but that is still an amount of data that is okay. I don't think it makes sense to worry about breaking up the data into smaller epochs.

# In[ ]:


pd.concat((res, df_val))[['Image', 'Id']].to_csv('oversampled_train.csv', index=False)
res_with_val[['Image', 'Id']].to_csv('oversampled_train_and_val.csv', index=False)


# The naming here is not very fortunate, but the idea is that `oversampled_train` has single entries for images in `val_fns` and `oversampled_train_and_val` is both `val` and `train` combined. Meaning, `oversampled_train_and_val` is one we might want to use when retraining on the entire train set.

# In[ ]:


df = pd.read_csv('oversampled_train.csv')


# In[ ]:


data = (
    ImageItemList
        .from_df(df[df.Id != 'new_whale'], '../input/train', cols=['Image'])
        .split_by_valid_func(lambda path: path2fn(path) in val_fns)
        .label_from_func(lambda path: fn2label[path2fn(path)])
        .add_test(ImageItemList.from_folder('../input/test'))
        .transform(get_transforms(do_flip=False, max_zoom=1, max_warp=0, max_rotate=2), size=SZ, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=BS, num_workers=NUM_WORKERS, path='../input')
        .normalize(imagenet_stats)
)


# In[ ]:


data


# In[ ]:





# In[ ]:




