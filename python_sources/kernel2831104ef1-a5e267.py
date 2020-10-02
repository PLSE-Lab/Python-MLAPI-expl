#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# settings
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# load libraries
from fastai import *
from fastai.vision import *
import pandas as pd


# In[ ]:


folder = 'beautiful_dog'
file = 'beautiful_dog.csv'
path = Path('../input/dogbreed/dogs/dogs/')
dataPath = Path('/tmp/.fastai/data')
dest = dataPath/folder
dest.mkdir(parents=True, exist_ok=True)
download_images(path/folder/file, dest, max_pics=200)


# In[ ]:


folder = 'shorthair_dog'
file = 'shorthair_dog.csv'
dest = dataPath/folder
dest.mkdir(parents=True, exist_ok=True)
download_images(path/folder/file, dest, max_pics=200)


# In[ ]:


folder = 'wolf_dog'
file = 'wolf_dog.csv'
dest = dataPath/folder
dest.mkdir(parents=True, exist_ok=True)
download_images(path/folder/file, dest, max_pics=200)


# In[ ]:


classes = ['beautiful_dog','shorthair_dog','wolf_dog']


# In[ ]:



for c in classes:
    print(c)
    verify_images(dataPath/c, delete=True, max_size=500)


# In[ ]:


size = 16 # ssize of input images
bs = 64 # batch size
tfms = get_transforms(do_flip=False)


# In[ ]:


dataPath


# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(dataPath,train='.',valid_pct=0.2,ds_tfms=tfms, size=size, bs=bs).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,8))


# In[ ]:


get_ipython().system(' mkdir ../working/data/dogs')

