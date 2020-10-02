#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


inputPath = Path('../input')
path = Path('data/cats')


# In[ ]:


folder = 'MaineCoon'
file = 'URLs_MaineCoon.txt'
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
download_images(inputPath/file, dest, max_pics=200)


# In[ ]:


folder = 'NFC'
file = 'URLs_NFC.txt'
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
download_images(inputPath/file, dest, max_pics=200)


# In[ ]:


folder = 'Siberian'
file = 'URLs_Siberian.txt'
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
download_images(inputPath/file, dest, max_pics=200)


# In[ ]:


get_ipython().system('cp ../input/* {path}/')


# Remove any images that can't be opened

# In[ ]:


classes = ['NFC', 'Siberian', 'MaineCoon']
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)


# **View data**

# In[ ]:


np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=256, num_workers=0).normalize(imagenet_stats)


# In[ ]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# In[ ]:


data.show_batch(rows=3, figsize=(7,8))


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('stage-1')
learn.unfreeze()
learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(2, max_lr=slice(1e-6, 1e-3))


# In[ ]:


learn.save('stage-2')
interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.plot_confusion_matrix()


# In[ ]:


interp.most_confused(min_val = 2)


# **Cleaning Up**

# In[ ]:


from fastai.widgets import *


# In[ ]:


# Manually remove top-losses
ds, idxs = DatasetFormatter().from_toplosses(learn, n_imgs=100)
ImageCleaner(ds, idxs, path)


# In[ ]:


# Manually remove duplicates
ds, idxs = DatasetFormatter().from_similars(learn)
ImageCleaner(ds, idxs, path)


# An update csv file is supposed to be created at this point, from which I can make an optimized dataset.
# But where is this csv file?
