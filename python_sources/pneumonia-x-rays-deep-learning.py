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


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


path = Path("../input/chest_xray/chest_xray/")
path


# In[ ]:


path.ls()


# In[ ]:


fnames = get_image_files(path/'train'/'PNEUMONIA')
fnames[:5]


# In[ ]:


# Training data has 3875 images of Pneumonia Cases
fnames_train_pneumonia = np.array(fnames)
fnames_train_pneumonia.shape


# In[ ]:


# Training data has 1341 images of Normal Cases
fnames = get_image_files(path/'train'/'NORMAL')
np.array(fnames).shape


# In[ ]:


# Lets see how many validation folder has
fnames = get_image_files(path/'val'/'NORMAL')
print(np.array(fnames).shape)

fnames = get_image_files(path/'val'/'PNEUMONIA')
print(np.array(fnames).shape)


# In[ ]:


# Lets see how many test folder has
fnames = get_image_files(path/'test'/'NORMAL')
print(np.array(fnames).shape)

fnames = get_image_files(path/'test'/'PNEUMONIA')
print(np.array(fnames).shape)


# In[ ]:


np.random.seed(42)
tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(path, train="train", valid_pct=0.20,
                                  ds_tfms = tfms, classes = ['PNEUMONIA', 'NORMAL'], bs=64, size=224).normalize(imagenet_stats)


# In[ ]:


print(data.classes)
print(len(data.classes))
print(data.c)


# In[ ]:


data.show_batch(3, figsize=(12,12))


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=accuracy, model_dir = "/temp/model/")
learn.fit_one_cycle(4)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
losses, idxs = interp.top_losses()
len(data.valid_ds) == len(losses) == len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(12,12))


# In[ ]:


interp.plot_confusion_matrix(figsize=(10,10), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.save('stage-1')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2)


# In[ ]:


learn.load('stage-1')
learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(5, max_lr=slice(1e-6, 1e-4))


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(4, figsize=(10,8), heatmap=False)
plt.show()


# In[ ]:


interp.plot_confusion_matrix(figsize=(10, 8), dpi=60)


# In[ ]:


learn.show_results()

