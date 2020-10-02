#!/usr/bin/env python
# coding: utf-8

# This is a standard model from Fast AI course getting ridiculous levels of accuracy - amazing what you can get without applying any intelligent thought to it.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/plantvillage/PlantVillage"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

from fastai.vision import *
from fastai.metrics import error_rate

bs = 32
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart


# In[ ]:


path_img = '../input/plantvillage/PlantVillage'


# In[ ]:


fnames = get_image_files(path_img,recurse=True)

fnames[:20]


# In[ ]:


def get_labels(file_path): 
    dir_name = os.path.dirname(file_path)
    split_dir_name = dir_name.split("/")
    dir_levels = len(split_dir_name)
    label  = split_dir_name[dir_levels - 1]
    return(label)


# In[ ]:


np.random.seed(3)
data = ImageDataBunch.from_name_func(path_img, fnames, get_labels, ds_tfms=get_transforms(), size=299, bs=bs
                                  ).normalize(imagenet_stats)


# In[ ]:


print(data.classes)
len(data.classes),data.c


# In[ ]:


data.show_batch(rows=3, figsize=(15,15))


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate)
learn.model_dir='/kaggle/working'


# In[ ]:


learn.fit_one_cycle(8)


# In[ ]:


learn.save('stage-1-50')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,15))


# In[ ]:


interp.plot_confusion_matrix(figsize=(12,12), dpi=60)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


learn.load('stage-1-50')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(8, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.save('stage-2-50')


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp.most_confused(min_val=1)

