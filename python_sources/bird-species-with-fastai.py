#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
        
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

# Any results you write to the current directory are saved as output.


# In[ ]:


from fastai import *
from fastai.vision import *


# In[ ]:


bs = 64


# In[ ]:


path = Path('/kaggle/input/100-bird-species/175')


# In[ ]:


path.ls()


# In[ ]:


path_valid = path/'valid'
path_train = path/'train'
path_test = path/'test'


# In[ ]:


data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=224, bs=bs
                                  ).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6))


# In[ ]:


print(data.classes)

print(len(data.classes), data.c) # .c is number of classes for classification problems


# In[ ]:


#Training the model


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=error_rate)


# In[ ]:


learn.fit_one_cycle(4) # 4 Epochs


# In[ ]:


learn.model_dir='/kaggle/working/'


# In[ ]:


learn.save('birds_stage-1')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

#Quick check
len(data.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# In[ ]:


interp.plot_confusion_matrix(figsize=(40,40), dpi=400)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:


#Tweaking our model to make it better


# In[ ]:


learn.unfreeze()


# In[ ]:


learn.fit_one_cycle(2)


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.load('birds_stage-1')


# In[ ]:


#Training: resnet50


# In[ ]:


data = ImageDataBunch.from_folder(path, ds_tfms=get_transforms(), size=299, bs=bs//2, num_workers=0
                                  ).normalize(imagenet_stats)


# In[ ]:


learn = cnn_learner(data, models.resnet50, metrics=error_rate)


# In[ ]:


#learn.model_dir='/kaggle2/workingresnet50/'
learn.model_dir='/kaggle/working'


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(4)


# In[ ]:


learn.save('birds_stage-1-50')


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))


# In[ ]:


learn.load('birds_stage-1-50')


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)


# In[ ]:


interp.most_confused(min_val=2)


# In[ ]:




