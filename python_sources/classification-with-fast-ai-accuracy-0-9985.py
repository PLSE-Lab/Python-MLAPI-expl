#!/usr/bin/env python
# coding: utf-8

# ## Import libs

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


# We import all the necessary packages. We are going to work with the [fastai V1 library](http://www.fast.ai/2018/10/02/fastai-ai/) which sits on top of [Pytorch 1.0](https://hackernoon.com/pytorch-1-0-468332ba5163). The fastai library provides many useful functions that enable us to quickly and easily build neural networks and train our models.

# In[ ]:


from fastai import *
from fastai.vision import *
from fastai.utils.collect_env import *


# In[ ]:


show_install(True)


# In[ ]:


np.random.seed(42)


# ## Looking at the data

# In[ ]:


path = Path('../input/fruits-360_dataset/fruits-360')


# In[ ]:


get_ipython().system('ls {path}')


# In[ ]:


train_path = path/'Training'
test_path = path/'Test'

ls_trn = train_path.ls()
ls_tst = test_path.ls()
len(ls_trn), len(ls_tst), ls_trn[90:], ls_tst[90:]


# In[ ]:


def add_test_folder(iil, test_path):
    iil.test = ImageItemList.from_folder(test_path).no_split().label_from_folder().train
    
iil = (ImageItemList.from_folder(train_path)
                     .random_split_by_pct(.2)
                     .label_from_folder())

add_test_folder(iil, test_path)

data = iil.transform(tfms=None, size=100, bs=32).databunch().normalize(imagenet_stats)


# In[ ]:


len(data.train_dl.dataset), len(data.valid_dl.dataset), len(data.test_dl.dataset)


# In[ ]:


data.show_batch(ds_type=DatasetType.Train ,rows=3, figsize=(7,7))


# In[ ]:


print(data.classes)
len(data.classes),data.c


# ## Training Model

# In[ ]:


metrics = [accuracy]


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


learn = create_cnn(data, models.resnet34, model_dir='/kaggle/working/models',  metrics=metrics)


# In[ ]:


learn.loss_func


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


lr = 1e-3


# In[ ]:


learn.fit_one_cycle(5, max_lr=lr)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('fruits-stg1-rn34')
get_ipython().system('ls -l ./models')


# ## Evaluate Model

# In[ ]:


results = learn.validate(dl=learn.data.test_dl)


# In[ ]:


print('loss: {:.6f}; accuracy: {}'.format(results[0].item(), results[1].item()))


# In[ ]:




