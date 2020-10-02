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


from fastai.vision import *


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

df_train.head()


# In[ ]:


path = Path("../input")


# In[ ]:


np.random.seed(42)
src = (ImageList.from_csv(path, 'train.csv', folder='train_images', suffix='.png')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' '))


# In[ ]:


data = (src.transform(tfms = get_transforms(), size=128)
        .databunch(bs = 128).normalize(imagenet_stats))


# In[ ]:


data.show_batch()


# In[ ]:


def acc(input:Tensor, targs:Tensor)->Rank0Tensor:
    "Computes accuracy with `targs` when `input` is bs * n_classes."
    n = targs.shape[0]
    input = input.argmax(dim=-1).view(n,-1)
    targs = targs.view(n,-1).long()
    return (input==targs).float().mean()


# In[ ]:


learn = cnn_learner(data, models.resnet34, metrics=[acc], model_dir="/tmp/model/", callback_fns=ShowGraph)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(5, slice(1e-3))


# In[ ]:


learn.unfreeze()


# In[ ]:


# learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10, slice(1e-5, 1e-3))


# In[ ]:




