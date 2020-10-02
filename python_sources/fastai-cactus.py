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
import fastai


# In[ ]:


path = Path('../input')


# In[ ]:


train_df = pd.read_csv(path/'train.csv')
train_df.head()


# In[ ]:


test_df = pd.read_csv(path/'sample_submission.csv')
print(test_df.shape)
test_df.head()


# In[ ]:


tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=10.0, max_zoom=1.1, max_lighting=0.2, max_warp=0.2, p_affine=0.75, p_lighting=0.75)
train_data = ImageDataBunch.from_df(path/'train'/'train', train_df, ds_tfms=tfms, size=128)


# In[ ]:


train_data.show_batch(rows=3, figsize=(5,6))


# In[ ]:


train_data.classes,train_data.c


# In[ ]:


learn = cnn_learner(train_data, models.densenet161, metrics=[accuracy],model_dir="/tmp/model/")


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot(suggestion=True)


# In[ ]:


lr =1.0e-2
learn.fit_one_cycle(7,slice(lr))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


solution = pd.DataFrame(columns=test_df.columns)
solution


# In[ ]:


for index,row in test_df.iterrows():
  img_name = row['id']
  img = open_image(path/'test'/'test'/img_name)
  pred_class,pred_idx,outputs = learn.predict(img)
  solution.loc[len(solution)] = [img_name,outputs.numpy()[1]]


# In[ ]:


solution.to_csv('submission.csv', index=False)


# In[ ]:




