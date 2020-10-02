#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.vision import *
from pathlib import Path
import os
import gc


# In[ ]:


path = Path('/kaggle/input/plant-pathology-2020-fgvc7/')


# In[ ]:


train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')
sample_df = pd.read_csv(path/'sample_submission.csv')


# In[ ]:


test_data = ImageList.from_csv(path, 'test.csv', folder='images', suffix='.jpg')


# In[ ]:


tfms = get_transforms(
    flip_vert=True,
    max_lighting=0.3,
    max_rotate=45.0,
)


# In[ ]:


src = (ImageList
       .from_df(train_df, path, folder='images', suffix='.jpg')
       .split_by_rand_pct(0.2)
       .label_from_df(cols=[1,2,3,4], label_cls=FloatList)
       .add_test(test_data))


# In[ ]:


train_data = (src
              .transform(tfms, size=128)
              .databunch(bs=64))


# In[ ]:


train_data.show_batch(rows=3, figsize=(10,7))


# In[ ]:


train_data = train_data.normalize()


# In[ ]:


learn = cnn_learner(train_data, models.resnet152, metrics=[accuracy], pretrained=True, wd=1e-1)


# In[ ]:


del train_data, test_data, train_df, test_df
gc.collect()


# In[ ]:


learn.path = Path('/')


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


min_lr = learn.recorder.min_grad_lr


# In[ ]:


learn.fit_one_cycle(10, min_lr)


# In[ ]:


learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


min_lr = learn.recorder.min_grad_lr


# In[ ]:


learn.fit_one_cycle(10, max_lr=slice(min_lr, min_lr/10))


# In[ ]:


preds, y = learn.get_preds(DatasetType.Test)


# In[ ]:


sample_df = pd.read_csv(path/'sample_submission.csv')
sample_df.iloc[:,1:] = preds.numpy()
sample_df.to_csv('submission.csv', index=False)

