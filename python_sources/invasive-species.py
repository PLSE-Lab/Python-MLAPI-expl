#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install fastai==1.0.47')


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


# !head ../input/train_labels.csv


# In[ ]:


import fastai
from fastai.vision import *


# In[ ]:


batch_size = 32


# In[ ]:


tfms = get_transforms()


# In[ ]:


data = (ImageList.from_csv(csv_name='train_labels.csv', path='../input', folder='train', suffix='.jpg')
            .split_by_rand_pct()
            .label_from_df(cols='invasive')
            .transform(tfms,size=224)
            .add_test_folder('test/')
            .databunch(bs=batch_size)
           .normalize(imagenet_stats))


# In[ ]:


# data.show_batch(rows=3, figsize=(9, 9))


# In[ ]:


# data.classes, data.c


# In[ ]:


learn = cnn_learner(data, models.densenet161, metrics=accuracy, path='./')


# In[ ]:


# learn.lr_find()
# learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(5,slice(1e-2))


# In[ ]:


learn.unfreeze()


# In[ ]:


# learn.lr_find()
# learn.recorder.plot(suggestion=True)


# In[ ]:


learn.fit_one_cycle(2,slice(1e-6,1e-4))


# In[ ]:


# interp = ClassificationInterpretation.from_learner(learn)
# losses, idxs = interp.top_losses()
# interp.plot_top_losses(9, figsize=(7, 8))


# In[ ]:


# interp.plot_confusion_matrix(figsize=(3, 3))


# In[ ]:


probs, _ = learn.get_preds(ds_type=DatasetType.Test)


# In[ ]:


ilst = data.test_ds.x


# In[ ]:


fnames = [item.name.split('.')[0] for item in ilst.items]; fnames[:10]


# In[ ]:


test_df = pd.DataFrame({'name': fnames, 'invasive': probs[:,1]}) ; test_df.head()


# In[ ]:


test_df.to_csv('submission.csv', index=None)

