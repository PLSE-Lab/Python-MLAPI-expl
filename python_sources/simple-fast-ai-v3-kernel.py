#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


from fastai.vision import *


# In[3]:


# import everything into databunch
path = Path("../input/")
tfms = get_transforms(flip_vert=True, do_flip=True)
data = ImageDataBunch.from_csv(path=path, csv_labels='train.csv', folder='train/train',
                              test='test/test',ds_tfms = tfms)
data


# OK, so now we have our ImageDataBunch. We now need a model to train the data on.

# In[4]:


learn = cnn_learner(data, models.resnet50, model_dir="/tmp/model/",
                   metrics = [error_rate, accuracy])


# In[5]:


learn.lr_find()


# In[6]:


learn.recorder.plot()


# In[7]:


# Let's now try to fit a first epoch.
learn.fit_one_cycle(1, 1e-2)


# That is already quite remarkable (98% accuracy). Now the next step is to unfreeze the model and check if we can get even better accuracy (one also needs to know what is the metric used to score the exercise).

# In[8]:


learn.save('fit-first-epoch')


# In[22]:


learn.load('fit-first-epoch');


# In[23]:


learn.unfreeze()


# In[24]:


learn.fit_one_cycle(11, slice(1e-4, 2e-3))


# In[25]:


learn.save('after-11-epochs');


# In[ ]:


#learn.load('after-6-epochs');


# In[26]:


# submit a prediction
# copied from https://www.kaggle.com/interneuron/fast-fastai-with-condensenet
preds,_ = learn.get_preds(ds_type=DatasetType.Test)


# In[33]:


test_df = pd.read_csv("../input/sample_submission.csv")
test_df.has_cactus = preds.numpy()[:, 1]


# In[34]:


test_df.head()


# In[ ]:


to_np(p)


# In[ ]:


#submission = pd.DataFrame({'id':[i.name for i in learn.data.test_ds.items],'has_cactus': p[:,1]})


# In[ ]:


#submission.head()


# In[35]:


#submission.to_csv('submission.csv', index=False)
test_df.to_csv('submission.csv', index=False)


# In[ ]:




