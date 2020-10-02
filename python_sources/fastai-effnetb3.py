#!/usr/bin/env python
# coding: utf-8

# In[42]:


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


# In[43]:


get_ipython().system('pip install efficientnet_pytorch')


# In[44]:


from fastai.vision import *
path = Path('../input/')


# In[45]:


path.ls()


# In[46]:


train = pd.read_csv(path/'train.csv')
test = pd.read_csv(path/'sample_submission.csv')


# In[55]:


test_img = ImageList.from_df(test, path=path/'test', folder='test')
tfms = get_transforms()
data = (ImageList.from_df(train, path=path/'train', folder='train')
        .split_by_rand_pct(0.01)
        .label_from_df()
        .add_test(test_img)
        .transform(tfms, size=224, resize_method=ResizeMethod.SQUISH)
        .databunch(path='.', bs=64, device= torch.device('cuda:0'))
        .normalize(imagenet_stats)
       )


# In[48]:


from efficientnet_pytorch import EfficientNet


# In[49]:


model_name = 'efficientnet-b3'
def getModel(pret):
    model = EfficientNet.from_pretrained(model_name)
    model._fc = nn.Linear(model._fc.in_features,data.c)
    return model


# In[50]:


learn = Learner(data,getModel(True),metrics=[accuracy])


# In[51]:


# learn.lr_find()
# learn.recorder.plot()


# In[52]:


lr=5e-3


# In[53]:


learn.fit_one_cycle(3,lr)


# In[57]:


preds,_ = learn.get_preds(ds_type=DatasetType.Test)
# preds,_ = learn.TTA(ds_type=DatasetType.Test)
idx = preds.numpy()[:,0]


# In[58]:


test.has_cactus = idx
test.to_csv('submission.csv', index=False)


# In[ ]:




