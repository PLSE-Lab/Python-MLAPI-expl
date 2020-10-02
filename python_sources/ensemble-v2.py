#!/usr/bin/env python
# coding: utf-8

# This kernel is created to see if other top scores can boost results.
# kernal used
# 1. https://www.kaggle.com/konradb/rather-silly-1am-ensemble
# 2. https://www.kaggle.com/meaninglesslives/unet-with-efficientnet-encoder-in-keras
# 3. https://www.kaggle.com/iafoss/hypercolumns-pneumothorax-fastai-0-831-lb
# 
# >  Its rather really silly to emsemble it just gives you nearly mean of two scores rather Increasing it.
# 
# I have done ensemble in both possible ways
# 1. **Left = xf1 , Right = xf2**   which gives 
# ![](https://i.imgur.com/ICPWjg8.png)
# 2.** Left = xf2 , Right= xf1**   which gives 
# ![](https://i.imgur.com/AfIi8hH.png)
# 
# No change using sample submission leak kernal to
# https://www.kaggle.com/raddar/sample-submission-leak
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


# load the two reference submissions 
#xf1 = pd.read_csv('../input/hypercolumns-pneumothorax-fastai-0-831-lb/submission.csv')
xf2 = pd.read_csv('../input/lungv1/leaky_unet_submission.csv')
xf1 = pd.read_csv('../input/unet-with-efficientnet-encoder-in-keras/orig_submission.csv')


# In[ ]:


# align indices
xf1.columns = ['ImageId', 'enc1']
xf2.columns = ['ImageId', 'enc2']

xf3 = pd.merge(left = xf1, right = xf2, on = 'ImageId', how = 'inner')
print(xf1.shape, xf2.shape, xf3.shape)

# identify the positions where xf1 has empty predictions but xf2 does not
xf3[xf3['enc1'] != xf3['enc2']]
id1 = np.where(xf3['enc1'] == '-1')[0]
id2 = np.where(xf3['enc2'] != '-1')[0]
idx = np.intersect1d(id1,id2)

# map non-empty xf2 slots to empty ones in xf1
xf3['EncodedPixels'] = xf3['enc1']
xf3['EncodedPixels'][idx] = xf3['enc2'][idx]


# In[ ]:


xf3[['ImageId','EncodedPixels']].to_csv('hybrid_1_2.csv', index = False)

