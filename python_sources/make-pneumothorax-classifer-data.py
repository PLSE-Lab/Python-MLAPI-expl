#!/usr/bin/env python
# coding: utf-8

# In this kernel i generate data for training pneumothorax clasifier. The classifier gets around 97.5 % accuracy.
# You can find the classifier training notebook [here](https://www.kaggle.com/meaninglesslives/siim-classifier-seresnext50)

# # load libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import shutil
from tqdm import tqdm_notebook


# # unzip the data

# In[ ]:


get_ipython().system('mkdir masks')
get_ipython().system('unzip -q ../input/data-repack-and-image-statistics/masks.zip -d masks ')


# In[ ]:


get_ipython().system('mkdir train')
get_ipython().system('unzip -q ../input/data-repack-and-image-statistics/train.zip -d train ')


# In[ ]:


get_ipython().system('mkdir test')
get_ipython().system('unzip -q ../input/data-repack-and-image-statistics/test.zip -d test ')


# # Use train_rle to create classifier training set

# In[ ]:


train_rle = pd.read_csv('../input/siim-acr-pneumothorax-segmentation-data/pneumothorax/train-rle.csv')
train_rle.columns = ['ImageId', 'EncodedPixels']


# In[ ]:


train_rle.head()


# In[ ]:


im_id_disease = train_rle[train_rle.EncodedPixels!=' -1'].ImageId
im_id_no_disease = train_rle[train_rle.EncodedPixels==' -1'].ImageId


# In[ ]:


get_ipython().system('mkdir -p classifier_data/disease')


# In[ ]:


train_path = './train'
cls_data_path = './classifier_data/disease'
for im_id in tqdm_notebook(im_id_disease):
    im_id = im_id +'.png'
    shutil.copy(os.path.join(train_path, im_id),os.path.join(cls_data_path, im_id))
    
len(os.listdir(cls_data_path))


# In[ ]:


get_ipython().system('mkdir classifier_data/no_disease')


# In[ ]:


train_path = './train'
cls_data_path = './classifier_data/no_disease'
for im_id in tqdm_notebook(im_id_no_disease):
    im_id = im_id +'.png'
    shutil.copy(os.path.join(train_path, im_id),os.path.join(cls_data_path, im_id))
    
len(os.listdir(cls_data_path))


# In[ ]:


get_ipython().system('tar -zcf classifier_data.tar.gz ./classifier_data')
get_ipython().system('tar -zcf test_data.tar.gz ./test')


# In[ ]:


get_ipython().system('rm -r */')


# In[ ]:




