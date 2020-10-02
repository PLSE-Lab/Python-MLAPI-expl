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

# Any results you write to the current directory are saved as output.fname


# In[ ]:


fname = os.listdir('../input/severstal-steel-defect-detection/train_images')


# In[ ]:


fname[:5]


# In[ ]:


#fname = df.ImageId.to_list()
cname = list(range(1,5))
fname = ['{}_{}'.format(a,b) for b in cname for a in fname]
fname[:5]


# In[ ]:


df = pd.DataFrame()
df['ImageId'] = fname


# In[ ]:


df['ClassId'] = df['ImageId'].str.split('_', expand=True)[1]
df['ImageId'] = df['ImageId'].str.split('_', expand=True)[0]


# In[ ]:


df =  df.sort_values(['ImageId','ClassId']).reset_index(drop=True)


# In[ ]:


df.head()


# In[ ]:


tr_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv')
tr_df.head()


# In[ ]:


tr_df.dtypes


# In[ ]:


df.dtypes


# In[ ]:


df['ClassId'] = df['ClassId'].astype(int)


# In[ ]:


train = df.merge(tr_df, on=['ImageId','ClassId'], how='left')


# In[ ]:


train.head()


# In[ ]:


train.ClassId.value_counts()


# In[ ]:


train.dtypes


# In[ ]:


train['ImageId_ClassId'] = train['ImageId']+"_"+train['ClassId'].astype(str)


# In[ ]:


train = train[['ImageId_ClassId','EncodedPixels']]
train.head()


# In[ ]:


train.to_csv("train.csv", index=False)

