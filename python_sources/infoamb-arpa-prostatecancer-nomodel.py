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

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LinearRegression

# Any results you write to the current directory are saved as output.


# In[ ]:


# Data acquisition
data = pd.read_csv('../input/prostate.data',sep='\t')


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data=data.drop(data.columns[0], axis=1)


# In[ ]:


data.head()


# In[ ]:


train=data['train']
type(train)
train.head()


# In[ ]:


data=data.drop('train',axis=1)
data.head()


# In[ ]:


lpsa=data['lpsa']
lpsa.head()


# In[ ]:


predictors=data.drop('lpsa',axis=1)
predictors.head()


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.hist(figsize=(20,15))


# In[ ]:


data.describe()


# In[ ]:


type(train)


# In[ ]:


dataTrain=data.loc[train=="T"]
dataTrain.head(10)


# In[ ]:


dataTrain.shape


# In[ ]:


dataTest=data.loc[train=="F"]
dataTest.head(10)


# In[ ]:


dataTest.shape


# In[ ]:


lpsaTrain=lpsa.loc[train=="T"]
lpsaTrain.head(10)


# In[ ]:


lpsaTrain.size


# In[ ]:


lpsaTest=lpsa.loc[train=="F"]
lpsaTest.head(10)


# In[ ]:


lpsaTest.size


# In[ ]:


dataTrain.corr()


# In[ ]:


predictorsTrain=dataTrain.drop('lpsa',axis=1)
predictorsTrain.head(10)


# In[ ]:


predictorsTrain.shape


# In[ ]:


predictorsTest=dataTest.drop('lpsa',axis=1)
predictorsTest.head(10)


# In[ ]:


predictorsTest.shape


# In[ ]:


predictorsTrainMeans=predictorsTrain.mean()
predictorsTrainStd=predictorsTrain.std()
print(predictorsTrainMeans)
print(predictorsTrainStd)
predictorsTrain_std=(predictorsTrain-predictorsTrainMeans)/predictorsTrainStd
predictorsTrain_std.head()


# In[ ]:


predictorsTrain_std.hist(figsize=(20,15))


# In[ ]:


predictorsTest_std=(predictorsTest-predictorsTrainMeans)/predictorsTrainStd
predictorsTest_std.head()

