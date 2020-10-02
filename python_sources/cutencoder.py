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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # CutEncoder

# In this notebook i will share with you a solution for an encoding problem.
# 
# let us asume that we have some numeric feature in two data sets - train and test, if we want to make some feature engineering adding a column with $n$ cuts of the values, using the popular $pandas.cut(x, n)$ function for each one of the data sets will oftenly give us different cuts for each one of the data sets, which will badly influce machine learning models and other encoders, let us see an example of the famous titanic data.
# 

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train.info()


# for simplicity i will drop null values

# In[ ]:


train.dropna(inplace=True)
test.dropna(inplace=True)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# now let us add new Age_range column.

# In[ ]:


train['Age_Range'] = pd.cut(train['Age'], bins=4)
sns.countplot(train['Age_Range'])


# In[ ]:


test['Age_Range'] = pd.cut(test['Age'], bins=4)
sns.countplot(test['Age_Range'])


# we can notice immediatly that the bins is totaly different, so this will be bad choice for us, because this will lie for the model and also we can not encode that unseen values, and if execute that code 
# ```
# age_encoder = LabelEncoder().fit(train['Age'])
# train['Age'] = age_encoder.transform(train['Age'])
# test['Age'] = age_encoder.transform(test['Age'])
# ```
# 
# we will get this error :
# 
# ```
# y contains previously unseen labels: Interval(19.128, 38.085, closed='right')
# ```

# we could fix the problem by getting bins intervals for the combined data, and perform a cut again with the defined intervals for both the data sets.

# In[ ]:


cutted = pd.cut(pd.concat([train['Age'], test['Age']]), 4, retbins=True)
intervals = cutted[1]


# In[ ]:


sns.countplot(pd.cut(train['Age'], intervals))


# In[ ]:


sns.countplot(pd.cut(test['Age'], intervals))


# we can see that the bins are the same for both data sets. now we need to add new column with the the data and encode labels.

# In[ ]:


cutted = pd.cut(pd.concat([train['Age'], test['Age']]), 4, retbins=True)
intervals = cutted[1]
train['Age_Range'] = pd.cut(train['Age'], intervals)
test['Age_Range'] = pd.cut(test['Age'], intervals)
age_encoder = LabelEncoder().fit(pd.concat([train['Age_Range'], test['Age_Range']]))
train['Age_Range'] = age_encoder.transform(train['Age_Range'])
test['Age_Range'] = age_encoder.transform(test['Age_Range'])


# In[ ]:


sns.countplot(train['Age_Range'])


# In[ ]:


sns.countplot(test['Age_Range'])


# Imagine that we have the so called **```CutEncoder```**, which will cut the data into equal bins and encode it.

# In[ ]:


class CutEncoder():
    """Encode numeric values with equal interval cut value between 0 and n_classes-1."""
    def __init__(self):
        self.intervals = None
    def fit(self, x, bins):
        self.x = x
        self.bins = bins
        cutted = pd.cut(self.x, self.bins, retbins=True)
        self.intervals = cutted[1]
    def transform(self, y):
        return pd.cut(y, self.intervals, labels=list(range(len(self.intervals)-1)))
        


# In[ ]:


cut_encoder = CutEncoder()
cut_encoder.fit(pd.concat([train['Age'], test['Age']]), 4)
train['Age_Range'] = cut_encoder.transform(train['Age'])
test['Age_Range'] = cut_encoder.transform(test['Age'])


# In[ ]:


sns.countplot(train['Age_Range'])


# In[ ]:


sns.countplot(test['Age_Range'])


# we got the same result with easier and less code
# ```
# cut_encoder = CutEncoder()
# cut_encoder.fit(pd.concat([train['Age_Range'], test['Age_Range']]), 4)
# train['Age_Range'] = c.transform(train['Age'])
# test['Age_Range'] = c.transform(test['Age'])
# ```
# 
# instead of 
# 
# ```
# cutted = pd.cut(pd.concat([train['Age'], test['Age']]), 4, retbins=True)
# intervals = cutted[1]
# train['Age_Range'] = pd.cut(train['Age'], intervals)
# test['Age_Range'] = pd.cut(test['Age'], intervals)
# age_encoder = LabelEncoder().fit(pd.concat([train['Age_Range'], test['Age_Range']]))
# train['Age_Range'] = age_encoder.transform(train['Age_Range'])
# test['Age_Range'] = age_encoder.transform(test['Age_Range'])
# ```
# 

# Good Luck

# Ayoub Abuzer
