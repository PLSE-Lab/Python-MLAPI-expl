#!/usr/bin/env python
# coding: utf-8

# **We are going to generate a baseline solution using plain statistics without any model. From this we can guess the pattern in test data**

# ### Imports

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ***In this Baseline Solution, we will be using the non null values of 2016 for both 2017 & 2018. We can see that for some buildings, the initial data points have target 0, and later the pattern changes. So, we treat the data points with six consecutive non nulls as our first data. We drop the rest of initial data.***

# ### Read data

# In[ ]:


train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')
train[:5]


# In[ ]:


test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')
test[:5]


# ### Clean train data 
# Remove the initial data points where the target is mostly zero. Here we impose condition that, there should be non-null target for six consecutive hours.

# In[ ]:


train_len = len(train)
print ("Initial Train Rows :",train.shape[0])
train['nonZero'] = (train.meter_reading>0).astype(int)
lag = lambda x,l: x.shift(l).fillna(1)
mask = train.groupby(['building_id','meter'])['nonZero'].transform(lambda x: (x*lag(x,1)*lag(x,2)*lag(x,3)*lag(x,4)*lag(x,5)).cumsum())
train=train[mask>0]
train= train.drop('nonZero',axis=1)
print ("Clean Train Rows : ",train.shape[0])
print ('Number of Rows removed : ', train_len-len(train))
gc.collect()
train[:5]


# ### Merge datasets
# Merge the train and test set based on dates. (i.e. 2016-10-10 10:00:00,  2017-10-10 10:00:00 and  2017-10-10 10:00:00 will all have same meter reading)

# In[ ]:


test.timestamp = test.timestamp.apply(lambda x: x.replace('2017','2016').replace('2018','2016'))
test = pd.merge(test, train, how='left')
print(pd.isna(test).sum())
test[:5]


# In[ ]:


test.groupby(['building_id','meter'])['meter_reading'].count().sort_values()[:5]


# ### Handle Null Targets

# We can see that there are over 2.6M missing meter readings in the merged set. We should impute them using interpolation.
# Also there are few buildings which do not have any training data. We can impute them with zeros/mean.

# In[ ]:


mean_target = np.expm1(np.log1p(train.meter_reading).mean())  ## Computing the geometric mean
del train
gc.collect()


# In[ ]:


def impute(x):
    if len(x.dropna())==0:
        return x.fillna(0)
        # return x.fillna(mean_target)    ## Uncomment if you want to use geometric mean
    if pd.isna(x.iloc[0]):
        last_val = x.dropna().iloc[-1]
        x.iloc[0]=last_val
    if pd.isna(x.iloc[-1]):
        first_val = x.dropna().iloc[0]
        x.iloc[-1]=first_val
    return x.interpolate()


# In[ ]:


test.meter_reading = test.groupby(['building_id','meter'])['meter_reading'].transform(impute)
gc.collect()
test[:5]


# ### Write submission

# In[ ]:


# test.loc[test.meter_reading==0, 'meter_reading'] = mean_target   ## Uncomment if you want to replace zeros with geometric mean of target

test[['row_id','meter_reading']].to_csv('submission.csv',index=False)

