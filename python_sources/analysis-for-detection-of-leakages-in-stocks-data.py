#!/usr/bin/env python
# coding: utf-8

# <h2>Data Analysis for detection of leakages in stocks data generation</h2>
# One can follow the article at: https://medium.com/@hrshtsharma2012/detecting-data-leakages-84cdd1ed5eb4 for introductory information about data leakages.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing the data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


data_cleaner = [df_train,df_test]


# Let us now explore the dataset.<br>
# **Exploratory data Analysis**

# In[ ]:


df_train.head().transpose()


# In[ ]:


df_train.describe()


# In[ ]:


df_train['time'].describe() #the date is in object format. Let us convert to datetime format for analysis


# In[ ]:


#It can be done directly as the data has been provided in a convertible format
df_train['time']= pd.to_datetime(df_train['time']) 
df_test['time']= pd.to_datetime(df_test['time']) 


# In[ ]:


df_train['time'].describe()


# In[ ]:


df_test['time'].describe()


# In[ ]:


df_train.shape[0]


# The above description of the test and train dataset shows that the train data belongs to a time period between January 2017 to November 2018 while the test data belongs to a time period between January 2017 to January 2018. This suggests that the provider has data till the end of 2018 but unfortunately has not divided data randomly using the same random seed which result in the train data having more futuristic dataset. The test set renders the prediction purpose useless.

# In[ ]:


#Let us now explore the price dataset.
for dataset in data_cleaner:
    dataset['pricebin'] = pd.cut(dataset['price'],df_train.shape[0]/2500)


# In[ ]:


df_train['pricebin'].unique()


# In[ ]:


pricebin_tr = df_train['pricebin'].value_counts(sort=False)
pricebin_t = df_test['pricebin'].value_counts(sort=False)


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
axes[0].set_ylabel('Frequency')
axes[0].set_title('Training set')
plt.ylabel('Frequency')
pricebin_tr.plot(ax=axes[0],kind='bar')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Testing set')
pricebin_t.plot(ax=axes[1],kind='bar')
fig.tight_layout()


# In[ ]:


df_test['pricebin'].unique()


# In[ ]:


df_test.describe()


# from the above results it seems as if the price data has been picked from a normal distribution. The train data has a mean of nearly 110 and the test data has a mean of around 119. Let us confirm our hypothesis by checking the kurtosis and skewness of the price provided.

# In[ ]:


fig, ax = plt.subplots()
sns.kdeplot(df_train['price'], ax=ax,shade=True)
sns.kdeplot(df_test['price'], ax=ax,shade=True)


# In[ ]:


for dataset in data_cleaner:
    #skewness and kurtosis
    print("Skewness: %f" % dataset['price'].skew())
    print("Kurtosis: %f" % dataset['price'].kurt())


# As the value of the Skewness (Sidedness from mean postion) and the kurtosis (hike from normal value) are very close to one one can assume that the data has been picked from a normal distribution. **This is the leak that we have found**. Since the training set also contains data from the future we can assume that prices in future are going to decrease.
