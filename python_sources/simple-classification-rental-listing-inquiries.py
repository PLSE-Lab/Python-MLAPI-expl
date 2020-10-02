#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Load Data

# In[ ]:


train = pd.read_json('/kaggle/input/two-sigma-connect-rental-listing-inquiries/train.json.zip')
train.head()


# In[ ]:


test = pd.read_json('/kaggle/input/two-sigma-connect-rental-listing-inquiries/test.json.zip')
test.head()


# # Data Exploration

# In[ ]:


train.shape


# In[ ]:


train.columns


# In[ ]:


train.describe()


# * Summary of numeric variables

# In[ ]:


train.dtypes


# * Types of variables

# In[ ]:


train['interest_level'].unique()


# * Target Variable

# In[ ]:


train.dtypes


# In[ ]:


train.drop(['features','photos'],1).nunique()


# * After dropping list columns('features','photos'), check the number of unique values of each column

# # Data Visualization

# In[ ]:


import matplotlib.pyplot as plt 
import seaborn as sns

plt.figure(figsize=(10,8))
sns.boxplot(train['interest_level'],train['bedrooms'])


# * There is no difference in this boxplot. 
# * So, should we drop 'bedrooms' column?
# * If no, then how can we visualize in other way?

# In[ ]:


train['bedrooms'].nunique()


# * 'bedrooms' is categorical column.
# * Also, our target variable(interest_level) has categorical attribute.

# In[ ]:


train['bathrooms'].nunique()


# In[ ]:


figure, (a,b) = plt.subplots(nrows= 2)
figure.set_size_inches(10,15)
sns.countplot(train['bedrooms'],ax=a)
sns.countplot(train['bathrooms'],ax=b)


# ### 'bedrooms', 'bathrooms' (Category - Category Values)

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(train['bedrooms'],hue=train['interest_level'])


# * bedrooms from 1 -> 2, interest level increases since number of 'low' decreases while number of 'medium' and 'high' increase.

# In[ ]:


train.groupby('interest_level')['building_id'].count() 
# Since 'low' has the most values, it is natural that low has the highest shape


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(train['bathrooms'],hue=train['interest_level'])


# ### 'created'

# In[ ]:


train['created'] = train['created'].astype('datetime64')
train['day'] = train['created'].dt.day
train['month'] = train['created'].dt.month
train['year'] = train['created'].dt.year


# * Make 'day', 'month', 'year' column to analysis whether there is meaningful difference between each interest_level values.

# In[ ]:


train['month'].unique()


# In[ ]:


train['year'].unique()


# #### day

# In[ ]:


plt.figure(figsize=(20,10))
sns.boxplot(train['interest_level'],train['day'])


# * Can't find the difference with boxplot.

# In[ ]:


plt.figure(figsize=(20,10))
sns.countplot(train['day'],hue=train['interest_level'])


# * Does this column help predicting target variable?
# * day 2->3, 5->6, 20->21..

# #### month

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(train['month'],hue=train['interest_level'])


# In[ ]:


train.groupby('month')['interest_level'].value_counts()


# # Prediction

# * Drop features, photos column (just for my convenience)

# In[ ]:


train = pd.read_json('/kaggle/input/two-sigma-connect-rental-listing-inquiries/train.json.zip')
alldata = pd.concat([train,test])


# In[ ]:


alldata.isnull().any() 


# * Check NA

# In[ ]:


alldata2 = alldata.drop(['features','interest_level','photos'],axis=1)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for i in alldata2.columns[alldata2.dtypes == object] :
    alldata2[i] = le.fit_transform(alldata2[i])


# In[ ]:


train2 = alldata2[:len(train)]
test2 = alldata2[len(train):]


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0, n_jobs=-1)
rf.fit(train2, train['interest_level'])
result = rf.predict_proba(test2)


# In[ ]:


result


# In[ ]:


sub = pd.read_csv('/kaggle/input/two-sigma-connect-rental-listing-inquiries/sample_submission.csv.zip')
sub.head()


# In[ ]:


sub['high'] = result[:,0]
sub['medium'] = result[:,2]
sub['low'] = result[:,1]

sub.head()


# In[ ]:


sub.to_csv('rental.csv',index=False)

