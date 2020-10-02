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

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read data
data = pd.read_csv('../input/train.csv')


# In[ ]:


# Take a look at top 5 rows of dataset 
data.head()


# In[ ]:


# dataframe consist rows * columns
# Columns indicates number of variables or features in dataset
# Rows indicates number of training examples.
data.shape


# In[ ]:


# As columns indicates features , we can take a look into all 12 features
data.columns


# In[ ]:


# We can get general information about dataframe
data.info()
# Information we get from .info() command
# data type : float (2 features), int64(5 features),object(5 features)
# Missing values : from .shape method we got number of rows are 891 , from below data we can see that
#                  'Age','Cabin'.'Embarked' features have some missing values.


# In[ ]:


# describe method statistical characteristics of each feature
data.describe()
# by default describe method gives chracteristics of numerical features.


# In[ ]:


# We can include other data type features as well in describe method.
data.describe(include=['object'])


# In[ ]:


# For Categorical Object features value counts can be used to find distribution of variables.
data['Sex'].value_counts()


# In[ ]:


data['Embarked'].value_counts()


# In[ ]:


# We can get normalized output as well from value counts
data['Sex'].value_counts(normalize=True)


# In[ ]:


# What is praportion of Passengers Survived in our dataset?
# We can get this information by indexing single column data.
data['Survived'].mean()


# In[ ]:


# Boolean Indexing
# We can use boolean indexing to more insight about dataset.
# What is average age of Male passengers who survived .
data[(data['Sex']=='male') & data['Survived']==1]['Age'].mean()


# In[ ]:


# What is average age of Female passengers who not survived .
data[(data['Sex']=='female') & data['Survived']==0]['Age'].mean()


# In[ ]:


# what is average fare of passengers who survived
data[data['Survived']==1]['Fare'].mean()


# In[ ]:


# what is average fare of passengers who not survived
data[data['Survived']==0]['Fare'].mean()


# In[ ]:


# Most of algorithm in ML take numerical data ,so we can convert ordered categirical data to have numerical values.
data['Sex'] = data['Sex'].map({'male':1,'female':0})


# In[ ]:


data['Sex'].head()


# In[ ]:


# For categorical variables we can fin all unique values
data['Embarked'].nunique() # Number of Non Unique values are 


# In[ ]:


# Which are those non-unique values?
data['Embarked'].unique()


# In[ ]:


data['Embarked']= data['Embarked'].map({'S':0,'C':1,'Q':2})


# In[ ]:


data['Embarked'].value_counts()


# In[ ]:


# df.groupby(by=grouping_columns)[columns_to_show].function()
data.groupby(['Survived'])['Sex'].value_counts()
# We can see that Survival rate of female is more than male


# In[ ]:


data.groupby(['Survived'])['Pclass'].value_counts()
# We can see that Survival rate of passengers in class 3 is less than other 2 classes


# In[ ]:


# Summary tables
# Cross table : Can be use  to see how the observations in our sample are distributed in the context of two variables
pd.crosstab(data['Survived'],data['Sex'])


# In[ ]:


pd.crosstab(data['Survived'],data['Pclass'],normalize=True)


# In[ ]:


# Pivot Tables :
# Pivot taables consists of values ,index and function
# values : Columns to show data for
# Index : column to group data by
# function : statistics needs to calculate for groups
data.pivot_table(values=['Pclass','Sex','Embarked'],index=['Survived'],aggfunc=[np.mean])


# In[ ]:




