#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# loading the dataset into the kernel using pandas read_csv function. head() function prints the first 5 rows of the data. similarly tail function prints last 5 rows

# In[ ]:


train=pd.read_csv('/kaggle/input/students-performance-in-exams/StudentsPerformance.csv')
train.head()


# missingno is a library which is used to find the missing values in the dataset ana also their location. To our surprise our data doesnot ocmtain any missing values

# In[ ]:


import missingno as msno
plt.figure(figsize=(5,5))
msno.matrix(train)
plt.savefig('f18')


# dtypes function prints the data tyeps of all the columns in the dataset

# In[ ]:


train.dtypes


# from the above dtypes we can see that all the traning features in the data are object data types and the output are integer data types

# describe function prints the general statistics of the dataset like count, mean of each column, standard deviation,min value. interquartile range values,maximum values

# In[ ]:


train.describe()


# In[ ]:


train.info()


# In[ ]:


train.isnull().sum()


# it is a good sign that the data does not contain any null values!!

# In[ ]:


train.gender.value_counts()


# distplot plot is used to plot frequency of the unique value in the data and we can also plot them according to the differnt classes for better visualisatin and understanding of the data.

# In[ ]:


plt.title('female vs male math score distribution')
sns.distplot(train[train.gender=='female']['math score'],label='female')
sns.distplot(train[train.gender=='male']['math score'],label='male')
plt.legend(['female','male'])
plt.savefig('f1')


# In[ ]:


plt.title('female vs male reading score distribution')
sns.distplot(train[train.gender=='female']['reading score'],label='female')
sns.distplot(train[train.gender=='male']['reading score'],label='male')
plt.legend()
plt.savefig('f1')


# In[ ]:


plt.title('female vs male writing score distribution')
sns.distplot(train[train.gender=='female']['writing score'],label='female')
sns.distplot(train[train.gender=='male']['writing score'],label='male')
plt.legend()
plt.savefig('f2')


# scatter plot is used to find the relation between how two columns are related

# In[ ]:



sns.scatterplot(x='math score',y='reading score',hue='gender',data=train)
plt.savefig('f4')


# In[ ]:


sns.scatterplot(x='writing score',y='reading score',hue='gender',data=train)
plt.savefig('f5')


# In[ ]:


sns.scatterplot(x='math score',y='reading score',hue='gender',data=train)
plt.savefig('f6')


# pairplot function of seaborn is used to plot graphs between all the column pairs in the data

# In[ ]:


sns.pairplot(train,hue='gender')
plt.savefig('f7')


# In[ ]:


train['race/ethnicity'].value_counts()


# In[ ]:


sns.scatterplot(x='reading score',y='math score',hue='race/ethnicity',data=train)


# In[ ]:


sns.scatterplot(x='reading score',y='writing score',hue='race/ethnicity',data=train)


# In[ ]:


sns.scatterplot(x='math score',y='writing score',hue='race/ethnicity',data=train)


#  counplot gives the frequecy of uniques values in eac column

# In[ ]:


sns.countplot(x='race/ethnicity',hue='gender',data=train)
plt.savefig('f8')


# In[ ]:


train['parental level of education'].value_counts()


# In[ ]:


sns.countplot(hue='gender',x='parental level of education',data=train)
plt.savefig('f9')


# In[ ]:


sns.countplot(train['test preparation course'],hue='parental level of education',data=train)
plt.savefig('f10')


# In[ ]:


sns.scatterplot(x='math score',y='writing score',hue='test preparation course',data=train)
plt.savefig('f11')


# In[ ]:


sns.scatterplot(x='reading score',y='writing score',hue='test preparation course',data=train)
plt.savefig('f12')


# In[ ]:


sns.scatterplot(x='math score',y='reading score',hue='test preparation course',data=train)
plt.savefig('f13')


# jointplot is another kind of a plotting function which is a combination of regplot and distplot functions. 

# In[ ]:


sns.jointplot(x='math score',y='reading score',kind='reg',data=train)
plt.savefig('f14')


# In[ ]:


sns.jointplot(x='math score',y='reading score',kind='hex',data=train)
plt.savefig('f15')


# heatmap is the function which can be used for examining the feature importance betweeen all the feature column pairs!! it is very useful for feature selection

# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(train.corr())
plt.savefig('f16')


# label encoder is used to encoding differnt caterories in column from objectc label to numerica labels so that it could be interpreted by the models

# In[ ]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
col=[col for col in train.columns if train[col].dtype=='O']
for c in col:
    train[c]=encoder.fit_transform(train[c])
    


# In[ ]:


train.head()


# In[ ]:


train=train.drop_duplicates()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(train.corr())
plt.savefig('f17')


# In[ ]:


sns.swarmplot(x='gender',y='math score',data=train)
plt.savefig('f19')


# In[ ]:


sns.swarmplot(x='parental level of education',y='reading score',data=train)
plt.savefig('f20')


# In[ ]:


sns.swarmplot(x='race/ethnicity',y='writing score',data=train)
plt.savefig('f21')


# In[ ]:


sns.swarmplot(x='test preparation course',y='writing score',data=train)
plt.savefig('f22')


# Lastly , I would like to add that that the plotting functions in seaborn are no less and this was just a fraction of it .
# In conclusion, seaborn is very convenient with many features, and plotly is also quite easy with plotly express.
# I would like to share some tips on visualization with many people. Thank you for reading so far.

# In[ ]:




