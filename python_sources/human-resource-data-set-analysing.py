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


# In[ ]:


data=pd.read_csv('/kaggle/input/human-resources-data-set/HRDataset_v13.csv')


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.isnull()


# In[ ]:


data['Sex'].unique()


# In[ ]:


data['Sex'].replace('F','female',inplace=True)


# In[ ]:


data['Sex'].dropna(inplace=True)


# In[ ]:


data['Sex'].isnull()


# In[ ]:


data['Sex'].replace('M','Male',inplace=True)


# In[ ]:


data['Sex'].unique()


# In[ ]:


data['Sex'].value_counts()


# In[ ]:


data['Sex'].value_counts().plot(kind='bar')


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
# plot through sns


# In[ ]:


ax=sns.countplot(data['Sex'])


# In[ ]:


# Gender diversity across departmets
plt.figure(figsize=(16,9))
ax=sns.countplot(data['Department'],hue=data['Sex'])


# In[ ]:


#Conclusions from graph :

#No males in executive office and no females in software engineering department.
#Gender diversity is not maintained in production department and software engineering.
#No.of females is nearly double the number of males


# In[ ]:


plt.figure(figsize=(10,6))
data['MaritalDesc'].value_counts().plot(kind='pie')


# In[ ]:


data['CitizenDesc'].unique()


# In[ ]:


data['CitizenDesc'].value_counts().plot(kind='bar')


# In[ ]:


data['Position'].value_counts()


# In[ ]:


plt.figure(figsize=(20,12))
data['Position'].value_counts().plot(kind='bar')


# **How is the performance score related to pay rate?*

# In[ ]:


data['PerformanceScore'].unique()


# In[ ]:


data['PerformanceScore'].dropna(inplace=True)


# In[ ]:


data['PerformanceScore'].value_counts().plot(kind='bar')


# In[ ]:


df_perf = pd.get_dummies(data,columns=['PerformanceScore'])


# In[ ]:


df_perf.head()


# In[ ]:


data['PerformanceScore'].unique()


# In[ ]:


col_plot= [col for col in df_perf if col.startswith('Performance')]
col_plot


# In[ ]:


fig, axes = plt.subplots(2, 2, figsize=(16,9))
for i,j in enumerate(col_plot):
    df_perf.plot(x=j,y='PayRate',ax = axes.flat[i],kind='scatter')


# **Which manager has the best performers?**

# In[ ]:


data['ManagerName'].unique()


# In[ ]:


data['ManagerName'].dropna(inplace=True)


# In[ ]:


plt.figure(figsize=(16,20))
sns.countplot(y=data['ManagerName'],hue=data['PerformanceScore'])


# ***** Davind Stanley and Kelly Spirea have highest number of employees who fully meet the expectation.
# Simon and Brannon have a highest number of exceptional employess!
# Employees working with Michael need to improve their performance.********

# Which departmet gets pay more?****

# In[ ]:



plt.figure(figsize=(16,9))
data.groupby('Department')['PayRate'].sum().plot(kind='bar')


# **Which position gives away more money? **This doesn't mean that all employees in this position get maximum pay. The number of employees could be more for this dept. 

# In[ ]:


plt.figure(figsize=(16,9))
data.groupby('Position')['PayRate'].sum().plot(kind='bar')


# In[ ]:


data.columns


# Who's gets  the higest salary?  The CEO

# In[ ]:


data.loc[data['PayRate'].idxmax()]


# Who's gets  the lowest salary?  

# In[ ]:


data.loc[data['PayRate'].idxmin()]


# END****

# In[ ]:




