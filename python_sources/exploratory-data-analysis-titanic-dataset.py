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


# In[ ]:


df_train=pd.read_csv('/kaggle/input/titanic/train.csv')


# In[ ]:


df_train.head()


# In[ ]:


df_train.info()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# Creating heatmap for finding the missing values

# In[ ]:


sns.heatmap(df_train.isnull(),cmap='YlGnBu')


# In[ ]:


df_train.Cabin.isnull().value_counts()


# As Cabin columnhas null values for more then 75% of rows. So we can drop this column.

# In[ ]:


df_train=df_train.drop('Cabin',axis=1)


# In[ ]:


df_train.head()


# In[ ]:


df_train.Age.isnull().value_counts()

Around 20% of Age data is missing. So we can replace the null values with the mean value.
# In[ ]:


plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=df_train)


# If we classify the age with respect to class, the average age of passengers in each class is different. So we can replace the missing values according to the class mean of age.

# In[ ]:




