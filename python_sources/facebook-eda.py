#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/facebook-data/pseudo_facebook.csv')


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df = pd.get_dummies(df,columns=['gender'])


# In[ ]:


df.head()


# In[ ]:


df = df.dropna()


# In[ ]:


df.info()


# In[ ]:


df['age'].value_counts()


# In[ ]:


labels=['10-20','21-30','31-40','41-50','51-60','61-70','71-80','81-90','91-100','101-110','111-120']
df['age_group'] = pd.cut(df.age,bins=np.arange(10,121,10),labels=labels,right=True)


# In[ ]:


df.head()


# In[ ]:


df['age_group'].unique()


# In[ ]:


df['age_group'].value_counts()


# In[ ]:


sns.pairplot(data = df, hue="gender_female")


# In[ ]:


sns.countplot(x='age',data=df)


# In[ ]:


sns.countplot(x='age',hue='gender_male',data=df)


# In[ ]:


sns.barplot(df['gender_female'],df['likes'])
sns.barplot(df['gender_male'],df['likes'])


# In[ ]:


sns.lmplot( x="age", y="friend_count", data=df, fit_reg=False, hue='tenure', legend=False)


# In[ ]:


sns.jointplot(x='age',y='friend_count',data=df)


# In[ ]:


sns.stripplot(x='likes',y='friend_count',data=df,jitter=False)


# In[ ]:


df.corr()


# In[ ]:


plt.subplots(figsize=(12,12))
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[ ]:





# In[ ]:




