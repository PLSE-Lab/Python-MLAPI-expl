#!/usr/bin/env python
# coding: utf-8

# In[41]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

try:
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import seaborn as sns
    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
    import matplotlib.pyplot as plt
    import os
except ImportError:
    pass
# Any results you write to the current directory are saved as output.


# In[8]:


data = pd.read_csv('../input/StudentsPerformance.csv')


# In[9]:


data.head(5)


# In[16]:


data.columns = ['Gender', 'Race', "Parent's Education", 'Lunch', 
                'Preparation Course', 'Math Score', 'Reading Score', 'Writing Score']


# In[17]:


print(data.isnull().any())
print('----------------------')
print(data.isna().sum())


# In[18]:


data.describe()


# In[33]:


data['Total Score'] = data['Math Score'] + data['Reading Score'] + data['Writing Score']


# In[34]:


data.dtypes


# In[27]:


data[['Race', "Parent's Education", 'Preparation Course']].head(10)


# In[36]:


print("Average math score is    : {}".format(np.mean(data['Math Score'])))
print("Average reading score is : {}".format(np.mean(data['Reading Score'])))
print("Average writing score is : {}".format(np.mean(data['Writing Score'])))
print("Average total score is   : {}".format(np.mean(data['Total Score'])/3))


# In[55]:


data['Math Score'].describe()


# In[68]:


math_mean = data['Math Score'].mean()
math_max = data['Math Score'].max()
print(math_mean)
print(math_max)


# In[39]:


data['Preparation Course'].value_counts()


# In[73]:


sns.pairplot(data, hue='Math Score')


# In[97]:


plt.figure(figsize=(15,10))

plt.subplot(1,3,1)
sns.barplot(x = 'Preparation Course', y = 'Math Score', data = data, hue="Gender", palette='winter')

plt.subplot(1,3,2)
sns.barplot(x = 'Preparation Course', y = 'Writing Score', data = data, hue="Gender", palette='winter')

plt.subplot(1,3,3)
sns.barplot(x = 'Preparation Course', y = 'Reading Score', data = data, hue="Gender", palette='winter')

plt.show()


# In[98]:


plt.figure(figsize=(15,10))

plt.subplot(1,3,1)
sns.barplot(x = 'Preparation Course', y = 'Math Score', data = data, hue="Parent's Education", palette='winter')

plt.subplot(1,3,2)
sns.barplot(x = 'Preparation Course', y = 'Writing Score', data = data, hue="Parent's Education", palette='winter')

plt.subplot(1,3,3)
sns.barplot(x = 'Preparation Course', y = 'Reading Score', data = data, hue="Parent's Education", palette='winter')

plt.show()


# In[102]:


data[(data['Math Score'] > 90) & (data['Reading Score'] > 90) & (data['Writing Score']>90)].sort_values(by=['Total Score'],ascending=False).head(5)


# In[109]:


sns.countplot(x = "Parent's Education", data = data)


# In[ ]:




