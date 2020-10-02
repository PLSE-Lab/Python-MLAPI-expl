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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('/kaggle/input/udemy-courses/udemy_courses.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df['subject'].unique()


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[ ]:


df['is_paid'] = pd.get_dummies(df['is_paid'])


# In[ ]:


df.head()


# In[ ]:


df['price']=df.price.str.replace('Free','0')


# In[ ]:


df['price'].value_counts()


# In[ ]:


df['price']=df.price.str.replace('TRUE','0')


# In[ ]:


df['price'] = df['price'].astype(float)
df.info()


# In[ ]:


df.head(1)


# In[ ]:


df["content_duration"]= df["content_duration"].str.split(" ", n = 0, expand = True) 


# In[ ]:


df.head(1)


# In[ ]:


df['content_duration'] = df.content_duration.str.replace('Beginner','0')


# In[ ]:


df['content_duration'] = pd.to_numeric(df['content_duration'])
df.info()


# In[ ]:


sns.pairplot(df)


# In[ ]:


df.info()


# In[ ]:


fig = plt.figure(figsize=(6,6))
sns.barplot(x='is_paid', y='num_subscribers', data=df, hue='subject')


# In[ ]:


sns.relplot(x='num_reviews' , y = 'num_subscribers', data=df , kind='line', hue='subject' )


# In[ ]:


fig = plt.figure(figsize=(12,40))

sns.relplot(x='price' , y= 'num_subscribers', data=df , kind='line', hue='subject')


# In[ ]:


sns.regplot(x='num_subscribers', y='content_duration' ,data=df)


# In[ ]:


df['num_subscribers'].corr(df['content_duration'])


# In[ ]:


sns.regplot(x=df['num_subscribers'], y=df['price'], data=df)


# In[ ]:


df['num_subscribers'].corr(df['price'])


# In[ ]:


sns.barplot(x='is_paid', y='num_subscribers', data=df, hue='level')


# In[ ]:





# In[ ]:




