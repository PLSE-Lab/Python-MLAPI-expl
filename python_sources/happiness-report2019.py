#!/usr/bin/env python
# coding: utf-8

# # World Happiness Report 2019

# Exploring the influencing factors of happiness

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


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/world-happiness-report-2019.csv')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df[df.isnull().any(1)]


# In[ ]:


df = df.fillna(df.mean().round())


# In[ ]:


plt.figure(figsize=(10,8))
corr = df.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)]= True
sns.heatmap(corr,mask=mask,annot=True,cmap=plt.cm.RdBu)


# Ladder has strong correlation with Social support, Log of GDP per capita and Healthy life expectancy.The rest of the correlation is not strong

# Now,let us discuss it in detail

# In[ ]:


sns.lmplot(x='Ladder',y='SD of Ladder',data=df)


# It's is a small relationship

# In[ ]:


sns.lmplot(x='Ladder',y='Positive affect',data=df)


# In[ ]:


sns.lmplot(x='Ladder',y='Negative affect',data=df)


# In[ ]:


sns.lmplot(x='Ladder',y='Social support',data=df)


# This is a strong correlation. Social support has a great influence on Ladder.

# In[ ]:


sns.lmplot(x='Ladder',y='Freedom',data=df)


# In[ ]:


sns.lmplot(x='Ladder',y='Corruption',data=df)


# This has little relevance

# In[ ]:


sns.lmplot(x='Ladder',y='Generosity',data=df)


# In[ ]:


sns.lmplot(x='Ladder',y='Log of GDP\nper capita',data=df)


# In[ ]:


sns.lmplot(x='Ladder',y='Healthy life\nexpectancy',data=df)


# The longer human life, the greater the happiness

# ## Conclusion

#  Social support, Log of GDP per capita and Healthy life expectancy the main influencing factor of Ladder

# SD of Ladder,Positive affect,egative affect,Freedom,Corruption,Generosit are not the directly related with Ladder
