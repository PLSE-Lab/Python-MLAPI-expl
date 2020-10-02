#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import xlrd

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Read the Data
df = pd.read_excel('/kaggle/input/worldbankdata.xlsx')


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.info()


# In[ ]:


print(df.isnull().values.any())


# In[ ]:


print(df.isnull().sum().sum())


# In[ ]:


print(df.isnull().sum())


# In[ ]:


df1 = df.copy()
df1.dropna(inplace=True)


# In[ ]:


df1.isnull().sum()


# In[ ]:


df1.info()


# In[ ]:


sns.pairplot(df1)


# In[ ]:


sns.scatterplot(x='gdp.cap', y='population', data=df1, hue='population')


# In[ ]:


sns.kdeplot(df1['population'], shade=True, color='orangered')


# In[ ]:


df1.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(20, 20), color='deeppink')


# In[ ]:


df1.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(20, 20))


# In[ ]:


mask = np.tril(df1.corr())
sns.heatmap(df1.corr(), fmt='.1g', annot = True, cmap= 'cool', mask=mask)

