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


import pandas as pd


# In[ ]:


df = pd.read_csv("../input/naukri-dataset/naukri.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum().sort_values(ascending=False)


# In[ ]:


round(100*df.isnull().sum()/len(df),2).sort_values(ascending=False)


# In[ ]:


import seaborn as sns
sns.heatmap(df.isnull(),yticklabels=False)


# In[ ]:


df.dropna(inplace=True)
df.shape


# In[ ]:


df['Crawl Timestamp'].min()


# In[ ]:


df['Crawl Timestamp'].max()


# In[ ]:


df['Job Salary'].value_counts()


# In[ ]:


df['Key Skills'].value_counts()


# In[ ]:


df['Role Category'].value_counts()


# In[ ]:


df['Location'].value_counts()


# In[ ]:


df['Functional Area'].value_counts()


# In[ ]:


df['Industry'].value_counts()


# In[ ]:


df['Role'].value_counts()


# In[ ]:


df.drop(['Uniq Id'],inplace = True,axis=1)


# In[ ]:


df.rename(columns = {'Job Experience Required':'Experience'}, inplace=True)


# In[ ]:


df['Experience'].replace('30 years and above','30 - 50 Yrs',inplace = True)


# In[ ]:


df['Experience'] = df['Experience'].str.strip()


# In[ ]:


df['Min_Exp'] = df.Experience.str.split("-",expand=True,)[0]


# In[ ]:


df['Max_Exp'] = df.Experience.str.split("-",expand=True,)[1]


# In[ ]:


df['Max_Exp'] = df.Max_Exp.str.extract('(\d+)')


# In[ ]:


df.head()


# In[ ]:




