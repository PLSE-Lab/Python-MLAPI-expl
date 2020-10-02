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


# In[ ]:


df = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
df


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df["parental level of education"].value_counts()


# In[ ]:



print(df.describe())


# In[ ]:


df["gender"].value_counts()


# In[ ]:


plt.figure(figsize=(4,5))
sns.countplot(x="gender", data=df)


# According to plot above, female numbers are higher than  male.
# 
# 

# In[ ]:





# In[ ]:


df.head()


# In[ ]:


df["parental level of education"].value_counts()


# In[ ]:


plt.figure(figsize=(12,5))
sns.countplot(x="parental level of education", data=df)


# numbers of parental level of education

# In[ ]:


plt.figure(figsize=(16,6))
sns.scatterplot(x='gender' , y='math score',data= df)


# According to plot above, male scores are higher than female scores in math.
# 
# 

# In[ ]:


plt.figure(figsize=(16,6))
sns.scatterplot(x='gender' , y='test preparation course',data= df)


# In[ ]:





# In[ ]:


sns.pairplot(df)
plt.show()


# In[ ]:


df["reading score"]


# In[ ]:


df["math score"]


# In[ ]:


df["writing score"]


# In[ ]:




