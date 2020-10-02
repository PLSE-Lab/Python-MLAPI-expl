#!/usr/bin/env python
# coding: utf-8

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


import seaborn as sns
from matplotlib import pyplot as plt


# In[ ]:


df = pd.read_csv("../input/diabetes.csv")
df.head()


# In[ ]:


df.tail()


# In[ ]:


df.sample(6)


# In[ ]:


df.shape


# In[ ]:


df.dtypes


# In[ ]:


df.columns


# In[ ]:


df.describe()


# In[ ]:


print("pregnancies count are",df['Pregnancies'].count())


# In[ ]:


df.isnull().sum() #no missing values in the dataset...


# **data visualization**

# **barplot**

# In[ ]:


sns.barplot(x='Outcome',y='BloodPressure',data=df,hue="Outcome")


# In[ ]:


plt.figure(figsize=(12,12))
plt.subplot(3,3,1)
sns.barplot(x='Outcome',y='Glucose',data=df,hue="Outcome")
plt.subplot(3,3,2)
sns.barplot(x='Outcome',y='BloodPressure',data=df,hue="Outcome")
plt.subplot(3,3,3)
sns.barplot(x='Outcome',y='SkinThickness',data=df,hue="Outcome")
plt.subplot(3,3,4)
sns.barplot(x='Outcome',y='BMI',data=df,hue="Outcome")
plt.subplot(3,3,5)
sns.barplot(x='Outcome',y='DiabetesPedigreeFunction',data=df,hue="Outcome")
plt.subplot(3,3,6)
sns.barplot(x='Outcome',y='Age',data=df,hue="Outcome")


# **factorplot**

# In[ ]:


sns.factorplot(x='Pregnancies',y='Insulin',data=df,hue='Outcome')


# **swarmplot**

# In[ ]:


sns.swarmplot(x='Outcome',y='BloodPressure',data=df,hue='Outcome')


# In[ ]:


sns.lmplot(x='Pregnancies',y='Glucose',data=df,hue="Outcome")


# **pairplot***

# In[ ]:


aa=sns.pairplot(df,hue='Outcome')


# **distplot**

# In[ ]:


plt.figure(figsize=(12,12))
plt.subplot(3,3,1)
sns.distplot(df.Pregnancies)
plt.subplot(3,3,2)
sns.distplot(df.Glucose)
plt.subplot(3,3,3)
sns.distplot(df.BloodPressure)
plt.subplot(3,3,4)
sns.distplot(df.SkinThickness)
plt.subplot(3,3,5)
sns.distplot(df.BMI)
plt.subplot(3,3,6)
sns.distplot(df.DiabetesPedigreeFunction)


# **boxplot**

# In[ ]:


sns.boxplot(x="Pregnancies",y="Age",data=df,hue="Outcome")


# **countplot**

# In[ ]:


sns.countplot(x="Pregnancies",data=df)


# In[ ]:


sns.countplot(x="Outcome",data=df)


# In[ ]:




