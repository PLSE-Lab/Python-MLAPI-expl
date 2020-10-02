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


df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")


# In[ ]:


df


# **So this problem is a classical Classification problem whether someone has Heart Disease or not based on given features.**

# In[ ]:


X = df
y = df.target


# **Separating out target variable and training set.**

# In[ ]:


y


# In[ ]:


X.drop("target",axis = 1, inplace = True)


# In[ ]:


X


# **Checking out for features with positive correlation and negative correlation**

# In[ ]:


X.corr()


# In[ ]:


X.isnull().sum()


# **No Missing values**

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# **Line Plots**

# In[ ]:


plt.title("Age with Cholestrol level")
sns.lineplot(x = df['age'], y = df['chol'],data = X)


# **Inference:
#     1. As the age increases Cholestrol Level increases.
#     2. Reaches peak value between age 50 and 60.**

# In[ ]:


X.columns


# In[ ]:


plt.title("Age with resting blood pressure level")
sns.lineplot(x = df['age'], y = df['trestbps'],data = X)


# **cp = Constrictive pericarditis refers to the inflamation of pericardium**

# In[ ]:


plt.title("Age with Constrictive pericarditis i.e. cp")
sns.lineplot(x = df['age'], y = df['cp'],data = X)


# **Reaches peak value at 68 age, has mixed inference rises at 40 years old and dips then again**

# **fbs = fasting blood sugar**

# In[ ]:


plt.title("Age with Fasting blood sugar i.e. cp")
sns.lineplot(x = df['age'], y = df['fbs'],data = X)


# **Inference** : 
# >     1. Below 40 age groups don't have fasting blood sugar rather much less in comparison to othe age groups**

# In[ ]:


plt.title("Age with resting electrocardiograph i.e. cp")
sns.lineplot(x = df['age'], y = df['restecg'],data = X)


# In[ ]:


plt.title("Age with Constrictive pericarditis i.e. cp")
sns.lineplot(x = df['age'], y = df['thalach'],data = X)


# Decremental change with respect to age

# In[ ]:


plt.title("Cholestrol level vs Maximum heart rate of an individual")
sns.lineplot(x = df['chol'], y = df['thalach'],data = X)


# Lower Cholestrol levels signifies higher heart rate. As the colestrol level increases in an individual the maximum heart rate tends to lower down.

# In[ ]:


plt.title("Age vs exang")
sns.lineplot(x = df['age'], y = df['exang'],data = X)


# **Attains Peak value at age 40 and age ranging from 50 to 60**

# In[ ]:


plt.title("Age vs oldpeak")
sns.lineplot(x = df['age'], y = df['oldpeak'],data = X)


# **Attains peak value at age range from 55 to 65**

# In[ ]:


X.columns


# In[ ]:


plt.title("Age vs exang")
sns.lineplot(x = df['age'], y = df['thal'],data = X)


# **Max value 40 year age and lowest value at 65**

# In[ ]:


sns.heatmap(data = X,annot = True)


# In[ ]:




