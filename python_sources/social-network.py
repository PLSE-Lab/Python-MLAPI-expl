#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("/kaggle/input/social-network-ads/Social_Network_Ads.csv")
df.head()


# The average salary based on age

# In[ ]:


mean_salary = df[['Age', 'EstimatedSalary']].groupby(['Age'], as_index=False).mean().sort_values(by='Age')
mean_salary


# In[ ]:


plt.xlabel("Age")        
plt.ylabel("meanSalary")    
plt.grid()     

plt.plot(mean_salary.EstimatedSalary)


# **The average salary based on gender**

# In[ ]:


gender_salary = df[['Gender', 'EstimatedSalary']].groupby(['Gender'], as_index=False).mean().sort_values(by='Gender')
gender_salary


# In[ ]:


gender_age = df[['Gender', 'Age']].groupby(['Gender'], as_index=False).mean().sort_values(by='Gender')
gender_age

