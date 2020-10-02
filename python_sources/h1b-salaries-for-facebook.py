#!/usr/bin/env python
# coding: utf-8

# I scraped this data from a website that posts H1B salaries for numerous companies. I wanted to visualize the data for some specific companies and eventually implement some machine learning models to make some predictions.

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
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Any results you write to the current directory are saved as output.

df1 = pd.read_csv("../input/Facebook_Salaries_2018.csv")
df1.head()


# In[ ]:


plt.figure(figsize=(12,6))
ax = sns.boxplot(x="State", y="Pay", data=df1,palette='rainbow')

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
ax.set_title("State vs Salary for Facebook")


# In[ ]:


df2=df1.set_index('Job_Role')


# In[ ]:


df2.head()


# In[ ]:


df2.loc['DATA SCIENTIST']


# In[ ]:


plt.figure(figsize=(10,6))
sns.distplot(df2['Pay'])


# In[ ]:


sns.pairplot(df2)


# In[ ]:


df2.describe()


# In[ ]:


df2.info()


# In[ ]:


corr_matrix = df2.corr()
corr_matrix ["Pay"].sort_values(ascending = False)


# In[ ]:




