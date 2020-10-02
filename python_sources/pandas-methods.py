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


df = pd.read_csv('../input/telecom_churn.csv')
df.head()
df.columns
df.info()
df.sort_values(by='Total day charge', ascending=False).head()
df['Churn'] = df['Churn'].astype('int64')
df['Churn'].value_counts()


# In[ ]:


df.loc[0:5, 'State':'Area code']


# In[ ]:


df.iloc[0:5, 4:6]


# In[ ]:


df.iloc[-1:]


# Appling function to cells, columns and rows. Use **apply()** function

# In[ ]:


df[df['State'].apply(lambda state:state[0] == 'W')].head()


# **map** method can be used to **replace the values in the columns** by passing a dictionary of the form *{old_value: new_value}* as arguments.

# In[ ]:


d = {'No' : False, 'Yes' : True}
df['International plan'] = df['International plan'].map(d)
df.head()


# In[ ]:


df = df.replace({'Voice mail plan': d})
df.head()


# **Group by** - Syntax **df.groupby(by=grouping_columns)[columns_to_show].function()**

# In[ ]:


columns_to_show = ['Total day minutes', 'Total eve minutes', 'Total night minutes']
df.groupby(['Churn'])[columns_to_show].describe(percentiles=[])


# **Summary tables** - Helps us how the observations for our sample are distributed in context of two variables.
# We can use **crosstab** Contingency Table

# In[ ]:


pd.crosstab(df['Churn'], df['International plan'])


# **DataFrame transformations** - Add/Remove columns to existing DataFrame
# Calculate the total number of calls for all users

# In[ ]:


total_calls = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls']
df.insert(loc=len(df.columns), column='Total calls', value=total_calls) 
df.head()


# Add a column more easily without creating an intermediate Series instance

# In[ ]:


df['Total charge'] = df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']
df.head()


# **Delete columns and rows**
# Method - **drop**
# Parameters - **indexes**, **axis**: **1** to delete columns and **0** to delete rows and **inplace**: **True** change in original DataFrame and **False** doesn't change original DataFrame

# In[ ]:


# Delete columns
df.drop(['Total calls', 'Total charge'], axis=1, inplace=True)
# Delete rows
df.drop([1, 2]).head()


# **Predicting Telecom Churn**
# Relation between churn rate and international plan and visual analysis with **Seaborn**

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (8, 6)
sns.countplot(x='International plan', hue='Churn', data=df);


#     Relation between **Customer Service Call** with **Churn**

# In[ ]:


pd.crosstab(df['Churn'], df['Customer service calls'], margins=True)
sns.countplot(x='Customer service calls', hue='Churn', data=df)


# Picture clearly states that the **Churn** rate strongly increases starting from 4 calls to the service center.
# 
# Add a binary attribute to our DataFrame - **Customer service calls > 3**

# In[ ]:


df['Many_service_calls'] = (df['Customer service calls'] > 3).astype('int')
df.head()


# Relationship between **Many_service_calls** and **Churn**

# In[ ]:


pd.crosstab(df['Many_service_calls'], df['Churn'], margins=True)
sns.countplot(x='Many_service_calls', hue='Churn', data=df);


# Relationship between **Many_service_calls** and **International plan** with **Churn**

# In[ ]:


pd.crosstab(df['Many_service_calls'] & df['International plan'], df['Churn'])


# In[ ]:


sns.countplot(x='International plan', hue='Churn', data=df);

