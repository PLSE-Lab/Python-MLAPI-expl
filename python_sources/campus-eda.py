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


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print("all libraries are imported")


# In[ ]:


df=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')


# In[ ]:


df=pd.DataFrame(df)


# In[ ]:


df.sample(5)


# In[ ]:


df.info()


# In[ ]:


s=df.shape
print("There are {} number of rows.".format(s[0]))
print("There are {} number of columns.".format(s[1]))


# In[ ]:


np.sum(pd.isnull(df))


# There are some missing values in salary column we replace them with zero

# In[ ]:


df['salary'] = df['salary'].replace(np.nan, 0)


# In[ ]:


np.sum(pd.isnull(df['salary']))


# In[ ]:


df.columns


# In[ ]:


df1 = df
df1['status'].values[df1['status']=='Not Placed'] = 0 
df1['status'].values[df1['status']=='Placed'] = 1
df1.status = df1.status.astype('int')
df1.head(2)


# **setting index to si_no**

# In[ ]:


df.set_index(df['sl_no'],inplace=True)


# ## Questions
# 1. Which factor influenced a candidate in getting placed?
# 2. Does percentage matters for one to get placed?
# 3. Which degree specialization is much demanded by corporate?
# 4. Play with the data conducting all statistical tests

# In[ ]:


features=['gender','ssc_p','ssc_b','hsc_p','hsc_b','hsc_s','degree_p','degree_t','workex','etest_p','specialisation','status','mba_p']
sns.heatmap(df[features].corr(),linewidth=0.2,cmap="YlOrRd",annot=True)


# * percentage  in ssc,hsc,degree has higher chance of geeting hired
# * percentage in etest doesn't play a role in hiring process

# In[ ]:


sns.countplot(df['gender'],palette=['#FF7799','#AABBFF'])


# * There are more number of males than females

# In[ ]:


sns.countplot(df["status"],palette=['#999900','#555555'])
plt.title("no of students placed")
plt.ylabel("no:of:students")


# In[ ]:


sns.catplot(x="status", y="ssc_p", jitter = False,data=df)
sns.catplot(x="status", y="hsc_p", jitter = False,data=df)
sns.catplot(x="status", y="degree_p", jitter = False,data=df)
sns.catplot(x="status", y="mba_p", jitter = False,data=df)


# * **percentage plays an role in getting placd**
# * **The bluedots indicates the candidate not placed and the orange indicates the candidate placed,we can observe that there is some percentage as minimum candidate whose percentage below the minimum are not placed**
# * **higher the percentage doesnot guarantee the placement**

# In[ ]:


df1=df[df['status']==1]
sns.countplot(x="specialisation",data=df)
plt.title("specialisation")
plt.ylabel("no_of_students_place")


# * marketing and finace has higher chance of getting selected 

# ## conclusion
# * **There is a correlation between the percentage of a candidate and placement status**
# * **There is some minimum percentage as the candidate whose percentage is less than minimum are not placed**
# * **The marketing and finance has highest placements when compared to marketing and HR**
# * **The percentage in employability test has no relation with placement status**
#      

# In[ ]:




