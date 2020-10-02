#!/usr/bin/env python
# coding: utf-8

# In[21]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# READ DATA

# In[22]:


temp=pd.read_csv('../input/Salaries.csv')
temp.head()


# In[23]:


temp.describe()


# In[24]:


temp.dtypes


# TIME PERIODS

# In[25]:


temp.Year.unique()


# JOB STATUS

# In[26]:


temp.Status.unique()


# HOW MANY JOB POSITION WE ARE TALKING ABOUT

# In[27]:


temp['Id'].count()


# In[28]:


temp.groupby('Status').count()


# EACH STATUS JOB MEDIAN

# In[29]:


tempFT=temp[temp['Status']=='FT']
tempPT=temp[temp['Status']=='PT']
temp0=temp[~temp['Status'].isin(['PT','FT'])]


# In[30]:


print('PT',tempPT['TotalPay'].median(),',','FT',tempFT['TotalPay'].median(),',','unknown',temp0['TotalPay'].median())


# WHAT ABOUT THE JOB POSITION NUMBER

# In[31]:


tempa=temp.groupby('JobTitle').count()
tempb=tempa.loc[:,['Id']]
tempc=tempb.sort_values(by='Id')


# In[32]:


tempc.head()


# In[33]:


tempc.tail()


# MEDIAN PAY OF GROUPY BY JOBTITLE 

# In[34]:


tempa=temp.loc[:,['JobTitle','TotalPay']].groupby('JobTitle').median()


# In[35]:


tempb=tempa.sort_values(by='TotalPay')


# In[36]:


tempb.median()


# WELFARE OF POLICE OFFICER

# In[37]:


tempa=temp.loc[:,['JobTitle','TotalPay']]


# In[38]:


tempa[tempa['JobTitle']=='Police Officer'].mean()


# In[39]:


tempa=temp[temp['JobTitle'].str.startswith('Pol')]
tempa['TotalPay'].median()


# In[40]:


tempa['JobTitle'].unique()


# In[41]:


tempb=tempa.loc[:,['JobTitle','TotalPay']]
tempb.groupby('JobTitle').median()


# In[42]:


tempb.groupby('JobTitle').count()

