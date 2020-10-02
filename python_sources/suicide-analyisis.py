#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#7 This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from textblob import TextBlob,Word
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


df=pd.read_csv("/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv")


# Description and analysis of null values

# In[ ]:


df.describe()
df.isnull().sum()


# Removing unwanted columns

# In[ ]:


df=df.drop(['HDI for year','generation'],axis=1)
df.head()


# **yearwise suicide rate **

# In[ ]:



su=df.groupby(['year'])['suicides_no'].sum()
su=su.to_frame()
su.reset_index(level=0, inplace=True)
su.plot(x='year',y='suicides_no',kind='line')
su.head()

#su.loc[su['year']==2015]


# counts MALE OR FEMALE 
# 

# In[ ]:


df.head(100)
s=df.groupby(['sex'])['suicides_no'].sum()
print(s)


# number of suicides based on age

# In[ ]:


s=df.groupby(['age','sex'])['suicides_no'].sum()
print(s)


# In[ ]:


sns.barplot(x='sex',y='suicides_no',hue='age',data=df)


# In[ ]:


male=df.loc[df['sex']=='male']
female=df.loc[df['sex']=='female']


male_plot=sns.lineplot(x='year',y='suicides_no',data=male)
female_plot=sns.lineplot(x='year',y='suicides_no',data=female)
plt.legend(['males','females'])


# countrywise suicide analysis

# In[ ]:


ca=pd.DataFrame(df.groupby(['country'])['suicides_no'].sum())
ca.reset_index(level=0, inplace=True)
ca.sort_values(by=['suicides_no'],inplace=True,ascending=False)
ca.head(10)


# 
