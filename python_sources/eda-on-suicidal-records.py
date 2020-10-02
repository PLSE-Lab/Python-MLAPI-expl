#!/usr/bin/env python
# coding: utf-8

# In[35]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.


# In[36]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[37]:


df = pd.read_csv('../input/master.csv')


# In[38]:


df.head()


# In[39]:


df.describe()


# In[40]:


df.info()


# In[41]:


df.dtypes


# In[42]:


#lets drop all those columns which contains more the 30% null value
#we can drop HDI and country-year columns
df2 = df[[col for col in df if df[col].count()/len(df)>=.50]]


# In[43]:


df2.drop('country-year',axis=1,inplace=True)


# In[44]:


df2.head()


# In[45]:


#Now lets change the catagorical datato numeric category
age = df['age'].unique()
A ={}
j=0
for i in age:
    A[i] = j
    j+=1
df2['age'] = df2['age'].map(A)
a = {v: k for k, v in A.items()}
df2.head()


# In[46]:


gen = df['generation'].unique()
G ={}
j=0
for i in gen:
    G[i] = j
    j+=1
df2['generation'] = df2['generation'].map(G)
g = {v: k for k, v in G.items()}
df2.head()


# In[47]:


sex = df['sex'].unique()
S ={}
j=0
for i in sex:
    S[i] = j
    j+=1
df2['sex'] = df['sex'].map(S)
S
s = {v: k for k, v in S.items()}
df2.head()


# In[48]:


country = df['country'].unique()
C ={}
j=0
for i in country:
    C[i] = j
    j+=1
df2['country'] = df['country'].map(C)
c = {v: k for k, v in C.items()}
df2.head()


# In[49]:


df2.tail()


# In[50]:


df2.corr()


# In[51]:


sns.heatmap(df2.corr());


# In[52]:


county = df[['country','suicides_no']].groupby('country',as_index=False).sum().sort_values(by='suicides_no',ascending=False)
fig=plt.figure(figsize=(20,10))
sns.barplot(x=county['country'],y=county['suicides_no'],data=county)
plt.xticks(rotation=90)
plt.title('World-wide total suicides from 1985-2016');


# #### World wide most of the people who commit suicides are from russia, united states,japan and most of them are male[](http://)

# In[53]:


country = df[['country','suicides_no']].groupby('country',as_index=False).sum().sort_values(by='suicides_no',ascending=False).head(10)
sns.barplot(x='country',y='suicides_no',data=country)
plt.xticks(rotation=90)
plt.title('Top 10 countries in total suicides');


# #### most of the people who commited suicide are between 35-54 age and most of them were male.

# In[54]:


age_suicide = df[['age','sex','suicides_no']].groupby(['age','sex'],as_index=False).sum()
sns.barplot(x='age',y='suicides_no',hue='sex',data=age_suicide)
plt.xticks(rotation=90)
plt.title('Age wise total suicides');


# #### Generation boomers have more tendency to commit suicides

# In[55]:


sns.barplot(x='generation',y='suicides_no',hue='sex',data=df[['generation','suicides_no','sex']].groupby(['generation','sex'],as_index=False).sum())
plt.xticks(rotation=90)
plt.title('Generation wise total suicides');


# In[56]:


a = df[['sex','suicides_no']].groupby('sex',as_index=False).sum()
sns.barplot(x='sex',y='suicides_no',data=a)
plt.title('Gender wise total suicides');


# In[57]:


sns.barplot(x='year',y='suicides_no',data=df[['year','suicides_no']].groupby('year',as_index=False).sum())
plt.xticks(rotation=90)
plt.title('Year wise total suicides');


# #### between 2015 and 2016 united states ans russian govt. tends to control suicides also world wide number of suicides decreased

# In[58]:


data=df[['year','suicides_no','sex','country']].groupby(['year','country','sex'],as_index=False).sum().sort_values(by=['suicides_no'],ascending=False)
data[data['year']==2015].head(10)


# In[59]:


data=df[['year','suicides_no','sex','country']].groupby(['year','country','sex'],as_index=False).sum().sort_values(by=['suicides_no'],ascending=False)
data[data['year']==2016].head(15)

