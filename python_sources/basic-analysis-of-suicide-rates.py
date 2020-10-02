#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


df = pd.read_csv('../input/master.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.groupby('sex').agg({'population':'sum'})


# In[6]:


df.groupby('sex').agg({'suicides_no':'sum'})


# In[8]:


d = df.groupby(['sex','year']).agg({'suicides/100k pop':'mean'})


# In[9]:


d = d.reset_index()


# In[10]:


d.head()


# In[11]:


female = d.loc[d['sex']=='female',:]


# In[12]:


female.head()


# In[13]:


male = d.loc[d['sex']=='male',:]


# In[14]:


male.head()


# In[15]:


import matplotlib.pyplot as plt


# In[17]:


female.plot(x='year',y='suicides/100k pop',kind='bar')
plt.xlabel('Year')
plt.ylabel('Suicides/100k pop')
plt.title('Female Sucide rates/100k population')


# In[18]:


male.plot(x='year',y='suicides/100k pop',kind='bar')
plt.xlabel('Year')
plt.ylabel('Suicides/100k pop')
plt.title('Male Sucide rates/100k population')


# In[19]:


countrywise = df.groupby('country').agg({'suicides/100k pop':'mean'}).sort_values(by='suicides/100k pop')


# In[20]:


countrywise.head()


# In[21]:


countrywise = countrywise.reset_index()


# In[22]:


countrywise.plot(kind='barh')


# In[23]:


#Case study of south africa

sa = df[df['country']=='South Africa'].groupby('year').agg({'suicides_no':'sum','population':'mean'})


# In[24]:


sa = sa.reset_index()


# In[25]:


sa.head()


# In[26]:


sa = sa.astype(int)


# In[27]:


sa.head()


# In[28]:


year1 = sa.iloc[:,0].values


# In[29]:


pop1 = sa.iloc[:,2].values


# In[30]:


plt.plot(year1,pop1,color='green')


# Populaion increase vs year in South Africa

# 

# In[31]:


suicide_no1 = sa.iloc[:,1].values


# In[32]:


plt.plot(year1,suicide_no1)


# In[33]:


plt.subplot(1,2,1)
plt.plot(year1,suicide_no1)
plt.xlabel('Year')
plt.ylabel('Suicides Numbers')
plt.title('Year vs Sucide number in South Africa')

plt.subplot(1,2,2)
plt.plot(pop1,suicide_no1)
plt.xlabel('Population')
plt.ylabel('Suicides Numbers')
plt.title('Population vs Sucide number in South Africa')


# Thus we can see that as year increase, there is similar increase in population, due to which there are a similar pattern in increase in suicide rates.

# In[34]:


globaldata = df.groupby('year').agg({'suicides_no':'sum'})


# In[35]:


globaldata.head()


# In[36]:


globaldata = globaldata.reset_index()


# In[38]:


plt.plot(globaldata.year,globaldata.suicides_no)
plt.title('Year vs Suicides Globally')
plt.xlabel('Year')
plt.ylabel('Suicides')


# Thus, there is a rapid decrease in suicide rate during 2013-2015 .

# In[ ]:




