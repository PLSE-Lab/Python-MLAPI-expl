#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ***First of all, we are looking general information for our data.***

# In[ ]:


df = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')
df.info()


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(df.corr(),annot=True, lw=.5,fmt='.2f',ax=ax)
plt.show()


# In[ ]:


df["country"].unique() #All countries that we have in our data.


# In[ ]:


df = df[df["country"] == 'Turkey'] #Filtering for Turkey
df.head()


# In[ ]:


df.describe() #Turkey's numerical data information.


# In[ ]:


#correlation map for Turkey
f,ax = plt.subplots(figsize = (15,15))
sns.heatmap(df.corr(),annot=True, lw=.5,fmt='.2f',ax=ax)
plt.show()


# In[ ]:


#scatter plot
#year and suicide number

df.plot(kind='scatter',x='year',y='suicides_no')
plt.xlabel('year')
plt.ylabel('suicide no')
plt.show()


# In[ ]:


#Lets look at maximum suicide number per year and visualize it.
max_suicides= df.groupby(['year']).suicides_no.max() #max suicide numbers
print(max_suicides)
f=plt.subplots(figsize=(10,5))
plt.bar(df['year'].unique(),max_suicides)
plt.xlabel('year')
plt.ylabel('suicide number')
plt.title('maximum number of suicides per year')
plt.show()


# In[ ]:


#finding min suicide numbers and its bar graph
min_suicides = df.groupby(['year']).suicides_no.min()

plt.figure(figsize=(10,5))
plt.bar(df.year.unique(),min_suicides)
plt.xlabel('year')
plt.ylabel('suicide number')
plt.title('minimum number of suicides per year')

plt.show()


# In[ ]:


#now we are going to look at total suicide numbers per year
total_suicides = df.groupby(['year']).suicides_no.sum()
total_suicides


# In[ ]:


#lets visualize it.
#line graph
plt.figure(figsize=(10,10))
plt.plot(df['year'].unique(),total_suicides,lw=2,marker='o')
plt.xlabel('year')
plt.ylabel('total suicide no')
plt.title('relationship between year and total suicide numbers -line graph')
plt.show()


# ****There is a dramatic increase between 2011-2013 years.****

# In[ ]:


#bar graph
plt.figure(figsize=(10,5))
plt.bar(df['year'].unique(),total_suicides)
plt.xlabel('year')
plt.ylabel('total suicide no')
plt.title('relationship between year and total suicide numbers -bar graph')
plt.show()


# ****After this graph, we can easily say that maximum suicide number graph and total suicide number graphs are almost same.****

# #### Now we are going to look genders.

# In[ ]:


plt.figure(figsize=(13,5))
sns.barplot(df.sex,df.suicides_no,hue=df.age)
plt.title('Suicide number by Gender')
plt.show()


# ### In this graph, we can conclude that most of the suicides in Turkey are between 35-54 years old.
# ### Also, we can conclude that males are more likely to suicide than females.

# In[ ]:


#now we are going to look at the relationship between GDP per capita and suicide number
plt.figure(figsize=(10,8))
gdp_suicides = df.groupby(['gdp_per_capita ($)']).suicides_no.sum() #total suicide numbers for each gdp value.
sns.lineplot(df['gdp_per_capita ($)'].unique(),gdp_suicides)
plt.xlabel('gdp per capita ($)')
plt.ylabel('suicides no')
plt.title('relationship between GDP per capita and suicide number')
plt.show()


# In[ ]:




