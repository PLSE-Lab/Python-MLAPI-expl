#!/usr/bin/env python
# coding: utf-8

# This is a countrywide traffic accident dataset, which covers 49 states of the United States. The data is collected from February 2016 to December 2019, using several data providers, including two APIs that provide streaming traffic incident data.This data is always being updated and hence as a result pretty good data to work for real-life problem of accidents and hopefully we can find Answers to some meaningful question.
# ### Please Upvote if you find this kernel useful

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


df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_Dec19.csv")


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.sample(5)


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


#Lets check for missing values'


# In[ ]:





# In[ ]:


df.isna().sum()


# lets check them now as percentage

# In[ ]:


((df.isna().sum())/df.shape[0])*100


# In[ ]:


#Lets start off with dropping End_Lat,End_Lng,Number,Wind chill,Precipitation(in)       


# In[ ]:


df.columns


# In[ ]:


df2 = df.drop(['End_Lat', 'End_Lng','Number','Wind_Chill(F)','Precipitation(in)'],axis =1
    )


# In[ ]:


df2.head()


# In[ ]:


#Now lets check their correlation


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:



sns.set_palette('deep')


# In[ ]:


plt.figure(figsize = (16,12))
sns.heatmap(df2.corr(),cmap = 'Blues',annot = True,center = 0,linewidths=.5)


# In[ ]:


sns.countplot(df2['Source'])


# In[ ]:


plt.figure(figsize = (16,8))
df_pie = df2.groupby('Source').agg({'Source':'count'})
labels = df_pie.index
explode = (0, 0.1,0.1)
plt.pie(df_pie['Source'],labels = labels,explode =explode,autopct='%1.1f%%', shadow=True, startangle=140)
plt.show()


# In[ ]:


#Most of Data is from MapQuest actually followed by Bing and finally MapQuest-Bing.


# In[ ]:


df2.columns


# In[ ]:


df_1 = df2.groupby('State').agg({'Severity':'mean'})


# In[ ]:





# In[ ]:


plt.figure(figsize = (16,12))
sns.barplot(x  = df_1.Severity , y = df_1.index,orient = 'h')
plt.show()


# In[ ]:


#Seems like Tenesse,Wyomingand and Arizona on average get the most severe traffic due to accidents.


# In[ ]:





# In[ ]:


fig = plt.figure(figsize = (12,6))
fig.add_subplot(1, 2, 1)
sns.countplot(df2['Severity'])
fig.add_subplot(1, 2, 2)
df3 = df2['Severity'].value_counts()
plt.pie(df3,labels = df3.index)
plt.show()


# In[ ]:


#0 and 1 are almost negligible as compared to others


# In[ ]:


#This is a work in progress.Hopefully you Liked it.


# In[ ]:




