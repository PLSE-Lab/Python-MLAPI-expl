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


import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


zomato_df=pd.read_csv("/kaggle/input/zomato-bangalore-restaurants/zomato.csv")
zomato_df.head()


# Removing unwanted columns

# In[ ]:


zomato_df.drop(['url','address','menu_item','reviews_list'],axis=1,inplace=True)
zomato_df.head()


# Renaming coumns in proper way

# In[ ]:


zomato_df.rename(columns={'approx_cost(for two people)':'Aprox_For_2','listed_in(type)':'Type','listed_in(city)':'City','name':'Name'},inplace=True)
zomato_df.head()


# Grouping Data by City

# In[ ]:


gk=zomato_df.groupby('City')
gk.head()


# Checking the null values

# In[ ]:


sns.heatmap(zomato_df.isnull())


# Removing null values

# In[ ]:


zomato_df['rate' and 'phone'].fillna(0,inplace=True)
zomato_df.drop('dish_liked',axis=1,inplace=True)
zomato_df.head()


# In[ ]:


sns.heatmap(zomato_df.isnull())


# In[ ]:


zomato_df['Aprox_For_2' and 'rest_type' and 'rate'].fillna(0,inplace=True)
zomato_df['rest_type'].fillna(0,inplace=True)
zomato_df['Aprox_For_2'].fillna(0,inplace=True)


# Checking after removing all the null values

# In[ ]:


sns.heatmap(zomato_df.isnull())


# Data Visualization

# In[ ]:


gk=zomato_df.groupby(['City','Name'])
gk.head()


# In[ ]:


zomato_df.shape


# Ploting graph for maximum rating

# In[ ]:


sns.catplot('rate',data=zomato_df,kind="count")
plt.title('No. of Restaurants with maximum rating')


# In[ ]:


zomato_df.rate.unique()


# Removing redundant values in rate column

# In[ ]:


zomato_df['rate']=zomato_df['rate'].astype(str)
zomato_df['rate']=zomato_df['rate'].apply(lambda x:x.replace('NEW','NAN'))
zomato_df['rate']=zomato_df['rate'].apply(lambda x:x.replace('-','NAN'))


# In[ ]:


zomato_df['rate']=zomato_df['rate'].apply(lambda x:x.replace('/5',''))
zomato_df.head()


# Ploting Grapgh between rate and Type

# In[ ]:


type_plt=pd.crosstab(zomato_df['rate'],zomato_df['Type'])
type_plt.plot(kind='bar',stacked=True);
plt.title('Type - Rating',fontsize=15,fontweight='bold')
plt.ylabel('Type',fontsize=10,fontweight='bold')
plt.xlabel('Rating',fontsize=10,fontweight='bold')
plt.xticks(fontsize=8,fontweight='bold')
plt.yticks(fontsize=5,fontweight='bold');


# Changing in column Aprox_For_2

# In[ ]:


zomato_df['Aprox_For_2']=zomato_df['Aprox_For_2'].astype(str)
zomato_df['Aprox_For_2']=zomato_df['Aprox_For_2'].apply(lambda x:x.replace(',',''))
zomato_df.info()


# Creating a new column which tell the restaurant is in budget

# In[ ]:


IN_Budget=[]
for Aprox_For_2 in zomato_df.Aprox_For_2:
    if int(Aprox_For_2) <= 800:
        IN_Budget.append('In Budget')
    else:
        IN_Budget.append('Expensive')


# Inserting new column in dataset
# 

# In[ ]:


zomato_df['In_Budget']=IN_Budget
zomato_df.head(77)


# Ploting Graph of In_Budget column which gives the no of restaurant in budget

# In[ ]:


sns.catplot('In_Budget',data=zomato_df,kind="count")
plt.title('No. of in budget reataurants')


# Ploting graph how many restaurant delivering online

# In[ ]:


sns.catplot('online_order',data=zomato_df,kind='count')
plt.title('Restaurants delivering online or Not')

