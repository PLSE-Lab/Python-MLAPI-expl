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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
sns.set(font_scale=1.25)
#%matpotlib inline


# In[ ]:


data = pd.read_csv("/kaggle/input/housing-in-london/housing_in_london_monthly_variables.csv")
data.head()


# In[ ]:


data.info()


# In[ ]:


import datetime


# In[ ]:


data['date'] = pd.to_datetime(data['date'])


# In[ ]:


#for column in data.columns.unique():
 #   print('\n', column,'\n', data[column].unique(), '\n')
    


# In[ ]:


data.isna().any()


# In[ ]:


data = data.dropna(axis = 0)


# In[ ]:


#data = data.fillna(method = 'bfill')


# In[ ]:


#data = data.fillna(method = 'ffill')


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


plt.figure(figsize=(10,7))
sns.distplot(data['average_price'])
plt.show()


# In[ ]:


plt.figure(figsize=(13,7))
sns.lineplot(x= 'date', y='average_price', data= data)
plt.title("Average House Price over the Years")


# In[ ]:


plt.figure(figsize=(15,7))
sns.lineplot(x= 'date', y='no_of_crimes', data= data, color = 'r')
plt.title("Crime rate over the Years")


# In[ ]:


plt.figure(figsize=(13,7))
sns.lineplot(x= 'date', y='houses_sold', data= data)
plt.title("Number of house sold each years")


# In[ ]:


plt.figure(figsize=(25,15))
sns.scatterplot(x= 'date', y='houses_sold', hue = 'area', data= data)
plt.title("Number of house sold over the years")


# In[ ]:


plt.figure(figsize=(20,13))
sns.lineplot(x= 'date', y='average_price', hue = 'area', data= data)
plt.title("Area_wise house prices")


# In[ ]:


plt.figure(figsize=(20,13))
sns.lineplot(x= 'date', y='no_of_crimes', hue = 'area', data= data)
plt.title("Area wise number of crimes")


# In[ ]:


df = pd.read_csv("/kaggle/input/housing-in-london/housing_in_london_yearly_variables.csv")
df.head()


# In[ ]:


df.isna().sum()


# In[ ]:


df = df.fillna(method = 'bfill')
df = df.fillna(method = 'ffill')


# In[ ]:


df.isnull().sum()


# In[ ]:


#df = df.dropna(axis = 0)


# In[ ]:


#df.info()


# In[ ]:


df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year


# In[ ]:


df.info()


# In[ ]:


df.isnull().any().sum()


# In[ ]:


#for col in df.columns:
#    print('\n', col , '\n', df[col].unique(), '\n')


# In[ ]:


df['mean_salary'] = df['mean_salary'].replace('-' , 0)
df['mean_salary'] = df['mean_salary'].replace('#' , 0)
df['mean_salary'] = df['mean_salary'].astype(int)
df['recycling_pct'] = df['recycling_pct'].replace('na', 0)
df['recycling_pct'] = df['recycling_pct'].astype(int)


# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(10,7))
sns.distplot(df['median_salary'], color = 'g')
plt.legend()
sns.distplot(df['mean_salary'], color= 'r')
plt.legend()
plt.show()


# In[ ]:


plt.figure(figsize=(10,5))
sns.lineplot(x= 'year', y = 'median_salary', data =  df)
sns.lineplot(x= 'year', y = 'mean_salary', data =  df, color= 'r')


# In[ ]:


plt.figure(figsize=(15,7))

sns.lineplot(x='date', y='life_satisfaction', data=df)

plt.show()


# In[ ]:




