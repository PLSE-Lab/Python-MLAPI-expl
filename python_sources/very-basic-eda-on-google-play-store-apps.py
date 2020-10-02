#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# load file into dataframe
apps = pd.read_csv('../input/googleplaystore.csv')


# In[ ]:


# check top 5 records
apps.head()


# In[ ]:


#check last 5 records
apps.tail()


# In[ ]:


# get details of columns(names,row count,data type,missing value count),rows(total row counts) & index 
apps.info()


# In[ ]:


# total count of missing values column wise
apps.isnull().sum()


# In[ ]:


# total count of non missing values column wise
apps.notnull().sum()


# In[ ]:


# print column names with no missing values
apps.columns[apps.notnull().all()]


# In[ ]:


# print columns with at least one missing value
apps.columns[apps.isnull().any()]


# In[ ]:


# desribe qunatitative data column
# average rating for app is 4.193338
# data shows there is outlier in rating column
apps.describe()


# In[ ]:


# describe qualitative data columns
# App column has 9660 unique values/Out of which "ROBLOX" is most common value with highest count(9)
# Category column has 34 unique values/Out of which "Family" is most common value with highest count(1972)
# Reviews column has 6002 unique values/Out of which "0" is most common value with highest count(596)
# Size columns has 462 unique values/Out of which "Varies with device" is the most common value with highest count(1695)
# Installs columns has 22 unique values/Out of which "1,000,000+" is the most common value with highest count(1579)
# Type column has 3 unique values/Out of which "Free" is the most common value with highest count(10039) 
# Price column has 93 unique value/Out of which "0" is the most common value with highest count(10040)
# Content Rating column has 6 unique values/Out of which "Everyone" is the most common value with highest count(8714)
# Genres column has 120 unique values/Out of which "Tools" is the most common value with highest count(842)
# Last Updated column has 1378 unique values/Out of which "August 3, 2018" is the most common value with highest count(326)
# Current Ver has 2832 unique values/ Out of which "Varies with device" is the most common value with highest count(1459)
# Android Ver has 33 unique values/Out of which "4.1 and up" is the most common value with highest count(2451)
apps.describe(include=[np.object])


# In[ ]:


#Make column names to lower case & remove space between them with "_"
apps.columns = ['_'.join(col.split(" ")).lower() for col in apps.columns.tolist()]


# In[ ]:


#Remove Outlier from rating column
apps.drop(apps[apps.rating==19].index[0], inplace=True)


# In[ ]:


#Replace missing value with zero in rating column
apps.rating = apps.rating.fillna(0)


# In[ ]:


#Change reviews column datatype to int
apps.reviews = apps.reviews.apply(lambda x:int(x))


# In[ ]:


#Change size column datatype to float by removing character M,k and "Varies with device" with None
#Convert size unit to MB only
def clean_size_column(x):
    if 'M' in x:
        return float(x.replace('M',''))
    elif 'Varies with device' in x:
        return None
    elif 'k' in x:
        return (float(x.replace('k',''))/1000)

apps['size'] = apps['size'].apply(clean_size_column)


# In[ ]:


# Remove + and , from installs column to make it integer column type
apps["installs"] = apps["installs"].apply(lambda x: int(x.replace("+","").replace(",","")))


# In[ ]:


# Remove $ character from price column values and make it float column type
apps["price"] = apps["price"].apply(lambda x: float(x.replace("$","")))


# In[ ]:


#Convert last_updated colum to date time column type
apps["last_updated"] = pd.to_datetime(apps['last_updated'], format='%B %d, %Y')


# In[ ]:


#Remove Na's from original dataset without zero values in installs,reviews column
#Log of zero values will create problem while creating pair plot
apps_clean = apps.drop_duplicates(subset="app")
apps_clean = apps_clean[(apps_clean.installs !=0) & (apps_clean.reviews!=0)].dropna()


# In[ ]:


# Plot pair plot for new dataset
# Plot plot shows the relationship between different columms of data
# If will generate plot only for numreic value by default
sns.pairplot(apps_clean)
plt.show()
# By looking at plot one can observed that install,reviews has high number of overlapping data so histogram is not clear for them.
# Apply log scale on above mentioned columns and then plot again


# In[ ]:


apps_clean["reviews"] = np.log(apps_clean["reviews"])
apps_clean["installs"] = np.log(apps_clean["installs"])


# In[ ]:


# Pair plot after applying Logarithmic on installs,reviews column
sns.pairplot(apps_clean)
plt.show()


# 
# **Observations from above pair plot**
# #High number of the android application has rating greator than equal to 4
# #Reviews and installs fields has strong positive co-relations
# #Size column is right skewed becuase most of applications size between 0 to 50

# In[ ]:


# Pair plot for diffrenent value of type categorical varibale
sns.pairplot(apps_clean,hue="type")
plt.show()


# **Observations**
# #free/less priced applications has higher review/rating than paid application
# #review/install has strong co-relation for paid/free application
# #free app has been installed more than paid app
# #review and price are negatively co-related

# ****Category Wise App Split****

# In[ ]:


apps_clean.category.value_counts().plot(kind='pie', autopct='%1.0f%%', pctdistance=0.9, radius=1.2,figsize=(10,10))
plt.ylabel('')
plt.xlabel('')
plt.show()


# ****Content Rating Wise Split****

# In[ ]:


apps_clean.content_rating.value_counts().plot(kind='pie', autopct='%1.0f%%', pctdistance=0.9, radius=1,figsize=(10,10))
plt.ylabel('')
plt.xlabel('')
plt.show()


# ***Type wise split***

# In[ ]:


apps_clean.type.value_counts().plot(kind='pie', autopct='%1.0f%%')
plt.ylabel('')
plt.xlabel('')
plt.show()


# In[ ]:


apps_clean.info()


# **Categories wise average rating**

# In[ ]:


apps_clean.groupby('category').rating.mean().sort_values(ascending=False).plot(kind='bar',figsize=(10,10))
plt.ylabel('Avg. Rating')
plt.show()


# **Category wise total review**

# In[ ]:


apps_clean.groupby('category').reviews.sum().sort_values(ascending=False).plot(kind='bar',figsize=(10,10))
plt.ylabel('Total Review Count')
plt.show()


# **Category wise mean size**

# In[ ]:


apps_clean.groupby('category')["size"].mean().sort_values(ascending=False).plot(kind='bar',figsize=(10,10))
plt.ylabel('Avg.Size')
plt.show()


# **Category wise installs**

# In[ ]:


apps_clean.groupby('category').installs.mean().sort_values(ascending=False).plot(kind='bar',figsize=(10,10))
plt.ylabel('Avg. Installation')
plt.show()


# **Category wise free/paid apps**

# In[ ]:


apps_clean.groupby(['category','type']).size().unstack().plot(kind='bar',stacked=True,figsize=(9,9))
#installs.mean().sort_values(ascending=False).plot(kind='bar',figsize=(10,10))
plt.ylabel('Count')
plt.show()


# In[ ]:




