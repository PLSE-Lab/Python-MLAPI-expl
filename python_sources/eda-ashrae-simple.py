#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Hello  
# This is my first attempt at exploratory data analysis. In this notebook, I would like to try to describe the data of the current competition, to find oddities and patterns in the data.  
# I will be glad to advice and comments!)

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # building_metadata.csv exploration

# Load data and describe it a bit

# In[ ]:


building_df = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")
print(building_df.shape)
building_df.head()


# In[ ]:


building_df.describe()


# Check data types of our columns. Many columns have a larger data type than they should:

# In[ ]:


building_df.dtypes


# Missed Data Percentage

# In[ ]:


building_df.isna().sum() / building_df.shape[0]


# We set a smaller data type for columns that do not have missing values (we will work with the remaining columns later)

# In[ ]:


building_df['site_id'] = building_df['site_id'].astype(np.uint8)
building_df['building_id'] = building_df['building_id'].astype(np.uint16)
building_df['square_feet'] = building_df['square_feet'].astype(np.uint32)
building_df.dtypes


# Let's look at the distribution of data on various features. For the site_id and primary_use, you can use the bar plot

# In[ ]:


cat_columns = ['site_id', 'primary_use']
plt.figure(figsize=(15, 5))
for ind, col in enumerate(cat_columns):
    plt.subplot(1, len(cat_columns), ind+1)
    plt.title(col)
    building_df[col].value_counts(sort=False).plot(kind='bar')


# For year_build we can use simple plot, and see how many building was build in each year

# In[ ]:


plt.figure(figsize=(10,5))
y = building_df['year_built'].value_counts(sort=False)
plt.title('year_built')
plt.xticks(rotation=45)
y.index = pd.to_datetime(y.index.astype(int), format='%Y')
y = y.sort_index()
plt.grid()
plt.plot(y)
del y


# Let's convert prinary_use column to the number:

# In[ ]:


le = LabelEncoder()
building_df["primary_use_enc"] = le.fit_transform(building_df["primary_use"]).astype(np.uint8)


# And create correlation matrix for our data. As we can see squeree_feet and floor_count is correlate well. This fact may be useful in recovering missing values for the number of floors.

# In[ ]:


plt.figure(figsize=(6,6))
sns.heatmap(building_df.corr(), square=True, annot=True)


# Let's see the dependence of the number of floors on the logarithm of the area

# In[ ]:


plt.figure(figsize=(5, 5))
plt.xlabel('square_feet')
plt.ylabel('floor_count')
X = building_df[pd.notnull(building_df['floor_count'])]
plt.scatter(np.log1p(X['square_feet']), X['floor_count'])
del X


# # weather_train.csv exploration

# Load data and describe it a bit

# In[ ]:


weather_train = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")
print(weather_train.shape)
weather_train.head()


# In[ ]:


weather_train.describe()


# In[ ]:


weather_train.dtypes


# Missed data percentage

# In[ ]:


weather_train.isna().sum() / weather_train.shape[0]


# Change dtypes of the columns

# In[ ]:


float_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature',
        'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']


# In[ ]:


weather_train['site_id'] = weather_train['site_id'].astype(np.uint8)
weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])
for i in float_cols:
    weather_train[i] = weather_train[i].astype(np.float32)
weather_train.dtypes


# Check correlations between features 

# In[ ]:


plt.figure(figsize=(8, 8))
sns.heatmap(weather_train.corr(), square=True, annot=True)


# In[ ]:


tmp = weather_train[weather_train['site_id'] == 0]
plt.figure(figsize=(20, 15))
for ind, col in enumerate(float_cols):
    plt.subplot(4, 2, ind + 1)
    plt.xticks(rotation=30)
    plt.grid()
    plt.title(col)
    plt.plot(tmp['timestamp'], tmp[col])


# ### more later...
