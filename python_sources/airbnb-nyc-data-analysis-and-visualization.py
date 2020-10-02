#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


# Importing the libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

#Reading the data file.
data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv', 
                   parse_dates=['last_review'])


# In[ ]:


# Check the dataframe, columns & other properties.
print(f'Shape : {data.shape}')


# In[ ]:


print({data.info()})


# In[ ]:


print(f'Data Five number summary: \n {data.describe()}')


# ### Five number summary incudes min, 25%, 50%, 75% and max values of the feature.
# With these summary we can figure which features are normaly distributed or which are skewed.
# 
# And, we can see from above numbers, **Following features are "Right Skewed" (Max values are too far from Median, i.e., 50 percentile)
# Price, minimum_nights,  number_of_reviews,  reviews_per_month, minimum_nights,  number_of_reviews  reviews_per_month**
# 
# **Following are Normaly distributed:
# latitude     longitude**
# 
# We can summarize this from boxplot also.
# 

# In[ ]:


# Checking for NULL values
data.isnull().sum()


# For current analysis, we will not considering name, hostname and last_review features.
# And dropping name and hostname as ethical standards.

# In[ ]:


data.drop(['name', 'host_name'],axis=1, inplace=True)


# #Filling the null values of reviews_per_month to Zero

# In[ ]:


min(data['last_review'])


# In[ ]:


data['number_of_reviews'][data['last_review'].isnull()].sum()


# 'last review' with null are data issue, actually these host were never reviewed. As sum of reviews is zero. Updating NAN of 'last_review' to 20000101, as min value is from 2011. 20000101 will be kind of 0 for NAN date values.

# In[ ]:


data.fillna({'reviews_per_month':0, 'last_review': pd.to_datetime('2000-01-01')}, inplace=True)


# In[ ]:


data.isnull().sum()


# ### Now, We will do Descriptive Statistical Analysis of the Data.

# In[ ]:


data.hist(column=['latitude', 'longitude', 'price', 'minimum_nights','number_of_reviews',
                 'reviews_per_month','calculated_host_listings_count','availability_365' ], figsize=(14,12));


# In[ ]:


fig,a =  plt.subplots(2,3, figsize=(16,10))

a[0][0].set_title('Price')
a[0][1].set_title('minimum_nights')
a[0][2].set_title('reviews_per_month')
a[1][0].set_title('number_of_reviews')
a[1][1].set_title('minimum_nights')
a[1][2].set_title('calculated_host_listings_count')

a[0][0].hist(data['price'][data['price']< 400]);
a[0][1].hist(data['minimum_nights'][data['minimum_nights']<25],bins=25);
a[0][2].hist(data['reviews_per_month'][data['reviews_per_month']<10]);
a[1][0].hist(data['number_of_reviews'][data['number_of_reviews']<200]);
a[1][1].hist(data['minimum_nights'][data['minimum_nights']<40]);
a[1][2].hist(data['calculated_host_listings_count'][data['calculated_host_listings_count']<10]);


# In[ ]:


sns.pairplot(data[['price', 'minimum_nights','number_of_reviews','reviews_per_month',
                  'calculated_host_listings_count','availability_365','room_type']])

