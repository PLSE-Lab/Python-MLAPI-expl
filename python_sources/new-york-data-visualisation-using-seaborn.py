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


#     What can we learn about different hosts and areas? 
#     What can we learn from predictions? (ex: locations, prices, reviews, etc)
#     Which hosts are the busiest and why?
#     Is there any noticeable difference of traffic among different areas and what could be the reason for it?

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


file_path = '/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv'
data = pd.read_csv(file_path)


# In[ ]:


data.head()


# **First glance information**
# 
# * name, host_name does not seems to be too important for this dataset. 
# * Should id and host-id should always be unique. if not how it will affect data have geo-related information. 
# * Both latitude/longitude and 2 categorical data neighbourhood_group and neighbourhood. 
# * It will be interesting to see if we can avoid either of these feature evaluate the price.
# * Room type will definately be affecting price. one this can be evalauted room_type co-relation with neighbour and reviews.
# * Have to evaluate availability , what is actually mean.

# In[ ]:


print('Size of the dataset: ', data.shape)
print('Number of feature is not very large, as compared to number of samples')


# In[ ]:


# check for values for each feature
data.isnull().sum()


# In[ ]:


# unique values count for each coloumn
data.nunique()


# ### Points to visualize 
# - Data distribution for neighbourhood_group and room_type.
# - Price compared to neighbourhood_group and room_type.
# - How review per month affect the Price.
# - How number of review affect the price for neighbourhood_group and room_type.
# - How availability and minimum nights affects the price.
# - Linear Relation of number of reviews vs review per month.
# - How calculated_host_listings_count affecting number of reviews and price.
# - Analyse data where host listed more than 1 property compared to others.
# - Analyse data when availability is 0 or 365 , to undertand this feature in more detail.
# - Can evalute the listing duration based on total number of reviews and review per month. And compare the price with the age of the listing.
# - To analyse the trend of region , minimum_nights requirement for each neighbourhood_group  and group type.
# - Room type compared to number of review , to analyse which room type get most reviews.

# In[ ]:


sns.set(rc={'figure.figsize':(20,30)})


# In[ ]:


## Data distribution for neighbourhood_group and room_type.
sns.catplot(x="neighbourhood_group", col='room_type', kind="count", data=data);


# In[ ]:


## Price compared to neighbourhood_group and room_type.
sns.catplot(x="neighbourhood_group", y="price", col="room_type", col_wrap=3,            data=data);


# In[ ]:


## How review per month affect the Price.
sns.relplot(x="reviews_per_month", y="price", data=data);


# In[ ]:


## How number of review affect the price for neighbourhood_group and room_type.
sns.relplot(x="reviews_per_month", y="price", hue='neighbourhood_group', col='room_type', 
            data=data);


# In[ ]:


## How availability and minimum nights affects the price.
sns.relplot(x="minimum_nights", y="price", hue = 'availability_365', data=data);


# In[ ]:


### Linear Relation of number of reviews vs review per month.
sns.relplot(x="reviews_per_month", y="number_of_reviews", data=data);


# In[ ]:


## How calculated_host_listings_count affecting number of reviews and price.
sns.relplot(x="calculated_host_listings_count", y="price", data=data);
sns.relplot(x="number_of_reviews", y="price", data=data);


# In[ ]:


sns.regplot(x="calculated_host_listings_count", y="price", data=data);


# In[ ]:


## Analyse data when availability is 0 or 365 , to undertand this feature in more detail.
sns.catplot(x="neighbourhood_group", y="price", col="availability_365", col_wrap=2,
            data=data.query("availability_365 == 0 or availability_365 == 365 "));


# In[ ]:


## Can evalute the listing duration based on total number of reviews and review per month.
## And compare the price with the age of the listing.
data['number_of_months'] = data['number_of_reviews']/ data['reviews_per_month']
sns.regplot(x="number_of_months", y="price", data=data);


# In[ ]:


## To analyse minimum_nights requirement for each neighbourhood_group.
sns.catplot(x="neighbourhood_group", y='minimum_nights', kind="bar", data=data);
sns.catplot(x="neighbourhood_group", y='minimum_nights', kind="box", data=data);


# In[ ]:


## Room type compared to number of review , to analyse which room type get most reviews.
sns.catplot(x="room_type", y='number_of_reviews', data=data);

