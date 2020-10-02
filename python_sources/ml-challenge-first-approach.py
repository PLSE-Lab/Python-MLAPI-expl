#!/usr/bin/env python
# coding: utf-8

# This kernel shows our very first attempt to solve the ML Challenge problem

# In[ ]:


# import required libraries

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


# # 1. Loading data
# We use Python Pandas package to load and preprocess our datasets. We start by loading data from csv file into DataFrames.

# In[ ]:


train_df = pd.read_csv('../input/train_data.csv',index_col=None)
test_df = pd.read_csv('../input/test_data.csv',index_col=None)

# print some information
print(train_df.info())
print(test_df.info())


# ### Which features are categorical?
# 
# A categorical variable is a variable that can take on one of a limited, and usually fixed number of possible values. The most common two types of categorical features are nominal, which has no quantitative value and ordinal, which has some logical order
# 
# * Nominal: town, flat_type, block, street_name, flat_model, nearest_mrt_station, lease_commence_date
# * Ordinal: storey_range, max_floor_lvl, month
# 
# ### Which features are numerical? 
# Numerical variables are variables where the measurement or number has a numerical meaning. They can be further classified into discrete and continuous variables
# 
# * Discrete: remaining_lease, nearest_mrt_distance
# * Continous: floor_area_sqm

# In[ ]:


# preview the data
train_df.head()


# # 2. Feature Engineering
# 

# ## Check for null value

# In[ ]:


# Count number of NaN values in each column
train_df.isna().sum()


# ## Analysing features
# ### Assumptions
# * Units in center area are more expensive than other
# * Units on higher storeys are more expensive
# * remaining_lease, nearest_mrt_distance, floor_area_sqm are also important features when calculating unit price

# In[ ]:


# Average unit price grouped by storey_range
train_df[['storey_range', 'resale_price']].groupby(['storey_range'], as_index=False).mean().sort_values(by='resale_price', ascending=False)


# In[ ]:


# Average unit price grouped by town
train_df[['town', 'resale_price']].groupby(['town'], as_index=False).mean().sort_values(by='resale_price', ascending=False)


# In[ ]:


# correlation between remaining_lease, nearest_mrt_distance and floor_area_sqm and retail_price
train_df[['remaining_lease', 'nearest_mrt_distance', 'floor_area_sqm', 'resale_price']].corr()['resale_price']


# ## Preprocess data
# We concatenate train and test datasets and apply the same operations on both datasets together

# In[ ]:


# remove target column from train data

train_label = train_df['resale_price']
train_df = train_df.drop(columns=['resale_price'])


# In[ ]:


# combine train and test
train_length = len(train_df)
all_data = pd.concat(objs=[train_df, test_df], axis=0)


# * We should split column 'month' into two separate columns

# In[ ]:


def process_month(df):
    df['month_year'] = df.apply(lambda row: row['month'].split('-')[0], axis=1)
    df['month_month'] = df.apply(lambda row: row['month'].split('-')[1], axis=1)

process_month(all_data)
    
# 'month' column now can be removed
all_data = all_data.drop(columns=['month'])


# * id column should also be removed.
# * It can be observed that remaining_lease (in years) is the difference from current date and lease_commence_date so we can remove one columnn. In this example, we keep 'remaining_lease' column

# In[ ]:


all_id = all_data['id']
all_data = all_data.drop(columns=['id', 'lease_commence_date'])


# Convert categorical variables to dummy variables

# In[ ]:


all_data = pd.get_dummies(all_data)
all_data.head()


# Normalize numerical variables to (0, 1) scale

# In[ ]:


min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(all_data)
df = pd.DataFrame(x_scaled)

df.head()


# # 3. Train model
# Split data to train and test again.

# In[ ]:


train_data = x_scaled[:train_length]
test_data = x_scaled[train_length:]
train_label = train_label.values


# Create and train model

# In[ ]:


# create model
model = RandomForestRegressor()

# fit model with train data
model.fit(train_data, train_label)


# Prediction

# In[ ]:


prediction = model.predict(test_data)

test_id = all_id[train_length:].values

result_df = pd.DataFrame({'id': test_id,'resale_price': prediction})
result_df.to_csv('submission.csv',index=False)

