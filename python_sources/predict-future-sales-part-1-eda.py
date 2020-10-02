#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-size:30px">Predict Future Sales - EDA</h1>
# <br>
# The problem is based on a time-series dataset consisting of daily sales data provided by one of the largest Russian software firms - 1C Company, which is available as a Kaggle competition.<br>
# <hr>
# <h1 style="font-size:18px">Objective</h1>
# <br>
# Initially we aim to understand and analyze the data, and then to predict total sales for every product and store in the next month.<br>
# This kernel is the first part of the scope.
# <hr>
# <h1 style="font-size:18px">Content</h1>
# <br>
# This kernel will be divided into:
# 1. <a href="#basic">Basic Information</a>
# 2. <a href="#engineering">Feature Engineering</a>
# 3. <a href="#numeric">Numeric Features</a>

# <h1 style="font-size:18px">File description</h1>
# * sales_train.csv - the training set. Daily historical data from January 2013 to October 2015.
# * test.csv - the test set. You need to forecast the sales for these shops and products for November 2015.
# * sample_submission.csv - a sample submission file in the correct format.
# * items.csv - supplemental information about the items/products.
# * item_categories.csv  - supplemental information about the items categories.
# * shops.csv- supplemental information about the shops.

# <h1 style="font-size:18px">About the data</h1>
# * ID - an Id that represents a (Shop, Item) tuple within the test set
# * shop_id - unique identifier of a shop
# * item_id - unique identifier of a product
# * item_category_id - unique identifier of item category
# * item_cnt_day - number of products sold. You are predicting a monthly amount of this measure
# * item_price - current price of an item
# * date - date in format dd/mm/yyyy
# * date_block_num - a consecutive month number, used for convenience. January 2013 is 0, February 2013 is 1,..., October 2015 is 33
# * item_name - name of item
# * shop_name - name of shop
# * item_category_name - name of item category

# <h1 style="font-size:18px">Import libraries</h1>

# In[ ]:


# Numpy for numerical computing
import numpy as np

# Pandas for Dataframes
import pandas as pd
pd.set_option('display.max_columns',100)

# Matplolib for visualization
from matplotlib import pyplot as plt
# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns

# Datetime deal with dates formats
import datetime as dt


# <span style="font-size:18px">**Load files**</span>

# In[ ]:


train = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv')
items = pd.read_csv('../input/items.csv')
items_categories = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')


# <br id="basic">
# # 1. Basic Information
# Let's first check some informations about the dataset for each loaded file, as:
# * Dimension
# * Features type
# * Number of missing values
# * View the first 3 rows

# In[ ]:


# Dataframe dimensions
print('The dimension of the training set is:',train.shape,'\n')
print('The feature types are:\n', train.dtypes,'\n')
print('Number of missing values:\n',train.isnull().sum())
train.head(3)


# In[ ]:


print('The dimension of the test set is:',test.shape,'\n')
print('The feature types are:\n', test.dtypes,'\n')
print('Number of missing values:\n',test.isnull().sum())
test.head(3)


# In[ ]:


print('The dimension of the items set is:',items.shape,'\n')
print('The feature types are:\n', items.dtypes,'\n')
print('Number of missing values:\n',items.isnull().sum())
items.head(3)


# In[ ]:


print('The dimension of the items categories is:',items_categories.shape,'\n')
print('The feature types are:\n', items_categories.dtypes,'\n')
print('Number of missing values:\n', items_categories.isnull().sum())
items_categories.head(3)


# In[ ]:


print('The dimension of the shops set is:',shops.shape,'\n')
print('The feature types are:\n', shops.dtypes,'\n')
print('Number of missing values:\n', shops.isnull().sum())
shops.head(3)


# So far the data looks pretty good. There is no missing values, and the features seems to have the correct type.

# <br id="engineering">
# # 2. Feature Engineering
# 
# Before we continue, let's do some preliminary feature engineering to make the data easier to deal with.<br>
# <br>
# Firstly, let's change the date type, from object to datetime. Then divide de **date** feature to create 3 new columns for **year**, **month** and **day**.

# In[ ]:


# Change the date type
date = train.date.apply(lambda x:dt.datetime.strptime(x, '%d.%m.%Y'))

# Create 3 new features for year, month and day
train['year'] = date.dt.year
train['month'] = date.dt.month
train['day'] = date.dt.day
train.head()

# Remove the "date" feature
train = train.drop('date', axis=1)


# Checking the dataset we can see that there is a feature that might be missing in the **train set**: item_category_id.<br> 
# The **item_category_id** and the **item_id** are related at **items set**. Thus, we can create another feature for these categories.

# In[ ]:


# Add the "item_category_id" to the dataset
train = pd.merge(train, items.drop('item_name', axis=1), on='item_id')
train.head()


# As we have the current price of the item (item_price) and the number of items sold (item_cnt_day) features, we can create another feature called "revenue" by their dot multiplication.

# In[ ]:


# Create "revenue" feature
train['revenue'] = train.item_price*train.item_cnt_day
train.head()


# <br id="numeric">
# # 3. Numeric features
# To ease up the study of the numeric features, we will look the data throught grouping it by year and month to see if there is any possible seasonability.<br>
# <br>
# Firstly we will analyze the feature item_cnt_day, which gives the number of products sold.

# In[ ]:


# Plot the total number of products sold by year
train.groupby('year').item_cnt_day.sum().plot()
plt.xticks(np.arange(2013, 2016, 1))
plt.xlabel('Year')
plt.ylabel('Total number of products sold')
plt.show()

# Plot the total number of products sold by month for each year
train.groupby(['month','year']).sum()['item_cnt_day'].unstack().plot()
plt.xlabel('Month')
plt.ylabel('Total number of products sold')
plt.show()


# We can see that the number of sold products are decreasing over the years.<br>
# Looking at the months, the sales seems to vary in a certain range until October, and then the sales start to increase greatly.<br>
# <br>
# Now let's check the revenue behavior.

# In[ ]:


# Plot the total revenue by year
train.groupby('year').revenue.sum().plot()
plt.xticks(np.arange(2013, 2016, 1))
plt.xlabel('Year')
plt.ylabel('Total revenue')
plt.show()

# Plot the total revenue by month for each year
train.groupby(['month','year']).sum()['revenue'].unstack().plot()
plt.xlabel('Month')
plt.ylabel('Total revenue')
plt.show()


# The revenue behavior is a little different from the number of total sales.<br>
# In 2014 the total revenue increased, even though the number of total sales decreased from 2013. This is due to the "item_price" variable, which can fluctuate by the time.<br>
# We can also observe that, over the months, even though the number of sales product decreased, the revenue seems similar for the three years.<br>
# <br>
# Let's look at the top 10 items and the top 10 shops.

# In[ ]:


# Plot the top 10 items
sns.countplot(y='item_id', hue='year', data=train, order = train['item_id'].value_counts().iloc[:10].index)
plt.xlim(0,20000)
plt.xlabel('Number of times the item was sold')
plt.ylabel('Identifier of the item')
plt.show()

# Plot the top 10 shops
sns.countplot(y='shop_id', hue='year', data=train, order = train['shop_id'].value_counts().iloc[:10].index)
plt.xlabel('Number of times the shop sold')
plt.ylabel('Identifier of the shop')
plt.show()


# The item 20949 is the sales champion over the years by far!<br>
# The top 10 shops have similar sales behavior over the years.
