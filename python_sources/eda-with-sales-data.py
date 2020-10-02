#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import *
import nltk, datetime
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings(action='once')

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


# # Reading Data

# In[ ]:


train = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
items = pd.read_csv('../input/items.csv')
item_cats = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')

print('train:', train.shape, 'test:', test.shape)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


submission.head()


# # 1. Business Understanding
# 
# * **Training set**: Daily historical data from January 2013 to October 2015.
# * **Test set**: Forecast the sales for these shops and products for the month of November 2015.
# 
# # 2. Data Understanding

# In[ ]:


# Joining datasets
train_full = pd.merge(train, items, how='left', on=['item_id','item_id'])
train_full = pd.merge(train_full, item_cats, how='left', on=['item_category_id','item_category_id'])
train_full = pd.merge(train_full, shops, how='left', on=['shop_id','shop_id'])

# Adding date features
train_full['date'] = pd.to_datetime(train_full['date'], format='%d.%m.%Y')
train_full['month'] = train_full['date'].dt.month
train_full['year'] = train_full['date'].dt.year

# Selecting feature set
col = ['date_block_num', 'date', 'month', 'year', 'shop_id', 'item_category_id', 'item_id', 'item_price',
       'item_cnt_day', 'shop_name', 'item_category_name','item_name',
      ]
train_full = train_full[col]
train_full['revenue'] = train_full.item_price * train_full.item_cnt_day

print("Description of training set")
print("Shape: \t\t\t", train_full.shape)
print("#NaNs: \t\t\t", train_full.isna().sum().sum()) # No NaN
print("#Shops: \t\t", train_full.shop_id.nunique())
print("#Item Categories: \t", train_full.item_category_id.nunique())
print("#Items: \t\t", train_full.item_id.nunique())
print("#Months: \t\t", train_full.date_block_num.nunique())
print("Date range from: \t", train_full.date.min(), " to ", train_full.date.max())
print("Price range from: \t", train_full.item_price.min(), " to ", train_full.item_price.max())
print("Units Sold range from: \t", train_full.item_cnt_day.min(), " to ", train_full.item_cnt_day.max())
print("Revenue range from: \t", train_full.revenue.min(), " to ", train_full.revenue.max())

train_full.head(3)


# In[ ]:


# Average item price
item_price = train_full[['item_id', 'item_price']].groupby('item_id')['item_price'].mean()

item_price = item_price.reset_index()
item_price = item_price.rename(columns={'item_price':'avg_item_price'})

item_price.head(3)


# In[ ]:


# On avergage no item has a negative price
(item_price.avg_item_price < 0).sum()


# In[ ]:


train_agg = train_full[['date_block_num', 'month', 'year', 'shop_id', 'item_category_id', 'item_id', 'item_cnt_day', 'revenue']].groupby(
    ['date_block_num', 'shop_id', 'item_category_id', 'item_id'])[
    ['item_cnt_day', 'revenue']].sum()

train_agg = train_agg.reset_index()
train_agg = train_agg.rename(columns={'item_cnt_day':'units_sold'})

print("Shape of complete training set:", train_agg.shape)

train_agg = train_agg[train_agg.units_sold >= 0.0] # Subsetting for sales between 0 to 20 units per month
train_agg = train_agg[train_agg.units_sold <= 20.0]
print("Shape of selected training set:", train_agg.shape)

# Add average price
train_agg = pd.merge(train_agg, item_price, how='left', left_on='item_id', right_on='item_id')

train_agg.head(3)


# In[ ]:


train_agg[['units_sold', 'avg_item_price', 'revenue']].describe(percentiles=[.01, .5, .95, .99]).T


# In[ ]:


df = train_agg[['item_category_id', 'units_sold', 'revenue']].groupby(
    ['item_category_id'])['units_sold', 'revenue'].sum()

df = df.reset_index()
df['avg_item_price'] = round(df.revenue / df.units_sold, 2)

f, ax = plt.subplots(figsize=(16, 4))
plt.scatter(y="units_sold", x="avg_item_price", data=df, c="b", s=30, linewidth=1, marker="+")
plt.title("For each item category \nUnits Sold ~ Avg item price");


# ## Plotting Overall Monthly trend

# In[ ]:


df = train_agg[['date_block_num', 'item_id']].groupby(by='date_block_num', as_index=False).nunique()

fig, ax = plt.subplots(figsize=(16,4))
plt.plot(df.item_id, sns.xkcd_rgb["denim blue"], lw=4)
plt.title("Count of Unique items sold")


df = train_agg[['date_block_num', 'units_sold']].groupby(by='date_block_num', as_index=False).sum()

f, ax = plt.subplots(figsize=(16, 4))
plt.plot(df.units_sold, sns.xkcd_rgb["denim blue"], lw=4)
plt.title("Sum of items sold")


df = train_agg[['date_block_num', 'revenue']].groupby(by='date_block_num', as_index=False).sum()

f, ax = plt.subplots(figsize=(16, 4))
plt.plot(df.revenue, sns.xkcd_rgb["denim blue"], lw=4)
plt.title("Total revenue generated");


# Number of Unique items and Total Units Sold are decreasig with time.  Revenue trend looks relatively flat
# 
# ## Plotting same metrics for each shop

# In[ ]:


stores_hm = train_agg.pivot_table(index='date_block_num', columns='shop_id', values='item_id', aggfunc='nunique', fill_value=0)

fig, ax = plt.subplots(figsize=(16,4))
cmap = sns.cubehelix_palette(8, as_cmap=True, dark=0, light=1, gamma=0.8, reverse=True)
plt.title("Count of Unique items sold")
_ = sns.heatmap(stores_hm, ax=ax, cmap=cmap, cbar=False, xticklabels=False)


stores_hm = train_agg.pivot_table(index='date_block_num', columns='shop_id', values='units_sold', aggfunc='sum', fill_value=0)

fig, ax = plt.subplots(figsize=(16,4))
cmap = sns.cubehelix_palette(8, as_cmap=True, dark=0, light=1, gamma = .8, reverse=True)
plt.title("Sum of items sold")
_ = sns.heatmap(stores_hm, ax=ax, cmap=cmap, cbar=False, xticklabels=False)


stores_hm = train_agg.pivot_table(index='date_block_num', columns='shop_id', values='revenue', aggfunc='sum', fill_value=0)

fig, ax = plt.subplots(figsize=(16,4))
cmap = sns.cubehelix_palette(8, as_cmap=True, dark=0, light=1, gamma = .8, reverse=True)
plt.title("Total revenue generated")
_ = sns.heatmap(stores_hm, ax=ax, cmap=cmap, cbar=False, xticklabels=False)


# Black areas indicate 0
# Not all 60 shops are in operations; few shops with light color are the main drivers
# 
# ## Plotting same metrics for each item category

# In[ ]:


stores_hm = train_agg.pivot_table(index='date_block_num', columns='item_category_id', values='item_id', aggfunc='nunique', fill_value=0)

fig, ax = plt.subplots(figsize=(16,4))
cmap = sns.cubehelix_palette(8, as_cmap=True, dark=0, light=1, gamma=0.8, reverse=True)
plt.title("Count of Unique items sold")
_ = sns.heatmap(stores_hm, ax=ax, cmap=cmap, cbar=False, xticklabels=False)


stores_hm = train_agg.pivot_table(index='date_block_num', columns='item_category_id', values='units_sold', aggfunc='sum', fill_value=0)

fig, ax = plt.subplots(figsize=(16,4))
cmap = sns.cubehelix_palette(8, as_cmap=True, dark=0, light=1, gamma = .8, reverse=True)
plt.title("Sum of items sold")
_ = sns.heatmap(stores_hm, ax=ax, cmap=cmap, cbar=False, xticklabels=False)


stores_hm = train_agg.pivot_table(index='date_block_num', columns='item_category_id', values='revenue', aggfunc='sum', fill_value=0)

fig, ax = plt.subplots(figsize=(16,4))
cmap = sns.cubehelix_palette(8, as_cmap=True, dark=0, light=1, gamma = .8, reverse=True)
plt.title("Total revenue generated")
_ = sns.heatmap(stores_hm, ax=ax, cmap=cmap, cbar=False, xticklabels=False)


# Only a very few item categories (out of 84) are sold in high numbers; distribution of revenue generation is spread among a few more item categories

# In[ ]:




