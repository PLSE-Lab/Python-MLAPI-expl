#!/usr/bin/env python
# coding: utf-8

# # Predict Future Sales : Exploratory Data Analysis
# ___
# `Started on 04 April 2018`
# 
# `Rhodium Beng`

# ### This is a competition that require participants to work with a challenging time-series dataset consisting of daily sales data, provided by one of the largest Russian software firms - 1C Company.  The Task is to predict the total sales for every item in every shop in the next month.
# ### This kernel explore the data.

# In[ ]:


import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ## Load the data

# In[ ]:


sales = pd.read_csv('../input/sales_train.csv')
items = pd.read_csv('../input/items.csv')
item_categories = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')


# ## Take a look and explore the data

# In[ ]:


sales.head()


# In[ ]:


items.head()


# In[ ]:


item_categories.head()


# In[ ]:


shops.head()


# In[ ]:


test.head()


# In[ ]:


# Examine the size & shape of the data
print(sales.shape, items.shape, item_categories.shape, shops.shape, test.shape)


# ### Observations:
# There are almost 3 million sales transactions in the sales data (mostly sales with some returns [indicated by negative `item_cnt_day`). The sales period is about 1.5 years, from Jan 2013 to Oct 2015. Column `date_block_num` also indicates the month; with 0 representing Jan 2013, and 33 representing Oct 2015.
# 
# The testing period for the test data (`test`) is the month, i.e. Nov 2015, that follows the last month of the training data (`sales`).
# 
# There are a total of 22,170 unique items, 84 unique item categories and 60 unique shops.

# ### Plot some charts to look at patterns & trends

# Let's look at the number of transactions by month.

# In[ ]:


transactions = sales.groupby('date_block_num')['date'].count()


# In[ ]:


sns.set()
transactions.plot.line(title='Number of transactions by month', color='gray')


# ### Observations:
# Overall, the total number of transactions by month for all shops is on a downward trend. 
# 
# The two peaks correspond to the two year-end periods (Dec 2013 & Dec 2014) in the `sales` dataset. 

# Let's examine the number of unique shops and items in each of the month block that have transactions in the sales data.

# In[ ]:


shop_counts = sales.groupby('date_block_num')['shop_id'].nunique()
item_counts = sales.groupby('date_block_num')['item_id'].nunique()


# In[ ]:


fig, axarr = plt.subplots(1, 2, figsize=(12, 5))
shop_counts.plot.line(ax=axarr[0], color='gray')
item_counts.plot.line(ax=axarr[1], color='gray')
axarr[0].set_title('Number of shops with transactions')
axarr[1].set_title('Number of items with transactions')


# In[ ]:


print(sales['item_id'].nunique(), sales['shop_id'].nunique())


# ### Observations:
# In `sales`, there are 60 unique shops and 21,807 unique items. For the shops, the number corresponds to the total number of shops in the `shops` dataset. However, for the items, the number is less than the total number of items (22,170) in the `items` dataset.
# 
# Looking at the line charts, we can see that the number of shops with transactions started to increase in the second year, but drop to the same level at the end of the second year. It seems that the number of shops drops slightly further in the third year.
# 
# Corresponding to the downward trend in transactions, the range of items sold over the months also show a downward trend after the first year.
# 
# From the charts, we can see that not all the shops has got transactions every month, and on the average only a third and less of the items have transactions every month.
# 
# **We should keep these observations and trends in mind. Now, let's have a feel and some intuition for the `test` data.**

# ### Explore the test data

# In[ ]:


test_shops = test['shop_id'].unique()
print(len(test_shops))
print(np.sort(test_shops))


# There are 42 unique shops in the `test` data. 12 shops i.e. Shop 0, 1, 8, 9, 11, 13, 17, 20, 23, 27, 29, 30, 31, 33, 40, 43, 51 & 54 do not feature in the `test` data. We shall keep this in mind.
# 
# Are there `items` in the `sales` dataset that are not in the `test` data? Let's examine.

# In[ ]:


test_items = test['item_id'].unique()
print(len(test_items))


# In[ ]:


check = sales[np.isin(sales['item_id'], test_items, invert=True)]
print(len(check['item_id'].unique()))
print(np.sort(check['item_id'].unique()))


# There are 5,100 unique items in the `test` data. And some 17,070 items in the `sales` data are not in the `test` data. Similarly, we shall keep this in mind.

# ## Aggregate data
# Since the task is to make a monthly prediction, we will need to aggregate the data to monthly level.
# 
# We will also add the test data into this grid. The `date_block_num` for the test data will be `34`, following the last month (Oct 2015) block number.

# In[ ]:


# Create grid with columns
index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items from that month
grid = [] 
for block_num in sales['date_block_num'].unique():
    cur_shops = sales[sales['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sales[sales['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

# Add the shop_id from the test data and create date_block_num 34
block_num = 34
cur_shops = test['shop_id'].unique()
cur_items = test['item_id'].unique()
grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])), dtype='int32'))
    
# Turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)


# In[ ]:


len(grid)


# In[ ]:


# The size of the grid should be the same as the sum of the product of unique `shop_counts` & `item_counts` for
# both the sales & test data
shop_counts = sales.groupby('date_block_num')['shop_id'].nunique()
item_counts = sales.groupby('date_block_num')['item_id'].nunique()
test_shops = test['shop_id'].nunique()
test_items = test['item_id'].nunique()
print(shop_counts.dot(item_counts) + test_shops * test_items)


# In[ ]:


# Get aggregated values for (shop_id, item_id, month)
gb = sales.groupby(index_cols, as_index=False)['item_cnt_day'].agg('sum')
# Rename column
gb = gb.rename(columns={'item_cnt_day':'target'})
# Join aggregated data to the grid
all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)


# In[ ]:


all_data.head()


# #### After the monthly aggregated dataframe has been generated, next is to clip the true target values to the range of `[0,20]`. This could make a difference in model fitting later on, as there seems to be some very large outliers as shown below. 

# In[ ]:


all_data['target'].describe()


# In[ ]:


all_data['target'].plot.hist(color='gray')


# It was suggested to clip the true target values to [0,20] for better results in model fitting. Let's do this.

# In[ ]:


all_data['target'] = np.clip(all_data['target'], 0, 20)
all_data['target'].plot.hist(color='gray')


# ## Output 'all_data' dataframe

# In[ ]:


# Generate output file in csv format
all_data.to_csv('all_data.csv', index=False)


# ### Thanks for reading my kernel. Hope this helps to get to the next step of applying machine learning models, and predictions
# ### Tips and comments are most welcomed & appreciated.
# ### Please upvote if you find it useful.

# In[ ]:




