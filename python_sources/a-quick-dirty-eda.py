#!/usr/bin/env python
# coding: utf-8

# # Import libraries

# In[ ]:


from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Configure hyper-parameters

# In[ ]:


ROOT = Path('/kaggle/')
INPUT_DIR = ROOT / 'input/competitive-data-science-predict-future-sales'


# # Have a quick look at what we have

# In[ ]:


list(INPUT_DIR.glob('*'))


# In[ ]:


item_df = pd.read_csv(INPUT_DIR / 'items.csv')
item_df.head()


# In[ ]:


shop_df = pd.read_csv(INPUT_DIR / 'shops.csv')
shop_df.head()


# In[ ]:


item_cat_df = pd.read_csv(INPUT_DIR / 'item_categories.csv')
item_cat_df.head()


# In[ ]:


train_df = pd.read_csv(INPUT_DIR / 'sales_train.csv')
train_df.head()


# In[ ]:


test_df = pd.read_csv(INPUT_DIR / 'test.csv')
test_df.head()


# In[ ]:


sample_submission = pd.read_csv(INPUT_DIR / 'sample_submission.csv')
sample_submission.head()


# # Join all available data

# In[ ]:


train_df = train_df.join(item_df, on='item_id', how='outer', lsuffix='', rsuffix='_r')[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day', 'item_name', 'item_category_id']]


# In[ ]:


train_df.head()


# In[ ]:


train_df = train_df.join(item_cat_df, on='item_category_id', how='outer', lsuffix='', rsuffix='_r')[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day', 'item_name', 'item_category_id', 'item_category_name']]


# In[ ]:


train_df.head()


# In[ ]:


train_df = train_df.join(shop_df, on='shop_id', how='outer', lsuffix='', rsuffix='_r')[['date', 'date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day', 'item_name', 'item_category_id', 'item_category_name', 'shop_name']]


# In[ ]:


train_df.head()


# In[ ]:


train_df.reset_index(drop=True, inplace=True)


# # Clean the data

# In[ ]:


train_df.dropna(inplace=True)
train_df.drop_duplicates(inplace=True)


# # Modify some features (fields/columns)

# ## Split the date column into "day", "month" and "year" ones

# In[ ]:


train_df[['day', 'month', 'year']] = train_df.date.str.split('.', expand=True)
train_df.day = train_df.day.apply(lambda x: int(x))
train_df.month = train_df.month.apply(lambda x: int(x))
train_df.year = train_df.year.apply(lambda x: int(x))
train_df.head()


# # Have a quick computation for all number cols

# In[ ]:


train_df.describe()


# # Visualize some information

# In[ ]:


year_count = train_df.groupby('year').count().item_id.reset_index()
year_count.columns = ['year', 'total_bill']
month_count = train_df.groupby('month').count().item_id.reset_index()
month_count.columns = ['month', 'total_bill']
day_count = train_df.groupby('day').count().item_id.reset_index()
day_count.columns = ['day', 'total_bill']

fig, axes = plt.subplots(1, 3, figsize=(20, 4))
sns.barplot(x='year', y='total_bill', data=year_count, ax=axes[0])
sns.barplot(x='month', y='total_bill', data=month_count, ax=axes[1])
sns.barplot(x='day', y='total_bill', data=day_count, ax=axes[2])
plt.show()


# In[ ]:


date_block_count = train_df.groupby('date_block_num').count().item_id.reset_index()
date_block_count.columns = ['date_block', 'total_bill']
fig = plt.figure(figsize=(12, 4))
ax = fig.add_axes([0, 0, 1, 1])
sns.barplot(x='date_block', y='total_bill', data=date_block_count, ax=ax)
plt.show()

