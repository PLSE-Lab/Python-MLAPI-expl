#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
sns.set(rc={'figure.figsize':(7 ,5)})


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
merchants_df = pd.read_csv("../input/merchants.csv")
historical_transactions_df = pd.read_csv("../input/historical_transactions.csv")
new_merchant_transactions_df = pd.read_csv("../input/new_merchant_transactions.csv")


# # Train

# In[ ]:


print(train_df.shape)
train_df.head()


# In[ ]:


train_df.describe()


# In[ ]:


#There are some obvious outliers in 'target'
sns.distplot(train_df.target.values, kde=False)
print (f'{train_df[train_df.target < -10].target.shape[0]} outliers')


# In[ ]:


train_df = train_df[train_df.target > -10]
sns.pairplot(train_df[['feature_1', 'feature_2', 'feature_3', 'target']])


# # Merchants
# 
# merchants can be joined with the transaction sets to provide additional merchant-level information.

# In[ ]:


print(merchants_df.shape)
merchants_df.head()


# In[ ]:


#There are some NaN values
merchants_df.isna().sum()


# In[ ]:


merchants_df.dropna(inplace=True)
numerical_cols = ['numerical_1', 'numerical_2', 'avg_sales_lag3',
                  'avg_purchases_lag3','active_months_lag3',
                  'avg_sales_lag6', 'avg_purchases_lag6',
                  'active_months_lag6', 'avg_sales_lag12', 
                  'avg_purchases_lag12', 'active_months_lag12']

merchants_df[numerical_cols].describe()


# In[ ]:


cat_cols = [ 'category_1', 'category_2',  'category_4', 
          'most_recent_purchases_range', 'most_recent_sales_range',
           'merchant_group_id', 'merchant_category_id', 'subsector_id']

for col in cat_cols:
    cat_num = merchants_df[col].value_counts().index.shape[0]
    print(f'{cat_num} unique values in {col}')


# In[ ]:


print('Almost 14% of values  belong to merchant group #35')
rows = merchants_df.shape[0]
merchants_df.merchant_group_id.value_counts()[:5]/rows*100


# # Transactions
# 
# **historical_transactions.csv**  - up to 3 months' worth of historical transactions for each card_id

# In[ ]:


#Missing values
historical_transactions_df.isna().sum()


# In[ ]:


print(historical_transactions_df.shape)
historical_transactions_df.head()


# In[ ]:


cat_cols = [ 'authorized_flag', 'card_id', 'city_id', 'category_1', 
             'category_3', 'merchant_category_id', 'merchant_id', 
             'category_2', 'state_id', 'subsector_id']

for col in cat_cols:
    cat_num = historical_transactions_df[col].value_counts().index.shape[0]
    print(f'{cat_num} unique values in {col}')


# In[ ]:


#example of continuous data on single card_id
idx = np.random.choice(historical_transactions_df.card_id)
df = historical_transactions_df[historical_transactions_df.card_id == idx]
sns.pairplot(df[['month_lag', 'purchase_amount']])


# # New Merchant Transactions
# 
# **historical_transactions.csv** and **new_merchant_transactions.csv** are designed to be joined with **train.csv**, ** test.csv**, and **merchants.csv**. They contain information about transactions for each card.

# In[ ]:


print(new_merchant_transactions_df.shape)
new_merchant_transactions_df.head()


# In[ ]:


new_merchant_transactions_df.isna().sum()


# In[ ]:


cat_cols = [ 'authorized_flag', 'card_id', 'city_id', 'category_1', 
             'category_3', 'merchant_category_id', 'merchant_id', 
             'category_2', 'state_id', 'subsector_id']

for col in cat_cols:
    cat_num = new_merchant_transactions_df[col].value_counts().index.shape[0]
    print(f'{cat_num} unique values in {col}')


# In[ ]:


#example of continuous data on single card_id
idx = np.random.choice(new_merchant_transactions_df.card_id)
df = new_merchant_transactions_df[new_merchant_transactions_df.card_id == idx]
sns.pairplot(df[['month_lag', 'purchase_amount']])


# In[ ]:




