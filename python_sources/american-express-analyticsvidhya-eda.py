#!/usr/bin/env python
# coding: utf-8

# # American Express - Exploratory Analysis
# 
# This kernel is meant to explore the data and build a model eventually. I've written down my own thoughts regarding why I think some part are important and some parts are not. Do feel free to use any of these parts for inspiration or blatantly copy from it. I don't mind either way. Do let me know if you feel I could do better too!
# 
# * EDA
# * Merging the Data
# * EDA
# * Tackling Class Imbalance
# * Modelling - Hyperparameter and Model selection

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


# Loading the data

train_df = pd.read_csv("/kaggle/input/train_AUpWtIz/train.csv")
item_df = pd.read_csv("/kaggle/input/train_AUpWtIz/item_data.csv")
transactions_df = pd.read_csv("/kaggle/input/train_AUpWtIz/customer_transaction_data.csv")
campaign_df = pd.read_csv("/kaggle/input/train_AUpWtIz/campaign_data.csv")
demographics_df = pd.read_csv("/kaggle/input/train_AUpWtIz/customer_demographics.csv")


# ### Train DF

# In[ ]:


print("Train Shape ", train_df.shape)
print("Unique Campaign ", train_df.campaign_id.nunique())
print("Unique Coupon ", train_df.coupon_id.nunique())
print("Unique Customer ", train_df.customer_id.nunique())
train_df.head()


# In[ ]:


sns.countplot(train_df.redemption_status)


# Here we can see that our response variable (y) is highly imbalanced. We should factor for class imbalance before making any predictions. For help, we could refer to kernels which tackle class imbalance in banking data, for example, fraud detection.

# ### Item DF

# In[ ]:


print("Number of Unique Id ", item_df['item_id'].nunique())
print("Number of Brands ", item_df['brand'].nunique())
item_df.head()


# In[ ]:


sns.countplot(item_df['brand_type'])


# In[ ]:


plt.figure(figsize=(10, 5))
ax = sns.countplot(item_df['category'])

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()


# This describes the contents of the items. Established items comes from the very popular country wide services and local items like the name suggests should be from close by. Nonetheless it makes for some categorical features for our final analysis.

# ### Transactions

# In[ ]:


print("Number of Transactions ", transactions_df.shape[0])
transactions_df.head()


# In[ ]:


sns.distplot(transactions_df['selling_price'])


# In[ ]:


# lets check to see how many transactions used the coupon discount

transactions_df['coupon_discount'].value_counts().head()


# In[ ]:


# lets look at the largest discount

print(transactions_df['coupon_discount'].min())

# lets look at the row for this item

print(transactions_df[transactions_df['coupon_discount'] == transactions_df['coupon_discount'].min()])


# It's a very interesting row. It looks to me that either the shop had to pay the customer to \$171 to take home a commodity which costs around $2k or it's a mistake of some sort. We'll just let it be there for now. Other than that the price and discount variables provide valuble information to predict our final output if our coupon was redeemed or not.

# ### Campaign DF

# In[ ]:


print("Number of Campaigns ", campaign_df['campaign_id'].nunique())

campaign_df.head()


# In[ ]:


sns.countplot(campaign_df['campaign_type'])


# Except for the two types of campaign which doesn't yield us much info, we can generate features from the start date and the end date.

# ### Demographics DF

# In[ ]:


print("Number of Users ", demographics_df['customer_id'].nunique())

demographics_df.head()


# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

sns.countplot(demographics_df['age_range'], ax=axes[0])
sns.countplot(demographics_df['marital_status'].fillna('null'), ax=axes[1])


# Demographics are the most interesting type of data. We can by looking at the age say that mostly young people redeem coupons. 
# And chances are it's mostly single people too. So these are very important columns and we can't drop them.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

sns.countplot(demographics_df['rented'].fillna('null'), ax=axes[0])
sns.countplot(demographics_df['family_size'].fillna('null'), ax=axes[1])


# These are also important, renting a place could be done by frugal people who are also more likely to redeem coupons. So very important features. Family size would also play a significant role. A family of 5+ would have to do a lot of shopping, so it's likely that they use coupon code due to their high-frequency visits to the shop.

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

sns.countplot(demographics_df['no_of_children'].fillna('null'), ax=axes[0])
sns.countplot(demographics_df['income_bracket'].fillna('null'), ax=axes[1])


# Same as the above, no of children could be an important identified. Also I'd say income bracket is one of the most important feature as if you're rich, there's no reason for you to use coupons. So all these could be very valuable predictors.

# ## Merging the Data 

# In[ ]:




