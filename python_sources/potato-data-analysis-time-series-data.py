#!/usr/bin/env python
# coding: utf-8

# # My assumptions and explorations

# One of the facts about this dataset is that there are more number of features than the number to training samples (wide dataset). This may mean that the features are composition of one-hot encoded categorical data and/or time series data. The reason I assume that the data may contain one-hot encoded categorical data is because the data is very sparse and it may probably be because categorical values are one-hot encoded. Not only that it may also mean that the data is time series data which probably contains **each** customer's transactions with the bank over *x* number of years/months. Most financial datasets would include some sort of record of over a period of timeframe for **each** customer which will be useful for some sort of predictions (credit worthiness, future transactions, etc).
# 
# This is my initial assumption.

# In[49]:


import pandas as pd
import numpy as np


# In[50]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[51]:


target = train['target'].values


# ## Datatypes

# In training dataset, 3147 features are int64 but there are no int64 datatype in testing dataset. If there are one-hot encoded categorical data in the dataset, they may be converted to float in the testing dataset.

# In[53]:


train.info()


# In[54]:


test.info()


# Let's check other aspects of the dataset while we are at it. First, let's check for features with constant values in **training dataset** (i.e., there is only a one unique value in the whole column).
# Excluding the first and second columnn (which is ID and target):

# In[55]:


np.sum(train.iloc[:, 2:].nunique() == 1)


# But for **testing dataset**, there are no features with constant values. However, we are going to train our models with just training data so we are going to discard those features. Either the testing data has some fake rows or something else.

# In[56]:


np.sum(test.iloc[:, 1:].nunique() == 1)


# ## Plotting data for some of the customers

# Let's take a look at the data as it is without any sort of filters and decompositions.
# What I am interested in is to actually try to infer what this data is and how this data is generated. I want to plot some of the data rows to better understand how this transaction data look like for each customer of Santander.
# 
# Plotting some randomly picked rows as it is:

# In[57]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


RANDOM_CUSTOMERS = [30, 55, 67]


# In[59]:


def plot_customers(rand_cust):
    for i in rand_cust:
        plt.plot(train.iloc[i].values)
        plt.title("For {}th customer target value: {}".format(i, target[i]))
        plt.xlabel("Features")
        plt.ylabel("Values")
        plt.show()


# In[60]:


plot_customers(RANDOM_CUSTOMERS)


# Too messy. Let's get rid of those constant features from training set.

# In[61]:


unique_df = train.nunique().reset_index()
unique_df.columns = ['col_name', 'unique_count']
constant_df = unique_df[unique_df["unique_count"]==1]
train = train.drop(constant_df['col_name'].values, axis=1)


# How about now?

# In[62]:


plot_customers(RANDOM_CUSTOMERS)


# Let's do something more. Let's drop all the features where 98% of that feature is zero.

# In[63]:


TRESHOLD = 0.98
cols_to_drop = [col for col in train.columns[2:]
                    if [i[1] for i in list(train[col].value_counts().items()) 
                    if i[0] == 0][0] >= train.shape[0] * TRESHOLD]

exclude = ['ID', 'target']
train_features = []
for c in train.columns:
    if c not in cols_to_drop and c not in exclude:
        train_features.append(c)


# In[64]:


print("Number of training features after dropping values: {}".format(len(train_features)))


# In[65]:


train, test = train[train_features], test[train_features]
print("Train shape: {}\nTest shape: {}".format(train.shape, test.shape))


# In[66]:


plot_customers(RANDOM_CUSTOMERS)


# Okay, I think it is a bit cleaner now. with all those peasky values being removed. From what I can see, I assume this is a form of time series data where the transactions of each customer being recorded over **x** years/months. I think it is worth exploring some sort of time-series modelling technique. I will try doing that and update this kernel.
