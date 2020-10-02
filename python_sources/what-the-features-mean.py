#!/usr/bin/env python
# coding: utf-8

# # Introduction
# The dataset for this competition has masked feature names.  Even though in practice this information is usually available to in-house modellers, hiding it this way gives this problem a numerology flavor.  Nevertheless, we will make reasonable assumptions as to what the features and their order means.
# 
# This is Santander's third Kaggle competition.  One of their two earlier competitions had a dataset [with meaningful column names][3].  However, the features had different types and ranges compared to what we have in this competition.  So it may not help us decipher our features.
# 
# In this analysis, I will go beyond exploring the data and will try to understand what the features represent.
# 
# [1]: https://www.santander.com/csgs/Satellite/CFWCSancomQP01/es_ES/Corporativo.html?leng=en_GB
# [2]: http://banksdaily.com/topbanks/Europe/market-cap-2017.html
# [3]: https://www.kaggle.com/c/santander-product-recommendation/data

# # 4,990 Features
# The dataset is made of one traning (`train.csv`) and one test (`test.csv`) files.  Here is what the training data looks like:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

get_ipython().run_line_magic('matplotlib', 'inline')
train = pd.read_csv('../input/train.csv', index_col='ID')
test = pd.read_csv('../input/test.csv', index_col='ID')

train.head()


# In fact, this is what ~ 2% of the columns look like.  The tranining (and test) data set has a massive 4,990 numeric features + ID and target columns.  The `ID` column is the bank customer's identification value.
# 
# The test data is about 10 times the size of the training data.  Here is their shape:

# In[ ]:


print('\trows\tcolumns')
print('Train:\t{:>6,}\t{:>6,}'.format(*train.shape))
print('Test:\t{:>6,}\t{:>6,}'.format(*test.shape))


# # Breakdown the Features by Type
# Based on my own experience working in a financial institution and manipulating similar data sets, most of these columns most likely represent the sum of customers daily transactions.  Every day (or week or month), a new column is added with the total value of transactions for the customer in that time period.
# 
# To know more about the features, let's see their mean value distribution.

# In[ ]:


target = train.pop('target')

sns.set()
mn_train = train.mean()
std_train = train.std()

mn_test = test.mean()
std_test = test.std()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax = sns.distplot(mn_train, kde=False, norm_hist=False, ax=ax)
sns.distplot(mn_test, kde=False, norm_hist=False, ax=ax)
ax.get_xaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title('Distribution of the mean value of train/test features.')
ax.set_xlabel(r'Mean value ($\mu$)')
ax.set_ylabel('Number of features')
ax.legend(['train', 'test'])

ax = axes[1]
sns.distplot(std_train, kde=False, norm_hist=False, ax=ax)
sns.distplot(std_test, kde=False, norm_hist=False, ax=ax)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_title('Distribution of std value of train/test features.')
ax.set_xlabel(r'Standard Deviation ($\sigma^2$)')
ax.set_ylabel('Number of features')
ax.legend(['Train', 'Test']);


# The test set data features have a higher low mean value (left plot), and higher low variance (right plot).

# In[ ]:


cr = (train.max() == 0) & (train.min() == 0)
train_all_zero_feature_count = cr.index[cr].shape[0]
cr2 = (test.max() == 0) & (test.max() == 0)
test_all_zero_feature_count = cr2.index[cr2].shape[0]
print('Number of training features with all 0 values:\t{}'.format(train_all_zero_feature_count))
print('Number of test features with all 0 values:\t{}'.format(test_all_zero_feature_count))


# ### 256 All Zero Features
# In our training data we have 256 features with all zeros values.  None of the test data features have all zero values, though.
# 
# Since the test data by definition does not include the target feature, we cannot train our model on these features.  However, if we can infer the meaning of some of these features, we may be able to put them to good use.
# 
# We will drop these 256 features from our training data set and will not include them in our analysis moving forward.
# 
# Let's see if we have any binary features in our data.

# In[ ]:


train_all_zero_features = cr.index[cr]
train.drop(columns=train_all_zero_features, inplace=True)

count_of_binary_features = (train.max() == 1).sum()
print('Number of binary features: {}'.format(count_of_binary_features))


# ### No Binary or Percentage Features
# No binary features in our data.  Also, there is no percentage or ratio feature (continuous feature rangin from 0 to 1 ) in our data.
# 
# There are no remaining features with a maximum value less than 1,000.  This is probably a sign that these are commercial banking transactions with high values.  Also, the currency used is of high denomination.
# 

# In[ ]:


less_than_1000_count = (train.max() < 1000).sum()
print('Number of train features with max value < 1,000: {}'.format(less_than_1000_count))


# Here is the distribution of the maximum value for the remaining features.

# In[ ]:


plt.figure(figsize=(13, 5))
ax = sns.distplot(train.max(), kde=False, norm_hist=False, bins=1000)
ax = sns.distplot(test.max(), kde=False, norm_hist=False, bins=1000)
plt.xlim(left=-20000000, right=1e9)
ax.set_xlabel('Max feature value (axis clipped @ 1e9)')
ax.set_ylabel('Number of features')
ax.set_title('Distribution of max value of features')
ax.legend(['Train', 'Test']);


# The distribution is drastically different between the training and the test data sets.  The train and test data does not seem to be drawn at random.  Using the training data alone in feature engineering will negatively affect the model performance.
# 
# One thing to note about our features is their temporal nature, and some of them might be cyclic (monthly or weekly deposits) since they represent banking transactions.  However, unfortunately we don't know their order or even the time period they represent.

# # Target Feature
# We know from the competition description that the target feature is a transaction value.  Therefore, it's distribution should not be much different from the rest of the features, asuming they represent bank transactions as well.  The max feature transaction showed earlier may make us question that assumption though.
# 
# Here is what the target feature distribution looks like.

# In[ ]:


plt.figure(figsize=(13, 5))
ax = sns.distplot(target, kde=False, norm_hist=False, bins=200)
ax.get_yaxis().set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.set_xlabel('Transaction value')
ax.set_ylabel('Number of transactions')
ax.set_title('Distribution of target transaction values');


# # Warning
# The features we are working with are most likely temporal data.  However, we don't konw for sure the temporal order of the features.  We should think of this problem as a time series prediction.  We have past values in the series and we're trying to predict the next value.  Therfore, trying to glean insight from finding correlation between the target feature and the independent variables will not be useful at best, and may be misleading.
# 
# Also, we don't know the categories of the features.  In financial institutions, it's common to have such very wide tables where some columns represent monthly credit card transactions over the past five years, and other represent daily checking account transactions over the past 3 years, and so on.  Grouping the features by their mean value or the frequency of transactions may help in categorizing them.
