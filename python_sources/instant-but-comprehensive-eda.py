#!/usr/bin/env python
# coding: utf-8

# # Load Library and Data

# In[ ]:


import gc
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import lightgbm as lgb
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata

warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")


# In[ ]:


# Dimension of Train and Test Data
train_df.shape, test_df.shape


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


print(train_df.info())
print('\n')
print(test_df.info())


# Train Data
# - id
# - target (0 or 1: binary)
# - 256 features
# 
# Test Data
# - id
# - 256 features

# # Missing Data?

# In[ ]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
#     if np.tranpose(tt).loc['Total'].sum()==0:
#         print("No missing value in the entire data")
    return(np.transpose(tt))


# In[ ]:


# Checking for missing data for train data
missing_data(train_df)
print(missing_data(train_df).loc['Total'].sum())


# In[ ]:


# Checking for missing data for test data
missing_data(test_df)
print(missing_data(test_df).loc['Total'].sum())


# No missing values in both train and test data!

# # Summary Statistics of data

# In[ ]:


train_df.describe()


# In[ ]:


test_df.describe()


# # Distribution of Target

# In[ ]:


sns.countplot(train_df['target'], palette='Set3')


# In[ ]:


print("There are {}% target values with 1".format(100 * train_df["target"].value_counts()[1]/train_df.shape[0]))


# Target is very well balanced (surprisingly)!!!

# # Density Plots of Features

# In[ ]:


def plot_feature_distribution(df1, df2, label1, label2, features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))

    for feature in features:
        i += 1
        plt.subplot(8,8,i)
        sns.distplot(df1[feature], hist=False,label=label1)
        sns.distplot(df2[feature], hist=False,label=label2)
        plt.xlabel(feature, fontsize=9)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show()


# In[ ]:


# The first 64 features
t0 = train_df.loc[train_df['target'] == 0]
t1 = train_df.loc[train_df['target'] == 1]
features = train_df.columns.values[1:65]
plot_feature_distribution(t0, t1, '0', '1', features)
plt.tight_layout()


# In[ ]:


# Another 64 features
get_ipython().run_line_magic('time', '')
features = train_df.columns.values[65:129]
plot_feature_distribution(t0, t1, '0', '1', features)


# In[ ]:


# Another 64 features
get_ipython().run_line_magic('time', '')
features = train_df.columns.values[129:193]
plot_feature_distribution(t0, t1, '0', '1', features)


# In[ ]:


# Last 64 features
get_ipython().run_line_magic('time', '')
features = train_df.columns.values[193:257]
plot_feature_distribution(t0, t1, '0', '1', features)


# For all features, distribution of target==0 and target==1 seems almost identical (completely overlaid) :oooo

# But there is one feature that looks very different from all other features! That is the "wheezy-copper-turtle-magic"! Let's look at that feature a little bit more!

# In[ ]:


train_df['wheezy-copper-turtle-magic'].describe()


# In[ ]:


test_df['wheezy-copper-turtle-magic'].describe()


# While other features are usually within the range of -2 to 2 and are floats, this "wheezy-copper-turtle-magic" feature seems to be comprised of discrete integers and has very large values (e.g. max value is 511)

# In[ ]:


plt.figure(figsize=[20,5])

plt.subplot(1,2,1)
sns.distplot(train_df["wheezy-copper-turtle-magic"])
plt.title("train")

plt.subplot(1,2,2)
sns.distplot(test_df["wheezy-copper-turtle-magic"])
plt.title("test")


# In[ ]:


plt.figure(figsize=[9,5])
plt.subplot(1,2,1)
train_df.groupby("wheezy-copper-turtle-magic").size().sort_values()[::-1].hist(bins=50)
plt.title("train")

plt.subplot(1,2,2)
test_df.groupby("wheezy-copper-turtle-magic").size().sort_values()[::-1].hist(bins=50)
plt.title("test")


# # How is the distribution of each feature different between train and test?

# In[ ]:


# The first 64 features
features = train_df.columns.values[1:65]
plot_feature_distribution(train_df, test_df, 'train', 'test', features)
plt.tight_layout()


# In[ ]:


# Another 64 features
features = train_df.columns.values[65:129]
plot_feature_distribution(train_df, test_df, 'train', 'test', features)
plt.tight_layout()


# In[ ]:


# Another 64 features
features = train_df.columns.values[129:193]
plot_feature_distribution(train_df, test_df, 'train', 'test', features)
plt.tight_layout()


# We see that distributions of 'wheezy-copper-turtle-magic' in both train and test data here look different from others

# In[ ]:


# Last 64 features
features = train_df.columns.values[193:257]
plot_feature_distribution(train_df, test_df, 'train', 'test', features)
plt.tight_layout()


# # Correlation between features

# In[ ]:


# Correlations between features in training data
# Top10 lowest correlation pairs

correlations = train_df[features].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
correlations = correlations[correlations['level_0'] != correlations['level_1']]
correlations.head(10)


# In[ ]:


# Correlations between features in training data
# Top10 highest correlation pairs
correlations.tail(10)


# Correlation amongst features seems very low! probably because it's an artifical data designed to have less correlation amongst features?!

# # Correlation between features and target

# In[ ]:


cols = [c for c in train_df.columns if c not in ['id']]
corr = train_df[cols].corr().abs().unstack().sort_values(kind="quicksort").reset_index()
corr = corr[corr['level_0'] != corr['level_1']]
corr = corr[corr['level_0'] == 'target']


# In[ ]:


# features with lowest correlation with target
corr.head(10)


# In[ ]:


# features with highest correlation with target
corr.tail(10)

