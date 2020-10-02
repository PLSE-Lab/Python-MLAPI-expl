#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# In this competition, Santander Group is asking Kagglers to help them identify the value of transactions for each potential customer. This is a first step that Santander needs to nail in order to personalize their services at scale.
# 
# And as part of any good data analysis or machine learning project we need to do some exploratory data analysis first. One thing that makes things a little more tricky than in other competitions is that the features have no actual names as they are annonymised and there is no data dictionary. 
# 
# So, without further ado, let's get started.

# In[1]:


# Import libraries

import pandas as pd
import numpy as np

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', context='notebook', palette='deep') 
import matplotlib.style as style
style.use('fivethirtyeight')


# In[2]:


# Import the data

train = pd.read_csv('../input/train.csv', index_col='ID')
#test = pd.read_csv('../input/test.csv', index_col='ID')


# ## Data Summary

# In[16]:


print('train shape:', train.shape)
print('test shape:', test.shape)


# In[49]:


train.head()


# In[18]:


test.head()


# ## Missing Data

# After having had a look at the initial data, one of the first questions you should be asking is whether there anay features (columns) with missing data. This happens quite often but this time there are no missing data (NaNs) in either training or test set. The graphs below would have shown which fields contain Nans and how many. 

# In[19]:


# Capture the necessary data
variables = train.columns

count = []

for variable in variables:
    length = train[variable].count()
    count.append(length)
    
count_pct = np.round(100 * pd.Series(count) / len(train), 2)
count = pd.Series(count)

missing = pd.DataFrame()
missing['variables'] = variables
missing['count'] = len(train) - count
missing['count_pct'] = 100 - count_pct
missing = missing[missing['count_pct'] > 0]
missing.sort_values(by=['count_pct'], inplace=True)
missing_train = np.array(missing['variables'])

#Plot number of available data per variable
plt.subplots(figsize=(15,6))

# Plots missing data in percentage
plt.subplot(1,2,1)
plt.barh(missing['variables'], missing['count_pct'])
plt.title('Count of missing training data in percent', fontsize=15)

# Plots total row number of missing data
plt.subplot(1,2,2)
plt.barh(missing['variables'], missing['count'])
plt.title('Count of missing training data as total records', fontsize=15)

plt.show()


# In[20]:


# Capture the necessary data
variables = test.columns

count = []

for variable in variables:
    length = test[variable].count()
    count.append(length)
    
count_pct = np.round(100 * pd.Series(count) / len(test), 2)
count = pd.Series(count)

missing = pd.DataFrame()
missing['variables'] = variables
missing['count'] = len(test) - count
missing['count_pct'] = 100 - count_pct
missing = missing[missing['count_pct'] > 0]
missing.sort_values(by=['count_pct'], inplace=True)
missing_test = np.array(missing['variables'])

#Plot number of available data per variable
plt.subplots(figsize=(15,6))

# Plots missing data in percentage
plt.subplot(1,2,1)
plt.barh(missing['variables'], missing['count_pct'])
plt.title('Count of missing test data in percent', fontsize=15)

# Plots total row number of missing data
plt.subplot(1,2,2)
plt.barh(missing['variables'], missing['count'])
plt.title('Count of missing test data as total records', fontsize=15)

plt.show()


# The challenge with this dataset is that it is very sparse and contains many zeroes. In fact there 256 features in the training set that have only one value - zero. Let's delete those from the training set as they contain no information.

# In[ ]:


unique_df = train.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1]
constant_df.shape


# In[ ]:


# Delete columns with constant values

train = train.drop(constant_df.col_name.tolist(), axis=1)


# ## Data Types of Features

# Let's check whether the data types of all features are correct. From what I have seen so far I would expect everything to be floats. However, there are lots of integers. Could these be categorical features?

# In[50]:


dtype = train.dtypes.reset_index()
dtype.columns = ["Count", "Column Type"]
dtype.groupby("Column Type").aggregate('count').reset_index()


# ## Target Variable

# Next one up is the target value. What does the distribution look like and do we have to transform it?
# 
# From the plots below, we can clearly see that the distribution is skewed and we will have to transform it, so that it approaches a more 'normal' distributions as this will help when training the data. Below you can see the log transformation of the traget varaible, which looks more 'normal'. Also notice that I added 1 to every value before taking the log. This gets around the problem of having 0s as you can't take the log of 0.

# In[21]:


sns.set()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#plt.figure(figsize=(10,6))
ax = axes[0]
sns.distplot(train['target'], ax=ax)
ax.set_title('Histogram of Target')

ax = axes[1]
sns.boxplot(data=train, x='target', ax=ax)
ax.set_title('Boxplot of Target')

plt.show()


# In[22]:


plt.figure(figsize=(14,5))
sns.distplot(np.log1p(train['target']))
plt.title('Histogram of Log of Target')
plt.show()


# ## Features / Independent Variables

# We can't write much about the features or how we think they will affect the target as we don't have any names or any other prior information. What we can do however is to get an overview of  their statistical properties.
# 
# Let's start with a correlation matrix of the features with the highest absolute correlations with target - otherwise this would be a huge matrix. We have over 4000 features after all. The first thing we need to do though, is scale the data. Otherwise the correlation calculations can lead to incorrect results when the features are on different scales. 

# In[58]:


# Scale the data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train)
train_scaled = pd.DataFrame(train_scaled, columns=train.columns)


# In[63]:


labels = []
values = []
for col in train_scaled.columns:
    if col != 'target':
        labels.append(col)
        values.append(np.corrcoef(train_scaled[col].values, train_scaled['target'].values)[0,1])
corr = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr = corr.sort_values(by='corr_values')

cols_to_use = corr[(corr['corr_values']>0.25) | (corr['corr_values']<-0.25)].col_labels.tolist()

temp_df = train_scaled[cols_to_use]
corrmat = temp_df.corr(method='spearman')

# Generate a mask for the upper triangle
mask = np.zeros_like(corrmat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(14, 14))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corrmat, mask=mask, 
            square=True, linewidths=.5, annot=False, cmap=cmap)
plt.yticks(rotation=0)
plt.title("Correlation Matrix of Most Correlated Features", fontsize=15)
plt.show()


# ## Final Points

# I will add more over the coming days. One interesting point [Wesam Elshamy](https://www.kaggle.com/wesamelshamy/beyond-eda-santander-transaction-value) is making though, is that the distributions of the features of the training and test set are quite different. This is very important as it will impact you prediction score. Ideally, they would have the same distributions.
