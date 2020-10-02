#!/usr/bin/env python
# coding: utf-8

# SRK's notebook "[Simple Exploration Notebook](https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-zillow-prize)" has a nice set of visualizations for the general data.  This will attempt to explore the outliers instead.  

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
pd.options.display.max_rows = 200


# In[ ]:


train_df = pd.read_csv('../input/train_2016.csv', parse_dates=["transactiondate"])
prop_df = pd.read_csv('../input/properties_2016.csv')
sample_df = pd.read_csv('../input/sample_submission.csv')


# First let's reproduce the log error graph so we have another look at the outliers

# In[ ]:


plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.show()


# ### Isolate the outliers:
# 
# Now we can remove the middle values to isolate the outliers

# In[ ]:


ulimit = np.percentile(train_df.logerror.values, 99)
llimit = np.percentile(train_df.logerror.values, 1)
print("Upper limit: {}".format(ulimit))
print("Lower limit: {}".format(llimit))
outlier_df = train_df[(train_df.logerror.values > ulimit) | (train_df.logerror.values < llimit)]
print("Outlier shape: {}".format(outlier_df.shape))

plt.figure(figsize=(12,8))
sns.distplot(outlier_df.logerror.values, bins=50, kde=False)
plt.xlabel('logerror', fontsize=12)
plt.show()


# So we get the same normal distribution with the center cut out.  Exactly what we expected.  We can reproduce many on the same graphs as in SRK's notebook.
# 
# ### Transaction Date:
# 
# Now we can reproduce the date field graph.

# In[ ]:


outlier_df['transaction_month'] = outlier_df['transactiondate'].dt.month

cnt_srs = outlier_df['transaction_month'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
plt.xticks(rotation='vertical')
plt.xlabel('Month of transaction', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()


# Basically the same as the SRK version except scaled down.

# ### Properties 2016:
# 
# Now lets merge the data and start looking at the outliers.

# In[ ]:


outlier_df = pd.merge(outlier_df, prop_df, on='parcelid', how='left')
outlier_df.head()


# Now lets look at the role of NaN values in the outliers.

# In[ ]:


missing_df = outlier_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df[missing_df['missing_count']>0]

ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel('Count of missing values')
ax.set_title('Number of missing values in each column')
plt.show()


# Let us explore the latitude and longitude variable.

# In[ ]:


plt.figure(figsize=(12,12))
sns.jointplot(x=outlier_df.latitude.values, y=outlier_df.longitude.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()


# In[ ]:


dtype_df = outlier_df.dtypes.reset_index()
dtype_df.columns = ['Count', 'Column Type']
dtype_df


# In[ ]:


missing_df.head()


# In[ ]:




