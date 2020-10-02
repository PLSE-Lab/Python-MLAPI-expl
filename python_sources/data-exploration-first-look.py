#!/usr/bin/env python
# coding: utf-8

# **Imports Library**
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from ggplot import *

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


# In[ ]:


# Load Train data

train_df = pd.read_csv("../input/train_2016.csv", parse_dates=["transactiondate"])
train_df.head()


# **Considering all data make a plot of distribution of error**

# In[ ]:



plt.figure(figsize=(12,8))
sns.distplot(train_df.logerror.values, bins=50, kde=False)
plt.xlabel('logerror', fontsize=12)
plt.show()


# In[ ]:


train_df['transaction_month'] = train_df['transactiondate'].dt.month

cnt_srs = train_df['transaction_month'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[1])
plt.xticks(rotation='vertical')
plt.xlabel('Month of transaction', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()


# In[ ]:


train_df['transaction_day'] = train_df['transactiondate'].dt.day

cnt_srs = train_df['transaction_day'].value_counts()
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[5])
plt.xticks(rotation='vertical')
plt.xlabel('Day of transaction', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()


# In[ ]:


#load properties data frame
prop_df = pd.read_csv("../input/properties_2016.csv")
prop_df.head()


# How many NaN there are in this dataset ?

# In[ ]:


missing_df = prop_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.ix[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.4
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='red')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of NaN")
ax.set_title("Number of missing values in each column")
plt.show()


# In[ ]:


plt.figure(figsize=(12,12))
sns.jointplot(x=prop_df.latitude.values, y=prop_df.longitude.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()


# In[ ]:


#merge data on key 
train_df_merged = pd.merge(train_df, prop_df, on='parcelid', how='left')
train_df_merged.head()


# In[ ]:


#aggregate this
nan_df = train_df_merged.isnull().sum(axis=0).reset_index()
nan_df.columns = ['column_name', 'missing_count']
nan_df['missing_ratio'] = nan_df['missing_count'] / train_df_merged.shape[0]
nan_df.ix[nan_df['missing_ratio']>0.999]


# In[ ]:


# Let us just impute the missing values with mean values to compute correlation coefficients #
mean_values = train_df_merged.mean(axis=0)
train_df_new = train_df_merged.fillna(mean_values, inplace=True)

# Now let us look at the correlation coefficient of each of these variables #
x_cols = [col for col in train_df_new.columns if col not in ['logerror'] if train_df_new[col].dtype=='float64']

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(train_df_new[col].values, train_df_new.logerror.values)[0,1])
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
    
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
#autolabel(rects)
plt.show()


# In[ ]:


corr_df_sel = corr_df.ix[(corr_df['corr_values']>0.02) | (corr_df['corr_values'] < -0.01)]
#corr_df_sel

cols_to_use = corr_df_sel.col_labels.tolist()

temp_df = train_df_merged[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(10, 10))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Correlation HeatMap", fontsize=15)
plt.show()


# **Bathroom count**
# 
# ***There is an interesting 2.279 value in the bathroom count this is a mean value***

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="bathroomcnt", data=train_df_merged)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Bathroom', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Bathroom count", fontsize=15)
plt.show()


# **Dimension distribution**

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="bedroomcnt", data=train_df_merged)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Bedroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Beedroom count frequency", fontsize=15)
plt.show()


# In[ ]:


train_df_merged['bedroomcnt'].ix[train_df_merged['bedroomcnt']>7] = 7
plt.figure(figsize=(10,8))
sns.violinplot(x='bedroomcnt', y='logerror', data=train_df_merged)
plt.xlabel('Bedroom count', fontsize=12)
plt.ylabel('Log Error', fontsize=12)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x="yearbuilt", data=train_df_merged)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Bedroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Beedroom count frequency", fontsize=15)
plt.show()


# In[ ]:



ggplot(aes(x='latitude', y='longitude', color='logerror'), data=train_df_merged) +     geom_point() +     scale_color_gradient(low = 'red', high = 'blue')


# In[ ]:


ggplot(aes(x='finishedsquarefeet12', y='taxamount', color='logerror'), data=train_df_merged) +     geom_now_its_art()


# In[ ]:


#reload all data 
train_raw_data = pd.read_csv("../input/train_2016.csv", parse_dates=["transactiondate"])

#merge data on key 
train_df = pd.merge(train_raw_data, prop_df, on='parcelid', how='left')

train_y = train_raw_data['logerror'].values
cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate']+cat_cols, axis=1)
#clean all data NaN
train_df = train_df.fillna(0)
feat_names = train_df.columns.values

train_df.head()


# In[ ]:


from sklearn import ensemble
model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)
model.fit(train_df, train_y)
## plot the importances ##
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="b", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='90')
plt.xlim([-1, len(indices)])
plt.gca().invert_yaxis()
plt.show()

