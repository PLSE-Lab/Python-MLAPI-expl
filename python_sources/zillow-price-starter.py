#!/usr/bin/env python
# coding: utf-8

# Simple Start
# Reference: https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-zillow-prize

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
pd.options.display.max_rows = 100

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


train = pd.read_csv("../input/train_2016_v2.csv", parse_dates = ["transactiondate"])
train.head()


# In[ ]:


plt.figure(figsize = (8,6))
plt.scatter(range(len(train)), np.sort(train.logerror.values))
plt.xlabel('Sample Number')
plt.ylabel('logerror')
plt.show()


# In[ ]:


ulimit = np.percentile(train.logerror.values, 99)
llimit = np.percentile(train.logerror.values, 1)
train.logerror[train.logerror > ulimit] = ulimit
train.logerror[train.logerror < llimit] = llimit
plt.figure(figsize = (8,6))
sns.distplot(train.logerror.values, bins=50, kde=False)
plt.xlabel('logerror')
plt.show()


# In[ ]:


train['trainsaction_month'] = train.transactiondate.dt.month
cnt_month = train['trainsaction_month'].value_counts()
plt.figure(figsize=(8,6))
sns.barplot(cnt_month.index, cnt_month.values, alpha=0.7, color = color[4])
plt.xlabel('Transaction Month')
plt.ylabel('Transaction Count')
plt.show()


# In[ ]:


train.parcelid.value_counts().value_counts()
train[train.parcelid.duplicated(keep=False).values].groupby('parcelid').size().unique()
train['transN']  = 1
train.transN[train.parcelid.duplicated(keep='last').values] = 2
train.transN.unique()


# In[ ]:





# In[ ]:





# In[ ]:


prop = pd.read_csv("../input/properties_2016.csv")
prop.head()


# In[ ]:


missing_prop = prop.isnull().sum(axis=0)
missing_prop = missing_prop[missing_prop>0]
missing_prop.sort_values(inplace = True)
ind = np.arange(len(missing_prop))
fig, ax = plt.subplots(figsize=(12,18))
ax.barh(ind, missing_prop.values, color='blue',alpha=0.7)
ax.set_yticks(ind)
ax.set_yticklabels(missing_prop.index.values, rotation='horizontal')
ax.set_xlabel('Missing Values Count')
ax.set_ylabel('Number of missing values in each column')
plt.show()


# In[ ]:


plt.figure(figsize=(12,12))
sns.jointplot(x = prop.latitude.values, y = prop.longitude)
plt.ylabel('Longitude')
plt.xlabel('Latitude')
plt.show()


# In[ ]:


train_df = pd.merge(train, prop, on = 'parcelid', how='left')
train_df.head()


# In[ ]:


train_df.dtypes.value_counts()


# In[ ]:


missing_train = train_df.isnull().sum(axis=0)
missing_train =missing_train.to_frame('MissingCount')
missing_train['missing_ratio'] = missing_train.MissingCount.values/len(train_df)


# In[ ]:


missing_train[missing_train.missing_ratio>0.998]


# In[ ]:


train_df.fillna(train_df.mean(axis=0), inplace = True)


# In[ ]:


((train_df.isnull().sum(axis=0)).to_frame('MissingCount')).join(train_df.dtypes.to_frame('dtype'))


# In[ ]:


train_df['taxamount'].head()


# In[ ]:


x_cols = [col for col in train_df.columns if (col != 'logerror' and train_df[col].dtype == 'float64')]
labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(train_df[col].values, train_df.logerror.values)[0,1])
corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
corr_df = corr_df.sort_values(by = 'corr_values')
corr_df


# In[ ]:


ind = np.arange(len(labels))
fig, ax = plt.subplots(figsize = (14,50))
ax.barh(ind, corr_df.corr_values.values, color=color[3])
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation = 'horizontal')
ax.set_xlabel('Correlation coefficient')
ax.set_title('Correlation between variables and logerror')
plt.show()


# In[ ]:


train_df[corr_df[corr_df.corr_values.isnull()].col_labels.values].std()


# In[ ]:


corr_df_select = corr_df[ (corr_df['corr_values']>0.02).values | (corr_df['corr_values'] < -0.01).values ]
corr_df_select


# In[ ]:


train_df[['yearbuilt', 'bedroomcnt', 'fullbathcnt']] = train_df[['yearbuilt', 'bedroomcnt', 'fullbathcnt']].applymap(int)
col_corrs = corr_df_select.col_labels.tolist()+['logerror']
corrmat = train_df[col_corrs].corr(method='spearman')
plt.figure(figsize=(8,10))
sns.heatmap(corrmat, vmax = 1, square = True)
plt.title('Variables Correlation Map')
plt.show()


# In[ ]:


fig = plt.figure(figsize = (12,12))
sns.pairplot(train_df[col_corrs[:5] + ['logerror']])
plt.show()
fig = plt.figure(figsize = (12,12))
sns.pairplot(train_df[col_corrs[5:-1] + ['logerror']])
plt.show()


# In[ ]:


col = 'finishedsquarefeet12'
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit
plt.figure(figsize=(12,12))
sns.jointplot(x=train_df.finishedsquarefeet12.values, y=train_df.logerror.values, color= color[1])
plt.ylabel('Log Error')
plt.xlabel('Finished Square Feet 12')
plt.title('Finished Square Feet 12 VS Log Error')
plt.show()


# In[ ]:


col = 'calculatedfinishedsquarefeet'
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit
plt.figure(figsize=(12,12))
sns.jointplot(x=train_df.calculatedfinishedsquarefeet.values, y=train_df.logerror.values, color= color[0])
plt.ylabel('Log Error')
plt.xlabel('Calculated finished square feet')
plt.title('Calculated finished square feet 12 VS Log Error')
plt.show()


# In[ ]:


plt.figure(figsize=(12,12))
sns.boxplot(x = 'bathroomcnt', y = 'logerror', data=train_df)
plt.ylabel('Log Error')
plt.xlabel('Bathroom Count')
plt.xticks(rotation = 'vertical')
plt.title('Bathroom Count VS Log Error')
plt.show()


# In[ ]:


plt.figure(figsize=(12,12))
sns.boxplot(x = 'bedroomcnt', y = 'logerror', data=train_df)
plt.ylabel('Log Error')
plt.xlabel('Bedroom Count')
plt.xticks(rotation = 'vertical')
plt.title('Bedroom Count VS Log Error')
plt.show()


# In[ ]:


from ggplot import *
ggplot(aes(x='yearbuilt', y= 'logerror'), data= train_df) + geom_point(color='steelblue', size=1) + stat_smooth()

col = 'yearbuilt'
ulimit = np.percentile(train_df[col].values, 99.5)
llimit = np.percentile(train_df[col].values, 0.5)
train_df[col].ix[train_df[col]>ulimit] = ulimit
train_df[col].ix[train_df[col]<llimit] = llimit
plt.figure(figsize=(16,16))
sns.jointplot(x=train_df.yearbuilt.values, y=train_df.logerror.values, color= color[1])
plt.ylabel('Log Error')
plt.xlabel('Year Built')
plt.title('Year Built VS Log Error')
plt.show()

ggplot(aes(x='latitude', y='longitude', color='logerror'), data=train_df) +     geom_point() +     scale_color_gradient(low = 'blue', high = 'red')
train_y = train_df.logerror.values
cols_drop = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
train_X = train_df.drop(['parcelid', 'logerror', 'transactiondate'] + cols_drop, axis = 1)
featrure_name = train_X.columns.values

from sklearn import ensemble
model = ensemble.ExtraTreesRegressor(n_estimators = 25, max_depth=30, max_features = 0.3,n_jobs=-1, random_state=0)
model.fit(train_X, train_y)

importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis = 0)
indices = np.argsort(importances)[::-1][:10]

plt.figure(figsize=(12,12))
plt.title('Feature importance')
plt.bar(range(len(indices)), importances[indices], color = color[3], yerr = std[indices], align = 'center')
plt.xticks(range(len(indices)),featrure_name[indices], rotation = 'vertical' , fontsize =14)
plt.show()


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


sub.head()


# In[ ]:


train.head()
train.trainsaction_month.value_counts().sort_index()


# In[ ]:


prop.head()

