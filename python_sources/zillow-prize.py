#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None


# In[ ]:


import os
os.listdir("../input/zillow-prize-1")


# In[ ]:


train_df = pd.read_csv('../input/zillow-prize-1/train_2017.csv', parse_dates=["transactiondate"])
train_df.head()


# In[ ]:


train_df.shape


# In[ ]:


train_df.logerror.values


# In[ ]:


plt.figure(figsize=(8, 6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))
plt.xlabel('index')
plt.ylabel('logerror')
plt.show()


# Identify and replace the outliers

# In[ ]:


upper_limit = np.percentile(train_df.logerror.values, 99)
lower_limit = np.percentile(train_df.logerror.values, 1)
upper_limit, lower_limit


# Replace the outlier vals with upper and lower limit vals

# In[ ]:


train_df.loc[train_df['logerror'] > upper_limit, 'logerror'] = upper_limit
train_df.loc[train_df['logerror'] < lower_limit, 'logerror'] = lower_limit


# In[ ]:


plt.figure(figsize=(8, 6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))
plt.xlabel('index')
plt.ylabel('logerror')
plt.show()


# In[ ]:


plt.figure(figsize=(12, 8))
sns.distplot(train_df.logerror.values, bins=50, kde=False)
plt.xlabel('logerror', fontsize=12)
plt.show()


# Normal distribution obtained.

# In[ ]:


train_df['transaction_month'] = train_df['transactiondate'].dt.month

index_of_months = train_df['transaction_month'].value_counts().index
unique_months_vals = train_df['transaction_month'].value_counts().values

plt.figure(figsize=(12, 6))
sns.barplot(index_of_months, unique_months_vals, alpha=0.8, color=color[2])
plt.xlabel('Months of transaction')
plt.ylabel('Number of Transactions')
plt.show()


# In[ ]:


temp_df = train_df['parcelid'].value_counts().reset_index()
temp_df['parcelid'].value_counts()


# After counting appearance of each parcel id.
# It is seen that most parcelid appear only once in the dataset

# In[ ]:


prop_df = pd.read_csv("../input/zillow-prize-1/properties_2017.csv")
prop_df.shape


# In[ ]:


prop_df.head()


# In[ ]:


prop_df.isnull().sum().reset_index()


# In[ ]:


missing_df = prop_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.ix[missing_df['missing_count'] > 0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
fig, ax = plt.subplots(figsize=(12, 18))
rects = ax.barh(ind, missing_df.missing_count.values)
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


# In[ ]:


plt.figure(figsize=(12, 12))
sns.jointplot(x = prop_df.latitude.values, y = prop_df.longitude.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()


# In[ ]:


train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
train_df.head()


# In[ ]:


pd.options.display.max_rows = 65

dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df


# Almost all are float variables with few object (categorical) variables

# In[ ]:


dtype_df.groupby("Column Type").count().reset_index()


# In[ ]:


train_df.mean(axis=0)


# In[ ]:


train_df = train_df.fillna(0)

x_cols = [col for col in train_df.columns if col not in ['logerror'] if train_df[col].dtype == 'float64']

labels = []
values = []

# check corr of each col wrt 'logerror' col
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(train_df[col].values, train_df.logerror.values)[0,1])

corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
corr_df = corr_df.sort_values(by='corr_values')

ind = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(12,40))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='r')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()


# There are few variables at the top of this graph without any correlation values.

# In[ ]:


corr_zero_cols = ['assessmentyear', 'storytypeid', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'poolcnt', 'decktypeid', 'buildingclasstypeid']

for col in corr_zero_cols:
    print(col, train_df[col].nunique())


# Let us take the variables with high correlation values 

# In[ ]:


corr_df.loc[(corr_df['corr_values'] > 0.02) | (corr_df['corr_values'] < -0.01), ('col_labels', 'corr_values')]


# In[ ]:


corr_df_sel = corr_df.ix[(corr_df['corr_values']>0.02) | (corr_df['corr_values'] < -0.01)]
corr_df_sel


# Now to find corr among cols which are highly correlated to 'logerror' col

# In[ ]:


cols_to_use = corr_df_sel.col_labels.tolist()

temp_df = train_df[cols_to_use]
corrmat = temp_df.corr(method='spearman')

fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


# The important variables themselves are very highly correlated.

# In[ ]:


cols_to_use


# Now visualizing each cols against 'logerror' col

# In[ ]:


col = 'finishedsquarefeet12'

upper_limit = np.percentile(train_df[col].values, 99.5)
lower_limit = np.percentile(train_df[col].values, 0.5)

train_df.loc[train_df[col] > upper_limit, col] = upper_limit
train_df.loc[train_df[col] < lower_limit, col] = lower_limit

# train_df[col].ix[train_df[col] > upper_limit] = upper_limit
# train_df[col].ix[train_df[col] < lower_limit] = lower_limit

plt.figure(figsize=(12,12))
sns.jointplot(col, 'logerror', data = train_df, size=10, color=color[2])
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Finished Square Feet 12', fontsize=12)
plt.title("Finished square feet 12 Vs Log error", fontsize=15)
plt.show()


# Seems the range of logerror narrows down with increase in finished square feet 12 variable. Probably larger houses are easy to predict

# In[ ]:


col = "calculatedfinishedsquarefeet"

upper_limit = np.percentile(train_df[col].values, 99.5)
lower_limit = np.percentile(train_df[col].values, 0.5)

train_df.loc[train_df[col] > upper_limit, col] = upper_limit
train_df.loc[train_df[col] < lower_limit, col] = lower_limit

plt.figure(figsize=(12,12))
sns.jointplot(col, 'logerror', data = train_df, size=10, color=color[4])
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Calculated finished square feet', fontsize=12)
plt.title("Calculated finished square feet Vs Log error", fontsize=15)
plt.show()


# Here as well the distribution is very similar to the previous one. No wonder the correlation between the two variables are also high.

# In[ ]:


train_df['bathroomcnt'].value_counts()


# In[ ]:


col = "bathroomcnt"

plt.figure(figsize=(12,8))
sns.countplot(x = col, data=train_df)
plt.ylabel('Count', fontsize=12)
plt.xlabel('Bathroom', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Bathroom count", fontsize=15)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.boxplot(x = col, y='logerror', data=train_df)
plt.ylabel('Log error', fontsize=12)
plt.xlabel('Bathroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("How log error changes with bathroom count?", fontsize=15)
plt.show()


# In[ ]:


col = "bedroomcnt"

plt.figure(figsize=(12,8))
sns.countplot(col, data = train_df)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Bedroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Bedroom count", fontsize=15)
plt.show()


# In[ ]:


train_df.loc[train_df['bedroomcnt'] > 7, 'bedroomcnt'] = 7


# In[ ]:


col = "bedroomcnt"

plt.figure(figsize=(12,8))
sns.countplot(col, data = train_df)
plt.ylabel('Frequency', fontsize=12)
plt.xlabel('Bedroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Bedroom count", fontsize=15)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
sns.violinplot(x='bedroomcnt', y='logerror', data=train_df)
plt.xlabel('Bedroom count', fontsize=12)
plt.ylabel('Log Error', fontsize=12)
plt.show()


# In[ ]:


col = "taxamount"

upper_limit = np.percentile(train_df[col].values, 99.5)
lower_limit = np.percentile(train_df[col].values, 0.5)

train_df.loc[train_df[col] > upper_limit, col] = upper_limit
train_df.loc[train_df[col] < lower_limit, col] = lower_limit

plt.figure(figsize=(12,12))
sns.jointplot(col, 'logerror', data = train_df, size=10, color=color[6])
plt.ylabel('Log Error', fontsize=12)
plt.xlabel('Tax Amt', fontsize=12)
plt.title("Tax Amount Vs Log error", fontsize=15)
plt.show()


# In[ ]:


train_df.columns


# In[ ]:


train_y = train_df['logerror'].values
cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]

train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate', 'transaction_month'] + cat_cols, axis=1)

feat_names = train_df.columns


# In[ ]:


from sklearn import ensemble

model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)
model.fit(train_df, train_y)


# In[ ]:


model.estimators_


# In[ ]:


model.feature_importances_


# In[ ]:


importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()


# In[ ]:


import xgboost as xgb

xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}

dtrain = xgb.DMatrix(train_df, train_y, feature_names=train_df.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)


fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()

