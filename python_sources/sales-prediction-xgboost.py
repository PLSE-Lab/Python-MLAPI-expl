#!/usr/bin/env python
# coding: utf-8

# #### Contents<br>

# * Import Libraries
# * Load Data
# * Data Exploration
# * Data Pre processing
# * Feature Engineering
# * Checking Outliers
# * Train/ Validation Split
# * Create .pkl file
# * Create Model
# * Submission

# **Import Libraries**

# In[ ]:


import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='darkgrid')
pd.set_option('display.float_format',lambda  x: '%.2f' %x)


# In[ ]:


#sales = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")


# **Load Data**

# In[ ]:


sales = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')


# In[ ]:


print('sales:',sales.shape,'test:',test.shape,'items:',items.shape,'item_cats:',item_cats.shape,'shop:',shops.shape)


# Change date form to DDMMYYYY

# In[ ]:


sales['date'] = pd.to_datetime(sales['date'], format='%d.%m.%Y')


# In[ ]:


sales.head(3)


# In[ ]:


test.head(3)


# In[ ]:


items.head(3)


# In[ ]:


item_cats.head(3)


# In[ ]:


shops.head(3)


# **Joining Dataset **

# In[ ]:


train = sales.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id',
 rsuffix='_').join(item_cats,on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)


# In[ ]:


train.head(3)


# In[ ]:


train.shape


# In[ ]:


train.describe()


# **Data Exploration**

# In[ ]:


plt.figure(figsize=(35,10))
sns.countplot(x='date_block_num',data=train)
plt.show()


# In[ ]:


plt.figure(figsize=(35,10))
sns.countplot(x='shop_id', data=train)
plt.show()


# In[ ]:


plt.figure(figsize=(35,10))
sns.countplot(x='item_category_id',data= train)
plt.show()


# Monthly sales

# In[ ]:


monthwise_sale=pd.DataFrame(train.groupby(['date_block_num'])['item_cnt_day'].sum().reset_index())
plt.figure(figsize=(35,10))
sns.barplot(x='date_block_num',y='item_cnt_day',data= monthwise_sale,order=monthwise_sale['date_block_num'])
plt.xlabel('Months')
plt.ylabel('itemcount in a day')
plt.title('item count in a day per month')
plt.show()


# In[ ]:


ts = train.groupby(['date_block_num'])['item_cnt_day'].sum()
plt.figure(figsize=(35,10))
plt.xlabel = 'Time'
plt.ylabel = 'Sales'
plt.title = 'Total sale of the company'
plt.plot(ts)


# There is peak sale at begenning of the year and then decreasing trend .(In between 2 month there is peak in sale)

# **Period** <br>
#  Let us check the data from  date till date.

# In[ ]:


print('Min date from train set: %s'  %train['date'].min().date())
print('Max date from train set: %s' % train['date'].max().date())


# **Data Leakage** <br>
# In test we have the column shop_id,item_id

# In[ ]:


shop_id_test = test['shop_id'].unique()
item_id_test = test['item_id'].unique()


# In[ ]:


# Only shop that exist in the test set
train_k = train[train['shop_id'].isin(shop_id_test)]
# Only item that exist in the test set
train_k = train[train['item_id'].isin(item_id_test)]


# In[ ]:


print('Data set  size before leakage',train.shape[0])
print('Data set  size after leakage',train_k.shape[0])


# Ther is one item with one item below zero,fill it with median

# In[ ]:


train[train['item_price']<=0]


# In[ ]:


train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)]


# In[ ]:


median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&
               (train.item_price>0)].item_price.median()


# In[ ]:


median


# In[ ]:


train.loc[train.item_price<0,'item_price'] =median


# **Data Pre processing**<br>
# Select only numerical features as we are working with numerical features

# In[ ]:


train_k.shape


# In[ ]:


train_monthly = train_k[['date', 'date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'item_cnt_day']]


# In[ ]:


train_monthly.shape


# In[ ]:


# Group by month in this case "date_block_num" and aggregate features.
train_monthly = train_monthly.sort_values('date').groupby(['date_block_num', 'shop_id', 'item_category_id', 'item_id'], as_index=False)


# In[ ]:


train_monthly = train_monthly.agg({'item_price':['sum', 'mean'], 'item_cnt_day':['sum', 'mean','count']})


# In[ ]:


train_monthly.shape


# In[ ]:


# Rename features.
train_monthly.columns = ['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'mean_item_price', 'item_cnt', 'mean_item_cnt', 'transactions']


# In[ ]:


train_monthly.head(3)


# In[ ]:


train_monthly.isnull().sum()


# To get real behavior of the data we have to create the missing records from the loaded dataset, so for each month(date_block_num) we need to create the missing records for each shop and item.

# In[ ]:


# Build a data set with all the possible combinations of ['date_block_num','shop_id','item_id'] so we won't have missing records.
shop_ids = train_monthly['shop_id'].unique()
item_ids = train_monthly['item_id'].unique()
empty_df = []
for i in range(34):
    for shop in shop_ids:
        for item in item_ids:
            empty_df.append([i, shop, item])
    
empty_df = pd.DataFrame(empty_df, columns=['date_block_num','shop_id','item_id'])


# In[ ]:


empty_df.shape


# In[ ]:


empty_df.head(3)


# Merge the train set with the complete set (missing records will be filled with 0).

# In[ ]:


train_monthly = pd.merge(empty_df, train_monthly, on=['date_block_num','shop_id','item_id'], how='left')
train_monthly.fillna(0, inplace=True)


# In[ ]:


train_monthly.describe()


# In[ ]:


#Extract Year and Month features
train_monthly['year'] = train_monthly['date_block_num'].apply(lambda x: ((x//12+2013)))


# In[ ]:


train_monthly['month'] = train_monthly['date_block_num'].apply(lambda x: (x%12))


# #### Feature Engineering

# Create additional features

# In[ ]:


gp_month_mean = train_monthly.groupby(['month'], as_index=False)['item_cnt'].mean()
gp_month_sum = train_monthly.groupby(['month'],as_index=False)['item_cnt'].sum()
gp_category_mean = train_monthly.groupby(['item_category_id'],as_index=False)['item_cnt'].mean()
gp_category_sum = train_monthly.groupby(['item_category_id'],as_index=False)['item_cnt'].sum()
gp_shop_mean = train_monthly.groupby(['shop_id'],as_index=False)['item_cnt'].mean()
gp_shop_sum = train_monthly.groupby(['shop_id'],as_index=False)['item_cnt'].sum()


# Data exploration with additional features<br>
#  Sales by Shop

# In[ ]:


f,axes = plt.subplots(2,1,figsize=(35,10))
sns.barplot(x='shop_id',y='item_cnt',data= gp_shop_mean,ax= axes[0]).set_title('monthly average sale')
sns.barplot(x='shop_id',y='item_cnt',data= gp_shop_sum,ax = axes[1]).set_title('monthly total sale')
plt.show()


# Except 4 shpos most of the shops have similar sales rate <br>
#  
#  **Yearly sales behaviar**

# In[ ]:


f,axes = plt.subplots(2,1,figsize=(35,10))
sns.barplot(x='month',y='item_cnt',data= gp_month_mean,ax= axes[0]).set_title('monthly mean')
sns.lineplot(x='month',y='item_cnt',data= gp_month_mean,ax= axes[0]).set_title('monthly mean')
sns.barplot(x='month',y='item_cnt',data= gp_month_sum,ax = axes[1]).set_title('monthly sum')
sns.lineplot(x='month',y='item_cnt',data= gp_month_sum,ax = axes[1]).set_title('monthly sum')
plt.show()


# Sales increases towards the end of the year<br>
#  **Sales by Category**

# In[ ]:


f,axes = plt.subplots(2,1,figsize=(35,10))
sns.barplot(x='item_category_id',y='item_cnt',data= gp_category_mean,ax= axes[0]).set_title('monthly mean sale by category')
sns.barplot(x='item_category_id',y='item_cnt',data= gp_category_sum,ax = axes[1]).set_title('monthly total sale by category ')
plt.show()


# Only few category have more sales count

# #### Checking Outliers

# In[ ]:


plt.subplots(figsize=(20,6))

sns.boxplot(train_monthly['item_cnt'])
plt.show()


# In[ ]:


plt.subplots(figsize=(20,6))

sns.boxplot(train_monthly['item_price'])
plt.show()


# In[ ]:


train_monthly.shape


# #### Treating Outliers

# In[ ]:


train_monthly = train_monthly.query('item_cnt >= 0 and item_cnt <= 20 and item_price < 400000')


# In[ ]:


train_monthly.shape


# We need to forcast 'item_cnt',

# In[ ]:


train_monthly['item_cnt_month'] = train_monthly.sort_values('date_block_num').groupby(['shop_id','item_id'])['item_cnt'].shift(-1)


# In[ ]:


train_monthly.head(3)


# Unit price for the item

# In[ ]:


train_monthly['item_price_unit'] = train_monthly['item_price']//train_monthly['item_cnt']


# In[ ]:


train_monthly.isnull().sum()


# In[ ]:


train_monthly['item_price_unit'].fillna(0, inplace=True)


# More features

# In[ ]:


gp_item_price = train_monthly.sort_values('date_block_num').groupby(['item_id'], as_index=False).agg({'item_price':[np.min, np.max]})
gp_item_price.columns = ['item_id', 'hist_min_item_price', 'hist_max_item_price']

train_monthly = pd.merge(train_monthly, gp_item_price, on='item_id', how='left')


# We have 15 column now

# In[ ]:


train_monthly.shape

How much each item's price changed from its (lowest/highest) historical price
# In[ ]:


train_monthly['price_increase'] = train_monthly['item_price'] - train_monthly['hist_min_item_price']
train_monthly['price_decrease'] = train_monthly['hist_max_item_price'] - train_monthly['item_price']


# In[ ]:


train_monthly.shape


# In[ ]:


train_monthly.head()


# In[ ]:


train_monthly.shape


# Pandas dataframe. shift() function Shift index by desired number of periods with an optional time freq. This function takes a scalar parameter called period, which represents the number of shifts to be made over the desired axis.

# In[ ]:


lag_list = [1, 2, 3]

for lag in lag_list:
    ft_name = ('item_cnt_shifted%s' % lag)
    train_monthly[ft_name] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_category_id', 'item_id'])['item_cnt'].shift(lag)
    train_monthly[ft_name].fillna(0, inplace=True)


# In[ ]:


train_monthly['item_cnt_shifted3'].max()


# In[ ]:


train_monthly.isnull().sum()


# Item sales count trend.

# In[ ]:


train_monthly['item_trend'] = train_monthly['item_cnt']

for lag in lag_list:
    ft_name = ('item_cnt_shifted%s' % lag)
    train_monthly['item_trend'] -= train_monthly[ft_name]

train_monthly['item_trend'] /= len(lag_list) + 1


# Dataset after feature engineering

# In[ ]:


train_monthly.describe()


# **Train/validation split**<br>
#   Train set will be the  3~28 months, validation will be 29~32 months and test will be block 33.<br>

# In[ ]:


train_set = train_monthly.query('date_block_num>=3 and date_block_num <28').copy()
validation_set = train_monthly.query('date_block_num >= 28 and date_block_num < 33').copy()
test_set = train_monthly.query('date_block_num == 33').copy()

train_set.dropna(subset= ['item_cnt_month'],inplace = True)
validation_set.dropna(subset = ['item_cnt_month'],inplace = True)
train_set.dropna(inplace = True)
validation_set.dropna(inplace = True)

print('Train set records:',train_set.shape[0])
print('validation set records:',validation_set.shape[0])
print('Test set records:',test_set.shape[0])

print('Train set records: %s (%.f%% of complete data)'
      % (train_set.shape[0],((train_set.shape[0]/train_monthly.shape[0])*100)))

print('Validation set records: %s (%.f%% of complete data)'
          % (validation_set.shape[0],((validation_set.shape[0]/train_monthly.shape[0])*100)))


# In[ ]:


train_set.head(4)


# In[ ]:


train_set.shape


# In[ ]:


validation_set.head(4)


# In[ ]:


validation_set.shape


# In[ ]:


test_set.head(4)


# In[ ]:


test_set.shape


# **Mean Encoding** <br>
#  done after the train/validation split.<br>
#  We are calculating Shop mean,Item mean,Shop-item Mean,Year mean,Month mean

# In[ ]:


# Shop mean encoding.
gp_shop_mean = train_set.groupby(['shop_id']).agg({'item_cnt_month': ['mean']})
gp_shop_mean.columns = ['shop_mean']
gp_shop_mean.reset_index(inplace=True)
# Item mean encoding.
gp_item_mean = train_set.groupby(['item_id']).agg({'item_cnt_month': ['mean']})
gp_item_mean.columns = ['item_mean']
gp_item_mean.reset_index(inplace=True)
# Shop with item mean encoding.
gp_shop_item_mean = train_set.groupby(['shop_id', 'item_id']).agg({'item_cnt_month': ['mean']})
gp_shop_item_mean.columns = ['shop_item_mean']
gp_shop_item_mean.reset_index(inplace=True)
# Year mean encoding.
gp_year_mean = train_set.groupby(['year']).agg({'item_cnt_month': ['mean']})
gp_year_mean.columns = ['year_mean']
gp_year_mean.reset_index(inplace=True)
# Month mean encoding.
gp_month_mean = train_set.groupby(['month']).agg({'item_cnt_month': ['mean']})
gp_month_mean.columns = ['month_mean']
gp_month_mean.reset_index(inplace=True)

# Add meand encoding features to train set.
train_set = pd.merge(train_set, gp_shop_mean, on=['shop_id'], how='left')
train_set = pd.merge(train_set, gp_item_mean, on=['item_id'], how='left')
train_set = pd.merge(train_set, gp_shop_item_mean, on=['shop_id', 'item_id'], how='left')
train_set = pd.merge(train_set, gp_year_mean, on=['year'], how='left')
train_set = pd.merge(train_set, gp_month_mean, on=['month'], how='left')
# Add meand encoding features to validation set.
validation_set = pd.merge(validation_set, gp_shop_mean, on=['shop_id'], how='left')
validation_set = pd.merge(validation_set, gp_item_mean, on=['item_id'], how='left')
validation_set = pd.merge(validation_set, gp_shop_item_mean, on=['shop_id', 'item_id'], how='left')
validation_set = pd.merge(validation_set, gp_year_mean, on=['year'], how='left')
validation_set = pd.merge(validation_set, gp_month_mean, on=['month'], how='left')


# In[ ]:


train_set.head(4)


# In[ ]:


print('train shape:',train_set.shape,'validation shape:',validation_set.shape,'test shape:',test_set.shape)


# In[ ]:


# Create train and validation sets and labels. 
X_train = train_set.drop(['item_cnt_month', 'date_block_num'], axis=1)
Y_train = train_set['item_cnt_month'].astype(int)
X_validation = validation_set.drop(['item_cnt_month', 'date_block_num'], axis=1)
Y_validation = validation_set['item_cnt_month'].astype(int)


# In[ ]:


Y_train.head()


# In[ ]:


X_train.shape,Y_train.shape,X_validation.shape,Y_validation.shape


# In[ ]:


test_set.shape


# **Build test set** <br>
#  We want to predict for "date_block_num" -34 .We use block 33 because we want to forecast values for block 34.

# In[ ]:


latest_records = pd.concat([train_set, validation_set]).drop_duplicates(subset=['shop_id', 'item_id'], keep='last')


# In[ ]:


latest_records.shape


# In[ ]:


X_test = pd.merge(test, latest_records, on=['shop_id', 'item_id'], how='left', suffixes=['', '_'])


# In[ ]:


X_test.head()


# In[ ]:


X_train.head()


# In[ ]:


X_train.shape


# In[ ]:


int_features = ['shop_id', 'item_id', 'year', 'month']

X_train[int_features] = X_train[int_features].astype('int32')
X_validation[int_features] = X_validation[int_features].astype('int32')


# In[ ]:


X_test['year'] = 2015
X_test['month'] = 9
X_test.drop('item_cnt_month', axis=1, inplace=True)
X_test[int_features] = X_test[int_features].astype('int32')
X_test = X_test[X_train.columns]


# In[ ]:


X_test.head()


# Replacing missing values.

# In[ ]:


X_test.isnull().sum()


# In[ ]:


sets = [X_train, X_validation, X_test]
# Median of each shop is used for filling missing value.            
for dataset in sets:
    for shop_id in dataset['shop_id'].unique():
        for column in dataset.columns:
            shop_median = dataset[(dataset['shop_id'] == shop_id)][column].median()
            dataset.loc[(dataset[column].isnull()) & (dataset['shop_id'] == shop_id), column] = shop_median
            
# Fill remaining missing values 
X_test.fillna(X_test.mean(), inplace=True)


# In[ ]:


# Dropping "item_category_id",as given test dont have Item category
X_train.drop(['item_category_id'], axis=1, inplace=True)
X_validation.drop(['item_category_id'], axis=1, inplace=True)
X_test.drop(['item_category_id'], axis=1, inplace=True)


# In[ ]:


X_train.head()


# In[ ]:


X_validation.head()


# In[ ]:


X_test.head()


# In[ ]:


X_test.describe()


# #### Create .pkl file <br>
#  We are Saving the Train,Validation,Test data as .pkl file  to save the time. These file can be call any time when ever we required for create  model .

# In[ ]:


#X_train.to_pickle('../working/X_train.pkl')
#X_test.to_pickle('../working/X_test.pkl')
#Y_validation.to_pickle('../working/Y_validation.pkl')
#X_validation.to_pickle('../working/X_validation.pkl')
#Y_train.to_pickle('../working/Y_train.pkl')


# ### Create Model
#  #### Import required Libraries.

# In[ ]:


#import pickle
from xgboost import XGBRegressor
from xgboost import plot_importance


# In[ ]:


#X_train = pd.read_pickle('../working/X_train.pkl')
#X_test = pd.read_pickle('../working/X_test.pkl')
#Y_validation = pd.read_pickle('../working/Y_validation.pkl')
#X_validation = pd.read_pickle('../working/X_validation.pkl')
#Y_train = pd.read_pickle('../working/Y_train.pkl')


# In[ ]:


xg_reg = XGBRegressor(
    n_jobs=-1,
    tree_method='exact',
    learning_rate=0.2,
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)


# In[ ]:


xg_reg.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_validation, Y_validation)], 
    verbose=10, 
    early_stopping_rounds = 3)


# In[ ]:


Y_pred = xg_reg.predict(X_validation).clip(0, 20)
Y_test = xg_reg.predict(X_test).clip(0, 20)


# In[ ]:


submission = pd.DataFrame({
    "ID": X_test.index, 
    "item_cnt_month": Y_test
})


# In[ ]:


#submission.to_csv('xgb_submission.csv', index=False)


# In[ ]:


def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

plot_features(xg_reg, (10,14))


# In[ ]:


from sklearn.metrics import mean_squared_error


# In[ ]:


#xgboost_train_pred = xg_reg.predict(X_train)
xgboost_val_pred = xg_reg.predict(X_validation)


# In[ ]:



print('Validation rmse:', np.sqrt(mean_squared_error(Y_validation, xgboost_val_pred)))

