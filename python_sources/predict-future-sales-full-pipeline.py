#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import os
import math

import numpy as np
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb


# # Read the data

# In[ ]:


data_dir = '/kaggle/input'
os.listdir(data_dir)


# In[ ]:


sample_submission = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
items = pd.read_csv(os.path.join(data_dir, 'items.csv'))
shops = pd.read_csv(os.path.join(data_dir,'shops.csv'))
sales = pd.read_csv(os.path.join(data_dir, 'sales_train.csv'))
test_data = pd.read_csv(os.path.join(data_dir, 'test.csv'))


# # Exploratory data analysis

# In[ ]:


items.columns, shops.columns, sales.columns, test_data.columns


# In[ ]:


sales.shape, test_data.shape


# In[ ]:


sales.info()


# In[ ]:


sales.describe()


# In[ ]:


sales.nunique()


# In[ ]:


x = sales.groupby('date_block_num').agg({'item_cnt_day': 'sum'})
plt.title('Total sales by date_block_num')
plt.plot(x.index, x['item_cnt_day'])


# In[ ]:


plt.title('Total sales by month')

x_1 = x[x.index < 12]
plt.plot(x_1.index + 1, x_1['item_cnt_day'], color='red', label='2013')

x_2 = x[x.index >= 12]
x_2 = x_2[x_2.index < 24]
plt.plot(range(1, len(x_2.index) + 1), x_2['item_cnt_day'], color='green', label='2014')

x_3 = x[x.index >= 24]
plt.plot(range(1, len(x_3.index) + 1), x_3['item_cnt_day'], color='blue', label='2015')

plt.legend()


# In[ ]:


# There is only one item more expensive than 100000 (outlier).
sales[sales['item_price'] > 100000]


# In[ ]:


x_price = sales[sales['item_price'] < 40000] # drop outliers
x_price = x_price.groupby('item_price').agg({'item_cnt_day': 'sum'})
x_price = x_price[x_price['item_cnt_day'].values < 20000] # drop outliers
plt.title('Items sold by price')
plt.scatter(x_price.index, x_price['item_cnt_day'])
plt.xlabel('Price')
plt.ylabel('item_cnt_day')


# # Feature engineering

# Remove first year of sales data

# In[ ]:


sales = sales[sales['date_block_num'] > 11]


# Aggregate and sort the data

# In[ ]:


index_cols = ['shop_id', 'item_id', 'date_block_num']

# For every month we create a grid from all shops/items combinations from that month
grid = [] 
for block_num in sales['date_block_num'].unique():
    cur_shops = sales[sales['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sales[sales['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))

#turn the grid into pandas dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

#get aggregated values for (shop_id, item_id, month)
gb = sales.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum'}})

#fix column names
gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]
# #join aggregated data to the grid
all_data = pd.merge(grid,gb,how='left',on=index_cols).fillna(0)

#sort the data
all_data.sort_values(['date_block_num','shop_id','item_id'],inplace=True)


# In[ ]:


all_data.head()


# Fix the duplicated shop id

# In[ ]:


all_data.loc[all_data['shop_id'] == 0, 'shop_id'] = 57
all_data.loc[all_data['shop_id'] == 1, 'shop_id'] = 58
all_data.loc[all_data['shop_id'] == 11, 'shop_id'] = 10


# Generate lag features

# In[ ]:


lags = [1, 2, 3, 6, 12]

for lag in lags:
    lag_col_name = 'target_lag_' + str(lag)
    shifted = all_data[index_cols + ['target']].copy()
    shifted.columns = index_cols + [lag_col_name]
    shifted['date_block_num'] += lag
    all_data = pd.merge(all_data, shifted, on=index_cols, how='left')
    all_data[lag_col_name].fillna(0, inplace=True)


# Add item category id

# In[ ]:


item_category_mapping = items[['item_id','item_category_id']].drop_duplicates()
all_data = pd.merge(all_data, item_category_mapping, how='left', on='item_id')


# Expanding mean encoding sales by item id

# In[ ]:


cumsum = all_data.groupby('item_id')['target'].cumsum() - all_data['target']
cumcnt = all_data.groupby('item_id')['target'].cumcount()
all_data['item_target_enc_exp'] = cumsum / cumcnt

target_mean = all_data['target'].mean()
all_data['item_target_enc_exp'].fillna(target_mean, inplace=True)


# Expanding mean encoding sales by shop id

# In[ ]:


cumsum = all_data.groupby('shop_id')['target'].cumsum() - all_data['target']
cumcnt = all_data.groupby('shop_id')['target'].cumcount()
all_data['shop_target_enc_exp'] = cumsum / cumcnt

target_mean = all_data['target'].mean()
all_data['shop_target_enc_exp'].fillna(target_mean, inplace=True)


# Expanding mean encoding sales by item category id

# In[ ]:


cumsum = all_data.groupby('item_category_id')['target'].cumsum() - all_data['target']
cumcnt = all_data.groupby('item_category_id')['target'].cumcount()
all_data['item_category_target_enc_exp'] = cumsum / cumcnt

target_mean = all_data['target'].mean()
all_data['item_category_target_enc_exp'].fillna(target_mean, inplace=True)


# Add last sale for shop_id, item_id pairs

# In[ ]:


last_sale_df = []
for d in range(1, 35):
    df = sales[sales.date_block_num < d].groupby(['shop_id', 'item_id'], as_index=False)['date_block_num'].max()
    df['last_sale_ago'] = d - df.date_block_num
    df.date_block_num = d
    last_sale_df.append(df)
last_sale_df = pd.concat(last_sale_df)

all_data = pd.merge(all_data, last_sale_df, on=['shop_id', 'item_id', 'date_block_num'], how='left')
all_data['last_sale_ago'].fillna(0, inplace=True)


# Add last shop sale

# In[ ]:


last_shop_sale_df = []
for d in range(1, 35):
    df = sales[sales.date_block_num < d].groupby('shop_id', as_index=False)['date_block_num'].max()
    df['last_shop_sale_ago'] = d - df.date_block_num
    df.date_block_num = d
    last_shop_sale_df.append(df)
last_shop_sale_df = pd.concat(last_shop_sale_df)

all_data = pd.merge(all_data, last_shop_sale_df, on=['shop_id', 'date_block_num'], how='left')
all_data['last_shop_sale_ago'].fillna(0, inplace=True)


# Add last item sale

# In[ ]:


last_item_sale_df = []
for d in range(1, 35):
    df = sales[sales.date_block_num < d].groupby('item_id', as_index=False)['date_block_num'].max()
    df['last_item_sale_ago'] = d - df.date_block_num
    df.date_block_num = d
    last_item_sale_df.append(df)
last_item_sale_df = pd.concat(last_item_sale_df)

all_data = pd.merge(all_data, last_item_sale_df, on=['item_id', 'date_block_num'], how='left')
all_data['last_item_sale_ago'].fillna(0, inplace=True)


# Encode the time as year and month

# In[ ]:


all_data['year_index'] = all_data['date_block_num'] // 12
all_data['month'] = all_data['date_block_num'] % 12 + 1
all_data = all_data.drop(columns='date_block_num')


# Add number of days in a month

# In[ ]:


days = pd.Series([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) # There is no 0 month
all_data['days_in_month'] = all_data['month'].map(days)


# In[ ]:


all_data.head()


# One-hot encode year and month for linear models

# In[ ]:


year_oh = pd.get_dummies(all_data['year_index'], prefix='year')
month_oh = pd.get_dummies(all_data['month'], prefix='month')
all_data_oh = all_data.drop(columns=['shop_id', 'item_id', 'item_category_id', 'year_index', 'month', 'target', 'last_sale_ago', 'last_shop_sale_ago', 'last_item_sale_ago'])
all_data_oh = pd.concat([all_data_oh, year_oh, month_oh], axis=1)
all_data_oh.head()


# # Train/validation split by time

# In[ ]:


train_b_index = (all_data['year_index'] == 2) & (all_data['month'] == 9)
train_c_index = (all_data['year_index'] == 2) & (all_data['month'] == 10)
train_a_index = ~train_b_index & ~train_c_index

X_train_a = all_data[train_a_index]
y_train_a = X_train_a['target'].clip(0, 20)
X_train_a = X_train_a.drop(columns='target')

X_train_b = all_data[train_b_index]
y_train_b = X_train_b['target'].clip(0, 20)
X_train_b = X_train_b.drop(columns='target')

X_train_c = all_data[train_c_index]
y_train_c = X_train_c['target'].clip(0, 20)
X_train_c = X_train_c.drop(columns='target')


# Do the split for linear models dataset

# In[ ]:


X_train_a_oh = all_data_oh[train_a_index]
X_train_b_oh = all_data_oh[train_b_index]
X_train_c_oh = all_data_oh[train_c_index]


# # Train the models on train_a

# Train XGBoost

# In[ ]:


model_xgb = xgb.XGBRegressor(max_depth=4, learning_rate=0.5, n_jobs=-1)
model_xgb.fit(X_train_a, y_train_a)


# Train k-NN

# In[ ]:


# model_knn = KNeighborsRegressor(n_neighbors=3, n_jobs=-1, leaf_size=500)
# model_knn.fit(X_train_a_oh.values, y_train_a)


# Train MLP

# In[ ]:


model_mlp = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', learning_rate_init=0.01, max_iter=10, shuffle=False, verbose=True)
model_mlp.fit(X_train_a_oh.values, y_train_a)


# Train random forest regressor

# In[ ]:


model_rf = RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, n_jobs=-1, verbose=1)
model_rf.fit(X_train_a.values, y_train_a)


# # Stacking

# In[ ]:


y_pred_1 = model_xgb.predict(X_train_b)
rmse = math.sqrt(mean_squared_error(y_train_b, y_pred_1))
R_score = r2_score(y_train_b, y_pred_1)
print('XGBoost rmse: ' + str(rmse) + ', R2: ' + str(R_score))


# In[ ]:


y_pred_2 = model_mlp.predict(X_train_b_oh)
rmse = math.sqrt(mean_squared_error(y_train_b, y_pred_2))
R_score = r2_score(y_train_b, y_pred_2)
print('MLP rmse: ' + str(rmse) + ', R2: ' + str(R_score))


# In[ ]:


y_pred_3 = model_rf.predict(X_train_b)
rmse = math.sqrt(mean_squared_error(y_train_b, y_pred_3))
R_score = r2_score(y_train_b, y_pred_3)
print('Random forest rmse: ' + str(rmse) + ', R2: ' + str(R_score))


# In[ ]:


# y_pred_4 = model_knn.predict(X_train_b_oh)
# rmse = math.sqrt(mean_squared_error(y_train_b, y_pred_4))
# R_score = r2_score(y_train_b, y_pred_4)
# print('k-NN rmse: ' + str(rmse) + ', R2: ' + str(R_score))


# Level 2 model

# In[ ]:


X_train_b_2 = np.stack([y_pred_1, y_pred_2, y_pred_3], axis=-1)
model = LinearRegression(n_jobs=-1)
model.fit(X_train_b_2, y_train_b)


# # Evaluate model on train_c

# In[ ]:


y_pred_1 = model_xgb.predict(X_train_c)
y_pred_2 = model_mlp.predict(X_train_c_oh)
y_pred_3 = model_rf.predict(X_train_c)
# y_pred_4 = model_knn.predict(X_train_c_oh)
X_train_c_2 = np.stack([y_pred_1, y_pred_2, y_pred_3], axis=-1)


# In[ ]:


stack_pred = model.predict(X_train_c_2)
rmse = math.sqrt(mean_squared_error(y_train_c, stack_pred))
R_score = r2_score(y_train_c, y_pred_1)
print('Ensemble rmse: ' + str(rmse) + ', R2: ' + str(R_score))


# # Train level 2 model on train_b + train_c

# In[ ]:


X_train_bc_2 = np.concatenate([X_train_b_2, X_train_c_2], axis=0)
y_train_bc = np.concatenate([y_train_b, y_train_c], axis=0)


# In[ ]:


model.fit(X_train_bc_2, y_train_bc)


# # Make predictions

# Add additional features to the test data

# In[ ]:


# feature extraction: fix the duplicated shop id
test_data.loc[test_data['shop_id'] == 0, 'shop_id'] = 57
test_data.loc[test_data['shop_id'] == 1, 'shop_id'] = 58
test_data.loc[test_data['shop_id'] == 11, 'shop_id'] = 10

# Generate lag features
test_data['date_block_num'] = 34
lags = [1, 2, 3, 6, 12]
all_data['date_block_num'] = all_data['year_index'] * 12 + all_data['month'] - 1

for lag in lags:
    lag_col_name = 'target_lag_' + str(lag)
    shifted = all_data[index_cols + ['target']].copy()
    shifted.columns = index_cols + [lag_col_name]
    shifted['date_block_num'] += lag
    test_data = pd.merge(test_data, shifted, on=index_cols, how='left')
    test_data[lag_col_name].fillna(0, inplace=True)

# Add item category
test_data = pd.merge(test_data, item_category_mapping, how='left', on='item_id')

# Add expanding mean encoding for item_id
item_id_mean = all_data.groupby('item_id')['target'].mean()
test_data['item_target_enc_exp'] = test_data['item_id'].map(item_id_mean)
test_data['item_target_enc_exp'].fillna(target_mean, inplace=True)

# Add expanding mean encoding for shop_id
shop_id_mean = all_data.groupby('shop_id')['target'].mean()
test_data['shop_target_enc_exp'] = test_data['shop_id'].map(shop_id_mean)
test_data['shop_target_enc_exp'].fillna(target_mean, inplace=True)

# Add expanding mean encoding for item_id
item_id_mean = all_data.groupby('item_category_id')['target'].mean()
test_data['item_category_target_enc_exp'] = test_data['item_category_id'].map(item_id_mean)
test_data['item_category_target_enc_exp'].fillna(target_mean, inplace=True)

# Add last sale ago for shop_id, item_id pairs
test_data = pd.merge(test_data, last_sale_df, on=['shop_id', 'item_id', 'date_block_num'], how='left')
test_data['last_sale_ago'].fillna(0, inplace=True)

# Add last shop sale
test_data = pd.merge(test_data, last_shop_sale_df, on=['shop_id', 'date_block_num'], how='left')
test_data['last_shop_sale_ago'].fillna(0, inplace=True)

# Add last item sale
test_data = pd.merge(test_data, last_item_sale_df, on=['item_id', 'date_block_num'], how='left')
test_data['last_item_sale_ago'].fillna(0, inplace=True)

# Add year index and month
test_data['year_index'] = 2
test_data['month'] = 11

# Add days in month
test_data['days_in_month'] = 30

# Drop id column
test_data.drop(columns='ID', inplace=True)

# Drop date_block_num
test_data.drop(columns='date_block_num', inplace=True)

test_data.head()


# Create test dataset for linear models

# In[ ]:


test_data_oh = test_data.drop(columns=['shop_id', 'item_id', 'item_category_id', 'year_index', 'month', 'last_sale_ago', 'last_shop_sale_ago', 'last_item_sale_ago'])
for year in range(1,3):
    test_data_oh['year_' + str(year)] = int(year == 2)
for month in range(1, 13):
    test_data_oh['month_' + str(month)] = int(month == 11)
test_data_oh.head()


# Predict

# In[ ]:


X_test_2 = np.stack([model_xgb.predict(test_data), model_mlp.predict(test_data_oh), model_rf.predict(test_data)], axis=-1)
predictions = model.predict(X_test_2)

df_pred = pd.DataFrame({'item_cnt_month': predictions})
df_pred.to_csv('submission.csv', index_label='ID')


# In[ ]:


sub = pd.read_csv('submission.csv')
sub.head()

