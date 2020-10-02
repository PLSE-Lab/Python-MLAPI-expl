#!/usr/bin/env python
# coding: utf-8

# This notebook takes inspiration from:
# 
# https://www.kaggle.com/alexeyb/coursera-winning-kaggle-competitions, https://www.kaggle.com/anqitu/feature-engineer-and-model-ensemble-top-10
# 
# By adding some further techniques and improvements the LB score was improvement.
# 
# # Part 1
# 
# Understand our data better in "Exploratory Data Analysis" section, do necessary data wrangling
# 

# # Exploratory Data Analysis

# ## Import necessary libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import time

from math import sqrt
from numpy import loadtxt
from itertools import product
from tqdm import tqdm
from sklearn import preprocessing
from xgboost import plot_tree
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer

kernel_with_output = True


# ## Data loading
# Load all provided datasets provided

# In[ ]:


sales_train = pd.read_csv('../input/sales_train.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
items = pd.read_csv('../input/items.csv')
shops = pd.read_csv('../input/shops.csv')
item_categories = pd.read_csv('../input/item_categories.csv')
test = pd.read_csv('../input/test.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
sample_submission = pd.read_csv('../input/sample_submission.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')

# Cleaned up a little of sales data after some basic EDA:
#sales_train = sales_train[sales_train.item_price<110000]
#sales_train = sales_train[sales_train.item_cnt_day<=1100]


# ## Insert missing rows and aggregations

# In[ ]:


# For every month we create a grid from all shops/items combinations from that month
grid = []
for block_num in sales_train['date_block_num'].unique():
    cur_shops = sales_train[sales_train['date_block_num']==block_num]['shop_id'].unique()
    cur_items = sales_train[sales_train['date_block_num']==block_num]['item_id'].unique()
    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
index_cols = ['shop_id', 'item_id', 'date_block_num']
grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)

# Aggregations
sales_train['item_cnt_day'] = sales_train['item_cnt_day'].clip(0,20)
groups = sales_train.groupby(['shop_id', 'item_id', 'date_block_num'])
trainset = groups.agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()
trainset = trainset.rename(columns = {'item_cnt_day' : 'item_cnt_month'})
trainset['item_cnt_month'] = trainset['item_cnt_month'].clip(0,20)

trainset = pd.merge(grid,trainset,how='left',on=index_cols)
trainset.item_cnt_month = trainset.item_cnt_month.fillna(0)

# Get category id
trainset = pd.merge(trainset, items[['item_id', 'item_category_id']], on = 'item_id')
trainset.to_csv('trainset_with_grid.csv')

trainset.head()


# # Part2
# ## Set up global vars 

# In[ ]:


# Set seeds and options
np.random.seed(10)
pd.set_option('display.max_rows', 231)
pd.set_option('display.max_columns', 100)

# Feature engineering list
new_features = []

# Periods range 
lookback_range = [1,2,11,12] #[1,2,3,4,5,6,7,8,9,10,11,12] #[1 ,2 ,3 ,4, 5, 12]

tqdm.pandas()


# ## Load data

# In[ ]:


current = time.time()

trainset = pd.read_csv('trainset_with_grid.csv')
items = pd.read_csv('../input/items.csv')
shops = pd.read_csv('../input/shops.csv')


# Only use more recent data
start_month = 0
end_month = 33
trainset = trainset[['shop_id', 'item_id', 'item_category_id', 'date_block_num', 'item_price', 'item_cnt_month']]
trainset = trainset[(trainset.date_block_num >= start_month) & (trainset.date_block_num <= end_month)]

print('Loading test set...')
test_dataset = loadtxt('../input/test.csv', delimiter="," ,skiprows=1, usecols = (1,2), dtype=int)
testset = pd.DataFrame(test_dataset, columns = ['shop_id', 'item_id'])

print('Merging with other datasets...')
# Get item category id into test_df
testset = testset.merge(items[['item_id', 'item_category_id']], on = 'item_id', how = 'left')
testset['date_block_num'] = 34
# Make testset contains same column as trainset so we can concatenate them row-wise
testset['item_cnt_month'] = -1

train_test_set = pd.concat([trainset, testset], axis = 0) 

end = time.time()
diff = end - current
print('Took ' + str(int(diff)) + ' seconds to train and predict val set')


# ## Map Items Categorries
# Map Categories to more narrow onesory

# In[ ]:


item_cat = pd.read_csv('../input/item_categories.csv')

# Fix category
l_cat = list(item_cat.item_category_name)
for ind in range(0,1):
    l_cat[ind] = 'PC Headsets / Headphones'
for ind in range(1,8):
    l_cat[ind] = 'Access'
l_cat[8] = 'Tickets (figure)'
l_cat[9] = 'Delivery of goods'
for ind in range(10,18):
    l_cat[ind] = 'Consoles'
for ind in range(18,25):
    l_cat[ind] = 'Consoles Games'
l_cat[25] = 'Accessories for games'
for ind in range(26,28):
    l_cat[ind] = 'phone games'
for ind in range(28,32):
    l_cat[ind] = 'CD games'
for ind in range(32,37):
    l_cat[ind] = 'Card'
for ind in range(37,43):
    l_cat[ind] = 'Movie'
for ind in range(43,55):
    l_cat[ind] = 'Books'
for ind in range(55,61):
    l_cat[ind] = 'Music'
for ind in range(61,73):
    l_cat[ind] = 'Gifts'
for ind in range(73,79):
    l_cat[ind] = 'Soft'
for ind in range(79,81):
    l_cat[ind] = 'Office'
for ind in range(81,83):
    l_cat[ind] = 'Clean'
l_cat[83] = 'Elements of a food'

lb = preprocessing.LabelEncoder()
item_cat['item_category_id_fix'] = lb.fit_transform(l_cat)
item_cat['item_category_name_fix'] = l_cat
train_test_set = train_test_set.merge(item_cat[['item_category_id', 'item_category_id_fix']], on = 'item_category_id', how = 'left')
_ = train_test_set.drop(['item_category_id'],axis=1, inplace=True)
train_test_set.rename(columns = {'item_category_id_fix':'item_category_id'}, inplace = True)

_ = item_cat.drop(['item_category_id'],axis=1, inplace=True)
_ = item_cat.drop(['item_category_name'],axis=1, inplace=True)

item_cat.rename(columns = {'item_category_id_fix':'item_category_id'}, inplace = True)
item_cat.rename(columns = {'item_category_name_fix':'item_category_name'}, inplace = True)
item_cat = item_cat.drop_duplicates()
item_cat.index = np.arange(0, len(item_cat))
item_cat.head()


# # Add previous shop/item sales as feature (Lag feature)

# In[ ]:


for diff in tqdm(lookback_range):
    feature_name = 'prev_shopitem_sales_' + str(diff)
    trainset2 = train_test_set.copy()
    trainset2.loc[:, 'date_block_num'] += diff
    trainset2.rename(columns={'item_cnt_month': feature_name}, inplace=True)
    train_test_set = train_test_set.merge(trainset2[['shop_id', 'item_id', 'date_block_num', feature_name]], on = ['shop_id', 'item_id', 'date_block_num'], how = 'left')
    train_test_set[feature_name] = train_test_set[feature_name].fillna(0)
    new_features.append(feature_name)
train_test_set.head(3)


# # Add previous shop/item price as feature (Lag feature)

# In[ ]:


groups = train_test_set.groupby(by = ['shop_id', 'item_id', 'date_block_num'])
for diff in tqdm(lookback_range):
    feature_name = 'prev_shopitem_price_' + str(diff)
    result = groups.agg({'item_price':'mean'})
    result = result.reset_index()
    result.loc[:, 'date_block_num'] += diff
    result.rename(columns={'item_price': feature_name}, inplace=True)
    train_test_set = train_test_set.merge(result, on = ['shop_id', 'item_id', 'date_block_num'], how = 'left')
    train_test_set[feature_name] = train_test_set[feature_name]
    new_features.append(feature_name)        
train_test_set.head(3)


# # Add previous item price as feature (Lag feature)

# In[ ]:


groups = train_test_set.groupby(by = ['item_id', 'date_block_num'])
for diff in tqdm(lookback_range):
    feature_name = 'prev_item_price_' + str(diff)
    result = groups.agg({'item_price':'mean'})
    result = result.reset_index()
    result.loc[:, 'date_block_num'] += diff
    result.rename(columns={'item_price': feature_name}, inplace=True)
    train_test_set = train_test_set.merge(result, on = ['item_id', 'date_block_num'], how = 'left')
    train_test_set[feature_name] = train_test_set[feature_name]
    new_features.append(feature_name)        
train_test_set.head(3)


# # Modelling & Cross Validation (XGBoost using residual based boosting technique to minimize the MSE)
# It was used residual based boosting technique to minimize the root square error. By using XGBoost, it was found  the distance between the predictions and the target values and they used as new target (Y) to a new Linear Regression model. Then the predictions of this model are summed up with the first predictions to create a new target value. This new target was used again to XGBoost model to find the final (improvement) predictions. The max_depth and learning rate (eta) were also optimized in order the model to be sufficient. This technique helps to minimize the LB score.

# In[ ]:


current = time.time()

baseline_features = ['shop_id', 'item_id', 'item_category_id', 'date_block_num'] +  new_features + ['item_cnt_month']
#train_test_set.fillna(0)
# Clipping to range 0-20
train_test_set['item_cnt_month'] = train_test_set.item_cnt_month.fillna(0).clip(0,20)

# train: want rows with date_block_num from 0 to 31
train_time_range_lo = (train_test_set['date_block_num'] >= 0)
train_time_range_hi =  (train_test_set['date_block_num'] <= 32)

# val: want rows with date_block_num from 22
validation_time =  (train_test_set['date_block_num'] == 33)

# test: want rows with date_block_num from 34
test_time =  (train_test_set['date_block_num'] == 34)


# Retrieve rows for train set, val set, test set
cv_trainset = train_test_set[train_time_range_lo & train_time_range_hi]
cv_valset = train_test_set[validation_time]
cv_trainset = cv_trainset[baseline_features]
cv_valset = cv_valset[baseline_features]
testset = train_test_set[test_time]
testset = testset[baseline_features]

# Prepare numpy arrays for training/val/test
cv_trainset_vals = cv_trainset.values.astype(int)
trainx = cv_trainset_vals[:, 0:len(baseline_features) - 1]
trainy = cv_trainset_vals[:, len(baseline_features) - 1]

cv_valset_vals = cv_valset.values.astype(int)
valx = cv_valset_vals[:, 0:len(baseline_features) - 1]
valy = cv_valset_vals[:, len(baseline_features) - 1]

testset_vals = testset.values.astype(int)
testx = testset_vals[:, 0:len(baseline_features) - 1]

print('Fitting...')
model = xgb.XGBRegressor(max_depth = 8, min_child_weight = 0.5, subsample = 1, eta = 0.2, num_round = 1000, seed = 1, nthread = 16)
model.fit(trainx, trainy, eval_metric='rmse')
preds = model.predict(valx)

npreds = valy-preds #Distance between predictions and targets
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(valx, npreds)
secondpreds = regressor.predict(valx)

newy = secondpreds+preds
#set now max_depth param from 8 to 10 for improvement
metamodel = xgb.XGBRegressor(max_depth = 10, min_child_weight = 0.5, subsample = 1, eta = 0.2, num_round = 1000, seed = 1, nthread = 16)
ft = metamodel.fit(valx, newy, eval_metric='rmse')
metapredicts = metamodel.predict(testx)


# Clipping to range 0-20
metapredicts = np.clip(metapredicts, 0,20)
#print('val set rmse: ', sqrt(mean_squared_error(valy, p)))

df = pd.DataFrame(metapredicts, columns = ['item_cnt_month'])
df['ID'] = df.index
df = df.set_index('ID')
df.to_csv('test_preds.csv')
print('test predictions written to file')

end = time.time()
diff = end - current
print('Took ' + str(int(diff)) + ' seconds to train and predict val, test set')
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(metamodel, max_num_features=50, height=0.8, ax=ax)
plt.show()


# # Conclusion
# I got a rmse score of 0.992784 on public LB and 0.996122 on private.
# 
# 
# 
