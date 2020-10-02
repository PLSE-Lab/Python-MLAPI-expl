#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Building on my first simple benchmark kernel EDA where I used October 2015 sales to predict November 2015 sales (found here: https://www.kaggle.com/alexyau/previous-value-benchmark-simple-eda ), I will now try building some simple models.

# In[ ]:


import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))

from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

import math
import matplotlib.pyplot as plt


# In[ ]:


train = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
items = pd.read_csv('../input/items.csv')
item_cats = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')


# ## EDA
# We'll start here by splitting the data and arranging it in order to fit the model. 

# In[ ]:


train.head()


# In[ ]:


test.head()


# Now, we have the same problem as before, where the training set and testing set consist of different columns. 
# 
# The test set contains only the shop_id and item_id from the training set, but the training set also includes a daily date, the date_block_num indicating consecutive month since January 2013, as well as item_price.
# 
# For simplicity's sake, and to probe the feature relationships, we will try to build a model with only shop_id and item_id. 
# The dependent variable will be the aggregated item_cnt_day counts (i.e. item_cnt_month) for each value of date_block_num.

# In[ ]:


train_agg = train.drop(['date', 'item_price'], axis=1)
train_agg.describe()


# Column item_cnt_day in the training set has a min of -22 and max of 2169. This might be a realistic number where 22 items were returned or 2169 items were sold in one day, but are definitely outliers that will affect the prediction model. Let's keep note of this - later we will clip the item_cnt_day numbers like before between something like [0,20] and see if the results improve.

# In[ ]:


# Sum up item_cnt_day grouped by shop_id, item_id and date_block_num to get unique rows of items sold per month. 

df = train_agg.groupby(["shop_id", "item_id", "date_block_num"])

monthly = df.aggregate({"item_cnt_day":np.sum}).fillna(0)
monthly.reset_index(level=["shop_id", "item_id", "date_block_num"], inplace=True)
monthly = monthly.rename(columns={ monthly.columns[3]: "item_cnt_month" })


# In[ ]:


monthly.describe()


# In[ ]:


monthly['item_id'].value_counts()/34


# In[ ]:


test['item_id'].value_counts()


# Looking good so far. Quick sanity check - how many shops does item_id=5822 appear in?

# In[ ]:


monthly['shop_id'].loc[monthly['item_id'] == 5822].value_counts().sort_index()


# In[ ]:


test['shop_id'].loc[test['item_id'] == 5822].value_counts().sort_index()


# There seems to be more shops in the training set than in the test set for some items. 
# 
# This makes sense if you figure out that some shops may have closed over time. 
# 

# In[ ]:


monthly['shop_id'].value_counts()


# In[ ]:


test['shop_id'].value_counts()


# It appears that the test set has structured its data so that it has a total of 42 shops, with each shop containing 5100 of the same items. 
# 
# From the historical training data, we are not sure how many items each shop contains, but records show that some shops have sold far less than 5100 unique items - perhaps some items in stock never get sold. Also there may be a few shops that sell much more than the unique item count in the test set - this can be explained by some products being discontinued from the store over time with new ones added recently. 
# 
# 

# ## Train/validation split
# 
# Now that we have some more real-world insight on the differences between the train and test set and cleaned the train set a bit, we can make the train/validation split. We will want to structure the validation set roughly the same as the test set for our model. 
# 
# Normally for time-series data we will want to sort the data by date and make some latter portion of the data the validation set. Since we will try to work without the dates first, we can arbitrarily portion 214200 of the 1.61 million rows for validation, the same 'size' as the test set. Working without dates, we can drop the date_block_num column and shuffle the rows in the train set as we like.

# In[ ]:


monthly.describe()


# In[ ]:


test.describe()


# In[ ]:


train_simple = monthly.drop('date_block_num', axis=1)
#shuffle rows
train_simple = train_simple.sample(frac=1).reset_index(drop=True)

X_simple = train_simple[['shop_id', 'item_id']]
y_simple = train_simple['item_cnt_month']


# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 214200
n_trn = len(train_simple) - n_valid
X_train, X_valid = split_vals(X_simple, n_trn)
y_train, y_valid = split_vals(y_simple, n_trn)


# In[ ]:


plt.scatter(X_valid.iloc[:100,1], y_valid[:100], color='black')


# ## Basic model - decision tree regression
# 
# Judging by the discrete nature of the data in the dependent feature that we are predicting (the number of sales per item/shop) a linear or polynomial regression model would not work since there is no direct collinearity between the item_id/shop_id and the number of sales. Therefore we will build decision tree regression models that can predict the sales based on learning past behavior of sales for certain items in certain shops. 
# 
# We'll start with a single tree and see how it performs.

# In[ ]:


m = RandomForestRegressor(n_estimators=1, n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')


# In[ ]:


def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)

print_score(m)


# In[ ]:


plt.scatter(X_valid.iloc[:100,1], m.predict(X_valid)[:100], color='black')


# The results seem poor - RMSE of 6.8 and R^2 of 0.38 on the validation set. But this is a good baseline to improve from. A random forest with many trees should definitely perform better.

# In[ ]:


m_2 = RandomForestRegressor(n_estimators=100, n_jobs=-1)
get_ipython().run_line_magic('time', 'm_2.fit(X_train, y_train)')
get_ipython().run_line_magic('time', 'print_score(m_2)')


# In[ ]:


preds = np.stack([t.predict(X_valid) for t in m_2.estimators_])


# In[ ]:


plt.plot([metrics.mean_squared_error(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(100)]);


# Just a slight improvement here using a random forest regressor with 100 trees - RMSE of 6.7 and R^2 of 0.4. 
# 
# ### Improving the model
# 
# Let's try to clip the item_cnt_month values into [0,20] for both the training and validation set to see how much the models improve.

# In[ ]:


X_simple = train_simple[['shop_id', 'item_id']]
y_simple = train_simple['item_cnt_month'].clip(0,20)


# In[ ]:


n_valid = 214200
n_trn = len(train_simple) - n_valid
X_train, X_valid = split_vals(X_simple, n_trn)
y_train, y_valid = split_vals(y_simple, n_trn)


# In[ ]:


m = RandomForestRegressor(n_estimators=1, n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


m_2 = RandomForestRegressor(n_estimators=100, n_jobs=-1)
get_ipython().run_line_magic('time', 'm_2.fit(X_train, y_train)')
print_score(m_2)


# In[ ]:


preds = np.stack([t.predict(X_valid) for t in m_2.estimators_])
plt.plot([metrics.mean_squared_error(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(100)]);


# In[ ]:


pd.DataFrame(m_2.predict(X_valid)).describe()


# The model is significantly better when the target values are clipped within [0,20], with RMSE of 2.17 from 100-tree random forest. Compared to the benchmark kernel using October 2015 as prediction for November 2015, our model does not score as well. This is definitely because we have built a model with data that is not dependent on time.
# 
# Let us now see what happens when time dependency is put back into the data. We can start by sorting the dataset by the date_block_num column, and splitting the October 2015 dataset for validation.

# In[ ]:


train_td = monthly.sort_values(by=["date_block_num"])
valid_td = monthly[monthly["date_block_num"] == 33]

X_train = train_td[['shop_id', 'item_id']]
y_train = train_td['item_cnt_month'].clip(0,20)
X_valid = valid_td[['shop_id', 'item_id']]
y_valid = valid_td['item_cnt_month'].clip(0,20)


# In[ ]:


m_3 = RandomForestRegressor(n_estimators=60, n_jobs=-1)
get_ipython().run_line_magic('time', 'm_3.fit(X_train, y_train)')
print_score(m_3)


# In[ ]:


preds = np.stack([t.predict(X_valid) for t in m_3.estimators_])
plt.plot([metrics.mean_squared_error(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(60)]);


# Quite a bit better. RMSE of 1.71 on the validation set with random forest of 60 trees. 
# 
# 
# ###TODO: put into submission format and check score
# 

# ## Improved model? - XGBoost
# 
# Let's now try to build an XGBoost model.

# In[ ]:


import xgboost as xgb
param = {'max_depth':12,  # originally 10
         'subsample':1,  # 1
         'min_child_weight':0.5,  # 0.5
         'eta':0.3,
         'num_round':1000, 
         'seed':0,  # 1
         'silent':0,
         'eval_metric':'rmse',
         'early_stopping_rounds':100
        }

progress = dict()
xgbtrain = xgb.DMatrix(X_train, y_train)
watchlist  = [(xgbtrain,'train-rmse')]
m_4 = xgb.train(param, xgbtrain)


# In[ ]:


preds = m_4.predict(xgb.DMatrix(X_valid))

rmse = np.sqrt(mean_squared_error(preds, y_valid))
print(rmse)


# # Things to try
# * using daily values to predict, then aggregate the predictions and use the extra data for modeling
# * add a column to indicate number of days since last sale for each item
# * mapping categories of items (using the other csv files)
# * onehotencoding date_block_num for items sold to reflect the months where nothing is sold
# * adding column indicating the year
# * leave in item_price, map it to test set

# In[ ]:


new_submission = pd.merge(month_sum, test, how='right', left_on=['shop_id','item_id'], right_on = ['shop_id','item_id']).fillna(0)
new_submission.drop(['shop_id', 'item_id'], axis=1)
new_submission = new_submission[['ID','item_cnt_month']]


# In[ ]:


new_submission['item_cnt_month'] = new_submission['item_cnt_month'].clip(0,20)
new_submission.describe()


# In[ ]:


new_submission.to_csv('previous_value_benchmark.csv', index=False)

