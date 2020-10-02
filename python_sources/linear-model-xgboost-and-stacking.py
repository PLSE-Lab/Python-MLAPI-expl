#!/usr/bin/env python
# coding: utf-8

# In this kernel I will share how I managed to build stacking model wich consists on linear regression with regularization and xgboost. I will use alredy cleaned and prepared train data 'all_data_final.pkl' from my another kernel. https://www.kaggle.com/emaksone/eda-with-feature-engineering this is how obtained it.
# 
# Pipeline:
# 
# * Define validation strategy and leaderboard probing
# * Hyperparameter tuning
# * Stacking
# 

# In[1]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set()
from sklearn.externals import joblib

from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor

import gc
from itertools import product
import time


# In[2]:


all_data = pd.read_pickle('../input/extended-train-data/all_data_final.pkl')
all_data.head()


# **<font size=4>Validation strategy</font>**

# Our object is to predict sales for next month (i.e. November 2015, or 34 in terms of date_block_num). And we can't just randomly split our data on tran/validation sets for parameters tuning because we need validate parameters on data in month that we didn't use in train set like in test. So,I use rows with date_block_num less than 33 as train set and rows with date_block_num equal 33 I use as validation set.
# 
# To successfully predict sales on test data you also need to have same train and test data. In other words you train and test data should have the same distribution. But stop, distribution in train data we know but how do we know distribution in test data? Such information we can get by **leaderboard probing**. For example we can easily find out mean of sales in public leader board by simlpy submiting two files with all 0's prediction and 0.5's predictions and do some math.
# 
# 
# And the mean of sales equals 0.28394. So mean of train/validation should be close to this number in order to get good results. Let's check mean of the target in our whole train set

# In[34]:


print('mean for whole train set: {0}'.format(np.mean(all_data.loc[all_data['date_block_num']<34, 'target'].astype(np.float32))))
print('mean for validation train set: {0}'.format(np.mean(all_data.loc[all_data['date_block_num']<33, 'target'].astype(np.float32))))
print('mean for validation test set: {0}'.format(np.mean(all_data.loc[all_data['date_block_num']==33, 'target'].astype(np.float32))))


# As you can see every numbers are close to mean of test sales. 

# One thing to notice before start building a model. Let's check if we have new items in test set

# In[4]:


test_items = all_data.loc[all_data['date_block_num']==34,'item_id'].unique()
train_items = all_data.loc[all_data['date_block_num']<34,'item_id'].unique()
items_in_test_and_not_in_train = set(test_items).difference(set(train_items))
print('Items in test and not in train: {0}'.format(len(items_in_test_and_not_in_train)))
items_in_train_and_not_in_test = set(train_items).difference(set(test_items))
print('Items in train and not in test: {0}'.format(len(items_in_train_and_not_in_test)))

test_shops = all_data.loc[all_data['date_block_num']==34,'shop_id'].unique()
print('Number of unique shops: {0}'.format(len(test_shops)))


# So, we have items in the test set that do not appear in train. So our model will struggle when see unknown items when predict. To eliminate this issue I will add all missing shop/item pair to every month with 0's.
# But after this we decrese our target mean that will make our train/test data distributions different. To avoid this I remove rows with target equals 0 and item that does not appear in test. At the end it doesn't change mean of targets and train data set will contain all items that exist in test.

# In[5]:


missing_shop_item_count = 15876 # 372*42 all missing items per month
index_cols = ['shop_id', 'item_id', 'date_block_num']

grid = [] 
for block_num in all_data.loc[all_data['date_block_num']<34, 'date_block_num'].unique():
    print(block_num)
  
    zero_target_df = all_data[(all_data['date_block_num'] == block_num) & (all_data['target']==0) & 
                              (all_data['item_id'].isin(items_in_train_and_not_in_test))]

    idx_to_delete = zero_target_df.sample(missing_shop_item_count, random_state=block_num).index
    all_data.drop(idx_to_delete, inplace=True)
    temp = np.array(list(product(*[test_shops, items_in_test_and_not_in_train, [block_num]])),dtype='int32')
    grid.append(temp)
    
    del zero_target_df
    del idx_to_delete
    del temp
    gc.collect()

grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)


# In[6]:


grid['shop_id'] = grid['shop_id'].astype(np.int16)
grid['item_id'] = grid['item_id'].astype(np.int32)
grid['date_block_num'] = grid['date_block_num'].astype(np.int8)


# In[7]:


all_data = pd.concat([all_data, grid], ignore_index=True, sort=False, keys=index_cols)
all_data[['item_shop_last_sale', 'item_last_sale']].fillna(-1, inplace=True) #-1 is default value in this columns
all_data.fillna(0, inplace=True)

del grid
del test_items
del test_shops
del train_items
del items_in_test_and_not_in_train
del items_in_train_and_not_in_test
gc.collect()


# In[8]:


all_data['is_december'] = all_data['is_december'].astype(np.int8)
all_data['item_category_id'] = all_data['item_category_id'].astype(np.int8)
all_data['type_code'] = all_data['type_code'].astype(np.int8)
all_data['subtype_code'] = all_data['subtype_code'].astype(np.int8)
all_data['city_code'] = all_data['city_code'].astype(np.int16)

all_data['month'] = all_data['month'].astype(np.int8)
all_data['days'] = all_data['days'].astype(np.int8)
all_data['item_shop_last_sale'] = all_data['item_shop_last_sale'].astype(np.int8)
all_data['item_last_sale'] = all_data['item_last_sale'].astype(np.int8)
all_data['item_shop_first_sale'] = all_data['item_shop_first_sale'].astype(np.int8)
all_data['item_first_sale'] = all_data['item_first_sale'].astype(np.int8)


# Check our means

# In[9]:


print('mean for whole train set: {0}'.format(np.mean(all_data.loc[all_data['date_block_num']<34, 'target'].astype(np.float32))))
print('mean for validation train set: {0}'.format(np.mean(all_data.loc[all_data['date_block_num']<33, 'target'].astype(np.float32))))
print('mean for validation test set: {0}'.format(np.mean(all_data.loc[all_data['date_block_num']==33, 'target'].astype(np.float32))))


# Good, As it was before.

# In[10]:


# put added rows in right position
all_test_data = all_data[all_data['date_block_num'] == 34]
all_data = all_data[all_data['date_block_num'] < 34]
all_data.sort_values(['date_block_num'], inplace=True)
all_data = pd.concat([all_data, all_test_data], ignore_index=True, sort=False, keys=index_cols)

del all_test_data
gc.collect()


# Now we ready to define train/test split. And after we tune parameters we will use whole data to train a model on this parameters and can use it to predict sales on test data.

# In[11]:


dates = all_data['date_block_num']

last_block = dates.max()
print('Test `date_block_num` is {0}'.format(last_block))

X_train = all_data.loc[dates <  last_block]
X_test =  all_data.loc[dates == last_block]

y_train = all_data.loc[dates <  last_block, 'target'].values
y_test =  all_data.loc[dates == last_block, 'target'].values

X_valid_train = all_data.loc[dates <  last_block-1]
X_valid_test =  all_data.loc[dates == last_block-1]

y_valid_train = all_data.loc[dates <  last_block-1, 'target'].values
y_valid_test =  all_data.loc[dates == last_block-1, 'target'].values

all_data.to_pickle('all_data.pkl') # will use it later. Now free RAM

del dates
del all_data
gc.collect()


# Delete columns that we can't use in test

# In[12]:


columns_to_delete = ['date_block_num', 'target']
X_valid_train = X_valid_train.drop(columns_to_delete, axis=1)
X_valid_test = X_valid_test.drop(columns_to_delete, axis=1)

X_train = X_train.drop(columns_to_delete, axis=1)
X_test = X_test.drop(columns_to_delete, axis=1)


# **<font size=4>Hyperparameters tuning<font>**

# I create my own custom grid search method because I use XGBoost with GPU and I didn't find a way to make them work together. If you know how to do it please drop a comment.

# In[35]:


#validation
def validate(estimator, X_train_, y_train_, X_val_, y_val_, grid_params):
    keys = grid_params.keys()
    vals = grid_params.values()
    parameters = []
    rmses = []
    rmses_train = []
    return_obj ={}
    prods = product(*vals)
   
    for idx, instance in enumerate(prods):
        print('-'*50)
        print('model {0}:'.format(idx))
        model_params = dict(zip(keys, instance))
        parameters.append(model_params)
        
        print(model_params)
        model = estimator(**model_params)
        model.fit(X_train_, y_train_)
            
        pred_test = model.predict(X_val_)       
        mse = mean_squared_error(y_val_, pred_test)
        rmse = np.sqrt(mse)
        print('RMSE: {0}'.format(rmse))
        rmses = rmses + [rmse]
        
        best_rmse_so_far = np.min(rmses)
        print('Best rmse so far: {0}'.format(best_rmse_so_far))
        best_model_params_so_far = parameters[np.argmin(rmses)]
        print('Best model params so far: {0}'.format(best_model_params_so_far))
        
        del best_rmse_so_far
        del best_model_params_so_far
        del pred_test
        del model
        gc.collect()
    
    rmses = np.array(rmses)
    best_rmse = np.min(rmses)
    print('Best rmse: {0}'.format(best_rmse))
    best_model_params = parameters[np.argmin(rmses)]
    print('Best model params: {0}'.format(best_model_params))

    return_obj['rmses'] = rmses
    return_obj['best_rmse'] = best_rmse
    return_obj['best_model_params'] = best_model_params
      
    return return_obj


# Let's find out optimal paramter for Ridge regression

# In[14]:


alphas = [10, 100, 2000, 5000]
grid_params = {'alpha':alphas}
val_res = validate(Ridge, X_valid_train, y_valid_train, 
                   X_valid_test, y_valid_test, grid_params)


# So 2000 is an optimal parameter for our model. We can use it to train model with all train data.

# In[15]:


# best_alpha=2000
# ridge_model = Ridge(best_alpha)
# ridge_model.fit(X_train, y_train)
# predictions = ridge_model.predict(X_test)


# In the same way we can do for XGBoost. But this model has much more parameters to config. https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/ this article helps to define strategy to find them starting with some parameters that make highest impact on the model and ending with ones that give lowest.
# So I came up with this parameters. Note that tree_method, predictor, gpu_id was fixed parameters from the begining to enable gpu. It works extremely slow without it.
# 
# Note: Don't forget to enable gpu in kernel settings.

# In[16]:


best_params = {'learning_rate': 0.16, 'n_estimators': 500, 
               'max_depth': 6, 'min_child_weight': 7,
               'subsample': 0.9, 'colsample_bytree': 0.7, 'nthread': -1, 
               'scale_pos_weight': 1, 'random_state': 42, 
               
               #next parameters are used to enable gpu for fasting fitting
               'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor', 'gpu_id': 0}


# In[17]:


ts = time.time()
model = XGBRegressor(**best_params)
model.fit(X_valid_train.values, 
                y_valid_train,
                eval_metric="rmse", 
                eval_set=[(X_valid_test.values, y_valid_test)], 
                verbose=True, 
                early_stopping_rounds = 50)


time.time() - ts


# Look how better xgboost then simple regression model.

# In[18]:


joblib.dump(model, 'xgboost_model.pkl')
del model
del X_train
del X_test
del y_train
del y_test
del X_valid_train
del X_valid_test
del y_valid_train
del y_valid_test

gc.collect()


# **<font size=4>Stacking<font>**

# So now after we find optimal parameters for two models we can use them for stacking. I will do it in the following way: 
# 
# * take data that date_block_num < 27 as train
# * train a models(Ridge and XGBoost) and predict for date_block_num == 27
# * put this prediction in two columns (xgb_prediction and ridge_prediction)
# * do it for 28, 29, 30, 31, 32, 33, 34 monthes
# * concat all this prediction.
# * fit data from 28 to 33 to Linear regression
# * use this simple model to predict data with month 34

# In[19]:


all_data = pd.read_pickle('all_data.pkl')


# In[20]:


def compute_prediction_for_specifi_month(monthes_before, df, estimator, params, prefix):
    predictions = pd.DataFrame(columns=['date_block_num', 'pred_'+prefix])
    for before in monthes_before:
        last_valid_month = np.max(df['date_block_num'])
        print('train: 12 to {0}'.format(last_valid_month-before-1))
        print('test: {0}'.format(last_valid_month-before))
        
        cur_train = df[df['date_block_num'] < last_valid_month-before]
        cur_test = df[df['date_block_num'] == last_valid_month-before]
        
        cur_y_train = cur_train['target']
        cur_train.drop('target', axis=1, inplace=True)
        cur_test.drop('target', axis=1, inplace=True)
        
        model = estimator(**params)
        model.fit(cur_train.values, cur_y_train.values)
        pred = model.predict(cur_test.values)
        cur_df = pd.DataFrame(columns=['date_block_num', 'pred_'+prefix])
        
        cur_df['pred_'+prefix] = pred
        cur_df['date_block_num'] = (last_valid_month-before)
       
        predictions = pd.concat([predictions, cur_df])
        del model
        del pred
        del cur_test
        del cur_df
        del cur_y_train
        gc.collect()
        
    return predictions


# In[21]:


best_param = {
    'learning_rate' :0.16,
    'n_estimators':500,
    'max_depth':6,
    'min_child_weight':7,
    'subsample':0.9,
    'colsample_bytree':0.7,
    'nthread':-1,
    'scale_pos_weight':1,
     #next parameters are used to enable gpu for fasting fitting
    'random_state':42,
    'nthread': -1,
    'tree_method':'gpu_hist',
    'predictor':'gpu_predictor',
    'gpu_id': 0,
}

ts= time.time()
monthes_before = [6, 5, 4, 3, 2, 1, 0]

stacks_xgb = compute_prediction_for_specifi_month(monthes_before, all_data, XGBRegressor, best_params,'xgb')
time.time()-ts


# Clip predictions

# In[22]:


stacks_xgb['pred_xgb'] = stacks_xgb['pred_xgb'].clip(0, 20)


# And do it for second model

# In[23]:


ts = time.time()
monthes_before = [6, 5, 4, 3, 2, 1, 0]
best_param ={'alpha': 2000}
stacks_lin_reg = compute_prediction_for_specifi_month(monthes_before, all_data, Ridge, best_param,'lin')
time.time() - ts


# Clip predictions again

# In[26]:


stacks_lin_reg['pred_lin'] = stacks_lin_reg['pred_lin'].clip(0, 20)


# And construct train/validation/test sets

# In[27]:


X_valid_train = pd.concat([stacks_xgb[stacks_xgb['date_block_num'] < 33], 
                           stacks_lin_reg[stacks_lin_reg['date_block_num'] < 33]], axis=1)
X_valid_test = pd.concat([stacks_xgb[stacks_xgb['date_block_num'] == 33], 
                           stacks_lin_reg[stacks_lin_reg['date_block_num'] == 33]], axis=1)
y_valid_train = all_data.loc[(all_data['date_block_num']<33) & (all_data['date_block_num']>27), 'target']
y_valid_test = all_data.loc[all_data['date_block_num']==33, 'target']


# In[28]:


X_train = pd.concat([stacks_xgb[stacks_xgb['date_block_num'] < 34], 
                           stacks_lin_reg[stacks_lin_reg['date_block_num'] < 34]], axis=1)
X_test = pd.concat([stacks_xgb[stacks_xgb['date_block_num'] == 34], 
                           stacks_lin_reg[stacks_lin_reg['date_block_num'] == 34]], axis=1)
y_train = all_data.loc[(all_data['date_block_num']<34) & (all_data['date_block_num']>27), 'target']


# In[29]:


X_valid_train.drop('date_block_num', axis=1, inplace=True)
X_valid_test.drop('date_block_num', axis=1, inplace=True)
X_train.drop('date_block_num', axis=1, inplace=True)
X_test.drop('date_block_num', axis=1, inplace=True)


# And use simple linear regression.

# In[30]:


model = LinearRegression()
model.fit(X_valid_train, y_valid_train)
pred = model.predict(X_valid_test).clip(0,20)

rmse = np.sqrt(mean_squared_error(y_valid_test, pred))
print('RMSE on valid set: {0}'.format(rmse))


# As you see the rmse is lower then single xgboost model's rmse.
# 
# Let's train a stacking model on a whole train data.

# In[31]:


model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test).clip(0,20)


# In[32]:


submit = pd.DataFrame({'ID':range(len(pred)), 'item_cnt_month': pred})
submit.to_csv('submit.csv', index=False)


# **<font size=4>Summary</font>**

# In this kernel we learned how to make proper cross validation strategy, tune parameters for single models and finnaly use one of the ensambling methods called stacking to make better predictions.
