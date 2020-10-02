#!/usr/bin/env python
# coding: utf-8

# # Santander - Separate Aggregates for the dense and sparse feature clusters, applied to simple XGBoost model
# 
# 
# Note: most of the code in this kernel is taken from a number of other public kernels, including the pre-processing, aggregate feature types and XGBoost code & settings.  Thank you everyone for being so generous with sharing Kernels!!
# 
# My own work is primarily the creation of aggregate features for the most dense columns.  I did this because they standout so much from the other columns that they must come from a different distribution - they hold a different type of transctional or limit information.
# In other public kernels it can be seen that there are a number of other variable clusters, which suggests 3-4 groups of variables could be created, with aggregate features for each.
# 

# In[ ]:


import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib

import xgboost as xgb
import lightgbm as lgb


# In[ ]:


train_file = '../input/train.csv'
test_file = '../input/test.csv'


# In[ ]:


train = pd.read_csv(train_file)
test = pd.read_csv(test_file)


# In[ ]:


test_ID = test['ID']
y_train = train['target']
y_train = np.log1p(y_train)


# In[ ]:


train.drop("ID", axis = 1, inplace = True)
train.drop("target", axis = 1, inplace = True)
test.drop("ID", axis = 1, inplace = True)


# In[ ]:


NUM_OF_DECIMALS = 32
train = train.round(NUM_OF_DECIMALS)
test = test.round(NUM_OF_DECIMALS)


# ### Split out the dense/sparse clusters here

# In[ ]:


train_zeros = pd.DataFrame({'Percent_zero':((train.values)==0).mean(axis=0),
                           'Column' : train.columns})

high_vol_columns = train_zeros['Column'][train_zeros['Percent_zero'] < 0.70].values
low_vol_columns = train_zeros['Column'][train_zeros['Percent_zero'] >= 0.70].values


# In[ ]:


train = train.replace({0:np.nan})
test = test.replace({0:np.nan})


# In[ ]:


cluster_sets = {"low":low_vol_columns, "high":high_vol_columns}
for cluster_key in cluster_sets:
    for df in [train,test]:
        df["count_not0_"+cluster_key] = df[cluster_sets[cluster_key]].count(axis=1)
        df["sum_"+cluster_key] = df[cluster_sets[cluster_key]].sum(axis=1)
        df["var_"+cluster_key] = df[cluster_sets[cluster_key]].var(axis=1)
        df["median_"+cluster_key] = df[cluster_sets[cluster_key]].median(axis=1)
        df["mean_"+cluster_key] = df[cluster_sets[cluster_key]].mean(axis=1)
        df["std_"+cluster_key] = df[cluster_sets[cluster_key]].std(axis=1)
        df["max_"+cluster_key] = df[cluster_sets[cluster_key]].max(axis=1)
        df["min_"+cluster_key] = df[cluster_sets[cluster_key]].min(axis=1)
        df["skew_"+cluster_key] = df[cluster_sets[cluster_key]].skew(axis=1)
        df["kurtosis_"+cluster_key] = df[cluster_sets[cluster_key]].kurtosis(axis=1)


# #### I had another version that only dropped the sparse columns and kept the dense columns, but dropping those as well improved LB score.

# In[ ]:


train_more_simplified = train.drop(high_vol_columns,axis=1).drop(low_vol_columns,axis=1)
test_more_simplified = test.drop(high_vol_columns,axis=1).drop(low_vol_columns,axis=1)


# In[ ]:


train_more_simplified.head()


# #### Define XGBoost model.  Note this is taken (almost) straight from another public kernel, although I think I changed the max depth to try to control overfitting. No grid-search performed, could help.

# In[ ]:


def run_xgb(train_X, train_y, val_X, val_y, test_X):
    params = {'objective': 'reg:linear', 
          'eval_metric': 'rmse',
          'eta': 0.001,
          'max_depth': 6, 
          'subsample': 0.6, 
          'colsample_bytree': 0.6,
          'alpha':0.001,
          'random_state': 42, 
          'silent': True}
    print("Load matrices")
    tr_data = xgb.DMatrix(train_X, train_y)
    va_data = xgb.DMatrix(val_X, val_y)
    
    print("Set watchlist")
    watchlist = [(tr_data, 'train'), (va_data, 'valid')]

    print("Train model")
    model_xgb = xgb.train(params, tr_data, 20000, watchlist, maximize=False, early_stopping_rounds = 100, verbose_eval=100)
    
    dtest = xgb.DMatrix(test_X)
    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))
    
    return xgb_pred_y, model_xgb


# In[ ]:


dev_X, val_X, dev_y, val_y = train_test_split(train_more_simplified, y_train, test_size = 0.2, random_state = 40)


# In[ ]:


pred_test_xgb, model_xgb = run_xgb(dev_X, dev_y, val_X, val_y, test_more_simplified)
print("Finished!")


# # Create submission file

# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub["target"] = pred_test_xgb


# In[ ]:


sub.to_csv('sub_XGB_Aggregate_v2.csv', index=False)

