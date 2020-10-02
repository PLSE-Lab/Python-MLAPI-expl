#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gc
import glob
import os
import time

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold

gc.enable()

pd.options.display.max_rows = 96
pd.options.display.max_columns = 128


# ### Available inputs: 
# 
# 
# For this competition, dictionary containing information about dataset is available.
# That's helpful for feature engineering, as it provides a possible direction of engineering for each feature.
# One thing to keep in mind is that this set was created _artificially_:
# **_All data is simulated and fictitious, and is not real customer data_**
# 
# Kernel environment has it's memory and speed constraints, therefore `historical_transactions.csv` file will not be used, as it's the biggest one.
# We will base our workflow on remaining set of files.

# In[ ]:


input_dir = '../input/'
input_files = sorted(glob.glob(input_dir + '*'))

input_files


# ### Data loading:

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_csv('../input/train.csv', parse_dates=['first_active_month'])\ntest = pd.read_csv('../input/test.csv', parse_dates=['first_active_month'])\nsample_submission = pd.read_csv('../input/sample_submission.csv')\n\n\nmerchant = pd.read_csv('../input/merchants.csv')\nnew_merchant = pd.read_csv('../input/new_merchant_transactions.csv')\n# historical = pd.read_csv('../input/historical_transactions.csv')")


# ### Quick look at train and test data:
# 
# - `card_id` is the ID of card, some of information from other DFs can be merged by those
# - only 3 features (anonymized) + information about month, according to the description: 'YYYY-MM', month of first purchase
# - `train.csv` contains target, which is the feature we will try to predict. This one is defined as: Loyalty numerical score calculated 2 months after historical and evaluation period

# In[ ]:


train.head()


# In[ ]:


test.head()


# ### Quick look at other tables - _merchant_:
# 
# 
# - This table gives a richer set of features for possible exploration and feature engineering
# - Some of those contain NaN values (worth exploring!)
# - Some features contain already averaged information, for example _avg_sales_lag3 - Monthly average of revenue in last 3 months divided by revenue in last active month_
# - There is no card_id to merge this DF to train/test set

# In[ ]:


merchant.head()


# ### Quick look at other tables - _new merchant_:
# 
# 
# - Amount of features in this table is somewhere between train and merchant DFs
# - Good information is that it contains both `merchant_id` and `card_id`, to this may be a mean to connect `train`, `test` and `merchant` DFs

# In[ ]:


new_merchant.head()


# ### NaN structure:
# 
# - no NaN in train and test, that's good
# - some NaNs in both merchants DF, especially `category_2` feature.

# In[ ]:


print('Train NaN:\n\n{}\n'.format(np.sum(pd.isnull(train))))
print('Test NaN:\n\n{}\n'.format(np.sum(pd.isnull(test))))
print('Merchant NaN:\n\n{}\n'.format(np.sum(pd.isnull(merchant))))
print('New Merchant NaN:\n\n{}\n'.format(np.sum(pd.isnull(new_merchant))))


# ### Let's check how many of those occur in both and how many do not:
# 
# We must start with filling missing values in `merchant_id` with some value without meaning, like `NoID`.

# In[ ]:


new_merchant = new_merchant.dropna(subset=['merchant_id'])


# In[ ]:


merchant_id_num = len(merchant['merchant_id'].unique())
new_merchant_id_num = len(new_merchant['merchant_id'].unique())
merchant_id_intersect = len(np.intersect1d(new_merchant.merchant_id, merchant.merchant_id))

print('Merchant IDs: {}'.format(merchant_id_num))
print('New merchant IDs: {}'.format(new_merchant_id_num))
print('Merchants ID intersection: {}'.format(merchant_id_intersect))


# After checking the intersection, we know that all merchants from new_merchant DF are covered, so this won't be an issue when merging with main train/test DF.

# ### card_id intersection:

# In[ ]:


train_card_id_num = len(train['card_id'].unique())
test_card_id_num = len(test['card_id'].unique())
train_card_id_intersect = len(np.intersect1d(new_merchant.card_id, train.card_id))
test_card_id_intersect = len(np.intersect1d(new_merchant.card_id, test.card_id))

print('train card IDs: {}'.format(train_card_id_num))
print('test card IDs: {}'.format(test_card_id_num))
print('train card IDs intersection: {}'.format(train_card_id_intersect))
print('test card IDs intersection: {}'.format(test_card_id_intersect))


# In[ ]:


train_id_frac = train_card_id_intersect / train_card_id_num
test_id_frac = test_card_id_intersect / test_card_id_num

print('train frac: {:.3f}, test frac: {:.3f}'.format(train_id_frac, test_id_frac))


# Coverage of train and test card IDs in new_merchant is almost the same, 89%.

# ### Feature engineering:
# 
# 
# ### 1. Feature encoding
# 
# In order to know, which features must be encoded (to numerical values), let's create a reusable function to get different types of columns based on their data type. This may not work in 100% but will cover most of the cases.
# Manual check is always worth a few minutes.

# In[ ]:


# Get columns of each type
def get_column_types(df):

    categorical_columns = [
        col for col in df.columns if df[col].dtype == 'object']
    categorical_columns_int = [
        col for col in df.columns if df[col].dtype == 'int']
    numerical_columns = [
        col for col in df.columns if df[col].dtype == 'float']

    categorical_columns = [
        x for x in categorical_columns if 'id' not in x]
    categorical_columns_int = [
        x for x in categorical_columns_int if 'id' not in x]

    return categorical_columns, categorical_columns_int, numerical_columns


# Rename columns after grouping for easy merge and access
def rename_columns(df):
    
    df.columns = pd.Index(['{}{}'.format(
        c[0], c[1].upper()) for c in df.columns.tolist()])
    
    return df


# In[ ]:


merchant_cat_feats, merchant_catint_feats, merchant_num_feats = get_column_types(merchant)

print('Categorical features to encode: {}'.format(merchant_cat_feats))
print('\nCategorical int features: {}'.format(merchant_catint_feats))
print('\nNumerical features: {}'.format(merchant_num_feats))


# In[ ]:


new_merchant_cat_feats, new_merchant_catint_feats, new_merchant_num_feats = get_column_types(new_merchant)

print('Categorical features to encode: {}'.format(new_merchant_cat_feats))
print('\nCategorical int features: {}'.format(new_merchant_catint_feats))
print('\nNumerical features: {}'.format(new_merchant_num_feats))


# In[ ]:


# Let's create set of aggregates, which will be used for features grouping.
# One for categorical and one for numerical features.

aggs_num_basic = ['mean', 'min', 'max', 'sum']
aggs_cat_basic = ['mean', 'sum', 'count']


# In[ ]:


# Encode string features to numbers:
# If encoding train and test separately, remember to keep the features mapping between the two!

for c in new_merchant_cat_feats:
    print('Encoding: {}'.format(c))
    new_merchant[c] = pd.factorize(new_merchant[c])[0]
    
for c in merchant_cat_feats:
    print('Encoding: {}'.format(c))
    merchant[c] = pd.factorize(merchant[c])[0]
    
new_merchant


# ### Group merchant data by merchant_id:

# In[ ]:


merchant_card_id_cat = merchant.groupby(['merchant_id'])[merchant_cat_feats].agg(aggs_cat_basic)
merchant_card_id_num = merchant.groupby(['merchant_id'])[merchant_num_feats].agg(aggs_num_basic)

merchant_card_id_cat = rename_columns(merchant_card_id_cat)
merchant_card_id_num = rename_columns(merchant_card_id_num)


# In[ ]:


merchant_card_id_cat.head()


# ### join merchant features with new_merchant:

# In[ ]:


new_merchant_ = new_merchant.set_index('merchant_id').join(merchant_card_id_cat, how='left')
new_merchant_ = new_merchant_.join(merchant_card_id_num, how='left')


# ### Group new_merchant data by card_id:

# In[ ]:


_, new_merchant_catint_feats2, new_merchant_num_feats2 = get_column_types(new_merchant_)

print('\nCategorical int features: {}'.format(new_merchant_catint_feats2))
print('\nNumerical features: {}'.format(new_merchant_num_feats2))


# In[ ]:


new_merchant_card_id_cat = new_merchant_.groupby(['card_id'])[new_merchant_catint_feats2].agg(aggs_cat_basic)
new_merchant_card_id_num = new_merchant_.groupby(['card_id'])[new_merchant_num_feats2].agg(aggs_num_basic)

new_merchant_card_id_cat = rename_columns(new_merchant_card_id_cat)
new_merchant_card_id_num = rename_columns(new_merchant_card_id_num)


# ### join new_merchant with train/test by card_id:

# In[ ]:


train_ = train.set_index('card_id').join(new_merchant_card_id_cat, how='left')
train_ = train_.join(new_merchant_card_id_num, how='left')

test_ = test.set_index('card_id').join(new_merchant_card_id_cat, how='left')
test_ = test_.join(new_merchant_card_id_num, how='left')


del train, test
gc.collect()


# ### Prepare for training:

# In[ ]:


y = train_.target
X = train_.drop(['target'], axis=1)
X_test = test_.copy()


features_to_remove = ['first_active_month']

X = X.drop(features_to_remove, axis=1)
X_test = X_test.drop(features_to_remove, axis=1)


# Assert that set of features is the same for both train and test DFs:
assert np.all(X.columns == X_test.columns)


del train_, test_
gc.collect()


# ### check NaN structure of new features:

# In[ ]:


np.sum(pd.isnull(X)) / X.shape[0]


# In[ ]:


np.sum(pd.isnull(X_test)) / X_test.shape[0]


# ### KFold LGB model training:

# In[ ]:


# KFold splits
kf = KFold(n_splits=5, shuffle=True, random_state=1337)
# Column names:
train_cols = X.columns.tolist()


# LGB model parameters:
params = {'learning_rate': 0.03,
          'boosting': 'gbdt', 
          'objective': 'regression', 
          'metric': 'rmse',
          'num_leaves': 64,
          'min_data_in_leaf': 6,
          'max_bin': 255,
          'bagging_fraction': 0.7,
          'lambda_l2': 1e-4,
          'max_depth': 12,
          'seed': 1337,
          'nthreads': 6}


# Placeholders for out-of-fold predictions
oof_val = np.zeros((X.shape[0]))
oof_test = np.zeros((5, X_test.shape[0]))


i = 0 # Placeholder for fold indexing
for tr, val in kf.split(X, y):
    
    print('Fold: {}'.format(i + 1))
    
    # Split into training and validation part
    X_tr, y_tr = X.iloc[tr, :], y.iloc[tr]
    X_val, y_val = X.iloc[val, :], y.iloc[val]
    
    # Create Dataset objects for lgb model
    dtrain = lgb.Dataset(X_tr.values, y_tr.values, feature_name=train_cols)
    dvalid = lgb.Dataset(X_val.values, y_val.values,
                         feature_name=train_cols, reference=dtrain)
    
    # Train model
    lgb_model = lgb.train(params, dtrain, 
                      num_boost_round=1000, 
                      valid_sets=(dvalid,), 
                      valid_names=('valid',), 
                      verbose_eval=25, 
                      early_stopping_rounds=20)
    
    # Save predictions for each fold
    oof_val[val] = lgb_model.predict(X_val)
    oof_test[i, :] = lgb_model.predict(X_test)
    
    i += 1


# In[ ]:


# Check RMSE for training set:
valid_rmse = mean_squared_error(y, oof_val) ** .5

print('Valid RMSE: {:.4f}'.format(valid_rmse))


# ### Prepare submission:

# In[ ]:


# Average test predcitions across folds:
test_preds = oof_test.mean(axis=0)

# Create submission:
sample_submission['target'] = test_preds
sample_submission.to_csv("submission_trial.csv", index=False)
sample_submission.head()


# In[ ]:




