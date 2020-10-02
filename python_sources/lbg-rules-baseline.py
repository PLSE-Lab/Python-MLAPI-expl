#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import gc
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb


# ## Loading data

# In[ ]:


# Loading the data
train = pd.read_csv('../input/train.csv', parse_dates=['date'])
test = pd.read_csv('../input/test.csv', parse_dates=['date'])
sample_sub = pd.read_csv('../input/sample_submission.csv')
print('Train shape:{}, Test shape:{}'.format(train.shape, test.shape))
train.head()


# ## Feature Engineering

# In[ ]:


# Concatenating train & test
train['train_or_test'] = 'train'
test['train_or_test'] = 'test'
df = pd.concat([train,test], sort=False)
print('Combined df shape:{}'.format(df.shape))
del train, test
gc.collect()


# ### Date Features

# In[ ]:


# Extracting date features
df['dayofmonth'] = df.date.dt.day
df['dayofyear'] = df.date.dt.dayofyear
df['dayofweek'] = df.date.dt.dayofweek
df['month'] = df.date.dt.month
df['year'] = df.date.dt.year
df['weekofyear'] = df.date.dt.weekofyear
df['is_month_start'] = (df.date.dt.is_month_start).astype(int)
df['is_month_end'] = (df.date.dt.is_month_end).astype(int)
df.head()


# In[ ]:


# Sorting the dataframe by store then item then date
df.sort_values(by=['store','item','date'], axis=0, inplace=True)


# ### Features constructed from previous sales values

# In[ ]:


# Creating sales lag features
def create_sales_lag_feats(df, gpby_cols, target_col, lags):
    gpby = df.groupby(gpby_cols)
    for i in lags:
        df['_'.join([target_col, 'lag', str(i)])] =                 gpby[target_col].shift(i).values + np.random.normal(scale=5.0, size=(len(df),))
    return df

# Creating sales rolling mean features
def create_sales_rmean_feats(df, gpby_cols, target_col, windows, min_periods=2, shift=1):
    gpby = df.groupby(gpby_cols)
    for w in windows:
        df['_'.join([target_col, 'rmean', str(w)])] =             gpby[target_col].shift(shift).rolling(window=w, min_periods=min_periods).mean().values +            np.random.normal(scale=5.0, size=(len(df),))
    return df

# Creating sales rolling median features
def create_sales_rmed_feats(df, gpby_cols, target_col, windows, min_periods=2, shift=1):
    gpby = df.groupby(gpby_cols)
    for w in windows:
        df['_'.join([target_col, 'rmed', str(w)])] =             gpby[target_col].shift(shift).rolling(window=w, min_periods=min_periods).median().values+            + np.random.normal(scale=5.0, size=(len(df),))
    return df


# ### OHE of categorical features

# In[ ]:


def one_hot_encoder(df, ohe_cols=['store','item','dayofmonth','dayofweek','month','weekofyear']):
    '''
    One-Hot Encoder function
    '''
    print('Creating OHE features..\nOld df shape:{}'.format(df.shape))
    df = pd.get_dummies(df, columns=ohe_cols)
    print('New df shape:{}'.format(df.shape))
    return df


# ## Time-based Validation set

# In[ ]:


# For validation 
# We can choose last 3 months of training period(Oct, Nov, Dec 2017) as our validation set to gauge the performance of the model.
# OR to keep months also identical to test set we can choose period (Jan, Feb, Mar 2017) as the validation set.
# Here we will go with the former to keep it simple.
masked_series = (df.year==2017) & (df.month.isin([10,11,12]))
df.loc[(masked_series), 'train_or_test'] = 'val'
print('Validation shape: {}'.format(df.loc[df.train_or_test=='val',:].shape))


# ## Model Validation

# In[ ]:


# Converting sales of validation period to nan so as to resemble test period
train = df.loc[df.train_or_test.isin(['train','val']), :]
Y_val = train.loc[train.train_or_test=='val', 'sales'].values.reshape((-1))
Y_train = train.loc[train.train_or_test=='train', 'sales'].values.reshape((-1))
train.loc[train.train_or_test=='val', 'sales'] = np.nan

# Creating sales lag, rolling mean, rolling median, ohe features of the above train set
train = create_sales_lag_feats(train, gpby_cols=['store','item'], target_col='sales', 
                               lags=[91,98,105,112,119,126,182,364,546,728])
# train = create_sales_rmean_feats(train, gpby_cols=['store','item'], target_col='sales', 
#                                  windows=[98,119,182,364], min_periods=2)
train = create_sales_rmed_feats(train, gpby_cols=['store','item'], target_col='sales', 
                                windows=[182,364,546,728], min_periods=2) #98,119,
train = one_hot_encoder(train, ohe_cols=['store','item','dayofweek','month']) 
                        #,'dayofmonth','weekofyear'

# Final train and val datasets
val = train.loc[train.train_or_test=='val', :]
train = train.loc[train.train_or_test=='train', :]
print('Train shape:{}, Val shape:{}'.format(train.shape, val.shape))


# ## LightGBM Model

# ### Training features

# In[ ]:


avoid_cols = ['date', 'sales', 'train_or_test', 'id', 'year']
cols = [col for col in train.columns if col not in avoid_cols]
print('No of training features: {} \nAnd they are:{}'.format(len(cols), cols))


# In[ ]:


def smape(preds, target):
    '''
    Function to calculate SMAPE
    '''
    smape_val = 0
    for pred,true in zip(preds, target):
        if (pred==0) & (true==0):
            continue
        else:
            smape_val += abs(pred-true)/(abs(pred)+abs(true))
    smape_val=(200*smape_val)/len(preds)
    return smape_val

def lgbm_smape(preds, train_data):
    '''
    Custom Evaluation Function for LGBM
    '''
    labels = train_data.get_label()
    smape_val = smape(preds, labels)
    return 'SMAPE', smape_val, False


# In[ ]:


# LightGBM parameters
lgb_params = {'task':'train', 'boosting_type':'gbdt', 'objective':'regression', 'metric': {'rmse'}, 
              'num_leaves': 100, 'learning_rate': 0.05, 'feature_fraction': 0.8, 'verbose': 0, 
              'num_boost_round':15000, 'early_stopping_rounds':30, 'nthread':-1}


# In[ ]:


# Creating lgbtrain & lgbval
lgbtrain = lgb.Dataset(data=train.loc[:,cols].values, label=Y_train, feature_name=cols)
lgbval = lgb.Dataset(data=val.loc[:,cols].values, label=Y_val, reference=lgbtrain, feature_name=cols)


# In[ ]:


def lgb_validation(params, lgbtrain, lgbval, X_val, Y_val, verbose_eval):
    t0 = time.time()
    evals_result = {}
    model = lgb.train(params, lgbtrain, num_boost_round=params['num_boost_round'], 
                      valid_sets=[lgbtrain, lgbval], feval=lgbm_smape, 
                      early_stopping_rounds=params['early_stopping_rounds'], evals_result=evals_result, 
                      verbose_eval=verbose_eval)
    print(model.best_iteration)
    print('Total time taken to build the model: ', (time.time()-t0)/60, 'minutes!!')
    pred_Y_val = model.predict(X_val, num_iteration=model.best_iteration)
    val_df = pd.DataFrame(columns=['true_Y_val','pred_Y_val'])
    val_df['pred_Y_val'] = pred_Y_val
    val_df['true_Y_val'] = Y_val
    print(val_df.shape)
    print(val_df.head())
    print('SMAPE for validation data is:{}'.format(smape(pred_Y_val, Y_val)))
    return model, val_df


# In[ ]:


# Training lightgbm model and validating
model, val_df = lgb_validation(lgb_params, lgbtrain, lgbval, val.loc[:,cols].values, Y_val, verbose_eval=50)


# In[ ]:


# Let's see top 25 features as identified by the lightgbm model.
print("Features importance...")
gain = model.feature_importance('gain')
feat_imp = pd.DataFrame({'feature':model.feature_name(), 
                         'split':model.feature_importance('split'), 
                         'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print('Top 25 features:\n', feat_imp.head(25))


# ## Final Model

# In[ ]:


# Creating sales lag, rolling mean, rolling median, ohe features of the above train set
df = create_sales_lag_feats(df, gpby_cols=['store','item'], target_col='sales', 
                            lags=[91,98,105,112,119,126,182,364,546,728])
# df = create_sales_rmean_feats(df, gpby_cols=['store','item'], target_col='sales', 
#                               windows=[98,119,182,364], min_periods=2)
df = create_sales_rmed_feats(df, gpby_cols=['store','item'], target_col='sales', 
                             windows=[182,364,546,728], min_periods=2) #98,119,
df = one_hot_encoder(df, ohe_cols=['store','item','dayofweek','month']) #'dayofmonth',,'weekofyear'

# Final train and test datasets
test = df.loc[df.train_or_test=='test', :]
train = df.loc[~(df.train_or_test=='test'), :]
print('Train shape:{}, Test shape:{}'.format(train.shape, test.shape))


# In[ ]:


# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=train.loc[:,cols].values, label=train.loc[:,'sales'].values.reshape((-1,)), 
                           feature_name=cols)


# In[ ]:


def lgb_train(params, lgbtrain_all, X_test, num_round):
    t0 = time.time()
    model = lgb.train(params, lgbtrain_all, num_boost_round=num_round, feval=lgbm_smape)
    test_preds = model.predict(X_test, num_iteration=num_round)
    print('Total time taken in model training: ', (time.time()-t0)/60, 'minutes!')
    return model, test_preds


# In[ ]:


# Training lgb model on whole data(train+val)
lgb_model, test_preds = lgb_train(lgb_params, lgbtrain_all, test.loc[:,cols].values, model.best_iteration+20)
print('test_preds shape:{}'.format(test_preds.shape))


# ## Submission

# In[ ]:


# Create submission
sub = test.loc[:,['id','sales']]
sub['sales'] = test_preds
sub['id'] = sub.id.astype(int)
sub.to_csv('submission.csv', index=False)
sub.head()


# ### TODO:
# *  Exponential mean features
# *  Explore other win_type in rolling  
