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
#df.sort_values(by=['store','item','month','dayofweek'], axis=0, inplace=True)
df.sort_values(by=['store','item','date'], axis=0, inplace=True)


# ### Monthwise aggregated sales values

# In[ ]:


def create_sales_agg_monthwise_features(df, gpby_cols, target_col, agg_funcs):
    '''
    Creates various sales agg features with given agg functions  
    '''
    gpby = df.groupby(gpby_cols)
    newdf = df[gpby_cols].drop_duplicates().reset_index(drop=True)
    for agg_name, agg_func in agg_funcs.items():
        aggdf = gpby[target_col].agg(agg_func).reset_index()
        aggdf.rename(columns={target_col:target_col+'_'+agg_name}, inplace=True)
        newdf = newdf.merge(aggdf, on=gpby_cols, how='left')
    return newdf


# ### Features constructed from previous sales values

# In[ ]:


# Creating sales lag features
def create_sales_lag_feats(df, gpby_cols, target_col, lags):
    gpby = df.groupby(gpby_cols)
    for i in lags:
        df['_'.join([target_col, 'lag', str(i)])] =                 gpby[target_col].shift(i).values + np.random.normal(scale=1.6, size=(len(df),))
    return df

# Creating sales rolling mean features
def create_sales_rmean_feats(df, gpby_cols, target_col, windows, min_periods=2, 
                             shift=1, win_type=None):
    gpby = df.groupby(gpby_cols)
    for w in windows:
        df['_'.join([target_col, 'rmean', str(w)])] =             gpby[target_col].shift(shift).rolling(window=w, 
                                                  min_periods=min_periods,
                                                  win_type=win_type).mean().values +\
            np.random.normal(scale=1.6, size=(len(df),))
    return df

# Creating sales rolling median features
def create_sales_rmed_feats(df, gpby_cols, target_col, windows, min_periods=2, 
                            shift=1, win_type=None):
    gpby = df.groupby(gpby_cols)
    for w in windows:
        df['_'.join([target_col, 'rmed', str(w)])] =             gpby[target_col].shift(shift).rolling(window=w, 
                                                  min_periods=min_periods,
                                                  win_type=win_type).median().values +\
            np.random.normal(scale=1.6, size=(len(df),))
    return df

# Creating sales exponentially weighted mean features
def create_sales_ewm_feats(df, gpby_cols, target_col, alpha=[0.9], shift=[1]):
    gpby = df.groupby(gpby_cols)
    for a in alpha:
        for s in shift:
            df['_'.join([target_col, 'lag', str(s), 'ewm', str(a)])] =                 gpby[target_col].shift(s).ewm(alpha=a).mean().values
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


# ### Log Sales 

# In[ ]:


# Converting sales to log(1+sales)
df['sales'] = np.log1p(df.sales.values)
df.sample(2)


# ## Time-based Validation set

# In[ ]:


# For validation 
# We can choose last 3 months of training period(Oct, Nov, Dec 2017) as our validation set to gauge the performance of the model.
# OR to keep months also identical to test set we can choose period (Jan, Feb, Mar 2017) as the validation set.
# Here we will go with the latter choice.
masked_series = (df.year==2017) & (df.month.isin([1,2,3]))
masked_series2 = (df.year==2017) & (~(df.month.isin([1,2,3])))
df.loc[(masked_series), 'train_or_test'] = 'val'
df.loc[(masked_series2), 'train_or_test'] = 'no_train'
print('Train shape: {}'.format(df.loc[df.train_or_test=='train',:].shape))
print('Validation shape: {}'.format(df.loc[df.train_or_test=='val',:].shape))
print('No train shape: {}'.format(df.loc[df.train_or_test=='no_train',:].shape))
print('Test shape: {}'.format(df.loc[df.train_or_test=='test',:].shape))


# ## Model Validation

# In[ ]:


# Converting sales of validation period to nan so as to resemble test period
train = df.loc[df.train_or_test.isin(['train','val']), :]
Y_val = train.loc[train.train_or_test=='val', 'sales'].values.reshape((-1))
Y_train = train.loc[train.train_or_test=='train', 'sales'].values.reshape((-1))
train.loc[train.train_or_test=='val', 'sales'] = np.nan

# # Creating sales lag, rolling mean, rolling median, ohe features of the above train set
train = create_sales_lag_feats(train, gpby_cols=['store','item'], target_col='sales', 
                               lags=[91,98,105,112,119,126,182,364,546,728])

train = create_sales_rmean_feats(train, gpby_cols=['store','item'], 
                                 target_col='sales', windows=[364,546], 
                                 min_periods=10, win_type='triang') #98,119,91,182,

# # train = create_sales_rmed_feats(train, gpby_cols=['store','item'], 
# #                                 target_col='sales', windows=[364,546], 
# #                                 min_periods=10, win_type=None) #98,119,91,182,

train = create_sales_ewm_feats(train, gpby_cols=['store','item'], 
                               target_col='sales', 
                               alpha=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5], 
                               shift=[91,98,105,112,119,126,182,364,546,728])

# # Creating sales monthwise aggregated values
# agg_df = create_sales_agg_monthwise_features(df.loc[df.train_or_test=='train', :], 
#                                              gpby_cols=['store','item','month'], 
#                                              target_col='sales', 
#                                              agg_funcs={'mean':np.mean, 
#                                              'median':np.median, 'max':np.max, 
#                                              'min':np.min, 'std':np.std})

# # Joining agg_df with train
# train = train.merge(agg_df, on=['store','item','month'], how='left')

# One-Hot Encoding 
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
    n = len(preds)
    masked_arr = ~((preds==0)&(target==0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds-target)
    denom = np.abs(preds)+np.abs(target)
    smape_val = (200*np.sum(num/denom))/n
    return smape_val

def lgbm_smape(preds, train_data):
    '''
    Custom Evaluation Function for LGBM
    '''
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False


# In[ ]:


# LightGBM parameters
lgb_params = {'task':'train', 'boosting_type':'gbdt', 'objective':'regression', 
              'metric': {'mae'}, 'num_leaves': 10, 'learning_rate': 0.02, 
              'feature_fraction': 0.8, 'max_depth': 5, 'verbose': 0, 
              'num_boost_round':15000, 'early_stopping_rounds':200, 'nthread':-1}


# In[ ]:


# Creating lgbtrain & lgbval
lgbtrain = lgb.Dataset(data=train.loc[:,cols].values, label=Y_train, 
                       feature_name=cols)
lgbval = lgb.Dataset(data=val.loc[:,cols].values, label=Y_val, 
                     reference=lgbtrain, feature_name=cols)


# In[ ]:


def lgb_validation(params, lgbtrain, lgbval, X_val, Y_val, verbose_eval):
    t0 = time.time()
    evals_result = {}
    model = lgb.train(params, lgbtrain, num_boost_round=params['num_boost_round'], 
                      valid_sets=[lgbtrain, lgbval], feval=lgbm_smape, 
                      early_stopping_rounds=params['early_stopping_rounds'], 
                      evals_result=evals_result, verbose_eval=verbose_eval)
    print(model.best_iteration)
    print('Total time taken to build the model: ', (time.time()-t0)/60, 'minutes!!')
    pred_Y_val = model.predict(X_val, num_iteration=model.best_iteration)
    pred_Y_val = np.expm1(pred_Y_val)
    Y_val = np.expm1(Y_val)
    val_df = pd.DataFrame(columns=['true_Y_val','pred_Y_val'])
    val_df['pred_Y_val'] = pred_Y_val
    val_df['true_Y_val'] = Y_val
    print(val_df.shape)
    print(val_df.sample(5))
    print('SMAPE for validation data is:{}'.format(smape(pred_Y_val, Y_val)))
    return model, val_df


# In[ ]:


# Training lightgbm model and validating
model, val_df = lgb_validation(lgb_params, lgbtrain, lgbval, val.loc[:,cols].values, 
                               Y_val, verbose_eval=500)


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
df_whole = create_sales_lag_feats(df, gpby_cols=['store','item'], target_col='sales', 
                                  lags=[91,98,105,112,119,126,182,364,546,728])
df_whole = create_sales_rmean_feats(df_whole, gpby_cols=['store','item'], 
                                    target_col='sales', windows=[364,546], 
                                    min_periods=10, win_type='triang')
# df = create_sales_rmed_feats(df, gpby_cols=['store','item'], target_col='sales', 
#                              windows=[364,546], min_periods=2) #98,119,
df_whole = create_sales_ewm_feats(df_whole, gpby_cols=['store','item'], target_col='sales', 
                                  alpha=[0.95, 0.9, 0.8, 0.7, 0.6, 0.5], 
                                  shift=[91,98,105,112,119,126,182,364,546,728])

# # Creating sales monthwise aggregated values
# agg_df = create_sales_agg_monthwise_features(df.loc[~(df.train_or_test=='test'), :], 
#                                              gpby_cols=['store','item','month'], 
#                                              target_col='sales', 
#                                              agg_funcs={'mean':np.mean, 
#                                              'median':np.median, 'max':np.max, 
#                                              'min':np.min, 'std':np.std})

# # Joining agg_df with df
# df = df.merge(agg_df, on=['store','item','month'], how='left')

# One-Hot Encoding
df_whole = one_hot_encoder(df_whole, ohe_cols=['store','item','dayofweek','month']) 
#'dayofmonth',,'weekofyear'

# Final train and test datasets
test = df_whole.loc[df_whole.train_or_test=='test', :]
train = df_whole.loc[~(df_whole.train_or_test=='test'), :]
print('Train shape:{}, Test shape:{}'.format(train.shape, test.shape))


# In[ ]:


# LightGBM dataset
lgbtrain_all = lgb.Dataset(data=train.loc[:,cols].values, 
                           label=train.loc[:,'sales'].values.reshape((-1,)), 
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
lgb_model, test_preds = lgb_train(lgb_params, lgbtrain_all, test.loc[:,cols].values, model.best_iteration)
print('test_preds shape:{}'.format(test_preds.shape))


# ## Submission

# In[ ]:


# Create submission
sub = test.loc[:,['id','sales']]
sub['sales'] = np.expm1(test_preds)
sub['id'] = sub.id.astype(int)
sub.to_csv('submission.csv', index=False)
sub.head()


# ## WaveNet Model 

# In[ ]:


df.head(2)


# In[ ]:


df.date.min(), df.date.max()


# In[ ]:




