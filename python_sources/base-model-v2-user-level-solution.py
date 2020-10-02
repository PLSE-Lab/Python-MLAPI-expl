#!/usr/bin/env python
# coding: utf-8

# # Introduction

# ### This kernel is mainly made up of three parts:
# * [**1. Data loading**](#Data loading)
# * [**2. Data preprocessing**](#Data preprocessing)
# * [**3. Model building**](#Model building)
# 
# ###  Main of the kernel:
# *  Chunk whole of data set  by the period as such a structure: 
# * [210 days of training period, 45 days of gap period, 2 months of traget perod].  
# * Aggregating data from the training period, ignoring the  gap period, getting the target from the traget period. 
# * The valiation set is set to Dec-Jan which is the same monthly period  as the target peroid of the test set.
# 
# ### Summary:
#  In this competition, the data set is so unbalanced that it's hard to say whether our solution can beat all-zeros. After doing some basic EDA, there are some conclusions are for sure: 
# 
# 1. if a customer will pay,  the  transaction will be highly likely happened at the first month, and no longer than two months after the customer shows up in first time. 
# 2. the minimum of transaction revenue is no less than 1E+07.
# ---
# * For the first one, our test set has a 1.5 months' gap between the traget period  which means our taget is divided into two groups: the first  is the one who has already spent no less than 45 days on thinking whether to pay. The second is the  one who has payed for partial services and is going to pay for additional services. To the first group, the customers are terrific unlikely to pay. To the second one, the customers are likely to pay much the same as they payed before. Under those conditions, my prediction of the number of people to pay is 200 or so.
# * For the second one, as we see, the prediction of our model is full of numbers less than 1E+07. But you'll get a worse score if you set those numbers to zero. Our model keeps betting wisely on minimize RMSE but the result keeps leaving away from the real numbers. 
# 
# ### random thoughts:
# * To set a user-defined objective function, which gives a high penalty once the floor level is breached, will be good for avoiding small values.
# * Time features should be under the first priority.
# * To the second group people, if it's possible to specify them by clustering.
# * if the customers wil return after a full year of services are expired?
# * the data set is lack of some important features such as page views of user's website. To the low volume users, why do they pay the bill for advance services if the free account already meets all the needs?

# * Data are generated from this script : https://www.kaggle.com/qnkhuat/make-data-ready 
# * Stacking part is from this script: https://www.kaggle.com/ashishpatel26/updated-bayesian-lgbm-xgb-cat-fe-kfold-cv

# ## Data loading

# In[ ]:


import numpy as np 
import pandas as pd 
from datetime import datetime

import os
from os.path import join as pjoin

data_root = '../input/make-data-ready'
print(os.listdir(data_root))

pd.set_option('display.max_rows',200)

from sklearn.preprocessing import LabelEncoder

from pprint import pprint
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


def load_data(data='train',n=2):
    df = pd.DataFrame()
    for i in range(n) :
        if data=='train':
            if i > 8 :
                break
            dfpart = pd.read_pickle(pjoin(data_root,f'train_{i}.pkl'))
        elif data=='test':
            if i > 2 :
                break
            dfpart = pd.read_pickle(pjoin(data_root,f'test_{i}.pkl'))
        df = pd.concat([df,dfpart])
        del dfpart
    return df
        


# In[ ]:


df_train = load_data(n=9)
df_test = load_data('test',n=4)


# ## Data preprocessing

# In[ ]:


df = pd.concat([df_train, df_test])


# ### Drop some features and items

# In[ ]:


col_drop = ['Date_Year', 'Date_Month', 'Date_Week','Date_Hour','device_isMobile','device_deviceCategory',
       'Date_Day', 'Date_Dayofweek', 'Date_Dayofyear', 'Date_Is_month_end',
       'Date_Is_month_start', 'Date_Is_quarter_end', 'Date_Is_quarter_start',
       'Date_Is_year_end', 'Date_Is_year_start','totals_visits',
           'date','visitId','totals_totalTransactionRevenue','geoNetwork_city','geoNetwork_continent',
            'geoNetwork_metro','geoNetwork_networkDomain',
'geoNetwork_region','geoNetwork_subContinent','trafficSource_adContent',
            'trafficSource_adwordsClickInfo.adNetworkType','trafficSource_adwordsClickInfo.gclId',
'trafficSource_adwordsClickInfo.slot','trafficSource_campaign',
            'trafficSource_keyword','trafficSource_referralPath','trafficSource_medium',
            'customDimensions_value','customDimensions_index','trafficSource_source',
           'totals_bounces','visitNumber','totals_newVisits']
df.drop(col_drop, axis=1, inplace=True)


# In[ ]:


country_drop=df.groupby('geoNetwork_country')['totals_transactions'].sum()[df.groupby('geoNetwork_country')['totals_transactions'].sum().sort_values()<4].index.tolist()
df.loc[df[df.geoNetwork_country.isin(country_drop)].index,'geoNetwork_country'] = 'NaN'

df.loc[df[~df.device_browser.isin(['Edge', 'Internet Explorer', 'Firefox', 'Safari', 'Chrome'])].index,'device_browser'] = 'NaN'
df.loc[df[~df.device_operatingSystem.isin(['Android', 'iOS', 'Linux', 'Chrome OS', 'Windows', 'Macintosh'])].index,'device_operatingSystem'] = 'NaN'


# ### Label encoding

# In[ ]:


col_lb = ['channelGrouping','device_browser','device_operatingSystem', 'geoNetwork_country',
          'trafficSource_adwordsClickInfo.isVideoAd','trafficSource_isTrueDirect']
for col in col_lb:
    lb = LabelEncoder()
    df[col]=lb.fit_transform(df[col])


# ### Features to user level
# There is also a feature called time_diff, which is directly coded in generating part. And this time- relative feature really works well

# In[ ]:


to_median = ['channelGrouping','device_browser','device_operatingSystem','geoNetwork_country','trafficSource_adwordsClickInfo.isVideoAd','trafficSource_isTrueDirect','trafficSource_adwordsClickInfo.page']
to_sum =['totals_hits','totals_pageviews','totals_timeOnSite','totals_transactionRevenue', 'totals_transactions']
to_mean =['totals_hits','totals_pageviews','totals_sessionQualityDim']
to_std = ['totals_hits','totals_pageviews']
to_time = 'visitStartTime'


# ### Time period
# * the training set has a 45 days gap to its target set that is same as the test set 
# * the training set has almost the same duration as the test set
# * the valiation set is set to Dec-Jan which is the same monthly period  as the target peroid of the test set

# In[ ]:


target_period = pd.date_range(start='2016-08-01',end='2018-12-01', freq='2MS')
train_period = target_period.to_series().shift(periods=-210, freq='d',axis= 0)
time_to = train_period[train_period.index>np.datetime64('2016-08-01')].astype('int')//10**9
time_end = target_period.to_series().shift(periods=-45, freq='d',axis= 0)[4:]


# ### Test data

# In[ ]:


user_x = df.iloc[df_train.shape[0]:,:]
test_x = pd.concat([user_x.groupby('fullVisitorId')[to_median].median().add_suffix('_median'),
user_x.groupby('fullVisitorId')['visitStartTime'].agg(['first','last']).add_suffix('_time').sub(time_to.values[-1]).abs(),
user_x.groupby('fullVisitorId')['visitStartTime'].apply(lambda x: x.max() -x.min()).rename('time_diff'),
user_x.groupby('fullVisitorId')[to_sum].sum().add_suffix('_sum'),
user_x.groupby('fullVisitorId')[to_mean].mean().add_suffix('_mean'),
user_x.groupby('fullVisitorId')[to_std].std(ddof=0).add_suffix('_std')],axis=1).reset_index()

test_ID= test_x.fullVisitorId
test_x = test_x.drop(['fullVisitorId'], axis=1,errors='ignore')
test_x = test_x.astype('int')


# ### Valiation data

# In[ ]:


i=4
user_x = df[(df.visitStartTime>=(time_to.index.astype('int')//10**9)[i]) & (df.visitStartTime<(time_end.index.astype('int')//10**9)[i])]
user_y = df[(df.visitStartTime>=time_to.values[i]) & (df.visitStartTime<time_to.values[i+1])]
train_x = pd.concat([user_x.groupby('fullVisitorId')[to_median].median().add_suffix('_median'),
user_x.groupby('fullVisitorId')['visitStartTime'].agg(['first','last']).add_suffix('_time').sub(time_to.values[i]).abs(),
user_x.groupby('fullVisitorId')['visitStartTime'].apply(lambda x: x.max() -x.min()).rename('time_diff'),
user_x.groupby('fullVisitorId')[to_sum].sum().add_suffix('_sum'),
user_x.groupby('fullVisitorId')[to_mean].mean().add_suffix('_mean'),
user_x.groupby('fullVisitorId')[to_std].std(ddof=0).add_suffix('_std')],axis=1).reset_index()

merged=train_x.merge(user_y.groupby('fullVisitorId')['totals_transactionRevenue'].sum().reset_index(),                          how='left', on='fullVisitorId')
val_y = merged.totals_transactionRevenue
val_y.fillna(0, inplace=True)
val_x = merged.drop(['fullVisitorId','totals_transactionRevenue'], axis=1,errors='ignore')
val_x = val_x.astype('int')


# ## Model building

# In[ ]:


import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error


# In[ ]:


params={'learning_rate': 0.01,
        'objective':'regression',
        'metric':'rmse',
        'num_leaves': 31,
        'verbose': 1,
        'bagging_fraction': 0.9,
        'feature_fraction': 0.9,
        "random_state":42,
        'max_depth': 5,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "bagging_frequency" : 5,
        'lambda_l2': 0.5,
        'lambda_l1': 0.5,
        'min_child_samples': 36
       }
xgb_params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.02,
        'max_depth': 22,
        'min_child_weight': 57,
        'gamma' : 1.45,
        'alpha': 0.0,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 456,
        'importance_type': 'total_gain'
    }

cat_param = {
    'learning_rate' :0.03,
    'depth' :10,
    'eval_metric' :'RMSE',
    'od_type' :'Iter',
    'metric_period ' : 50,
    'od_wait' : 20,
    'seed' : 42
    
}


# In[ ]:


oof_reg_preds = np.zeros(val_x.shape[0])
oof_reg_preds1 = np.zeros(val_x.shape[0])
oof_reg_preds2 = np.zeros(val_x.shape[0])
merge_pred = np.zeros(val_x.shape[0])
merge_preds = np.zeros(val_x.shape[0])
sub_preds = np.zeros(test_x.shape[0])
alist = list(range(time_to.shape[0]-1))
alist.remove(4)
folds = alist
folds=range(len(alist)-1)

for i in alist:
    print(i)
    user_x = df[(df.visitStartTime>=(time_to.index.astype('int')//10**9)[i]) & (df.visitStartTime<(time_end.index.astype('int')//10**9)[i])]
    user_y = df[(df.visitStartTime>=time_to.values[i]) & (df.visitStartTime<time_to.values[i+1])]
    train_x = pd.concat([user_x.groupby('fullVisitorId')[to_median].median().add_suffix('_median'),
    user_x.groupby('fullVisitorId')['visitStartTime'].agg(['first','last']).add_suffix('_time').sub(time_to.values[i]).abs(),
    user_x.groupby('fullVisitorId')['visitStartTime'].apply(lambda x: x.max() -x.min()).rename('time_diff'),
    user_x.groupby('fullVisitorId')[to_sum].sum().add_suffix('_sum'),
    user_x.groupby('fullVisitorId')[to_mean].mean().add_suffix('_mean'),
    user_x.groupby('fullVisitorId')[to_std].std(ddof=0).add_suffix('_std')],axis=1).reset_index()
    
    merged=train_x.merge(user_y.groupby('fullVisitorId')['totals_transactionRevenue'].sum().reset_index(),                              how='left', on='fullVisitorId')
    train_y = merged.totals_transactionRevenue
    train_y.fillna(0, inplace=True)
    train_x = merged.drop(['fullVisitorId','totals_transactionRevenue'], axis=1,errors='ignore')
    train_x = train_x.astype('int')    
    
    reg = lgb.LGBMRegressor(**params,n_estimators=1100)
    xgb = XGBRegressor(**xgb_params, n_estimators=1000)
    cat = CatBoostRegressor(iterations=1000,learning_rate=0.03,
                            depth=10,
                            eval_metric='RMSE',
                            random_seed = 42,
                            bagging_temperature = 0.2,
                            od_type='Iter',
                            metric_period = 50,
                            od_wait=20)
    print("-"* 20 + "LightGBM Training" + "-"* 20)
    reg.fit(train_x, np.log1p(train_y),eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,verbose=100,eval_metric='rmse')
    print("-"* 20 + "XGboost Training" + "-"* 20)
    xgb.fit(train_x, np.log1p(train_y),eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,eval_metric='rmse',verbose=100)
    print("-"* 20 + "Catboost Training" + "-"* 20)
    cat.fit(train_x, np.log1p(train_y), eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,use_best_model=True,verbose=100)
    
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_x.columns
    imp_df['gain_reg'] = np.zeros(train_x.shape[1])
    imp_df['gain_xgb'] = np.zeros(train_x.shape[1])
    imp_df['gain_cat'] = np.zeros(train_x.shape[1])
    imp_df['gain_reg'] += reg.booster_.feature_importance(importance_type='gain')/ len(folds)
    imp_df['gain_xgb'] += xgb.feature_importances_/ len(folds)
    imp_df['gain_cat'] += np.array(cat.get_feature_importance())/ len(folds)
    
    # LightGBM
    oof_reg_preds = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    lgb_preds = reg.predict(test_x, num_iteration=reg.best_iteration_)
    lgb_preds[lgb_preds < 0] = 0
    
    
    # Xgboost
    oof_reg_preds1 = xgb.predict(val_x)
    oof_reg_preds1[oof_reg_preds1 < 0] = 0
    xgb_preds = xgb.predict(test_x)
    xgb_preds[xgb_preds < 0] = 0
    
    # catboost
    oof_reg_preds2 = cat.predict(val_x)
    oof_reg_preds2[oof_reg_preds2 < 0] = 0
    cat_preds = cat.predict(test_x)
    cat_preds[cat_preds < 0] = 0
        
    #merge all prediction
    merge_pred = oof_reg_preds * 0.4 + oof_reg_preds1 * 0.3 +oof_reg_preds2 * 0.3
    merge_preds += (oof_reg_preds / len(folds)) * 0.4 + (oof_reg_preds1 / len(folds)) * 0.3 + (oof_reg_preds2 / len(folds)) * 0.3
    sub_preds += (lgb_preds / len(folds)) * 0.4 + (xgb_preds / len(folds)) * 0.3 + (cat_preds / len(folds)) * 0.3
    
    
print("LGBM  ", mean_squared_error(np.log1p(val_y), oof_reg_preds) ** .5)
print("XGBoost  ", mean_squared_error(np.log1p(val_y), oof_reg_preds1) ** .5)
print("CatBoost  ", mean_squared_error(np.log1p(val_y), oof_reg_preds2) ** .5)
print("merged  ", mean_squared_error(np.log1p(val_y), merge_pred) ** .5)
print("stack_merged  ", mean_squared_error(np.log1p(val_y), merge_preds) ** .5)
print("Zeros  ", mean_squared_error(np.log1p(val_y), np.zeros(val_x.shape[0])) ** .5)


# ## Display feature importances

# In[ ]:


plt.figure(figsize=(8, 12))
sns.barplot(x='gain_reg', y='feature', data=imp_df.sort_values('gain_reg', ascending=False))


# In[ ]:


plt.figure(figsize=(8, 12))
sns.barplot(x='gain_xgb', y='feature', data=imp_df.sort_values('gain_xgb', ascending=False))


# In[ ]:


plt.figure(figsize=(8, 12))
sns.barplot(x='gain_cat', y='feature', data=imp_df.sort_values('gain_cat', ascending=False))


# ## Save result

# In[ ]:


sub_df = pd.DataFrame(test_ID)
sub_df["PredictedLogRevenue"] = sub_preds
sub_df.to_csv("stacked_result.csv", index=False)

