#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General imports
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random, datetime, math

from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold,GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


# In[ ]:


########################### Helpers
#################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    
## Global frequency encoding    
def frequency_encoding(df, columns, self_encoding=False):
    for col in columns:
        fq_encode = df[col].value_counts(dropna=False).to_dict()
        if self_encoding:
            df[col] = df[col].map(fq_encode)
        else:
            df[col+'_fq_enc'] = df[col].map(fq_encode)
    return df


def values_normalization(dt_df, periods, columns, enc_type='both'):
    for period in periods:
        for col in columns:
            new_col = col +'_'+ period
            dt_df[col] = dt_df[col].astype(float)  

            temp_min = dt_df.groupby([period])[col].agg(['min']).reset_index()
            temp_min.index = temp_min[period].values
            temp_min = temp_min['min'].to_dict()

            temp_max = dt_df.groupby([period])[col].agg(['max']).reset_index()
            temp_max.index = temp_max[period].values
            temp_max = temp_max['max'].to_dict()

            temp_mean = dt_df.groupby([period])[col].agg(['mean']).reset_index()
            temp_mean.index = temp_mean[period].values
            temp_mean = temp_mean['mean'].to_dict()

            temp_std = dt_df.groupby([period])[col].agg(['std']).reset_index()
            temp_std.index = temp_std[period].values
            temp_std = temp_std['std'].to_dict()

            dt_df['temp_min'] = dt_df[period].map(temp_min)
            dt_df['temp_max'] = dt_df[period].map(temp_max)
            dt_df['temp_mean'] = dt_df[period].map(temp_mean)
            dt_df['temp_std'] = dt_df[period].map(temp_std)
            
            if enc_type=='both':
                dt_df[new_col+'_min_max'] = (dt_df[col]-dt_df['temp_min'])/(dt_df['temp_max']-dt_df['temp_min'])
                dt_df[new_col+'_std_score'] = (dt_df[col]-dt_df['temp_mean'])/(dt_df['temp_std'])
            elif enc_type=='norm':
                 dt_df[new_col+'_std_score'] = (dt_df[col]-dt_df['temp_mean'])/(dt_df['temp_std'])
            elif enc_type=='min_max':
                dt_df[new_col+'_min_max'] = (dt_df[col]-dt_df['temp_min'])/(dt_df['temp_max']-dt_df['temp_min'])

            del dt_df['temp_min'],dt_df['temp_max'],dt_df['temp_mean'],dt_df['temp_std']
    return dt_df

def get_new_columns(temp_list):
    temp_list = [col for col in list(full_df) if col not in temp_list]
    temp_list.sort()

    temp_list2 = [col if col not in remove_features else '-' for col in temp_list ]
    temp_list2.sort()

    temp_list = {'New columns (including dummy)': temp_list,
                 'New Features': temp_list2}
    temp_list = pd.DataFrame.from_dict(temp_list)
    return temp_list


# In[ ]:


########################### Vars
#################################################################################
SEED = 42
seed_everything(SEED)
LOCAL_TEST = True
MAKE_TESTS = True
TARGET = 'isFraud'


# In[ ]:


########################### Model params
lgb_params = {
                    'objective':'binary',
                    'boosting_type':'gbdt',
                    'metric':'auc',
                    'n_jobs':-1,
                    'learning_rate':0.01,
                    'num_leaves': 2**8,
                    'max_depth':-1,
                    'tree_learner':'serial',
                    'colsample_bytree': 0.7,
                    'subsample_freq':1,
                    'subsample':0.7,
                    'n_estimators':80000,
                    'max_bin':255,
                    'verbose':-1,
                    'seed': SEED,
                    'early_stopping_rounds':100, 
                } 


# In[ ]:


########################### Model
import lightgbm as lgb

def make_test(old_score=0, output=False):

    features_columns = [col for col in list(full_df) if col not in remove_features]
    train_mask = full_df['TransactionID'].isin(local_train_id['TransactionID'])
    test_mask = full_df['TransactionID'].isin(local_test_id['TransactionID'])
    
    X,y = full_df[train_mask][features_columns], full_df[train_mask][TARGET]    
    P,P_y = full_df[test_mask][features_columns], full_df[test_mask][TARGET]  

    for col in list(X):
        if X[col].dtype=='O':
            X[col] = X[col].fillna('unseen_before_label')
            P[col] = P[col].fillna('unseen_before_label')

            X[col] = X[col].astype(str)
            P[col] = P[col].astype(str)

            le = LabelEncoder()
            le.fit(list(X[col])+list(P[col]))
            X[col] = le.transform(X[col])
            P[col]  = le.transform(P[col])

            X[col] = X[col].astype('category')
            P[col] = P[col].astype('category')
        
    tt_df = full_df[test_mask][['TransactionID','DT_W',TARGET]]        
    tt_df['prediction'] = 0
    
    tr_data = lgb.Dataset(X, label=y)
    vl_data = lgb.Dataset(P, label=P_y) 
    estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets = [tr_data, vl_data],
            verbose_eval = 200,
        )   
        
    tt_df['prediction'] = estimator.predict(P)
    feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), columns=['Value','Feature'])
    
    if output:
        tt_df[['TransactionID','prediction']].to_csv('oof.csv',index=False)
        print('---Wrote OOF to file---')
    
    m_results = []
    print('#'*20)
    g_auc = metrics.roc_auc_score(tt_df[TARGET], tt_df['prediction'])
    score_diff = g_auc - old_score
    print('Global AUC', g_auc)
    m_results.append(g_auc)
    
    for i in range(full_df[test_mask]['DT_W'].min(), full_df[test_mask]['DT_W'].max()+1):
        mask = tt_df['DT_W']==i
        w_auc = metrics.roc_auc_score(tt_df[mask][TARGET], tt_df[mask]['prediction'])
        print('Week', i, w_auc, len(tt_df[mask]))
        m_results.append(w_auc)
        
    print('#'*20)
    print('Features Preformance:', g_auc)
    print('Diff with previous__:', score_diff)
    
    return tt_df, feature_imp, m_results, estimator


# In[ ]:


########################### DATA LOAD
#################################################################################
print('Load Data')
train_df = pd.read_pickle('../input/ieee-data-minification-private/train_transaction.pkl')
test_df = pd.read_pickle('../input/ieee-data-minification-private/test_transaction.pkl')

# Full Data set (careful with target encoding)
full_df = pd.concat([train_df, test_df]).reset_index(drop=True)

# Local test IDs with one month gap
local_test_id  = train_df[train_df['DT_M']==train_df['DT_M'].max()].reset_index(drop=True)
local_train_id = train_df[train_df['DT_M']<(train_df['DT_M'].max()-1)].reset_index(drop=True)
local_train_id = local_train_id[['TransactionID']]
local_test_id  = local_test_id[['TransactionID']]
del train_df, test_df

# Identity Data set
train_identity = pd.read_pickle('../input/ieee-data-minification-private/train_identity.pkl')
test_identity = pd.read_pickle('../input/ieee-data-minification-private/test_identity.pkl')
identity_df = pd.concat([train_identity, test_identity]).reset_index(drop=True)
del train_identity, test_identity

print('Shape control (for local test):', local_train_id.shape, local_test_id.shape)


# In[ ]:


########################### All features columns
#################################################################################
# Add list of feature that we will remove for sure
remove_features = [
    'TransactionID','TransactionDT', 
    TARGET,
    'DT','DT_M','DT_W','DT_D','DTT',
    'DT_hour','DT_day_week','DT_day_month',
    'DT_M_total','DT_W_total','DT_D_total',
    'is_december','is_holiday','temp','weight',
    ]

# Make sure that TransactionAmt is float64
# To not lose values during aggregations
full_df['TransactionAmt'] = full_df['TransactionAmt'].astype(float)

# Base lists for features to do frequency encoding
# and saved initial state
fq_encode = []
base_columns = list(full_df)

# We don't need V columns in the initial phase 
# removing them to make predictions faster
remove_features += ['V'+str(i) for i in range(1,340)]

# Removing transformed D columns
remove_features += ['uid_td_D'+str(i) for i in range(1,16) if i!=9]

# Make sure we have m_results variable
m_results = [0]


# In[ ]:


########################### This is start baseline
if MAKE_TESTS:
    tt_df, feature_imp, m_results, model = make_test()


# In[ ]:


########################### Fix card columns and encode
print('Fix card4 and card6 values')
saved_state = list(full_df)
####

####
# card4 and card5 have strong connection
# with card1 - we can unify values
# to guarantee that it will be same combinations
# for all data.

# I've tried to fill others NaNs
# But seems that there are no more bad values.
# All rest NaNs are meaningful.
####

full_df['card6'] = np.where(full_df['card6']==30, np.nan, full_df['card6'])
full_df['card6'] = np.where(full_df['card6']==16, np.nan, full_df['card6'])

i_cols = ['card4','card6']

for col in i_cols:
    temp_df = full_df.groupby(['card1',col])[col].agg(['count']).reset_index()
    temp_df = temp_df.sort_values(by=['card1','count'], ascending=False)
    del temp_df['count']
    temp_df = temp_df.drop_duplicates(subset=['card1'], keep='first').reset_index(drop=True)
    temp_df.index = temp_df['card1'].values
    temp_df = temp_df[col].to_dict()
    full_df[col] = full_df['card1'].map(temp_df)
    
# Add cards features for later encoding
i_cols = ['card1','card2','card3','card4','card5','card6']
fq_encode += i_cols

####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### Client Virtual ID
print('Create client identification ID')
saved_state = list(full_df)
####

####
# Client subgroups:

# bank_type -> looking on card3 and card5 distributions
# I would say it is bank branch and country
# full_addr -> Client registration address in bank
# uid1 -> client identification by bank and card type
# uid2 -> client identification with additional geo information
####

# Bank type
full_df['bank_type'] = full_df['card3'].astype(str)+'_'+full_df['card5'].astype(str)

# Full address
full_df['full_addr'] = full_df['addr1'].astype(str)+'_'+full_df['addr2'].astype(str)

# Virtual client uid
i_cols = ['card1','card2','card3','card4','card5','card6']
full_df['uid1'] = ''
for col in i_cols:
    full_df['uid1'] += full_df[col].astype(str)+'_'

# Virtual client uid + full_addr
full_df['uid2'] = full_df['uid1']+'_'+full_df['full_addr'].astype(str)


# Add uids features for later encoding
i_cols = ['full_addr','bank_type','uid1','uid2']
fq_encode += i_cols

# We can't use this features directly because
# test data will have many unknow values
remove_features += i_cols

# We've created just "ghost" features -> no need to run test
if False: 
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### Client identification using deltas
print('Create client identification ID using deltas')
saved_state = list(full_df)
####

# Temporary list
client_cols = []

# Convert all delta columns to some date
# D8 and D9 are not days deltas -
# we can try convert D8 to int and 
# probably it will give us date
# but I'm very very unsure about it.

# We will do all D columns transformation
# (but save original values) as we will
# use it later for other features.

for col in ['D'+str(i) for i in range(1,16) if i!=9]: 
    new_col = 'uid_td_'+str(col)
    
    new_col = 'uid_td_'+str(col)
    full_df[new_col] = full_df['TransactionDT'] / (24*60*60)
    full_df[new_col] = np.floor(full_df[new_col] - full_df[col])    
    remove_features.append(new_col)
    
    # Date is useless itself -> add to dummy features
    #remove_features.append(new_col)


# The most possible deltas to identify account or client
# initial activity are 'D1','D10','D15'
# We can try to find certain client using uid and date
# If client is the same uid+date combination will be
# unique per client and all his transactions
for col in ['D1','D10','D15']:
    new_col = 'uid_td_'+str(col)

    # card1 + full_addr + date
    full_df[new_col+'_cUID_1'] = full_df['card1'].astype(str)+'_'+full_df['full_addr'].astype(str)+'_'+full_df[new_col].astype(str)
    
    # uid1 + full_addr + date
    full_df[new_col+'_cUID_2'] = full_df['uid2'].astype(str)+'_'+full_df[new_col].astype(str)

    # columns 'D1','D2' are clipped we can't trust maximum values
    if col in ['D1','D2']:
        full_df[new_col+'_cUID_1'] = np.where(full_df[col]>=640, 'very_old_client', full_df[new_col+'_cUID_1'])
        full_df[new_col+'_cUID_2'] = np.where(full_df[col]>=640, 'very_old_client', full_df[new_col+'_cUID_2'])

    full_df[new_col+'_cUID_1'] = np.where(full_df[col].isna(), np.nan, full_df[new_col+'_cUID_1'])
    full_df[new_col+'_cUID_2'] = np.where(full_df[col].isna(), np.nan, full_df[new_col+'_cUID_2'])

    # reset cUID_1 if both address are nan (very unstable prediction)
    full_df[new_col+'_cUID_1'] = np.where(full_df['addr1'].isna()&full_df['addr2'].isna(), np.nan, full_df[new_col+'_cUID_1'])

    # cUID is useless itself -> add to dummy features
    remove_features += [new_col+'_cUID_1',new_col+'_cUID_2']
    
    # Add to temporary list (to join with encoding list later)
    client_cols += [new_col+'_cUID_1',new_col+'_cUID_2']
    
## Best candidate for client complete identification
## uid_td_D1_cUID_1
        
# Add cUIDs features for later encoding
fq_encode += client_cols

# We will save this list and even append 
# few more columns for later use
client_cols += ['card1','card2','card3','card4','card5',
                'uid1','uid2']

####
# We've created just "ghost" features -> no need to run test
if False: 
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### Mark card columns "outliers"
print('Outliers mark')
saved_state = list(full_df)
####

####
# We are checking card and uid activity -
# weither activity is constant during the year
# or we have just single card/account use cases.

# These features are categorical ones and
# Catboost benefits the most from them.

# Strange things:
# - "Time window" should be big enough 
# - Doesn't work for DT_W and DT_D
# even when local test showing score boost.

# Seems to me that catboost start to combine 
# them with themselfs and loosing "magic".
####

i_cols = client_cols.copy()
periods = ['DT_M'] 

for period in periods:
    for col in i_cols:
        full_df[col+'_catboost_check_'+period] = full_df.groupby([col])[period].transform('nunique')
        full_df[col+'_catboost_check_'+period] = np.where(full_df[col+'_catboost_check_'+period]==1,1,0)
        
####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### V columns compact and assign groups
print('V columns / Nan groups')
saved_state = list(full_df)
####

####
# Nangroups identification are categorical features
# and Catboost benefits the most from them.

# Mean/std just occasion transformation.
####

nans_groups = {}
nans_df = full_df.isna()

i_cols = ['V'+str(i) for i in range(1,340)]
for col in i_cols:
    cur_group = nans_df[col].sum()
    try:
        nans_groups[cur_group].append(col)
    except:
        nans_groups[cur_group]=[col]

for col in nans_groups:
    # Very doubtful features -> Seems it works in tandem with other feature
    # But I'm not sure
    full_df['nan_group_sum_'+str(col)] = full_df[nans_groups[col]].to_numpy().sum(axis=1)
    full_df['nan_group_mean_'+str(col)] = full_df[nans_groups[col]].to_numpy().mean(axis=1)
        
    # lgbm doesn't benefit from such feature -> 
    # let's transform and add it to dummy features list
    full_df['nan_group_catboost_'+str(col)]  = np.where(nans_df[nans_groups[col]].sum(axis=1)>0,1,0).astype(np.int8)
    remove_features.append('nan_group_catboost_'+str(col))
        
####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### Mean encoding using M columns
print('Mean encoding, using M columns')
saved_state = list(full_df)
####

main_cols = {
             'uid_td_D1_cUID_1':   ['M'+str(i) for i in [2,3,5,7,8,9]],
             'uid_td_D1_cUID_2':   ['M'+str(i) for i in [2,3,5,6,9]],
             'uid_td_D10_cUID_1':  ['M'+str(i) for i in [5,7,8,9]],
             'uid_td_D10_cUID_2':  ['M'+str(i) for i in [3,6,7,8]],
             'uid_td_D15_cUID_1':  ['M'+str(i) for i in [2,3,5,6,8,]],
             'uid_td_D15_cUID_2':  ['M'+str(i) for i in [2,3,5,6,7,8]],
             'card1':  ['M'+str(i) for i in [2,3,5,6,7,8,9]],
             'card2':  ['M'+str(i) for i in [1,2,3,7,9]],
             'card4':  ['M'+str(i) for i in [3,7,8]],
             'card5':  ['M'+str(i) for i in [5,6,8]],
             'uid1':   ['M'+str(i) for i in [3,5,6,7,8,9]],
             'uid2':   ['M'+str(i) for i in [2,3,5,6,7,8,9]],
            }

for main_col,i_cols in main_cols.items():
    for agg_type in ['mean']:
        temp_df = full_df[[main_col]+i_cols]
        temp_df = temp_df.groupby([main_col])[i_cols].transform(agg_type)
        temp_df.columns = [main_col+'_'+col+'_'+agg_type for col in list(temp_df)]
        full_df = pd.concat([full_df,temp_df], axis=1)
        
####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### D Columns Mean/Std
print('D columns Mean/Std')
saved_state = list(full_df)
####

i_cols = ['D'+str(i) for i in range(1,16)]
main_cols = {
             'uid_td_D1_cUID_1': ['D'+str(i) for i in [1,2,3,10,11,14,15]],
            }

for main_col,i_cols in main_cols.items():
    print(main_col)
    for agg_type in ['mean','std']:
        temp_df = full_df.groupby([main_col])[i_cols].transform(agg_type)
        temp_df.columns = [main_col+'_'+col+'_'+agg_type for col in list(temp_df)]
        full_df = pd.concat([full_df,temp_df], axis=1)
        
####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### TransactionAmt
print('TransactionAmt normalization')
saved_state = list(full_df)
####

# Decimal part
full_df['TransactionAmt_cents'] = np.round(100.*(full_df['TransactionAmt'] - np.floor(full_df['TransactionAmt'])),0)
full_df['TransactionAmt_cents'] = full_df['TransactionAmt_cents'].astype(np.int8)

# Clip top values
full_df['TransactionAmt'] = full_df['TransactionAmt'].clip(0,5000)

# Normalization by product
main_cols = [
             'uid_td_D1_cUID_1','uid_td_D1_cUID_2',
             'uid_td_D10_cUID_1','uid_td_D10_cUID_2',
             'uid_td_D15_cUID_1','uid_td_D15_cUID_2',
             'card1','card3',
            ]

for col in main_cols:
    for agg_type in ['mean','std']:
        full_df[col+'_TransactionAmt_Product_' + agg_type] =                full_df.groupby([col,'ProductCD'])['TransactionAmt'].transform(agg_type)

    f_std = col+'_TransactionAmt_Product_std'
    f_mean = col+'_TransactionAmt_Product_mean'
    full_df[col+'_Product_norm'] = (full_df['TransactionAmt']-full_df[f_mean])/full_df[f_std]
    del full_df[f_mean], full_df[f_std]
    

####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### TransactionAmt clients columns encoding
print('TransactionAmt encoding clients columns')
saved_state = list(full_df)
####

i_cols = ['TransactionAmt']
main_cols = client_cols.copy()

for main_col in main_cols:
    print(main_col)
    for agg_type in ['mean','std']:
        temp_df = full_df.groupby([main_col])[i_cols].transform(agg_type)
        temp_df.columns = [main_col+'_'+col+'_'+agg_type for col in list(temp_df)]
        full_df = pd.concat([full_df,temp_df], axis=1)

####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### Mark card columns "outliers"
print('Categorical outliers')
## 
saved_state = list(full_df)
####

i_cols = ['TransactionAmt','ProductCD','P_emaildomain','R_emaildomain',]
periods = ['DT_M']

for period in periods:
    for col in i_cols:
        full_df[col+'_catboost_check_'+period] = full_df.groupby([col])[period].transform('nunique')
        full_df[col+'_catboost_check_'+period] = np.where(full_df[col+'_catboost_check_'+period]==1,1,0).astype(np.int8)

        
####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### D Columns Normalize and remove original columns
print('D columns transformations')
## 
saved_state = list(full_df)
####

# Remove original features
# test data will have many unknow values
i_cols = ['D'+str(i) for i in range(1,16)]
remove_features += i_cols

####### Values Normalization
i_cols.remove('D1')
i_cols.remove('D2')
i_cols.remove('D9')
periods = ['DT_D']
for col in i_cols:
    full_df[col] = full_df[col].clip(0)
full_df = values_normalization(full_df, periods, i_cols, enc_type='norm')

i_cols = ['D1','D2','D9']
for col in i_cols:
    full_df[col+'_scaled'] = full_df[col]/full_df[col].max()


####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### Dist
print('Distance normalization')
## 
saved_state = list(full_df)
####

i_cols = ['dist1','dist2']
main_cols = [
             'uid_td_D1_cUID_1',
             'card1',
            ]


for main_col in main_cols:
    print(main_col)
    for agg_type in ['mean','std']:
        temp_df = full_df.groupby([main_col])[i_cols].transform(agg_type)
        temp_df.columns = [main_col+'_'+col+'_'+agg_type for col in list(temp_df)]
        full_df = pd.concat([full_df,temp_df], axis=1)
    
    for col in i_cols:
        f_std = main_col+'_'+col+'_std'
        f_mean = main_col+'_'+col+'_mean'
        full_df[main_col+'_'+col+'_norm'] = (full_df[col]-full_df[f_mean])/full_df[f_std]
        del full_df[f_mean], full_df[f_std]


####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### Count similar transactions per period
print('Similar transactions per period')
## 
saved_state = list(full_df)
####

periods = ['DT_W','DT_D'] 

for period in periods:
    full_df['TransactionAmt_Product_counts_' + period] =        full_df.groupby([period,'ProductCD','TransactionAmt'])['TransactionAmt'].transform('count')
    full_df['TransactionAmt_Product_counts_' + period] /= full_df[period+'_total']

####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### Find nunique dates per client
print('Nunique dates per client')
## 
saved_state = list(full_df)
####

main_cols = {
            'uid_td_D1_cUID_1': ['uid_td_D'+str(i) for i in range(2,16) if i!=9] + ['D8','D9'],
            }

for main_col,i_cols in main_cols.items():
    for col in i_cols:
        full_df[col+'_catboost_check_'+main_col] = full_df.groupby([main_col])[col].transform('nunique')

####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### Email transformation
print('Email split')
saved_state = list(full_df)
####

p = 'P_emaildomain'
r = 'R_emaildomain'

full_df['full_email'] = full_df[p].astype(str) +'_'+ full_df[r].astype(str)
full_df['email_p_extension'] = full_df[p].apply(lambda x: str(x).split('.')[-1])
full_df['email_r_extension'] = full_df[r].apply(lambda x: str(x).split('.')[-1])
full_df['email_p_domain'] = full_df[p].apply(lambda x: str(x).split('.')[0])
full_df['email_r_domain'] = full_df[r].apply(lambda x: str(x).split('.')[0])

i_cols = ['P_emaildomain','R_emaildomain',
          'full_email',
          'email_p_extension','email_r_extension',
          'email_p_domain','email_r_domain']

full_df = frequency_encoding(full_df, i_cols, self_encoding=True)

####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### Device info and identity
print('Identity sets')
saved_state = list(full_df)
####

########################### Device info
identity_df['DeviceInfo'] = identity_df['DeviceInfo'].fillna('unknown_device').str.lower()
identity_df['DeviceInfo_device'] = identity_df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
identity_df['DeviceInfo_version'] = identity_df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))
    
########################### Device info 2
identity_df['id_30'] = identity_df['id_30'].fillna('unknown_device').str.lower()
identity_df['id_30_device'] = identity_df['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
identity_df['id_30_version'] = identity_df['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))
    
########################### Browser
identity_df['id_31'] = identity_df['id_31'].fillna('unknown_device').str.lower()
identity_df['id_31_device'] = identity_df['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    
########################### Merge Identity columns
temp_df = full_df[['TransactionID']]
temp_df = temp_df.merge(identity_df, on=['TransactionID'], how='left')
del temp_df['TransactionID']
full_df = pd.concat([full_df,temp_df], axis=1)
  
i_cols = [
          'DeviceInfo','DeviceInfo_device','DeviceInfo_version',
          'id_30','id_30_device','id_30_version',
          'id_31','id_31_device',
          'id_33','DeviceType'
         ]

####### Global Self frequency encoding
full_df = frequency_encoding(full_df, i_cols, self_encoding=True)

####
if MAKE_TESTS:
    print(get_new_columns(saved_state))
    tt_df, feature_imp, m_results, model = make_test(m_results[0])
####


# In[ ]:


########################### Export
full_df.to_pickle('baseline_full_df.pkl')

remove_features_df = pd.DataFrame(remove_features, columns=['features_to_remove'])
remove_features_df.to_pickle('baseline_remove_features.pkl')

