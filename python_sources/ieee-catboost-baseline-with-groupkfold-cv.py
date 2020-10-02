#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# General imports
import numpy as np
import pandas as pd
import os, sys, gc, warnings, random, datetime, psutil

from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, GroupKFold
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')


# In[ ]:


########################### Helpers
#################################################################################
## Seeder
# :seed to make all processes deterministic     # type: int
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
## Memory Reducer
# :df pandas dataframe to reduce size             # type: pd.DataFrame()
# :verbose                                        # type: bool
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

## Simple "Memory profilers" to see memory usage
def get_memory_usage():
    return np.round(psutil.Process(os.getpid()).memory_info()[0]/2.**30, 2) 
        
def sizeof_fmt(num, suffix='B'):
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


# In[ ]:


########################### Model
def make_predictions(tr_df, tt_df, features_columns, target, cat_params, NFOLDS=2, kfold_mode='grouped'):
    
    X,y = tr_df[features_columns], tr_df[target]    
    P,P_y = tt_df[features_columns], tt_df[target]  
    split_groups = tr_df['DT_M']

    tt_df = tt_df[['TransactionID',target]] 
    tr_df = tr_df[['TransactionID',target]] 
    
    predictions = np.zeros(len(tt_df))
    oof = np.zeros(len(tr_df))

    if kfold_mode=='grouped':
        folds = GroupKFold(n_splits=NFOLDS)
        folds_split = folds.split(X, y, groups=split_groups)
    else:
        folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
        folds_split = folds.split(X, y)

    for fold_, (trn_idx, val_idx) in enumerate(folds_split):        
        print('Fold:',fold_)
        
        estimator = CatBoostClassifier(**cat_params)        
        estimator.fit(
            X.iloc[trn_idx,:],y[trn_idx],
            eval_set=(X.iloc[val_idx,:], y[val_idx]),
            cat_features=categorical_features,
            use_best_model=True,
            verbose=True)
        
        pp_p = estimator.predict_proba(P)[:,1]
        predictions += pp_p/NFOLDS
        
        oof_preds = estimator.predict_proba(X.iloc[val_idx,:])[:,1]
        oof[val_idx] = (oof_preds - oof_preds.min())/(oof_preds.max() - oof_preds.min())
        
        del estimator
        gc.collect()
        
    tt_df['prediction'] = predictions
    print('OOF AUC:', metrics.roc_auc_score(y, oof))
    if LOCAL_TEST:
        print('Holdout AUC:', metrics.roc_auc_score(tt_df[TARGET], tt_df['prediction']))
    
    return tt_df
## -------------------


# In[ ]:


########################### Vars
SEED = 42
seed_everything(SEED)
LOCAL_TEST = False
TARGET = 'isFraud'
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')


# In[ ]:


########################### Model params
cat_params = {
                'n_estimators':5000,
                'learning_rate': 0.07,
                'eval_metric':'AUC',
                'loss_function':'Logloss',
                'random_seed':SEED,
                'metric_period':500,
                'od_wait':500,
                'task_type':'GPU',
                'depth': 8,
                #'colsample_bylevel':0.7,
                } 


# In[ ]:


########################### DATA LOAD
#################################################################################
print('Load Data')

if LOCAL_TEST:
    train_df = pd.read_pickle('../input/ieee-fe-for-local-test/train_df.pkl')
    test_df = pd.read_pickle('../input/ieee-fe-for-local-test/test_df.pkl')
else:
    train_df = pd.read_pickle('../input/ieee-fe-with-some-eda/train_df.pkl')
    test_df = pd.read_pickle('../input/ieee-fe-with-some-eda/test_df.pkl')
    
remove_features = pd.read_pickle('../input/ieee-fe-with-some-eda/remove_features.pkl')
remove_features = list(remove_features['features_to_remove'].values)
print('Shape control:', train_df.shape, test_df.shape)


# In[ ]:


########################### Encode NaN goups
nans_groups = {}
temp_df = train_df.isna()
temp_df2 = test_df.isna()
nans_df = pd.concat([temp_df, temp_df2])

for col in list(nans_df):
    cur_group = nans_df[col].sum()
    if cur_group>0:
        try:
            nans_groups[cur_group].append(col)
        except:
            nans_groups[cur_group]=[col]

add_category = []
for col in nans_groups:
    if len(nans_groups[col])>1:
        train_df['nan_group_'+str(col)] = np.where(temp_df[nans_groups[col]].sum(axis=1)>0,1,0).astype(np.int8)
        test_df['nan_group_'+str(col)]  = np.where(temp_df2[nans_groups[col]].sum(axis=1)>0,1,0).astype(np.int8)
        add_category.append('nan_group_'+str(col))
        
del temp_df, temp_df2, nans_df, nans_groups


# In[ ]:


########################### Copy original Categorical features
categorical_features = ['ProductCD','M4',
                        'card1','card2','card3','card4','card5','card6',
                        'addr1','addr2','dist1','dist2',
                        'P_emaildomain','R_emaildomain',
                       ]

o_trans = pd.concat([pd.read_pickle('../input/ieee-data-minification/train_transaction.pkl'),
                     pd.read_pickle('../input/ieee-data-minification/test_transaction.pkl')])

o_ident = pd.concat([pd.read_pickle('../input/ieee-data-minification/train_identity.pkl'),
                     pd.read_pickle('../input/ieee-data-minification/test_identity.pkl')])

o_trans = o_trans.merge(o_ident, on=['TransactionID'], how='left')
o_trans = o_trans[['TransactionID'] + categorical_features]
o_features = categorical_features.copy()
categorical_features = [col+'_cat' for col in categorical_features]
o_trans.columns = ['TransactionID'] + categorical_features
del o_ident

temp_df = train_df[['TransactionID']]
temp_df = temp_df.merge(o_trans, on=['TransactionID'], how='left')
del temp_df['TransactionID']
train_df = pd.concat([train_df, temp_df], axis=1)

temp_df = test_df[['TransactionID']]
temp_df = temp_df.merge(o_trans, on=['TransactionID'], how='left')
del temp_df['TransactionID']
test_df = pd.concat([test_df, temp_df], axis=1)
del temp_df, o_trans

for col in o_features:
    if train_df[col].equals(train_df[col+'_cat']):
        print('No transformation (keep only categorical)', col)
        del train_df[col], test_df[col]
        
    col = col+'_cat'    
    train_df[col] = train_df[col].fillna(-999)
    test_df[col]  = test_df[col].fillna(-999)

categorical_features += add_category


# In[ ]:


########################### Transform Heavy Dominated columns
total_items = len(train_df)
keep_cols = [TARGET,'C3_fq_enc']

for col in list(train_df):
    if train_df[col].dtype.name!='category':
        cur_dominator = list(train_df[col].fillna(-999).value_counts())[0]
        if (cur_dominator/total_items > 0.85) and (col not in keep_cols):
            cur_dominator = train_df[col].fillna(-999).value_counts().index[0]
            print('Column:', col, ' | Dominator:', cur_dominator)
            train_df[col] = np.where(train_df[col].fillna(-999)==cur_dominator,1,0)
            test_df[col] = np.where(test_df[col].fillna(-999)==cur_dominator,1,0)

            train_df[col] = train_df[col].fillna(-999).astype(int)
            test_df[col] = test_df[col].fillna(-999).astype(int)

            if col not in categorical_features:
                categorical_features.append(col)
                
categorical_features +=['D8_not_same_day','TransactionAmt_check']


# In[ ]:


########################### Restore some categorical features
## These features weren't useful for lgbm
## but catboost can use it
restore_features = [
                    'uid','uid2','uid3','uid4','uid5','bank_type',
                    ]

for col in restore_features:
    categorical_features.append(col)
    remove_features.remove(col)


# In[ ]:


########################### Remove 100% duplicated columns
cols_sum = {}
bad_types = ['datetime64[ns]', 'category','object']

for col in list(train_df):
    if train_df[col].dtype.name not in bad_types:
        cur_col = train_df[col].values
        cur_sum = cur_col.mean()
        try:
            cols_sum[cur_sum].append(col)
        except:
            cols_sum[cur_sum] = [col]

cols_sum = {k:v for k,v in cols_sum.items() if len(v)>1}   

for k,v in cols_sum.items():
    for col in v[1:]:
        if train_df[v[0]].equals(train_df[col]):
            print('Duplicate', col)
            del train_df[col], test_df[col]


# In[ ]:


########################### Encode Str columns
# As we restored some original features
# we nned to run LabelEncoder to reduce
# memory usage and garant that there are no nans
for col in list(train_df):
    if train_df[col].dtype=='O':
        print(col)
        train_df[col] = train_df[col].fillna('unseen_before_label')
        test_df[col]  = test_df[col].fillna('unseen_before_label')
        
        train_df[col] = train_df[col].astype(str)
        test_df[col] = test_df[col].astype(str)
        
        le = LabelEncoder()
        le.fit(list(train_df[col])+list(test_df[col]))
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])
    
    elif col in categorical_features:
        train_df[col] = train_df[col].astype(float).fillna(-999)
        test_df[col]  = test_df[col].astype(float).fillna(-999)
        
        le = LabelEncoder()
        le.fit(list(train_df[col])+list(test_df[col]))
        train_df[col] = le.transform(train_df[col])
        test_df[col]  = le.transform(test_df[col])


# In[ ]:


########################### Final features list
features_columns = [col for col in list(train_df) if col not in remove_features]
categorical_features = [col for col in categorical_features if col in features_columns]

########################### Final Minification
## I don't like this part as it changes float numbers
## small change but change.
## To be able to train catboost without 
## minification we need to do some changes on model
## we will do it later.
if not LOCAL_TEST:
    train_df = reduce_mem_usage(train_df)
    test_df  = reduce_mem_usage(test_df)
    
train_df = train_df[['TransactionID','DT_M',TARGET]+features_columns]
test_df  = test_df[['TransactionID','DT_M',TARGET]+features_columns]
gc.collect()


# In[ ]:


########################### Cleaning
# Check what variables consume memory
for name, size in sorted(((name, sys.getsizeof(value)) for name,value in locals().items()),
                         key= lambda x: -x[1])[:10]:
    print("{:>30}: {:>8}".format(name,sizeof_fmt(size)))
print('Memory in Gb', get_memory_usage())

# Confirm thar variable exist
temp_df = 0

del temp_df
gc.collect()


# In[ ]:


########################### Model Train
if LOCAL_TEST:
    test_predictions = make_predictions(train_df, test_df, features_columns, TARGET, cat_params, 
                                        NFOLDS=4, kfold_mode='grouped')

else:    
    # Why NFOLDS = 6 -> we have 6 months -> let's split it by month))
    NFOLDS = 6
    folds = GroupKFold(n_splits=NFOLDS)

    X,y = train_df[features_columns], train_df[TARGET]    
    P,P_y = test_df[features_columns], test_df[TARGET]  
    
    split_groups = train_df['DT_M']
    # We don't need original sets anymore
    # let's reduce it
    train_df = train_df[['TransactionID',TARGET]] 
    test_df = test_df[['TransactionID',TARGET]] 
    test_df['prediction'] = 0
    gc.collect()
    
    oof = np.zeros(len(train_df))
    predictions = np.zeros(len(test_df))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):
        print('Fold:',fold_)
        
        estimator = CatBoostClassifier(**cat_params)        
        estimator.fit(
            X.iloc[trn_idx,:],y[trn_idx],
            eval_set=(X.iloc[val_idx,:], y[val_idx]),
            cat_features=categorical_features,
            use_best_model=True,
            verbose=True)

        oof_preds = estimator.predict_proba(X.iloc[val_idx,:])[:,1]
        oof[val_idx] = (oof_preds - oof_preds.min())/(oof_preds.max() - oof_preds.min())
        test_df['prediction'] += estimator.predict_proba(P)[:,1]/NFOLDS
        
        del estimator
        gc.collect()
        
    print('OOF AUC:', metrics.roc_auc_score(y, oof))


# In[ ]:


########################### Export
if not LOCAL_TEST:
    test_df['isFraud'] = test_df['prediction']
    test_df[['TransactionID','isFraud']].to_csv('submission.csv', index=False)

