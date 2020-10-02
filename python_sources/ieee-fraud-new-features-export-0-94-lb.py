#!/usr/bin/env python
# coding: utf-8

# 
# * I've added a few simple & generic fraud/anomaly features, such as anomality, isolation, missing values per group, datetime, occurences count and hash based identity reconstruction.
# * The kernel will be updated with additional features.
# * Isolation forest for anomaly detection feature from: https://www.kaggle.com/danofer/anomaly-detection-for-feature-engineering-v2
# 
# * Model code is from the baseline + [gpu/Xhulu's kernel](https://www.kaggle.com/xhlulu/ieee-fraud-xgboost-with-gpu-fit-in-40s)
# 
# * Start date set to 2.11.2017 instead of 20.12.2017, based on this kernel : https://www.kaggle.com/terrypham/transactiondt-timeframe-deduction
# 
# * To get my ~95 submission, just run this with more XGBoost iterations and depth / hyperparams (e.g. as in my kernel : https://www.kaggle.com/danofer/xgboost-using-optuna-fastauc-features) , also add V,C to the COLUMN_GROUP_PREFIXES
# 
# 
# 
# ###### PERFORMANCE notes: If running locally, set the `n_jobs` param in the isolation forests (+- XGBoost) to -1/-2 (it's problematic in kaggle), and if you have a GPU, set XGBoost to use `gpu_hist` - it's much much faster
#  ###### If runing locally - `FAST_RUN = False`  , otherwise you'll have poor results 

# In[ ]:


# Set this param to false for better performance, at thos ecost of longer run time. 
FAST_RUN = True


# In[ ]:


import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb

from sklearn.ensemble import IsolationForest

from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_predict, cross_val_score

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'if FAST_RUN:\n    print("using sampled data , fast run")\n    train_transaction = pd.read_csv(\'../input/train_transaction.csv\', index_col=\'TransactionID\').sample(frac=0.3)\nelse:\n    train_transaction = pd.read_csv(\'../input/train_transaction.csv\', index_col=\'TransactionID\')# ,nrows=12345)\n\ntest_transaction = pd.read_csv(\'../input/test_transaction.csv\', index_col=\'TransactionID\')#,nrows=123)\n\ntrain_identity = pd.read_csv(\'../input/train_identity.csv\', index_col=\'TransactionID\')\ntest_identity = pd.read_csv(\'../input/test_identity.csv\', index_col=\'TransactionID\')\n\nsample_submission = pd.read_csv(\'../input/sample_submission.csv\', index_col=\'TransactionID\')\n\ntrain = train_transaction.merge(train_identity, how=\'left\', left_index=True, right_index=True)\ntest = test_transaction.merge(test_identity, how=\'left\', left_index=True, right_index=True)\n\nprint(train.shape)\nprint(test.shape)\n\ny_train = train[\'isFraud\'].copy()\ndel train_transaction, train_identity, test_transaction, test_identity\n\n\ntrain.head()')


# In[ ]:


# ## join train+test for easier feature engineering:
# df = pd.concat([train,test],sort=False)
# print(df.shape)


# ### Add some features
# * missing values count
#     * TODO: nans per cattegory/group (e..g V columns)
#     * Could be more efficient with this code, but that's aimed at columnar, not row level summation: https://stackoverflow.com/questions/54207038/groupby-columns-on-column-header-prefix
# * Add some of the time series identified in external platform
# * ToDo: anomaly detection features. 
# * proxy for lack of an identifier, duplicate values. 
#     * TODO: try to understand what could be a proxy for a key/customer/card identifier (then apply features based on that).
#     
#     
# * ToDo: readd feature of identical transactions: this is typically a strong feature, but (surprisingly) gave no signal in this dataset. Both with and without transaction amount (and with transaction time removed ofc).
# 
# 
# * COLUMN_GROUP_PREFIXES - I don't calculate all due to memory issues/kernel instability. Not a very strogn feature, but can help. 

# In[ ]:


list(train.columns)

if FAST_RUN:
    COLUMN_GROUP_PREFIXES = ["card","M","id"]
    
else: COLUMN_GROUP_PREFIXES = ["card","C","D","M","V","id"]  # ,"V" # "C" , "V" # V has many values, slow, but does contribute. 

def column_group_features(df):
    """
    Note: surprisingly slow! 
    TODO: Check speed, e.g. with `$ pip install line_profiler`
    """
    df["total_missing"] = df.isna().sum(axis=1)
    print("total_missing",df["total_missing"].describe(percentiles=[0.5]))
    df["total_unique_values"] = df.nunique(axis=1,dropna=False)
    print("total_unique_values",df["total_unique_values"].describe(percentiles=[0.5]))
    
    for p in COLUMN_GROUP_PREFIXES:
        col_group = [col for col in df if col.startswith(p)]
        print("total cols in subset:", p ,len(col_group))
        df[p+"_missing_count"] = df[col_group].isna().sum(axis=1)
        print(p+"_missing_count", "mean:",df[p+"_missing_count"].describe(percentiles=[]))
        df[p+"_uniques_count"] = df[col_group].nunique(axis=1,dropna=False)
        print(p+"_uniques_count", "mean:",df[p+"_uniques_count"].describe(percentiles=[]))
#         df[p+"_max_val"] = df[col_group].max(axis=1)
#         df[p+"_min_val"] = df[col_group].min(axis=1)
    print("done \n")
    return df


# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
# WARNING! THIS CAN DAMAGE THE DATA 
def reduce_mem_usage(df,do_categoricals=False):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
        else:
            if do_categoricals==True:
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


# %%time

train = reduce_mem_usage(train,do_categoricals=False)
test = reduce_mem_usage(test,do_categoricals=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain = column_group_features(train)\nprint("train features generated")\n\ntest = column_group_features(test)\n\ntrain.head()')


# ## datetime features
# * try to guess date and datetime delta unit, then add features
# * TODO: strong features potential already found offline, need to validate
# * Try 01.12.2017 as start date: https://www.kaggle.com/kevinbonnes/transactiondt-starting-at-2017-12-01
# * ALT : https://www.kaggle.com/terrypham/transactiondt-timeframe-deduction 

# In[ ]:


import datetime

START_DATE = "2017-11-02"

# Preprocess date column
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
train['time'] = train['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
test['time'] = test['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

## check if time of day is morning/early night, and/or weekend/holiday:
train["hour_of_day"] = train['time'].dt.hour
test["hour_of_day"] = test['time'].dt.hour

## check if time of day is morning/early night, and/or weekend/holiday: (day of the week with Monday=0, Sunday=6.)
train["day_of_week"] = train['time'].dt.dayofweek
test["day_of_week"] = test['time'].dt.dayofweek

print(train['time'].describe())
print(test['time'].describe())


# In[ ]:


## no clear correlation, but we expect any such features to be categorical in nature, not ordinal/continous. the model can findi t
train[["isFraud","hour_of_day","day_of_week"]].sample(frac=0.07).corr()


# ## Identity hash: "magic feature"?
# * Top missing feature is user identity (for feature engineering). 
# * Previously (not shown), I extracted features based on  duplicate counts using most of the rows, and showed 1% + of rows as being duplicates (with `TransactionDT`, +- `TransactionAmt` excluded) - but this did not give a good feature (surprisingly).
#     * Fraud is often characterized by similar behavior/repeat behavior. 
# * I will create a proxy for identity from the identity/ `id` data! 
#     * This can doubtless be improved by keeping/dropping some. e.g. DeviceType , DeviceInfo may be too specific.
#     * `id_34` has "match_status:" values - may be only feature we need, or metric of noise (in which case we may want to drop it)
#     
# #### We see that due to the sparsity of the identifier data, 75% of the transactions are duplicates!! 
# * This is probably why this wasn't used to create  a "user ID" in the first place.
# * Presumably, it might still be salvageable using other data
# 
# * This kernel looks at just some card features and gets an approximate of missing car numbers: https://www.kaggle.com/smerllo/identify-unique-cards-id

# In[ ]:


# ID_COLS = [col for col in train if col.startswith("id")] # + (['DeviceType', 'DeviceInfo'])

ID_COLS =['id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 
 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26',
 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
 'DeviceType', 'DeviceInfo']
print(ID_COLS)


# In[ ]:


train[ID_COLS].sample(12).drop_duplicates()


# In[ ]:


train['DeviceType'].value_counts()


# In[ ]:


train['DeviceInfo'].value_counts()


# In[ ]:


print("percent of rows with duplicated identities in data: {:.5}% ".format(100*train[ID_COLS].duplicated().sum()/train.shape[0]))


# In[ ]:


train.duplicated(subset=ID_COLS).sum()


# #### Continued Identity hash: 
# * Let's try using some of the card data as well. (Ideally we'd use something like bank account or location, but we lack that :( )
# 
# * using transaction amt changes this from ~74% (when addin in card type) to 51% . Still a lot of redundancy, and it's a bad variable (too easy to change/game), but better than nothing ?
# * adding in the addr1/2 and email domains gets us to 30%. (The same without `TransactionAmt` goes back to 60% duplicates)
#     * Adding dist 1/2 gets us to 19%, but i'm scared of them - no way of knowing if they mean distance or location or something else
#     
#     
#     
#     ########### Ideal solution: brute force a half dozen combinations as features, add them all or one by one, check feature importance/feature selection, keep the best one(s). 
#     * I leave this for others kernels :) 
#     
#     * does hash respect nans? This kernel ensures they're kept : https://www.kaggle.com/smerllo/identify-unique-cards-id

# In[ ]:


ID_COLS_EXT =['id_01', 'id_02', 'id_03', 'id_04', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
 'DeviceType', 'DeviceInfo',"ProductCD","card1","card2","card3","card4","card5","card6",
              'addr1', 
              'addr2'
      , 'P_emaildomain', 
#               'R_emaildomain'
              ]


ID_COLS =[ 'DeviceType',
          "ProductCD",
          "card1","card2","card3","card4","card5","card6",
       'P_emaildomain', 
#           'addr1', 
              ]


# much siompler, basically just "did they use this trans.amount. Would benefit from loh10 rounding the amount" : 
MINI_ID_COLS = ["card1","card2"] # ,"card3","card4","card5","card6","ProductCD"  


# In[ ]:


train[ID_COLS].head(8).drop_duplicates()


# In[ ]:


train[MINI_ID_COLS].drop_duplicates().shape[0]


# In[ ]:


train[MINI_ID_COLS].head().drop_duplicates()


# In[ ]:


print("% EXT rows with duplicated identities: {:.4}% ".format(100*train[ID_COLS_EXT].duplicated().sum()/train.shape[0]))

print("Without TransactionAmt,  % rows with duplicated identities: {:.4}% ".format(100*train[ID_COLS].duplicated().sum()/train.shape[0]))


print("% duplicatecard: {:.4}% ".format(100*train[MINI_ID_COLS].duplicated().sum()/train.shape[0]))


# In[ ]:


train.columns[420:]


# ## make concatenated train+Test for this feature 
# * delete it after due to memory issues if using kernel
# * can use for anomaly detection features
# 
# 
# 
# todo: hash for counting duplicates (counts based on group by columns with missing values is tricky. We don't care about collisions much.
# 
# * integer (efficient) Hashing : https://stackoverflow.com/questions/25757042/create-hash-value-for-each-row-of-data-with-selected-columns-in-dataframe-in-pyt
#      * `df.apply(lambda x: hash(tuple(x)), axis = 1)`

# In[ ]:


df_all = pd.concat([train,test],sort=True)
print(df_all.shape)


# In[ ]:


# df_all["duplicated_extended"] = df_all[ID_COLS_EXT].duplicated().astype(int)

df_all["duplicated_base"] = df_all[ID_COLS].duplicated().astype(int)

df_all["duplicated_card"] = df_all[MINI_ID_COLS].duplicated().astype(int)


# In[ ]:


get_ipython().run_cell_magic('time', '', '## size includes NaN values, count does not.\n\ndf_all["card_hash"] = df_all[MINI_ID_COLS].apply(lambda x: hash(tuple(x)), axis = 1)\ndf_all["card_hash_total_counts"] = df_all.groupby("card_hash")["total_missing"].transform("size") \n\n\ndf_all["multIDcols_hash"] = df_all[ID_COLS].apply(lambda x: hash(tuple(x)), axis = 1)\ndf_all["multIDcols_total_counts"] = df_all.groupby(ID_COLS)["total_missing"].transform("size") \n\n# df_all["hash_PlusTransamt"] = df_all[ID_COLS_EXT].apply(lambda x: hash(tuple(x)), axis = 1)\n# df_all["hash_PlusTransamt_total_counts"] = df_all.groupby("hash_PlusTransamt")["total_missing"].transform("size") \n\n# # count # transaction amount reoccurred +- trans type\n\n# df_all["hash_Transamt"] = df_all[MINI_ID_COLS].apply(lambda x: hash(tuple(x)), axis = 1)\n# df_all["hash_Transamt_total_counts"] = df_all.groupby("hash_Transamt")["total_missing"].transform("size") \n\n\n\n# # drop some  the hashed ids that uses transaction amount\n# df_all.drop([\'hash_PlusTransamt\',"hash_Transamt"], axis=1, inplace=True)')


# In[ ]:


train.head(2)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n## Spending vs mean\ndf_all["TransactionAmt_count"] = df_all.groupby(["TransactionAmt"])["card1"].transform("size")\n\n## more (count) feats\n\ndf_all["TransactionAmt_count"] = df_all.groupby(["TransactionAmt"])["card1"].transform("size")\n\ndf_all["card1_count"] = df_all.groupby(["card1"])["ProductCD"].transform("size")\ndf_all["card2_count"] = df_all.groupby(["card2"])["ProductCD"].transform("size")\ndf_all["card3_count"] = df_all.groupby(["card3"])["ProductCD"].transform("size")\ndf_all["card5_count"] = df_all.groupby(["card5"])["ProductCD"].transform("size")\n\ndf_all["R_email_count"] = df_all.groupby(["R_emaildomain"])["ProductCD"].transform("size")\ndf_all["P_email_count"] = df_all.groupby(["P_emaildomain"])["ProductCD"].transform("size")\ndf_all["addr1_count"] = df_all.groupby(["addr1"])["ProductCD"].transform("size")\ndf_all["addr2_count"] = df_all.groupby(["addr2"])["ProductCD"].transform("size")\n# joint column count\ndf_all["P_R_emails_count"] = df_all.groupby(["P_emaildomain","R_emaildomain"])["ProductCD"].transform("size")\ndf_all["joint_addresses_count"] = df_all.groupby(["addr1","addr2"])["ProductCD"].transform("size")')


# In[ ]:


## transactions per hour
## https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#resampling
## https://stackoverflow.com/questions/45922291/average-number-of-actions-per-day-of-the-week-using-pandas

## results seem inconsistent between the 2 functions. need to debug. 
# df_all.head(8234).set_index("time").resample("H").size()
# df_all.head(2234).groupby([df_all.time.dt.hour,df_all.time.dt.dayofyear])["ProductCD"].transform("size")

df_all["events_this_hour_cnt"] = df_all.groupby([df_all.time.dt.hour,df_all.time.dt.dayofyear])["ProductCD"].transform("size")

####

# mean spend for an hour of day and DoW. could add more aggregations
df_all["mean_spend_hour_day"] = df_all.groupby([df_all.time.dt.hour,df_all.time.dt.dayofweek])["TransactionAmt"].transform("mean")
# spend vs mean
df_all["normalized_spend_vs_hour_day_mean"] = df_all["TransactionAmt"].div(df_all["mean_spend_hour_day"])

# spend vs that specific day of year
df_all["normalized_spend_vs_dayOfYear"] = df_all["TransactionAmt"].div(df_all.groupby([df_all.time.dt.day])["TransactionAmt"].transform("mean"))


# #### FE kernel - I forgot to add these originally!
# 
# https://www.kaggle.com/c/ieee-fraud-detection/discussion/108575
# * add mean transaction (even if I extract it anyway afterwards, externally).
# * card 1 counts..

# In[ ]:


## Frequency Encoding
temp = df_all['card1'].value_counts().to_dict()
df_all['card1_counts'] = df_all['card1'].map(temp)


# In[ ]:


## Additional mean Aggregations / Group Statistics
temp = df_all.groupby('card1')['TransactionAmt'].agg(['median']).rename({'median':'TransactionAmt_card1_median'},axis=1)
df_all = pd.merge(df_all,temp,on='card1',how='left')


temp = df_all.groupby('card_hash')['TransactionAmt'].agg(['mean']).rename({'mean':'TransactionAmt_card_hash_mean'},axis=1)
df_all = pd.merge(df_all,temp,on='card_hash',how='left')


# #### holiday possible features
# * Depends on our start date
# * Additional possible holidays - cyber monday, black friday... 
# * Holiday code from : https://www.kaggle.com/kyakovlev/ieee-fe-for-local-test
# * Original also did datetime aggregations, may be different than the ones I do here/above.. 

# In[ ]:


list(df_all.columns)


# In[ ]:


########################### TransactionDT
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
dates_range = pd.date_range(start='2017-10-01', end='2019-01-01')
us_holidays = calendar().holidays(start=dates_range.min(), end=dates_range.max())

## USA holidays
df_all['is_holiday'] = (df_all['time'].dt.date.astype('datetime64').isin(us_holidays)).astype(np.int8)


# In[ ]:


df_all.columns[432:]


# #### put our new features back into train, test. 
# 
# * Warning - requires that we manually add these columns here - could be improved. 
# 

# In[ ]:


ADDED_FEATS = [
       'duplicated_base', 'duplicated_card', 'card_hash',
       'card_hash_total_counts', 'multIDcols_total_counts', 'multIDcols_hash',
       'TransactionAmt_count', 'card1_count', 'card2_count', 'card3_count',
       'card5_count', 'R_email_count', 'P_email_count', 'addr1_count',
       'addr2_count', 'P_R_emails_count', 'joint_addresses_count',
       'events_this_hour_cnt', 'mean_spend_hour_day',
       'normalized_spend_vs_hour_day_mean', 'normalized_spend_vs_dayOfYear',
    'TransactionAmt_card1_median', 'TransactionAmt_card_hash_mean', 'card1_counts'
    ,'is_holiday']


# In[ ]:


# df_all = reduce_mem_usage(df_all,do_categoricals=False)


# In[ ]:


print(train.shape)
train = train.join(df_all[ADDED_FEATS])
print(train.shape)
print()
print(test.shape)
test = test.join(df_all[ADDED_FEATS])
print(test.shape)


# In[ ]:


# check if columns have unary vals
nunique = test.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index
test.drop(cols_to_drop, axis=1,inplace=True)

train.drop(cols_to_drop, axis=1,inplace=True)
print(train.shape)
print(test.shape)


# ### label-encode & model build
# * TODO: compare to OHE? +- other encoding/embedding methods

# In[ ]:


import gc


# In[ ]:


# Drop target, fill in NaNs ?
# consider dropping the TransactionDT column as well...
X_train = train.drop(['isFraud',"time",'TransactionDT'], axis=1).copy()
X_test = test.drop(["time",'TransactionDT'], axis=1).copy()

y_train = train['isFraud'].copy() # recopy


del train, test
gc.collect()

# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))   


# In[ ]:


### xgboost can handlke nans itself, often better (more efficient memory use) BUT our  test data has diff nan distrib ? 

X_train.fillna(-999,inplace=True)
X_test.fillna(-999,inplace=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nX_train = reduce_mem_usage(X_train,do_categoricals=True)\nX_test = reduce_mem_usage(X_test,do_categoricals=True)')


# ## Anomaly detection features
# * Isolation forest approach for now, can easily be improved with semisupervised approach, additional models, TSNE etc'
# * Based on this kernel: https://www.kaggle.com/danofer/anomaly-detection-for-feature-engineering-v2
# 
# * Note: potential improvement: train additional model on only positive (non fraud) samples on concatenated train+test. 
# 
# 
# 
# ##### Isolation forest (anomaly detection)
# * https://www.kaggle.io/svf/1100683/56c8356ed1b0a6efccea8371bc791ba7/__results__.html#Tree-based-techniques )
# * contamination = % of anomalies expected  (fraud class % in our case)
# 
# * isolation forest doesn't work on nan values!
#     * TODO: model +- transaction amount. +- nan imputation (at least/especially for important columns)

# In[ ]:


# df_all = pd.concat([X_train.dropna(axis=1),X_test.dropna(axis=1)]).drop(["TransactionDT"],axis=1).dropna(axis=1)
# TR_ROWS = X_train.shape[0]
# NO_NAN_COLS = df_all.columns
# print("num of no nan cols",len(NO_NAN_COLS))
# print("columns with no missing vals: \n",NO_NAN_COLS)
# print(df_all.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'clf = IsolationForest(random_state=82,  max_samples=0.05, bootstrap=False, n_jobs=2,\n                          n_estimators=100,max_features=0.98,behaviour="new",contamination= 0.035)\nclf.fit(pd.concat([X_train.dropna(axis=1),X_test.dropna(axis=1)]))\n# del (df_all)')


# In[ ]:


## add anomalous feature.
## Warning! this is brittle! be careful with the columns!!

# X_train["isolation_overall_score"] =clf.decision_function(X_train[NO_NAN_COLS])
# X_test["isolation_overall_score"] =clf.decision_function(X_test[NO_NAN_COLS])

X_train["isolation_overall_score"] =clf.decision_function(X_train)
X_test["isolation_overall_score"] =clf.decision_function(X_test)


print("Fraud only mean anomaly score",X_train.loc[y_train==1]["isolation_overall_score"].mean())
print("Non-Fraud only mean anomaly score",X_train.loc[y_train==0]["isolation_overall_score"].mean())


# In[ ]:


df_all["isolation_overall_score"] = pd.concat([X_train["isolation_overall_score"],X_test["isolation_overall_score"] ])

df_all["isolation_overall_score"].describe()


# In[ ]:


# train only on non fraud samples

clf = IsolationForest(random_state=31,  bootstrap=True,  max_samples=0.05,n_jobs=3,
                          n_estimators=100,behaviour="new",max_features=0.96) #
# clf.fit(X_train[NO_NAN_COLS].loc[y_train==1].values)
clf.fit(X_train.loc[y_train==1].values)

# X_train["isolation_pos_score"] =clf.decision_function(X_train[NO_NAN_COLS])
# X_test["isolation_pos_score"] =clf.decision_function(X_test[NO_NAN_COLS])

X_train["isolation_pos_score"] =clf.decision_function(X_train)
X_test["isolation_pos_score"] =clf.decision_function(X_test)

# del (clf)

print("Fraud only mean pos-anomaly score",X_train.loc[y_train==1]["isolation_pos_score"].mean())
print("Non-Fraud only mean pos-anomaly score",X_train.loc[y_train==0]["isolation_pos_score"].mean())


# In[ ]:


df_all["isolation_pos_score"] = pd.concat([X_train["isolation_pos_score"],X_test["isolation_pos_score"] ])

df_all["isolation_pos_score"].describe()


# ##### Model training
# 
# * todo: do cross_val_predict (sklearn) using sklearn api for convenience
# * Temporal split :  use sklearn's TimeSeriesSplit (or manual) for early stopping/validation + validation
# 
# 
# * First version - cross validated predictions - ensemble approach
# * secodn appproach - single model

# In[ ]:


## some hyperparams, made faster for fast runs. 

if FAST_RUN:
    EPOCHS = 2
    model_num_estimators = 300
    model_lr = 0.2
else: 
    EPOCHS = 4 # use more for better perf, but greater risk of overfitting
    model_num_estimators = 700
    model_lr = 0.06
    
    
kf = KFold(n_splits = EPOCHS, shuffle = False)
kf_time = TimeSeriesSplit(n_splits = EPOCHS) # temporal validation. use this to evaluate performance better , not necessarily as good for OOV ensembling though!


# In[ ]:


# %%time


# y_preds = np.zeros(sample_submission.shape[0])
# y_oof = np.zeros(X_train.shape[0])
# for tr_idx, val_idx in kf.split(X_train, y_train):
#     clf = xgb.XGBClassifier(#n_jobs=2,
#         n_estimators=500,  # 500 default
#         max_depth=9, # 9
#         learning_rate=0.05,
#         subsample=0.9,
#         colsample_bytree=0.9,
# #         tree_method='gpu_hist' # #'gpu_hist', - faster, less exact , "gpu_exact" - better perf
# #         ,min_child_weight=2 # 1 by default
#     )
    
#     X_tr, X_vl = X_train.iloc[tr_idx, :], X_train.iloc[val_idx, :]
#     y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[val_idx]
#     clf.fit(X_tr, y_tr)
#     y_pred_train = clf.predict_proba(X_vl)[:,1]
#     y_oof[val_idx] = y_pred_train
#     print('ROC AUC {}'.format(roc_auc_score(y_vl, y_pred_train)))
    
#     y_preds+= clf.predict_proba(X_test)[:,1] / EPOCHS


# In[ ]:


get_ipython().run_cell_magic('time', '', '## second approach:\n\nclf = xgb.XGBClassifier(n_jobs=3,\n    n_estimators=model_num_estimators,  # 500 default\n    max_depth=11, # 9\n    learning_rate=model_lr, # 0.05 better\n    subsample=0.9,\n    colsample_bytree=0.9\n    ,tree_method= \'hist\' # #\'gpu_hist\', - faster, less exact , "gpu_exact" - better perf , "auto", \'hist\' (cpu)\n        ,min_child_weight=2 # 1 by default\n)\n\n\nclf.fit( X_train,y_train)\ny_preds = clf.predict_proba(X_test)[:,1]')


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# fi = pd.DataFrame(index=clf.feature_names_)
fi = pd.DataFrame(index=X_train.columns)
fi['importance'] = clf.feature_importances_
fi.loc[fi['importance'] > 0.0004].sort_values('importance').head(50).plot(kind='barh', figsize=(14, 32), title='Feature Importance')
plt.show()


# In[ ]:


# make submissions
sample_submission['isFraud'] = y_preds
sample_submission.to_csv('dan_xgboost_cpu_hist_singlemodel.csv')


# In[ ]:


print("mean AUC",cross_val_score(clf, X_train,y_train, cv=kf, scoring='roc_auc').mean())


# In[ ]:


print("mean temporal CV AUC",cross_val_score(clf, X_train,y_train, cv=kf_time, scoring='roc_auc').mean())


# ## A model stacking feature
# #### warning - may overfit!!
# * temporal CV self score (vs random CV). 

# In[ ]:


X_train["random_cv_preds"] = cross_val_predict(clf, X_train,y_train, method='predict_proba')[:,1]
X_test["random_cv_preds"] = y_preds


# In[ ]:


# X_train["temporal_cv_preds"] = cross_val_predict(clf, X_train,y_train, cv=kf_time.split(X_train),  method='predict_proba')[:,1]
# X_test["temporal_cv_preds"] = y_preds


# In[ ]:


get_ipython().run_cell_magic('time', '', '## second approach:\n\nclf = xgb.XGBClassifier(n_jobs=2,\n    n_estimators=model_num_estimators,  # 500 default\n    max_depth=15, # 19\n    learning_rate=model_lr, # 0.06 is better\n    subsample=0.9,\n    colsample_bylevel=0.9\n    ,tree_method= \'auto\' # #\'gpu_hist\', / "hist" - faster, less exact , "gpu_exact" - better perf , "auto", \'hist\' (cpu)\n     ,min_child_weight=3 # 1 by default\n    ,missing=-999\n#     ,scale_pos_weight=3\n)\n\n\nclf.fit( X_train,y_train)\ny_preds = clf.predict_proba(X_test)[:,1]')


# In[ ]:


fi = pd.DataFrame(index=X_train.columns)
fi['importance'] = clf.feature_importances_
fi.loc[fi['importance'] > 0.0004].sort_values('importance').head(40).plot(kind='barh', figsize=(14, 32), title='Feature Importance')
plt.show()


# In[ ]:


sample_submission['isFraud'] = y_preds
sample_submission.to_csv('dan_xgboost_stack_model2.csv')


# In[ ]:


df_all["random_cv_preds"] = pd.concat([X_train["random_cv_preds"],X_test["random_cv_preds"] ])
# df_all["temporal_cv_preds"] = pd.concat([X_train["temporal_cv_preds"],X_test["temporal_cv_preds"] ])


# ### Store the extracted, novel features in a new dataframe for export/sharing
# 
# * Store it before any memory saving resizing if possible
# * can concat the anomaly model based features to it, or seperately. 
# 
# * Remember: `TransactionID` is the index, not a column

# In[ ]:


df_all.head()


# In[ ]:


df_all.columns[:65]


# In[ ]:


df_all.columns[:100]


# In[ ]:


df_all.columns[100:200]


# In[ ]:


df_all.columns[200:300]


# In[ ]:


df_all.columns[300:390]


# In[ ]:


df_all.columns[390:]


# In[ ]:


df_all.head()


# In[ ]:


len(df_all.columns[432:])


# In[ ]:


df_all.columns[431:]


# In[ ]:


EXTRA_FEAT_NAMES = df_all.columns[432:]


# In[ ]:


df_all[EXTRA_FEAT_NAMES].tail()


# In[ ]:


df_all[EXTRA_FEAT_NAMES].to_csv("extra_fraud_feats_baseV2.csv")#.gz",compression="gzip")

# del df_all


# In[ ]:


df_all.drop(['TransactionDT'],axis=1).loc[~df_all['isFraud'].isna()].sample(frac=0.005).to_csv("sample_fraud_train_augV1.csv")


# In[ ]:


df_all.drop(['TransactionDT'],axis=1).loc[~df_all['isFraud'].isna()].to_csv("fraud_train_augV1.csv")
print("train full saved")
df_all.loc[df_all['isFraud'].isna()].drop(['TransactionDT','isFraud'],axis=1).to_csv("fraud_test_augV1.csv")
print("test full saved")


# #### transactions only for TS

# In[ ]:


TRANSACT_COLS = ['isFraud', 'time', 'card_hash', 
       'multIDcols_hash','normalized_spend_vs_hour_day_mean',
       'normalized_spend_vs_dayOfYear', 'isolation_overall_score',
       'isolation_pos_score','TransactionAmt','P_emaildomain', 'ProductCD', 'R_emaildomain','addr1', 'addr2',
                'id_01', 'id_02', 'id_03', 'id_04', 'id_05',
       'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13',
       'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21',
       'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
       'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37',
       'id_38',"V307","V179","V120","V258", "V185"]


# In[ ]:


df_all.head()


# In[ ]:


df_all.loc[~df_all['isFraud'].isna()][TRANSACT_COLS].to_csv("tr_eventsTS.csv.gz",compression="gzip")


# In[ ]:


df_all.loc[df_all['isFraud'].isna()][TRANSACT_COLS].to_csv("test_eventsTS.csv.gz",compression="gzip")


# #### Simple model based feature importance plot
# * TODO: shapley, interactions
# 
# * It looks like our grouped missing values are **valuable**, although the datetime features seemingly didn't (likely, some of the anonymized variables already capture them). They may have some marginal contribution.
#     * toDo: check that run models with and without them
