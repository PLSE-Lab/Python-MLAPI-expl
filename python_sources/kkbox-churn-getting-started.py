#!/usr/bin/env python
# coding: utf-8

# ### Forked from: Regressing During Insomnia [0.21496]
# 
# * Modified to save out output merged files/data + FE + dates parsing

# In[ ]:


from multiprocessing import Pool, cpu_count
import gc; gc.enable()
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import *
import sklearn
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report


# In[ ]:


train = pd.read_csv('../input/train.csv')
# test = pd.read_csv('../input/sample_submission_zero.csv')
test = pd.read_csv('../input/sample_submission_zero.csv',dtype = {'msno' : str}) # new: alt

transactions = pd.read_csv('../input/transactions.csv', usecols=['msno'])
transactions = pd.DataFrame(transactions['msno'].value_counts().reset_index())
transactions.columns = ['msno','trans_count']
train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')
transactions = []; print('transaction merge...')

user_logs = pd.read_csv('../input/user_logs.csv', usecols=['msno'])
user_logs = pd.DataFrame(user_logs['msno'].value_counts().reset_index())
user_logs.columns = ['msno','logs_count']
train = pd.merge(train, user_logs, how='left', on='msno')
test = pd.merge(test, user_logs, how='left', on='msno')
user_logs = []; print('user logs merge...')

members = pd.read_csv('../input/members.csv')
train = pd.merge(train, members, how='left', on='msno')
test = pd.merge(test, members, how='left', on='msno')
members = []; print('members merge...') 


# In[ ]:


gender = {'male':1, 'female':2}
train['gender'] = train['gender'].map(gender)
test['gender'] = test['gender'].map(gender)

# train = train.fillna(-1)
# test = test.fillna(-1)


# In[ ]:


transactions = pd.read_csv('../input/transactions.csv')
transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
transactions = transactions.drop_duplicates(subset=['msno'], keep='first')

train = pd.merge(train, transactions, how='left', on='msno')
test = pd.merge(test, transactions, how='left', on='msno')
transactions=[]


# ### ALT + Some Aggregated feature engineering in advance.
# * Source: https://www.kaggle.com/talysacc/lgbm-starter-lb-0-23434
# * Different pipeline

# In[ ]:


## Add features to test & train

# df_members = pd.read_csv('../input/members.csv',dtype={'registered_via' : np.uint8,
#                                                       'gender' : 'category'})
# df_test = pd.merge(left=df_test,right=df_members,how='left',on=['msno'])
# del df_members

df_transactions = pd.read_csv('../input/transactions.csv',dtype = {'payment_method' : np.uint8,
                                                                  'payment_plan_days' : np.uint8,
                                                                  'plan_list_price' : np.uint8,
                                                                  'actual_amount_paid': np.uint8,
                                                                  'is_auto_renew' : np.bool,
                                                                  'is_cancel' : np.bool})
# orig: left = test ...
df_transactions = pd.merge(left = test[['msno']],right = df_transactions,how='left',on='msno')
grouped  = df_transactions.copy().groupby('msno')

df_stats = grouped.agg({'msno' : {'total_order' : 'count'},
                         'plan_list_price' : {'plan_net_worth' : 'sum'},
                         'actual_amount_paid' : {'mean_payment_each_transaction' : 'mean',
                                                  'total_actual_payment' : 'sum'},
                         'is_cancel' : {'cancel_times' : lambda x : sum(x==1)}})
             
df_stats.columns = df_stats.columns.droplevel(0)
df_stats.reset_index(inplace=True)

test = pd.merge(left = test,right = df_stats,how='left',on='msno')
del df_transactions


# In[ ]:


## Add features to train
df_transactions = pd.read_csv('../input/transactions.csv',dtype = {'payment_method' : np.uint8,
                                                                  'payment_plan_days' : np.uint8,
                                                                  'plan_list_price' : np.uint8,
                                                                  'actual_amount_paid': np.uint8,
                                                                  'is_auto_renew' : np.bool,
                                                                  'is_cancel' : np.bool})
# orig: left = test ...
df_transactions = pd.merge(left = train[['msno']],right = df_transactions,how='left',on='msno')
grouped  = df_transactions.copy().groupby('msno')

df_stats = grouped.agg({'msno' : {'total_order' : 'count'},
                         'plan_list_price' : {'plan_net_worth' : 'sum'},
                         'actual_amount_paid' : {'mean_payment_each_transaction' : 'mean',
                                                  'total_actual_payment' : 'sum'},
                         'is_cancel' : {'cancel_times' : lambda x : sum(x==1)}})
             
df_stats.columns = df_stats.columns.droplevel(0)
df_stats.reset_index(inplace=True)

train = pd.merge(left = train,right = df_stats,how='left',on='msno')
del df_transactions


# ### Back to original (insomnia) pipeline

# In[ ]:


def transform_df(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

def transform_df2(df):
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

df_iter = pd.read_csv('../input/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)
last_user_logs = []
i = 0 #~400 Million Records - starting at the end but remove locally if needed
for df in df_iter:
    if i>35:
        if len(df)>0:
            print(df.shape)
            p = Pool(cpu_count())
            df = p.map(transform_df, np.array_split(df, cpu_count()))   
            df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
            df = transform_df2(df)
            p.close(); p.join()
            last_user_logs.append(df)
            print('...', df.shape)
            df = []
    i+=1

last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
last_user_logs = transform_df2(last_user_logs)

train = pd.merge(train, last_user_logs, how='left', on='msno')
test = pd.merge(test, last_user_logs, how='left', on='msno')
last_user_logs=[]


# In[ ]:


train.dtypes


# In[ ]:


test.dtypes


# In[ ]:


test.info()


# In[ ]:


test.tail()


# In[ ]:


test.loc[test.msno=="oECkzJik4wKsbOEVY6UACLbmgM8qymFdb5cJaHrodY8="]


# #### We have at least 1 mystery user for whom we have no data.
# * this causes our test data to be stored using mostly floats instead of ints (due to nan handling). 

# In[ ]:


test.shape


# In[ ]:


# test.to_numeric().shape


# ## Parse as DateTime
# 
# * __Note__ that many types are doubles in test, but not in train (the last row in test in missing many vals)

# In[ ]:


dateCols = ["registration_init_time","transaction_date", "membership_expire_date","expiration_date","date"]


# In[ ]:


# train.registration_init_time = pd.to_datetime(train.registration_init_time.astype(int),format="%Y%m%d")
# test.registration_init_time = pd.to_datetime(test.registration_init_time.astype(int),format="%Y%m%d")

# train.transaction_date = pd.to_datetime(train.transaction_date.astype(int),format="%Y%m%d")
# test.transaction_date = pd.to_datetime(test.transaction_date.astype(int),format="%Y%m%d")

# train.membership_expire_date = pd.to_datetime(train.membership_expire_date.astype(int),format="%Y%m%d")
# test.membership_expire_date = pd.to_datetime(test.membership_expire_date.astype(int),format="%Y%m%d")


# In[ ]:


train.registration_init_time = pd.to_datetime(train.registration_init_time,format="%Y%m%d")
test.registration_init_time = pd.to_datetime(test.registration_init_time,format="%Y%m%d")

train.transaction_date = pd.to_datetime(train.transaction_date,format="%Y%m%d")
test.transaction_date = pd.to_datetime(test.transaction_date,format="%Y%m%d")

train.membership_expire_date = pd.to_datetime(train.membership_expire_date,format="%Y%m%d")
test.membership_expire_date = pd.to_datetime(test.membership_expire_date,format="%Y%m%d")

train.expiration_date = pd.to_datetime(train.expiration_date,format="%Y%m%d")
test.expiration_date = pd.to_datetime(test.expiration_date,format="%Y%m%d")

train.date = pd.to_datetime(train.date,format="%Y%m%d")
test.date = pd.to_datetime(test.date,format="%Y%m%d")


# In[ ]:


train[dateCols+["is_churn"]].head()


# In[ ]:


train["sum_nan"] = train.isnull().sum(axis=1)
# train["sum_nan"].describe()
test["sum_nan"] = test.isnull().sum(axis=1)
test["sum_nan"].describe()


# ### Add DateTime Features

# In[ ]:


def get_date_diffs(df):
    """
    Get time between the expiry date and other columns in days + add day of week, month features.
     - Could add more, e.g. time between other dates, is weekend, etc' .
     membership_expire_date is the deciding date for churn determination.
    """
    df["exp-registration-diff"] = (df.membership_expire_date - df.registration_init_time ).dt.days
    df["exp-transaction-diff"] = (df.membership_expire_date - df.transaction_date ).dt.days
    df["exp-expiration-diff"] = (df.membership_expire_date - df.expiration_date ).dt.days
    df["exp-logdate-diff"] = (df.membership_expire_date - df["date"] ).dt.days
    
    for col in dateCols:
        df["dayOfWeek_%s" %(col)] = df[col].dt.dayofweek
        df["dayOfMonth_%s" %(col)] = df[col].dt.day
    
    df["payment_plan_days_div-exp-expiration-diff"] = df.payment_plan_days / df["exp-expiration-diff"]
    df["payment_plan_days_div-exp-transaction-diff"] = df.payment_plan_days / df["exp-transaction-diff"]
    df["payment_plan_days_div-eexp-logdate-diff"] = df.payment_plan_days / df["exp-logdate-diff"]


# In[ ]:


print(train.shape)
get_date_diffs(train)
print(train.shape)


# In[ ]:


print(test.shape)
get_date_diffs(test)
print(test.shape)


# In[ ]:


set(train.columns)


# #### Feature: total amount of songs played, vs number of unique songs
#  * Could add more : e.g. number of 98.5+100 % / 25% played

# In[ ]:


train["played_songs_nonUnique_ratio"] = (train['num_100'] + train['num_25'] + train['num_50'] + train['num_75'] + train['num_985'])/train["num_unq"]
test["played_songs_nonUnique_ratio"] = (test['num_100'] + test['num_25'] + test['num_50'] + test['num_75'] + test['num_985'])/test["num_unq"]

train["played_songs_nonUnique_ratio"].describe()


# ### NaN imputing 
# * try to save as int after!

# In[ ]:


train.isnull().sum()


# In[ ]:


train.columns[~train.isnull().any()].tolist()


# In[ ]:


test.isnull().sum()


# ### fill na
# * Remove nans , downcast to int for better datatype consistency. 
# * may mess up features!!

# In[ ]:


### Ma ymess up date columns or other features.
## Could help with test vals having different types.. 
train = train.fillna(0,downcast="infer")
test = test.fillna(0,downcast="infer")


# In[ ]:


train["price_paid_diff"]  = train.plan_list_price -  train.actual_amount_paid
test["price_paid_diff"]  = test.plan_list_price -  test.actual_amount_paid


# In[ ]:


test.info()


# In[ ]:


len(set(train.columns) - set(test.columns) )


# In[ ]:


test["is_churn"]= 0


# # Adversarial validation: predict if train/test:
# 
# * https://www.kaggle.com/nlothian/adversarial-validation
# * http://fastml.com/adversarial-validation-part-two/
#     * https://github.com/zygmuntz/adversarial-validation/blob/master/numerai/sort_train.py
#     
#     
# * Currently doesn't work with sklearn - errors with inf/nans (downcasting didn't help, and data doesn't display nans). Likely downcasting related. 
# 
# * LAter: Add also isolation forest features! 

# In[ ]:


train["is_test"] = 0
test["is_test"] = 1


# In[ ]:


df = train.append(test)
df.drop("is_churn",axis=1,inplace=True)
df.shape


# #### Get only numeric columns
# * note that we drop duplicates and the ID column. 
# This will be for model training + predicting on the real data only!

# In[ ]:


df = df.select_dtypes(include=[np.number])
df.reset_index( inplace = True, drop = True )  # may not be needed


# In[ ]:


df.drop_duplicates(inplace=True)
print(df.shape)
df["is_test"].describe()


# In[ ]:


df.isnull().sum(axis=0)


# In[ ]:





# In[ ]:


# x = df.drop( ['is_test'], axis = 1 ).astype("float64")
# y = df.is_test.astype(int)


from sklearn import cross_validation as CV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy

from time import ctime

n_estimators = 100
clf = RF( n_estimators = n_estimators, n_jobs = 1 )

predictions = np.zeros( y.shape )

# cv = CV.StratifiedKFold( y, n_folds = 4, shuffle = True, random_state = 5678 )

# for f, ( train_i, test_i ) in enumerate( cv ):
#     print(f,  train_i, test_i )
#     print ("# fold {}, {}".format( f + 1, ctime()))

#     x_train = x.iloc[train_i]
#     x_test = x.iloc[test_i]
#     y_train = y.iloc[train_i]
#     y_test = y.iloc[test_i]

#     clf.fit( x_train, y_train )	

#     p = clf.predict_proba( x_test )[:,1]

#     auc = AUC( y_test, p )
#     print ("# AUC: {:.2%}\n".format( auc ))	

#     predictions[ test_i ] = p


# In[ ]:


# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# lr = LR()
# lr.fit(X_train.values, y_train.values)
# print (classification_report(lr.predict(X_test.values), y_test.values))


# ## Save data to disk

# In[ ]:


train.to_csv("kkbox_churn_v21.csv.gz",index=False,compression="gzip")


# In[ ]:


test.to_csv("test-kkbox_churn_v21.csv.gz",index=False,compression="gzip")


# In[ ]:




# cols = [c for c in train.columns if c not in ['is_churn','msno']]


# In[ ]:


# def xgb_score(preds, dtrain):
#     labels = dtrain.get_label()
#     return 'log_loss', metrics.log_loss(labels, preds)

# fold = 1
# for i in range(fold):
#     params = {
#         'eta': 0.02, #use 0.002
#         'max_depth': 7,
#         'objective': 'binary:logistic',
#         'eval_metric': 'logloss',
#         'seed': i,
#         'silent': True
#     }
#     x1, x2, y1, y2 = model_selection.train_test_split(train[cols], train['is_churn'], test_size=0.3, random_state=i)
#     watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
#     model = xgb.train(params, xgb.DMatrix(x1, y1), 150,  watchlist, feval=xgb_score, maximize=False, verbose_eval=50, early_stopping_rounds=50) #use 1500
#     if i != 0:
#         pred += model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
#     else:
#         pred = model.predict(xgb.DMatrix(test[cols]), ntree_limit=model.best_ntree_limit)
# pred /= fold
# test['is_churn'] = pred.clip(0.0000001, 0.999999)
# test[['msno','is_churn']].to_csv('submission3.csv.gz', index=False, compression='gzip')


# In[ ]:


# import matplotlib.pyplot as plt
# import seaborn as sns
# %matplotlib inline

# plt.rcParams['figure.figsize'] = (7.0, 7.0)
# xgb.plot_importance(booster=model); plt.show()

