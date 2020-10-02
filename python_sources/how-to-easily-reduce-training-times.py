#!/usr/bin/env python
# coding: utf-8

# ## How to Reduce Training Times
# 
# This notebook takes the code EDA and LightGBM training from the following notebook:
# - https://www.kaggle.com/yhn112/data-exploration-lightgbm-catboost-lb-3-760
# 
# After using the above code we optimise the pandas dataframe and then rerun the prediction and see the change in training time.  
# This work will show you that by using just a couple of simple functions, you can reduce your training time.   When dataframes become very large this may save you many hours 
# 
# For further info check out https://www.kaggle.com/richarde/random-forest-with-50-reduction-in-training-time
# 
# Go down to **DataFrame Optimisation** header to see the added work
# 

# ### Simple Exploration

# In[ ]:


# Loading packages
import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import gc


# In[ ]:


import os
print(os.listdir("../input"))


# - train.csv - the training set
# - test.csv - the test set
# - sample_submission.csv - a sample submission file in the correct format - contains all card_ids you are expected to predict for.
# - historical_transactions.csv - up to 3 months' worth of historical transactions for each card_id
# - merchants.csv - additional information about all merchants / merchant_ids in the dataset.
# - new_merchant_transactions.csv - two months' worth of data for each card_id containing ALL purchases that card_id made at merchant_ids that were not visited in the historical data.

# In[ ]:


train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
print("shape of train : ",train.shape)


# In[ ]:


train.head()


# - first_active_month : ''YYYY-MM', month of first purchase
# - feature_1,2,3 : Anonymized card categorical feature
# - target : Loyalty numerical score calculated 2 months after historical and evaluation period

# In[ ]:


test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
print("shape of test : ",test.shape)


# First_active_month of Train and Test looks similiar

# `feature_1 - feature_3` has 0.58 but `target - feature` low correaltion value

# feature_3 has 1 when feautre_1 high than 3

# feature_2 has not 3 when feature_1 == 5
# but what is target low than -30???

# ### Missing value

# In[ ]:


# checking missing data
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


# checking missing data
total = test.isnull().sum().sort_values(ascending = False)
percent = (test.isnull().sum()/test.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


import datetime

for df in [train,test]:
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['year'] = df['first_active_month'].dt.year
    df['month'] = df['first_active_month'].dt.month
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days

target = train['target']
del train['target']


# In[ ]:


train.head()


# ### Simple Exploration : historical_transactions

# In[ ]:


ht = pd.read_csv("../input/historical_transactions.csv")
print("shape of historical_transactions : ",ht.shape)


# In[ ]:


ht.head()


# - card_id	: Card identifier
# - month_lag	: month lag to reference date
# - purchase_date	: Purchase date
# - authorized_flag	: Y' if approved, 'N' if denied
# - category_3	: anonymized category
# - installments	: number of installments of purchase
# - category_1 : 	anonymized category
# - merchant_category_id	: Merchant category identifier (anonymized )
# - subsector_id	: Merchant category group identifier (anonymized )
# - merchant_id	: Merchant identifier (anonymized)
# - purchase_amount	: Normalized purchase amount
# - city_id	: City identifier (anonymized )
# - state_id	: State identifier (anonymized )
# - category_2 : anonymized category

# In[ ]:


ht['authorized_flag'] = ht['authorized_flag'].map({'Y':1, 'N':0})


# In[ ]:


def aggregate_historical_transactions(history):
    
    history.loc[:, 'purchase_date'] = pd.DatetimeIndex(history['purchase_date']).                                      astype(np.int64) * 1e-9
    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'purchase_date': [np.ptp],
        'month_lag': ['min', 'max']
        }
    agg_history = history.groupby(['card_id']).agg(agg_func)
    agg_history.columns = ['hist_' + '_'.join(col).strip() 
                           for col in agg_history.columns.values]
    agg_history.reset_index(inplace=True)
    
    df = (history.groupby('card_id')
          .size()
          .reset_index(name='hist_transactions_count'))
    
    agg_history = pd.merge(df, agg_history, on='card_id', how='left')
    
    return agg_history

history = aggregate_historical_transactions(ht)
del ht
gc.collect()


# In[ ]:


train = pd.merge(train, history, on='card_id', how='left')
test = pd.merge(test, history, on='card_id', how='left')


# ### Simple Exploration : merchants.csv

# In[ ]:


merchant = pd.read_csv("../input/merchants.csv")
print("shape of merchant : ",merchant.shape)


# In[ ]:


merchant.head()


# - merchant_id : Unique merchant identifier
# - merchant_group_id	: Merchant group (anonymized )
# - merchant_category_id	: Unique identifier for merchant category (anonymized )
# - subsector_id	: Merchant category group (anonymized )
# - numerical_1	: anonymized measure
# - numerical_2	: anonymized measure
# - category_1	: anonymized category
# - most_recent_sales_range	: Range of revenue (monetary units) in last active month --> A > B > C > D > E
# - most_recent_purchases_range	: Range of quantity of transactions in last active month --> A > B > C > D > E
# - avg_sales_lag3	: Monthly average of revenue in last 3 months divided by revenue in last active month
# - avg_purchases_lag3	: Monthly average of transactions in last 3 months divided by transactions in last active month
# - active_months_lag3	: Quantity of active months within last 3 months
# - avg_sales_lag6	: Monthly average of revenue in last 6 months divided by revenue in last active month
# - avg_purchases_lag6	: Monthly average of transactions in last 6 months divided by transactions in last active month
# - active_months_lag6	: Quantity of active months within last 6 months
# - avg_sales_lag12	: Monthly average of revenue in last 12 months divided by revenue in last active month
# - avg_purchases_lag12	: Monthly average of transactions in last 12 months divided by transactions in last active month
# - active_months_lag12	: Quantity of active months within last 12 months
# - category_4	: anonymized category
# - city_id	: City identifier (anonymized )
# - state_id	: State identifier (anonymized )
# - category_2	: anonymized category

# In[ ]:


# checking missing data
total = merchant.isnull().sum().sort_values(ascending = False)
percent = (merchant.isnull().sum()/merchant.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# ### Simple Exploration : new_merchants.csv

# In[ ]:


new_merchant = pd.read_csv("../input/new_merchant_transactions.csv")
print("shape of new_merchant_transactions : ",new_merchant.shape)


# In[ ]:


new_merchant.head()


# - card_id	: Card identifier
# - month_lag	: month lag to reference date
# - purchase_date	: Purchase date
# - authorized_flag	: Y' if approved, 'N' if denied
# - category_3	: anonymized category
# - installments	: number of installments of purchase
# - category_1	: anonymized category
# - merchant_category_id	: Merchant category identifier (anonymized )
# - subsector_id	: Merchant category group identifier (anonymized )
# - merchant_id	: Merchant identifier (anonymized)
# - purchase_amount	: Normalized purchase amount
# - city_id	: City identifier (anonymized )
# - state_id	: State identifier (anonymized )
# - category_2	: anonymized category
# 

# In[ ]:


# checking missing data
total = new_merchant.isnull().sum().sort_values(ascending = False)
percent = (new_merchant.isnull().sum()/new_merchant.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)


# In[ ]:


new_merchant['authorized_flag'] = new_merchant['authorized_flag'].map({'Y':1, 'N':0})


# In[ ]:


def aggregate_new_transactions(new_trans):    
    agg_func = {
        'authorized_flag': ['sum', 'mean'],
        'merchant_id': ['nunique'],
        'city_id': ['nunique'],
        'purchase_amount': ['sum', 'median', 'max', 'min', 'std'],
        'installments': ['sum', 'median', 'max', 'min', 'std'],
        'month_lag': ['min', 'max']
        }
    agg_new_trans = new_trans.groupby(['card_id']).agg(agg_func)
    agg_new_trans.columns = ['new_' + '_'.join(col).strip() 
                           for col in agg_new_trans.columns.values]
    agg_new_trans.reset_index(inplace=True)
    
    df = (new_trans.groupby('card_id')
          .size()
          .reset_index(name='new_transactions_count'))
    
    agg_new_trans = pd.merge(df, agg_new_trans, on='card_id', how='left')
    
    return agg_new_trans

new_trans = aggregate_new_transactions(new_merchant)


# In[ ]:


train = pd.merge(train, new_trans, on='card_id', how='left')
test = pd.merge(test, new_trans, on='card_id', how='left')


# In[ ]:


train.info(verbose=False)


# ### Make a Baseline model
# kernel : https://www.kaggle.com/youhanlee/hello-elo-ensemble-will-help-you

# In[ ]:


use_cols = [col for col in train.columns if col not in ['card_id', 'first_active_month']]

train = train[use_cols]
test = test[use_cols]

features = list(train[use_cols].columns)
categorical_feats = [col for col in features if 'feature_' in col]


# In[ ]:


for col in categorical_feats:
    print(col, 'have', train[col].value_counts().shape[0], 'categories.')


# In[ ]:


from sklearn.preprocessing import LabelEncoder
for col in categorical_feats:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))


# In[ ]:


df_all = pd.concat([train, test])
df_all = pd.get_dummies(df_all, columns=categorical_feats)

len_train = train.shape[0]

train = df_all[:len_train]
test = df_all[len_train:]


# ## DataFrame Optimisation
# **Lets look at the dataframe**

# In[ ]:


train.info(verbose=False)


# **There are several datatypes, below we convert to smaller types**

# In[ ]:


def mem_usage(pandas_obj):
    usage_b = pandas_obj.memory_usage(deep=True).sum()
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

train_mem_int = train.select_dtypes(include=['int'])
converted_int = train_mem_int.apply(pd.to_numeric,downcast='unsigned')

print("Size of integer types before {}".format(mem_usage(train_mem_int)))
print("Size of integer types after {}".format(mem_usage(converted_int)))

compare_ints = pd.concat([train_mem_int.dtypes,converted_int.dtypes],axis=1)
compare_ints.columns = ['No. of types before','No. of types after']
print(compare_ints.apply(pd.Series.value_counts))

train_col_int = list(train_mem_int.columns)
print(train_col_int)


# In[ ]:


train_mem_float = train.select_dtypes(include=['float'])
converted_float = train_mem_float.apply(pd.to_numeric,downcast='float')

print("Size of float types before: {}".format(mem_usage(train_mem_float)))
print("Size of float types after: {}".format(mem_usage(converted_float)))

compare_floats = pd.concat([train_mem_float.dtypes,converted_float.dtypes],axis=1)
compare_floats.columns = ['No. of types before','No. of types after']
print(compare_floats.apply(pd.Series.value_counts))

print(" ")

train_col_float = list(train_mem_float.columns)
print(train_col_float)


# **From above we have reduced the sizes of our int and float datatypes and reduced the size considerably.  We will run lightgbm with the original types then rerun with these downcast datatypes.**

# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import time


lgb_params = {"objective" : "regression", "metric" : "rmse", 
               "max_depth": 11, "min_child_samples": 20, 
               "reg_alpha": 1, "reg_lambda": 1,
               "num_leaves" : 128, "learning_rate" : 0.005, 
               "subsample" : 0.8, "colsample_bytree" : 0.8, 
               "verbosity": -1}

FOLDs = KFold(n_splits=8, shuffle=True, random_state=1989)

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

features_lgb = list(train.columns)
feature_importance_df_lgb = pd.DataFrame()

start = time.time()

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train)):
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])

    print("LGB " + str(fold_) + "-" * 50)
    num_round = 1000
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval = 400)
    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(test, num_iteration=clf.best_iteration) / FOLDs.n_splits
    
print(np.sqrt(mean_squared_error(oof_lgb, target)))
end = time.time()
print('Time taken in predictions: {0: .2f}'.format(end - start))
#time_first = end - start


# **Note the time taken above and final rmse score**
# 
# **We have run lightgbm with the original datatypes.  Below we change to the optimised types**

# In[ ]:


#replace the floats with smaller datatypes
columns_to_overwrite_float = train_col_float 
train.drop(labels=columns_to_overwrite_float, axis="columns", inplace=True)
train[columns_to_overwrite_float] = converted_float[columns_to_overwrite_float]


# In[ ]:


#replace the floats with smaller datatypes
columns_to_overwrite_int = train_col_int 
train.drop(labels=columns_to_overwrite_int, axis="columns", inplace=True)
train[columns_to_overwrite_int] = converted_int[columns_to_overwrite_int]


# In[ ]:


train.info(verbose=False)


# **Have reduced size of dataframe from 62MB to 33MB**
# 
# **Now rerun with optimised datatypes**

# In[ ]:


FOLDs = KFold(n_splits=8, shuffle=True, random_state=1989)

oof_lgb = np.zeros(len(train))
predictions_lgb = np.zeros(len(test))

features_lgb = list(train.columns)
feature_importance_df_lgb = pd.DataFrame()

start = time.time()

for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(train)):
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target.iloc[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx], label=target.iloc[val_idx])

    print("LGB " + str(fold_) + "-" * 50)
    num_round = 1000
    clf = lgb.train(lgb_params, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval = 400)
    oof_lgb[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)
    predictions_lgb += clf.predict(test, num_iteration=clf.best_iteration) / FOLDs.n_splits
    
print(np.sqrt(mean_squared_error(oof_lgb, target)))
end = time.time()
print('Time taken in predictions: {0: .2f}'.format(end - start))
#time_first = end - start


# **In this case the time saving has been quite small but as the size of the dataframe increases, applying these few simple tricks could save considerable time**

# In[ ]:




