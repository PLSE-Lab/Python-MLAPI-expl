#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering + Stacking lgb
# **Nguyen Dang Minh, PhD**
# 
# * [General functions and parameters](#general_function)
# * [Handling merchant data](#merchant_data)
# * [Handling transaction data](#transaction_data)
# * [Handling training and testing data](#training_data)
# * [Modeling](#modeling)

# ***If you are looking for high accuracy submission, you've come to the wrong place!!! There are lots of other kernels out there with much higher leaderboard score. ***
# 
# ***But if you enjoy exploring, playing with data and try out different ideas, then c'mon in!!!***

# ## Summary
# In this note book I will perform feature engineering and stacking ensemble on the [Elo merchant category recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation). Some part of the codes in this notebook is taken from [this excellent notebook](https://www.kaggle.com/fabiendaniel/elo-world) of [FabienDaniel](https://www.kaggle.com/fabiendaniel)
# 
# ### General model structure ###
# There are two layers. The first layer has:
# * 2 lightgbm
# * 1 xgboost
# * 1 catboost
# * 1 dense neural network
# 
# The result of the first layer is fitted to a Lars Regression to give final prediction
# 
# ### Some experience after several trials
# * Merging news and historical transaction data does not affect the result.
# * Separate transaction data into authorized and un-authorized transaction does help, but very little
# * The root-mean-squared error metrics is very sensitive to extreme case. Thus, in this problem, handling outliers (either by clipping or removing them) make the performance worse
# * Discretizing continuous variables (either by decisition tree or binning) make the performance worse
# * The most significant feature is how recent the customer makes the purchase.
# * Adding some knowledge-based feature such as: weeks before christmast, weekend frequency, etc. does help

# In[ ]:


import numpy as np
import pandas as pd
import datetime
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, LarsCV, RidgeCV, Lars
import warnings
import random
import datetime
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import catboost as cb
import scipy
from sklearn.cluster import DBSCAN
from pandas.api.types import is_numeric_dtype


from keras import Sequential
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

warnings.filterwarnings('ignore')
np.random.seed(1)
random.seed(1)
import time
import gc


# In[ ]:


from matplotlib import rcParams
rcParams['figure.figsize'] = (8,4)
rcParams['font.size'] = 12


# <a id='general_function'></a>
# 
# ## General functions and parameters

# In[ ]:


DEBUG = False
REF_DATE = datetime.datetime.strptime('2018-12-31', '%Y-%m-%d')


# In[ ]:


def skip_func(i, p=0.1, debug=DEBUG):
    if debug == True:
        return (i>0 and random.random()>p)
    else:
        return False


# In[ ]:


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


# In[ ]:


def print_null(df):
    for col in df:
        if df[col].isnull().any():
            print('%s has %.0f null values: %.3f%%'%(col, df[col].isnull().sum(), df[col].isnull().sum()/df[col].count()*100))


# In[ ]:


def impute_na(X_train, df, variable):
    # make temporary df copy
    temp = df.copy()
    
    # extract random from train set to fill the na
    random_sample = X_train[variable].dropna().sample(temp[variable].isnull().sum(), random_state=0, replace=True)
    
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = temp[temp[variable].isnull()].index
    temp.loc[temp[variable].isnull(), variable] = random_sample
    return temp[variable]


# In[ ]:


# Clipping outliers
def clipping_outliers(X_train, df, var):
    IQR = X_train[var].quantile(0.75)-X_train[var].quantile(0.25)
    lower_bound = X_train[var].quantile(0.25) - 6*IQR
    upper_bound = X_train[var].quantile(0.75) + 6*IQR
    no_outliers = len(df[df[var]>upper_bound]) + len(df[df[var]<lower_bound])
    print('There are %i outliers in %s: %.3f%%' %(no_outliers, var, no_outliers/len(df)))
    df[var] = df[var].clip(lower_bound, upper_bound)
    return df


# <a id='merchant_data'></a>
# 
# ## Merchant data
# ### Import and overview

# In[ ]:


df_merchants = pd.read_csv('../input/merchants.csv', 
                            skiprows=lambda i: skip_func(i,p=1))


# In[ ]:


df_merchants.head()


# In[ ]:


print('Merchant data types')
df_merchants.dtypes


# In[ ]:


df_merchants = df_merchants.replace([np.inf,-np.inf], np.nan)
print('Merchants null')
print_null(df_merchants)


# In[ ]:


print('Merchants unique values')
df_merchants[['merchant_id','merchant_group_id','merchant_category_id','subsector_id','category_1','most_recent_sales_range','most_recent_purchases_range','merchant_category_id',
         'active_months_lag3','active_months_lag6','category_4','category_2']].nunique()


# ### Preprocessing
# 
# #### Filling null

# Fill `ave_sales` with most frequent values. Fill `category_2` with random sampling from available data
# 

# In[ ]:


# Average sales null
null_cols = ['avg_purchases_lag3','avg_sales_lag3', 'avg_purchases_lag6','avg_sales_lag6','avg_purchases_lag12','avg_sales_lag12']
for col in null_cols:
    df_merchants[col] = df_merchants[col].fillna(df_merchants[col].mean())

# Category 2
df_merchants['category_2'] = impute_na(df_merchants, df_merchants, 'category_2')


# #### Discretize and mapping data
# 
# All `avg_sales` and `avg_purchases` data is discretized into 5 categories, following the 5 categories of most recent values

# In[ ]:


# Sales cut
sales_cut = df_merchants['most_recent_sales_range'].value_counts().sort_values(ascending=False).values
sales_cut = sales_cut/np.sum(sales_cut)
for i in range(1,len(sales_cut)):
    sales_cut[i] = sales_cut[i]+sales_cut[i-1]
    
# Purchases cut
purchases_cut = df_merchants['most_recent_purchases_range'].value_counts().sort_values(ascending=False).values
purchases_cut = purchases_cut/np.sum(purchases_cut)
for i in range(1,len(purchases_cut)):
    purchases_cut[i] = purchases_cut[i]+purchases_cut[i-1]


# In[ ]:


# Discretize data
discretize_cols = ['avg_purchases_lag3','avg_sales_lag3', 'avg_purchases_lag6','avg_sales_lag6','avg_purchases_lag12','avg_sales_lag12']

for col in discretize_cols:
    categories = pd.qcut(df_merchants[col].values,sales_cut, duplicates='raise').categories.format()
    df_merchants[col], intervals = pd.qcut(df_merchants[col], 5, labels=['A','B','C','D','E'], retbins=True, duplicates='raise')
    print('Discretize for %s:'%col)
    print(categories)


# In[ ]:


# Mapping data
df_merchants['category_1'] = df_merchants['category_1'].map({'Y':1, 'N':0})
df_merchants['category_4'] = df_merchants['category_4'].map({'Y':1, 'N':0})

map_cols = discretize_cols + ['most_recent_purchases_range', 'most_recent_sales_range']
for col in map_cols:
    df_merchants[col] = df_merchants[col].map({'A':5,'B':4,'C':3,'D':2,'E':1})


# In[ ]:


numeric_cols = ['numerical_1','numerical_2']+map_cols

colormap = plt.cm.RdBu
plt.figure(figsize=(12,12))
sns.heatmap(df_merchants[numeric_cols].astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1., square=True, cmap=colormap, linecolor='white', annot=True)
plt.title('Pair-wise correlation')


# #### Handling numerical data

# In[ ]:


numerical_cols = ['numerical_1','numerical_2']
for col in numerical_cols:
    df_merchants = clipping_outliers(df_merchants, df_merchants, col)
    plt.figure()
    sns.distplot(df_merchants[col])
print('Unique values:')
print(df_merchants[numerical_cols].nunique())


# After clipping outliers, there are only 5 uniques values left in these two columns. Thus, we map them into 3 categories: the lowest: `0`, the middle: `1`, and the extreme: `2`

# In[ ]:


for col in numerical_cols:
    b = df_merchants[col].unique()
    df_merchants[col] = df_merchants[col].apply(lambda x: 0 if x==b[0] else (1 if x in b[1:4] else 2))


# In[ ]:


df_merchants = reduce_mem_usage(df_merchants)


# In[ ]:


# Rename col
for col in df_merchants.columns:
    if col != 'merchant_id':
        df_merchants = df_merchants.rename(index=str, columns={col:'mer_'+col})


# <a id='transaction_data'></a>
# 
# ## Transaction data

# ### Import and merge transaction data

# In[ ]:


df_hist_trans = pd.read_csv('../input/historical_transactions.csv', 
                            skiprows=lambda i: skip_func(i), parse_dates=['purchase_date'])
df_new_trans = pd.read_csv('../input/new_merchant_transactions.csv', 
                           skiprows=lambda i: skip_func(i), parse_dates=['purchase_date'])


# In[ ]:


df_hist_trans['days_to_date'] = ((REF_DATE - df_hist_trans['purchase_date']).dt.days)
df_hist_trans['days_to_date'] = df_hist_trans['days_to_date'] #+ df_hist_trans['month_lag']*30
df_new_trans['days_to_date'] = ((REF_DATE - df_new_trans['purchase_date']).dt.days)#//30
df_trans = pd.concat([df_hist_trans, df_new_trans])
df_trans['months_to_date'] = df_trans['days_to_date']//30
df_trans = df_trans.drop(columns=['days_to_date'])

if DEBUG == False:
    del df_hist_trans, df_new_trans
    gc.collect()

df_trans = reduce_mem_usage(df_trans)
#df_trans = df_trans.sort_values(by=['purchase_date']).reset_index(drop=True)


# In[ ]:


df_trans.head()


# In[ ]:


# Merge with merchant data
df_trans = pd.merge(df_trans, df_merchants, how='left', left_on='merchant_id', right_on='merchant_id')

if DEBUG == False:
    del df_merchants
    gc.collect()


# In[ ]:


df_trans.head()


# In[ ]:


for col in df_trans.columns:
    if df_trans[col].nunique()<=15:
        plt.figure()
        sns.countplot(df_trans[col])


# In[ ]:


print('Null ratio')
print_null(df_trans)


# In[ ]:


#print('Unique values')
#df_trans[['card_id','city_id','category_1','city_id','category_1','installments','category_3','merchant_id','merchant_category_id',
         #'month_lag','category_2','state_id','subsector_id','days_to_date']].nunique()


# Some columns are duplicated and can be dropped

# In[ ]:


# Drop duplicate columns
df_trans = reduce_mem_usage(df_trans)
df_trans = df_trans.drop(columns=['mer_city_id', 'mer_state_id', 'mer_category_1', 'mer_category_2',
                          'mer_merchant_category_id','mer_subsector_id'])


# ### Filling null and encoding

# In[ ]:


# Fill null by most frequent data
df_trans['category_2'].fillna(1.0,inplace=True)
df_trans['category_3'].fillna('A',inplace=True)
df_trans['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)

# Fill null by random sampling
nan_cols = df_trans.columns[df_trans.isna().any()].tolist()
for col in nan_cols:
    df_trans[col] = impute_na(df_trans, df_trans, col)


# In[ ]:


# Encoding
df_trans['authorized_flag'] = df_trans['authorized_flag'].map({'Y':1,'N':0})
df_trans['category_1'] = df_trans['category_1'].map({'Y':1,'N':0})
dummies = pd.get_dummies(df_trans[['category_2', 'category_3']], prefix = ['cat_2','cat_3'], columns=['category_2','category_3'])
df_trans = pd.concat([df_trans, dummies], axis=1)


# In[ ]:


df_trans.head()
df_trans = reduce_mem_usage(df_trans)


# ### Knowledge-based features
# 
# * Weekend or not
# * Hour of the day: categorize into Morning (5 to 12), Afternoon (12 to 17), Evening (17 to 22) and Night (22 to 5) 
# * Day of month: categorize into Early (<10), Middle (>10 and <20) and Late (>20)
# * Time to christmas 2017 and time to Black Friday 2017 (purchase amount increase significantly around this time)

# In[ ]:


df_trans['weekend'] = (df_trans['purchase_date'].dt.weekday >=5).astype(int)
df_trans['hour'] = df_trans['purchase_date'].dt.hour
df_trans['day'] = df_trans['purchase_date'].dt.day
df_trans['weeks_to_Xmas_2017'] = ((pd.to_datetime('2017-12-25') - df_trans['purchase_date']).dt.days//7).apply(lambda x: x if x>=0 and x<=8 else 8)
df_trans['weeks_to_BFriday'] = ((pd.to_datetime('2017-11-25') - df_trans['purchase_date']).dt.days//7).apply(lambda x: x if x>=0 and x<=3 else 3)


# In[ ]:


# Categorize time
def get_session(hour):
    hour = int(hour)
    if hour > 4 and hour < 12:
        return 0
    elif hour >= 12 and hour < 17:
        return 1
    elif hour >= 17 and hour < 21:
        return 2
    else:
        return 3
    
df_trans['hour'] = df_trans['hour'].apply(lambda x: get_session(x))


# In[ ]:


# Categorize day
def get_day(day):
    if day <= 10:
        return 0
    elif day <=20:
        return 1
    else:
        return 2

df_trans['day'] = df_trans['day'].apply(lambda x: get_day(x))


# In[ ]:


'''
**Does authorize flag matter?**

In lots of other kernels, transaction data are splitted into two: authorized transaction and  un-authorized transaction. Let's see if there's a significant impact in doing that
    
There are no obvious difference between authorized and unauthorized transaction, thus we will NOT split transaction data into two separate sets.
'''


# In[ ]:


'''
# Categorical features
compare_cols = ['category_1', 'category_2', 'category_3', 'installments', 'mer_most_recent_sales_range',
               'mer_most_recent_purchases_range', 'mer_active_months_lag3', 'mer_active_months_lag6', 'mer_active_months_lag12',
               'mer_category_4', 'weekend','hour']
for col in compare_cols:
    fig = plt.figure()
    sns.countplot(x=col, hue='authorized_flag', data=df_trans)
'''


# In[ ]:


'''
# Numerical features
compare_cols = ['purchase_amount','months_to_date','mer_numerical_1','mer_numerical_2','mer_avg_sales_lag3',
               'mer_avg_purchases_lag3', 'mer_avg_sales_lag6', 'mer_avg_purchases_lag6', 'mer_avg_sales_lag12',
               'mer_avg_purchases_lag12']
for col in compare_cols:
    fig = plt.figure()
    temp_authorized = df_trans[col][df_trans['authorized_flag']==1]
    temp_unauthorized = df_trans[col][df_trans['authorized_flag']==0]
    sns.kdeplot(data=np.log(temp_unauthorized), label='unauthorized')
    sns.kdeplot(data=np.log(temp_authorized), label='authorized')
    plt.title('log-scale '+col)
    
if DEBUG==False:
    del temp_authorized,temp_unauthorized
    gc.collect()
'''


# ### Aggregating features

# In[ ]:


def most_frequent(agg_df, df, col):
    temp = df.groupby('card_id')[col].value_counts().index
    agg_df


# In[ ]:


def most_frequent(x):
    return x.value_counts().index[0]


# In[ ]:


def aggregate_trans(df):
    agg_func = {
        'authorized_flag': ['mean', 'std'],
        'category_1': ['mean'],
        'cat_2_1.0': ['mean'],
        'cat_2_2.0': ['mean'],
        'cat_2_3.0': ['mean'],
        'cat_2_4.0': ['mean'],
        'cat_2_5.0': ['mean'],
        'cat_3_A': ['mean'],
        'cat_3_B': ['mean'],
        'cat_3_C': ['mean'],
        'mer_numerical_1':['nunique','mean','std'],
        'mer_most_recent_sales_range': ['mean','std'],
        'mer_most_recent_purchases_range': ['mean','std'],
        'mer_avg_sales_lag12':['mean','std'],
        'mer_avg_purchases_lag12':['mean','std'],
        'mer_active_months_lag12':['nunique'],
        'merchant_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'state_id': ['nunique'],
        'city_id': ['nunique'],
        'subsector_id': ['nunique'],
        'mer_merchant_group_id': ['nunique'],
        'installments': ['sum','mean', 'max', 'min', 'std'],
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'weekend': ['mean', 'std'],
        'hour': ['mean', 'std'],
        'day': ['mean', 'std'],
        'weeks_to_Xmas_2017': ['mean'],
        'weeks_to_BFriday': ['mean'],
        'purchase_date': ['count'],
        'months_to_date': ['mean', 'max', 'min', 'std']
    }
    #'mer_category_4': ['mean'],
    #'mer_avg_sales_lag6':['nunique', 'mean','std'],
    #'mer_avg_purchases_lag6':['nunique', 'mean','std'],
    #'months_to_date': ['mean', 'max', 'min', 'std'],
    agg_df = df.groupby(['card_id']).agg(agg_func)
    agg_df.columns = ['_'.join(col)for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)
    return agg_df


# In[ ]:


def aggregate_per_month(df):
    agg_func = {
        'purchase_amount': ['count', 'sum', 'mean', 'min', 'max'],
        'installments': ['sum', 'mean', 'min', 'max'],
        'merchant_id': ['nunique'],
        'state_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'subsector_id': ['nunique']
    }
    agg_df = df.groupby(['card_id','months_to_date']).agg(agg_func)
    agg_df.columns = ['_'.join(col)for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)
    for col in agg_df.columns:
        if col != 'card_id':
            agg_df = agg_df.rename(index=str, columns={col:'monthly_'+col})
    final_group = agg_df.groupby('card_id').agg(['mean', 'std'])
    final_group.columns = ['_'.join(col)for col in final_group.columns.values]
    final_group = final_group.drop(columns=['monthly_months_to_date_mean','monthly_months_to_date_std'])
    final_group.reset_index(inplace=True)
    return final_group


# #### Splitting authorize and unthorize data

# In[ ]:


# To split authorized and un-authorized data
auth_trans = df_trans.groupby('card_id')['authorized_flag'].mean().reset_index()
df_trans_auth = df_trans[df_trans['authorized_flag']==1]
df_trans_unauth = df_trans[df_trans['authorized_flag']==0]
if DEBUG==False:
    del df_trans
    gc.collect
 
# Aggregate
agg_df_auth = aggregate_trans(df_trans_auth)
agg_df_auth_permonth = aggregate_per_month(df_trans_auth)
agg_df_unauth = aggregate_trans(df_trans_unauth)
agg_df_unauth_permonth = aggregate_per_month(df_trans_unauth)

if DEBUG==False:
    del df_trans_auth, df_trans_unauth
    gc.collect
    
# Merging
agg_df_auth = pd.merge(agg_df_auth, agg_df_auth_permonth, how='left', on='card_id')
agg_df_auth = reduce_mem_usage(agg_df_auth)
agg_df_unauth = pd.merge(agg_df_unauth, agg_df_unauth_permonth, how='left', on='card_id')
agg_df_unauth = reduce_mem_usage(agg_df_unauth)

# Replace null
agg_df_auth = agg_df_auth.replace([np.inf,-np.inf], np.nan)
agg_df_auth = agg_df_auth.fillna(value=0)

agg_df_unauth = agg_df_unauth.replace([np.inf,-np.inf], np.nan)
agg_df_unauth = agg_df_unauth.fillna(value=0)
for col in agg_df_unauth.columns:
        if col != 'card_id':
            agg_df_unauth = agg_df_unauth.rename(index=str, columns={col:'unauthorized_'+col})


# In[ ]:


'''
# Use when not splitting authorize/unauthorize
agg_df = aggregate_trans(df_trans)
agg_df_permonth = aggregate_per_month(df_trans)
auth_trans = df_trans.groupby('card_id')['authorized_flag'].mean().reset_index()
if DEBUG==False:
    del df_trans
    gc.collect

agg_df = pd.merge(agg_df, agg_df_permonth, how='left', on='card_id')
agg_df = reduce_mem_usage(agg_df)

agg_df = agg_df.replace([np.inf,-np.inf], np.nan)
print_null(agg_df)
agg_df = agg_df.fillna(value=0)

'''


# <a id='traning_data'></a>
# 
# ## Training and testing data
# ### Import and visualize

# In[ ]:


df_train = pd.read_csv('../input/train.csv', 
                            skiprows=lambda i: skip_func(i,p=1),parse_dates=['first_active_month'])
df_test = pd.read_csv('../input/test.csv',parse_dates=['first_active_month'])


# In[ ]:


df_test['first_active_month'] = impute_na(df_test, df_train, 'first_active_month')


# In[ ]:


cat_cols = ['feature_1','feature_2','feature_3']
for col in cat_cols:
    fig = plt.figure()
    sns.countplot(df_train[col])
    plt.title(col)


# In[ ]:


num_cols = ['target']
for col in num_cols:
    fig = plt.figure()
    sns.boxplot(df_train[col])
    plt.title(col)


# ### Features engineering

# #### Detecting outliers
# 
# This code is taken from this [notebook](https://www.kaggle.com/chauhuynh/my-first-kernel-3-699) by [Chau Ngoc Huynh](https://www.kaggle.com/chauhuynh)

# In[ ]:


df_train['outliers'] = 0
df_train.loc[df_train['target'] < -30, 'outliers'] = 1
df_train['outliers'].value_counts()


# In[ ]:


for f in ['feature_1','feature_2','feature_3']:
    order_label = df_train.groupby([f])['outliers'].mean()
    df_train[f] = df_train[f].map(order_label)
    df_test[f] = df_test[f].map(order_label)


# In[ ]:


# Clip outliers
df_train['target'] = df_train['target'].clip(-30,20)


# #### Merging with merchant data

# In[ ]:


#df_train = pd.merge(df_train, agg_df, on='card_id', how='left')
df_train = pd.merge(df_train, agg_df_auth, on='card_id', how='left')
df_train = pd.merge(df_train, agg_df_unauth, on='card_id', how='left')
df_train = pd.merge(df_train, auth_trans, on='card_id', how='left')
df_train['active_months'] = ((REF_DATE - df_train['first_active_month']).dt.days)//30
df_train['month_start'] = df_train['first_active_month'].dt.month

#df_test = pd.merge(df_test, agg_df, on='card_id', how='left')
df_test = pd.merge(df_test, agg_df_auth, on='card_id', how='left')
df_test = pd.merge(df_test, agg_df_unauth, on='card_id', how='left')
df_test = pd.merge(df_test, auth_trans, on='card_id', how='left')
df_test['active_months'] = ((REF_DATE - df_test['first_active_month']).dt.days)//30
df_test['month_start'] = df_test['first_active_month'].dt.month

if DEBUG==False:
    del agg_df_auth, agg_df_unauth
    gc.collect


# In[ ]:


print_null(df_train)


# In[ ]:


#df_train = pd.get_dummies(df_train, prefix=['feat1','feat2'],columns=['feature_1','feature_2'])
#df_test = pd.get_dummies(df_test, prefix=['feat1','feat2'],columns=['feature_1','feature_2'])


# In[ ]:


# Get numerical var
numerical = [var for var in df_train.columns if df_train[var].dtype!='O']
print('There are {} numerical variables'.format(len(numerical)))

# Get discrete var
discrete = []
for var in numerical:
    if len(df_train[var].unique())<8:
        discrete.append(var)
        
print('There are {} discrete variables'.format(len(discrete)))

# Get continuous var
continuous = [var for var in numerical if var not in discrete and var not in ['card_id', 'first_active_month','target']]
print('There are {} continuous variables'.format(len(continuous)))


# #### Filling null
# 

# In[ ]:


print('Null analysis of training data')
print_null(df_train)


# Since null data is less than 10%, we apply the following strategies:
# * Continuous variables filled with 0
# * Categorical variables filled with random sampling

# In[ ]:


# Detect all null columns
train_null = df_train.columns[df_train.isnull().any()].tolist()
test_null = df_test.columns[df_test.isnull().any()].tolist()

in_first = set(train_null)
in_second = set(test_null)

in_second_but_not_in_first = in_second - in_first

null_cols = train_null + list(in_second_but_not_in_first)


# In[ ]:


# Filling null
for col in null_cols:
    if col in continuous:
        df_train[col] = df_train[col].fillna(0)#df_train[col].astype(float).mean())
        df_test[col] = df_test[col].fillna(0)#df_train[col].astype(float).mean())
    if col in discrete:
        df_train[col] = impute_na(df_train, df_train, col)
        df_test[col] = impute_na(df_test, df_train, col)


# #### Dealing with outliers

# In[ ]:


# Discretize continuous variable
def tree_binariser(X_train, X_test, var):
    score_ls = []

    for tree_depth in [1,2,3,4]:
        # call the model
        tree_model = DecisionTreeRegressor(max_depth=tree_depth)

        # train the model using 3 fold cross validation
        scores = cross_val_score(tree_model, X_train[var].to_frame(), X_train['target'], cv=5, scoring='neg_mean_squared_error')
        score_ls.append(np.mean(scores))

    # find depth with smallest mse
    depth = [1,2,3,4][np.argmax(score_ls)]
    #print(score_ls, np.argmax(score_ls), depth)

    # transform the variable using the tree
    tree_model = DecisionTreeRegressor(max_depth=depth)
    tree_model.fit(X_train[var].to_frame(), X_train['target'])
    X_train[var] = tree_model.predict(X_train[var].to_frame())
    #X_val[var] = tree_model.predict(X_val[var].to_frame())
    X_test[var] = tree_model.predict(X_test[var].to_frame())
    return X_train, X_test


# In[ ]:


print('Clipping outliers ...')
#df_train = clipping_outliers(df_train, df_train, 'target')
#for col in continuous:
    #df_train, df_test = tree_binariser(df_train, df_test, col)
    #df_test = clipping_outliers(df_train,df_test,col)
    #df_train = clipping_outliers(df_train,df_train,col)


# In[ ]:


# Scaling
features = [c for c in df_train.columns if c not in ['card_id', 'first_active_month', 'target', 'outliers']]
scaler = StandardScaler()
#df_train[features] = scaler.fit_transform(df_train[features])
#df_test[features] = scaler.transform(df_test[features])


# In[ ]:


if DEBUG==False:
    df_train.to_csv('Train_final.csv')
    df_test.to_csv('Test_final.csv')


# In[ ]:


df_train.head()


# <a id='modeling'></a>
# 
# ## Modeling
# 
# Here we use [out of fold stacking ensemble](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-for-ensemble-models/). The architecture is as followed:
# 
# **Layer 1**:
# * 2 lightgbm
# * 1 xgboost
# * 1 catboost
# * 1 dense neural network
# 
# **Layer 2**:
# * Lasso regression
# * Ridge regression
# 

# In[ ]:


df_train = df_train.reset_index(drop=True)
target = df_train['target']
train = df_train.drop(columns=['target'])
test = df_test
if DEBUG == False:
    del df_train, df_test
    gc.collect


# ### First layer
# #### Tree-based model

# In[ ]:


# List of model to use
if DEBUG == True:
    ITERATIONS = 1
else:
    ITERATIONS = 4000
lgb1 = lgb.LGBMRegressor(num_leaves=111,
                        max_depth=9,
                        learning_rate=0.005,
                        n_estimators=ITERATIONS,
                        min_child_samples=149,
                        subsample=0.71,
                        subsample_freq=1,
                        feature_fraction=0.75,
                        reg_lambda=0.26,
                        random_state=1,
                        n_jobs=4,
                        metrics='rmse')

lgb2 = lgb.LGBMRegressor(num_leaves=200,
                        max_depth=9,
                        learning_rate=0.01,
                        n_estimators=ITERATIONS,
                        min_child_samples=40,
                        subsample=0.9,
                        subsample_freq=2,
                        feature_fraction=0.8,
                        reg_lambda=0.13,
                        random_state=2,
                        n_jobs=4,
                        metrics='rmse')

xgb1 = xgb.XGBRegressor(max_depth=9,
                       learning_rate=0.005,
                       n_estimators=ITERATIONS,
                       colsample_bytree=0.75,
                       sub_sample=0.75,
                       reg_lambda=0.15,
                       n_jobs=4,
                       random_state=3)

cb1 = cb.CatBoostRegressor(iterations=ITERATIONS, learning_rate=0.007, loss_function='RMSE', bootstrap_type='Bernoulli', depth=9, rsm=0.75, subsample=0.75, random_seed=4, reg_lambda=3)

ada1 = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=8), n_estimators=ITERATIONS, learning_rate=0.007, loss='square', random_state=2019)


# In[ ]:


if DEBUG==True:
    N_FOLDS=2
else:
    N_FOLDS=5
layer1_models = [lgb1, lgb2, xgb1, cb1]#, ada1]
layer1_names = ['lightgbm1', 'lightgbm2', 'xgboost1', 'catboost1']#, 'adaboost1']


# In[ ]:


folds = KFold(n_splits=N_FOLDS, shuffle=False, random_state=2019)
oof_train = np.zeros(shape=(len(train),len(layer1_models)))
oof_test = np.zeros(shape=(len(test),len(layer1_models)))

# Recording results
layer1_score = []
feature_importance = []


# In[ ]:


for i in range(len(layer1_models)):
    feature_importance_df = pd.DataFrame()
    print('\n')
    name = layer1_names[i]
    model = layer1_models[i]
    print('Training %s' %name)
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        print('Fold no %i/%i'%(fold_+1,N_FOLDS))
        trn_data = train.iloc[trn_idx][features]
        trn_label = target.iloc[trn_idx]
        val_data = train.iloc[val_idx][features]
        val_label = target.iloc[val_idx]
        if 'ada' in name:
            model.fit(X=trn_data, y=trn_label)
        else:
            model.fit(X=trn_data, y=trn_label,
                     eval_set=[(trn_data, trn_label), (val_data, val_label)],
                     verbose=200,
                     early_stopping_rounds=100)

        oof_train[val_idx,i] = model.predict(val_data)
        oof_test[:,i] += model.predict(test[features])/N_FOLDS
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
    score = mean_squared_error(oof_train[:,i], target)**0.5
    layer1_score.append(score)
    feature_importance.append(feature_importance_df)
    print('Training CV score: %.5f' %score)


# In[ ]:


for i in range(len(layer1_models)):
    feature_importance_df = feature_importance[i]
    cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:25].index)

    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

    plt.figure(figsize=(12,12))
    sns.barplot(x="importance",
                y="feature",
                data=best_features.sort_values(by="importance",
                                               ascending=False))
    plt.title('%s Features (avg over folds)' % layer1_names[i])
    plt.tight_layout()
    plt.savefig('%s_importances.png' % layer1_names[i])


# #### Neural network

# In[ ]:


# Preparation
oof_train_nn = np.zeros(shape=(len(train),1))
oof_test_nn = np.zeros(shape=(len(test),1))
scaler = StandardScaler()
scaler.fit(train[features])
X_train = scaler.transform(train.iloc[:][features].values)
X_test = scaler.transform(test.iloc[:][features].values)

X_train = pd.DataFrame(X_train, index=train[features].index, columns=train[features].columns)
X_test = pd.DataFrame(X_test, index=test[features].index, columns=test[features].columns)


if DEBUG == True:
    EPOCHS=1
else:
    EPOCHS=30 


# In[ ]:


def nn_model(input_shape):
    model = Sequential()
    model.add(Dense(128, input_dim = input_shape, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

early_stop = EarlyStopping(patience=5, verbose=True)


# In[ ]:


for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print('Fold no %i/%i'%(fold_+1,N_FOLDS))
    trn_data = X_train.iloc[trn_idx][features]
    trn_label = target.iloc[trn_idx]
    val_data = X_train.iloc[val_idx][features]
    val_label = target.iloc[val_idx]
    model = nn_model(trn_data.shape[1])
    hist = model.fit(trn_data,trn_label,
                     validation_data = (val_data, val_label),
                     epochs=EPOCHS, 
                     batch_size=512, 
                     verbose=True, 
                     callbacks=[early_stop])
    oof_train_nn[val_idx,0] = model.predict(val_data)[:,0]
    oof_test_nn[:,0] += model.predict(X_test[features])[:,0]/N_FOLDS


# In[ ]:


score_nn = mean_squared_error(oof_train_nn, target)**0.5
print('Training CV score for neural network: %.5f' %score)
layer1_names.append('neural_net')
layer1_score.append(score_nn)

oof_train = np.hstack((oof_train, oof_train_nn))
oof_test = np.hstack((oof_test, oof_test_nn))


# #### Layer 1 summary

# In[ ]:


# Print first layer result
layer1 = pd.DataFrame()
layer1['models'] = layer1_names
layer1['CV_score'] = layer1_score
layer1


# In[ ]:


layer1_corr = pd.DataFrame()
for i in range(len(layer1_names)):
    layer1_corr[layer1_names[i]] = oof_train[:,i]
layer1_corr['target'] = target
colormap = plt.cm.RdBu
plt.figure(figsize=(12,12))
sns.heatmap(layer1_corr.astype(float).corr(), linewidths=0.1, vmax=1.0, vmin=-1., square=True, cmap=colormap, linecolor='white', annot=True)
plt.title('Pair-wise correlation')


# ### Second layer

# In[ ]:


# Setup the model
ridge = Ridge(alpha=0.5, fit_intercept=False)
lasso = Lasso(alpha=0.5)
lars = Lars(fit_intercept=False, positive=True)
layer2_models = [lars]#[ridge]# lasso]
layer2_names = ['lars']#['ridge'] #, 'lasso']
#params_grid = {'alpha':[0.05,0.1,0.4,1.0]}

# Setup to record result
train_pred = np.zeros(len(train))
test_pred = np.zeros(len(test))

layer2 = pd.DataFrame()
layer2['models'] = layer2_names
layer2_score = []


# In[ ]:


# For regression

for i in range(len(layer2_models)):
    print('\n')
    name = layer2_names[i]
    model = layer2_models[i]
    print('Training %s' %name)
    #model, score = do_regressor((oof_train, target), model=model, parameters=params_grid)
    model.fit(oof_train, target)
    score = mean_squared_error(model.predict(oof_train), target)**0.5
    train_pred += model.predict(oof_train)/len(layer2_models)
    test_pred += model.predict(oof_test)/len(layer2_models)
    layer2_score.append(score)
    print('Training score: %.5f' % score)


# In[ ]:


#layer2['CV score'] = layer2_score
#layer2

layer2_coef = pd.DataFrame()
layer2_coef['Name'] = layer1_names
layer2_coef['Coefficient'] = model.coef_
#layer2_coef['Coefficient'] = coef
layer2_coef


# In[ ]:


np.sum(model.coef_)


# In[ ]:


plt.figure(figsize=(8,5))
plt.scatter(range(len(test_pred)), np.sort(test_pred))
plt.xlabel('index', fontsize=12)
plt.ylabel('Loyalty Score', fontsize=12)
plt.title('Loyalty score before scaling')
plt.show()


# In[ ]:


# Refit to the target
train_scaler = StandardScaler()
#train_scaler.fit(target.values.reshape(-1,1))
#test_pred = train_scaler.inverse_transform(test_pred.reshape(-1,1))


# ### Submission

# In[ ]:


sub_df = pd.DataFrame({"card_id":test["card_id"].values})
sub_df["target"] = test_pred
sub_df.to_csv("submit.csv", index=False)


# In[ ]:


plt.figure(figsize=(8,5))
plt.scatter(range(sub_df.shape[0]), np.sort(sub_df['target'].values))
plt.xlabel('index', fontsize=12)
plt.ylabel('Loyalty Score', fontsize=12)
plt.title('Loyalty score after scaling')
plt.show()

