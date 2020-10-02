#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# library 
import numpy as np 
import pandas as pd 
import itertools
from scipy import interp
import os
import datetime
import gc
import json
from numba import jit
from itertools import product
from tqdm import tqdm_notebook
import time
from contextlib import contextmanager
import psutil

# Suppr warning
import warnings
warnings.filterwarnings("ignore")

# Plots
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Data processing, metrics and modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from bayes_opt import BayesianOptimization
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from sklearn import metrics
from sklearn import preprocessing
from sklearn import linear_model

# ML
import lightgbm as lgb

# options
pd.set_option('display.max_columns', 500)


# In[ ]:


timer_depth = -1
@contextmanager
def timer(name):
    t0 = time.time()
    global timer_depth
    timer_depth += 1
    yield
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30
    print('----'*timer_depth + f'>>[{name}] done in {time.time() - t0:.0f} s ---> memory used: {memoryUse:.4f} GB', '')
    if(timer_depth == 0):
        print('\n')
    timer_depth -= 1


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


# ### read dataset

# In[ ]:


# read
train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
sub = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')

# merge 
train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')

# reduce_mem_usage
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


# In[ ]:


del train_transaction, train_identity, test_transaction, test_identity

print("Train shape : "+str(train.shape))
print("Test shape  : "+str(test.shape))


# In[ ]:


# sampling 

# train = train.sample(5000)
# test = test.sample(5000)


# In[ ]:


booleanDictionary = {'T':True, 'F':False}
train = train.replace(booleanDictionary)
test = test.replace(booleanDictionary)


# ### FE : missing value

# In[ ]:


train['nulls1'] = train.isna().sum(axis=1)
test['nulls1'] = test.isna().sum(axis=1)


# ### FE : time of day

# In[ ]:


START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')

for df in [train, test]:
    # Temporary
    df['DT'] = df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds=x)))
    df['DT_M'] = (df['DT'].dt.year - 2017) * 12 + df['DT'].dt.month
    df['DT_W'] = (df['DT'].dt.year - 2017) * 52 + df['DT'].dt.weekofyear
    df['DT_D'] = (df['DT'].dt.year - 2017) * 365 + df['DT'].dt.dayofyear

    df['hour'] = df['DT'].dt.hour
    df['dow'] = df['DT'].dt.dayofweek
    df['day'] = df['DT'].dt.day


# In[ ]:


def values_normalization(dt_df, periods, columns):
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

            dt_df[new_col+'_min_max'] = (dt_df[col]-dt_df['temp_min'])/(dt_df['temp_max']-dt_df['temp_min'])
            dt_df[new_col+'_std_score'] = (dt_df[col]-dt_df['temp_mean'])/(dt_df['temp_std'])
            del dt_df['temp_min'],dt_df['temp_max'],dt_df['temp_mean'],dt_df['temp_std']
    return dt_df

# D1 ~ D15
i_cols = ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15']
for col in i_cols:
    df = pd.concat([train[[col,'DT_M']],test[[col,'DT_M']]])
    df = values_normalization(df,['DT_M'],[col])
    print(df.head())
    train = pd.concat([train,df.iloc[:len(train),len(df.columns)-2:]],axis=1)
    test = pd.concat([test,df.iloc[len(train):,len(df.columns)-2:]],axis=1) 
    del df
    gc.collect() 


# > ### FE : id_31

# In[ ]:


def setbrowser(df):
    df.loc[df["id_31"]=="samsung browser 7.0",'lastest_browser']=1
    df.loc[df["id_31"]=="opera 53.0",'lastest_browser']=1
    df.loc[df["id_31"]=="mobile safari 10.0",'lastest_browser']=1
    df.loc[df["id_31"]=="google search application 49.0",'lastest_browser']=1
    df.loc[df["id_31"]=="firefox 60.0",'lastest_browser']=1
    df.loc[df["id_31"]=="edge 17.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 69.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 67.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 63.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 64.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 65.0 for ios",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for android",'lastest_browser']=1
    df.loc[df["id_31"]=="chrome 66.0 for ios",'lastest_browser']=1
    return df

train["lastest_browser"] = np.zeros(train.shape[0])
test["lastest_browser"] = np.zeros(test.shape[0])
train = setbrowser(train)
test = setbrowser(test)


# ### FE : email

# In[ ]:


def fe_email(df):   
    
    emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}
    us_emails = ['gmail', 'net', 'edu']
    
    df['P_email'] = (df['P_emaildomain']=='xmail.com')
    df['R_email'] = (df['R_emaildomain']=='xmail.com')
    
    df['P_isproton'] = (df['P_emaildomain']=='protonmail.com')
    df['R_isproton'] = (df['R_emaildomain']=='protonmail.com')

    df['email_check'] = np.where(df['P_emaildomain']==df['R_emaildomain'],1,0)
    df['email_check_nan_all'] = np.where((df['P_emaildomain'].isna())&(df['R_emaildomain'].isna()),1,0)
    df['email_check_nan_any'] = np.where((df['P_emaildomain'].isna())|(df['R_emaildomain'].isna()),1,0)    
    df['email_match_not_nan'] = np.where( (df['P_emaildomain']==df['R_emaildomain']) & (np.invert(df['P_emaildomain'].isna())) ,1,0)
    
    df['P_emaildomain_bin'] = df['P_emaildomain'].map(emails)    
    df['P_emaildomain_suffix'] = df['P_emaildomain'].map(lambda x: str(x).split('.')[-1])    
    df['P_emaildomain_suffix'] = df['P_emaildomain_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    df['P_emaildomain_prefix'] = df['P_emaildomain'].map(lambda x: str(x).split('.')[0])   

    df['R_emaildomain_bin'] = df['R_emaildomain'].map(emails)    
    df['R_emaildomain_suffix'] = df['R_emaildomain'].map(lambda x: str(x).split('.')[-1])    
    df['R_emaildomain_suffix'] = df['R_emaildomain_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    df['R_emaildomain_prefix'] = df['R_emaildomain'].map(lambda x: str(x).split('.')[0])   
    
    return df

train = fe_email(train)
test = fe_email(test)


# ### FE : card 

# In[ ]:


i_cols = ['TransactionID','card1','card2','card3','card4','card5','card6']

full_df = pd.concat([train[i_cols], test[i_cols]])

## I've used frequency encoding before so we have ints here
## we will drop very rare cards
full_df['card6'] = np.where(full_df['card6']==30, np.nan, full_df['card6'])
full_df['card6'] = np.where(full_df['card6']==16, np.nan, full_df['card6'])

i_cols = ['card2','card3','card4','card5','card6']

## We will find best match for nan values and fill with it
for col in i_cols:
    temp_df = full_df.groupby(['card1',col])[col].agg(['count']).reset_index()
    temp_df = temp_df.sort_values(by=['card1','count'], ascending=False).reset_index(drop=True)
    del temp_df['count']
    temp_df = temp_df.drop_duplicates(keep='first').reset_index(drop=True)
    temp_df.index = temp_df['card1'].values
    temp_df = temp_df[col].to_dict()
    full_df[col] = np.where(full_df[col].isna(), full_df['card1'].map(temp_df), full_df[col])  
    
i_cols = ['card1','card2','card3','card4','card5','card6']
for col in i_cols:
    train[col] = full_df[full_df['TransactionID'].isin(train['TransactionID'])][col].values
    test[col] = full_df[full_df['TransactionID'].isin(test['TransactionID'])][col].values


# In[ ]:


del full_df
gc.collect()


# In[ ]:


card1_thred = 2


# In[ ]:


# Reset values for "noise" card1
valid_card = train['card1'].value_counts()
valid_card = valid_card[valid_card>card1_thred]
valid_card = list(valid_card.index)
    
train['card1'] = np.where(train['card1'].isin(valid_card), train['card1'], np.nan)
test['card1']  = np.where(test['card1'].isin(valid_card), test['card1'], np.nan)


# In[ ]:


# card3/5 low freq values 
train.loc[train.card3.isin(train.card3.value_counts()[train.card3.value_counts() < 200].index), 'card3'] = "Others"
test.loc[test.card3.isin(test.card3.value_counts()[test.card3.value_counts() < 200].index), 'card3'] = "Others"

train.loc[train.card5.isin(train.card5.value_counts()[train.card5.value_counts() < 300].index), 'card5'] = "Others"
test.loc[test.card5.isin(test.card5.value_counts()[test.card5.value_counts() < 300].index), 'card5'] = "Others"


# In[ ]:


train['uid'] = train['card1'].astype(str)+'_'+train['card2'].astype(str)
test['uid'] = test['card1'].astype(str)+'_'+test['card2'].astype(str)

train['uid2'] = train['uid'].astype(str)+'_'+train['card3'].astype(str)+'_'+train['card5'].astype(str)
test['uid2'] = test['uid'].astype(str)+'_'+test['card3'].astype(str)+'_'+test['card5'].astype(str)

train['uid3'] = train['uid2'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['addr2'].astype(str)
test['uid3'] = test['uid2'].astype(str)+'_'+test['addr1'].astype(str)+'_'+test['addr2'].astype(str)

train['uid4'] = train['uid3'].astype(str)+'_'+train['P_emaildomain'].astype(str)
test['uid4'] = test['uid3'].astype(str)+'_'+test['P_emaildomain'].astype(str)

train['uid5'] = train['uid3'].astype(str)+'_'+train['R_emaildomain'].astype(str)
test['uid5'] = test['uid3'].astype(str)+'_'+test['R_emaildomain'].astype(str)


# In[ ]:


train['bank_type'] = train['card3'].astype(str)+'_'+train['card5'].astype(str)
test['bank_type']  = test['card3'].astype(str)+'_'+test['card5'].astype(str)

train['address_match'] = train['bank_type'].astype(str)+'_'+train['addr2'].astype(str)
test['address_match']  = test['bank_type'].astype(str)+'_'+test['addr2'].astype(str)

for col in ['address_match','bank_type']:
    tmp = pd.concat([train[[col]], test[[col]]])
    tmp[col] = np.where(tmp[col].str.contains('nan'), np.nan, tmp[col])
    tmp = tmp.dropna()
    fq_encode = tmp[col].value_counts().to_dict()   
    train[col] = train[col].map(fq_encode)
    test[col]  = test[col].map(fq_encode)

train['address_match'] = train['address_match']/train['bank_type'] 
test['address_match']  = test['address_match']/test['bank_type']


# ### FE : D9 (hour)

# In[ ]:


train['D9_na'] = np.where(train['D9'].isna(), 0, 1)
test['D9_na'] = np.where(test['D9'].isna(), 0, 1)

train['local_hour'] = train['D9']*24
test['local_hour']  = test['D9']*24

train['local_hour'] = train['local_hour'] - (train['TransactionDT']/(60*60))%24
test['local_hour']  = test['local_hour'] - (test['TransactionDT']/(60*60))%24

train['local_hour_dist'] = train['local_hour']/train['dist2']
test['local_hour_dist']  = test['local_hour']/test['dist2']


# ### FE : M1 ~ M9 (binary encoding, except M4)

# In[ ]:


i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']

train['M_sum'] = train[i_cols].sum(axis=1).astype(np.int8)
test['M_sum']  = test[i_cols].sum(axis=1).astype(np.int8)

train['M_na'] = train[i_cols].isna().sum(axis=1).astype(np.int8)
test['M_na']  = test[i_cols].isna().sum(axis=1).astype(np.int8)


# In[ ]:


i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']

train['M_type'] = train[i_cols].astype(str).apply(lambda x: '_'.join(x), axis=1)
test['M_type'] = test[i_cols].astype(str).apply(lambda x: '_'.join(x), axis=1)


# ### FE : device 

# In[ ]:


def id_split(df):
    df['device_name'] = df['DeviceInfo'].str.split('/', expand=True)[0]
    df['device_version'] = df['DeviceInfo'].str.split('/', expand=True)[1]

    df['OS_id_30'] = df['id_30'].str.split(' ', expand=True)[0]
    df['version_id_30'] = df['id_30'].str.split(' ', expand=True)[1]

    df['browser_id_31'] = df['id_31'].str.split(' ', expand=True)[0]
    df['version_id_31'] = df['id_31'].str.split(' ', expand=True)[1]

    df['screen_width'] = df['id_33'].str.split('x', expand=True)[0]
    df['screen_height'] = df['id_33'].str.split('x', expand=True)[1]

    df['id_34'] = df['id_34'].str.split(':', expand=True)[1]
    df['id_23'] = df['id_23'].str.split(':', expand=True)[1]

    df.loc[df['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'
    df.loc[df['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'
    df.loc[df['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'
    df.loc[df['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'
    df.loc[df['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'
    df.loc[df['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'
    df.loc[df['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'
    df.loc[df['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'
    df.loc[df['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'
    df.loc[df['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'

    df.loc[df.device_name.isin(df.device_name.value_counts()[df.device_name.value_counts() < 200].index), 'device_name'] = "Others"
    df['had_id'] = 1
    gc.collect()
    
    return df

train = id_split(train)
test = id_split(test)


# In[ ]:


for df in [train, test]:
    # Device info
    df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
    df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    df['DeviceInfo_version'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

    # Device info 2
    df['id_30'] = df['id_30'].fillna('unknown_device').str.lower()
    df['id_30_device'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    df['id_30_version'] = df['id_30'].apply(lambda x: ''.join([i for i in x if i.isnumeric()]))

    # Browser
    df['id_31'] = df['id_31'].fillna('unknown_device').str.lower()
    df['id_31_device'] = df['id_31'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))


# ### FE : TransactionAmt

# In[ ]:


train['TransactionAmt_log'] = np.log(train['TransactionAmt'])
test['TransactionAmt_log'] = np.log(test['TransactionAmt'])


# In[ ]:


train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)


# In[ ]:


train['TransactionAmt_check'] = np.where(train['TransactionAmt'].isin(test['TransactionAmt']), 1, 0)
test['TransactionAmt_check']  = np.where(test['TransactionAmt'].isin(train['TransactionAmt']), 1, 0)


# ### FE : two featrues label encoding

# In[ ]:


i_cols = [
    'id_02__id_20', 
    'id_02__D8', 
    'D11__DeviceInfo', 
    'DeviceInfo__P_emaildomain', 
    'P_emaildomain__C2', 
    'card2__dist1', 
    'card1__card5', 
    'card2__id_20', 
    'card5__P_emaildomain', 
    'addr1__card1'    
]

for feature in i_cols:

    f1, f2 = feature.split('__')
    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

    le = preprocessing.LabelEncoder()
    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
    train[feature] = le.transform(list(train[feature].astype(str).values))
    test[feature] = le.transform(list(test[feature].astype(str).values))


# ### FE : count encoding

# In[ ]:


i_cols = [
    'card1','card2','card3','card5','card4',
    'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
    'D1','D2','D3','D4','D5','D6','D7','D8','D9',
    'addr1','addr2',
    'dist1','dist2',
    'P_emaildomain', 'R_emaildomain',
    # 'id_01','id_02','id_03','id_04','id_05','id_06','id_07','id_08','id_09',
    # 'id_10','id_11','id_13','id_14','id_17','id_18','id_19',
    # 'id_20','id_21','id_22','id_24','id_25','id_26',
    # 'id_30','id_31','id_32','id_33','id_34','id_35','id_36',
    'DeviceInfo', 'DeviceInfo_device', 'DeviceInfo_version',
    'id_30', 'id_30_device', 'id_30_version',
    'id_31_device',
    'id_33',
    'uid', 'uid2', 'uid3','uid4', 'uid5'
]

for feature in i_cols:
    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))


# In[ ]:


# for feature in ['id_01', 'id_31', 'id_33', 'id_36', 'id_35']:
#    train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
#    test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))


# ### FE : ProductCD and M4 Target mean

# In[ ]:


i_cols = [
    'ProductCD',
    'M4', 
    'id_12', 'id_15', 'id_16', 'id_28', 'id_29', 'id_35', 'id_36', 'id_37', 'id_38',
    'DeviceType'    
]

for col in i_cols:
    temp_dict = train.groupby([col])['isFraud'].agg(['mean']).reset_index().rename(columns={'mean': col+'_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col+'_target_mean'].to_dict()

    train[col+'_target_mean'] = train[col].map(temp_dict)
    test[col+'_target_mean']  = test[col].map(temp_dict)


# ### FE : indicator features

# In[ ]:


def response_rate_by_group(df,x,y):
    tmp = pd.crosstab(df[x],df[y])
    tmp['Sum'] = tmp.apply(np.sum,axis=1)
    tmp['Response_rate'] = tmp.loc[:,1]/tmp['Sum']
    tmp = tmp.sort_values(['Response_rate'],ascending=False)
    print("It would be interesting to see if the amount percentual is higher or lower than 3.5% of total!")
    return(tmp)

# print(response_rate_by_group(df=train, x="card3", y="isFraud"))
# print(response_rate_by_group(df=train, x="card5", y="isFraud"))

train['card3_high_rate_fraud'] = np.where(train['card3'].isin([185,119,144]),1,0)
test['card3_high_rate_fraud'] = np.where(test['card3'].isin([185,119,144]),1,0)

train['card5_high_rate_fraud'] = np.where(train['card5'].isin([137,147,141,223,138]),1,0)
test['card5_high_rate_fraud'] = np.where(test['card5'].isin([137,147,141,223,138]),1,0)

train['card5_137'] = np.where(train['card5'].isin([137]),1,0)
test['card5_137'] = np.where(test['card5'].isin([137]),1,0)

train['day'] = np.where(train['day'].isin([1,29,30,31]),1,0)
test['day'] = np.where(test['day'].isin([1,29,30,31]),1,0)

train['hour_from6_to10'] = np.where(train['hour'].isin([6,7,8,9,10]),1,0)
test['hour_from6_to10'] = np.where(test['hour'].isin([6,7,8,9,10]),1,0)

train['hour_from5_to11'] = np.where(train['hour'].isin([5,6,7,8,9,10,11]),1,0)
test['hour_from5_to11'] = np.where(test['hour'].isin([5,6,7,8,9,10,11]),1,0)

train['hour_from4_to12'] = np.where(train['hour'].isin([4,5,6,7,8,9,10,11,12]),1,0)
test['hour_from4_to12'] = np.where(test['hour'].isin([4,5,6,7,8,9,10,11,12]),1,0)

train['P_emaildomain_mail'] = np.where(train['P_emaildomain'].isin(['mail.com']),1,0)
test['P_emaildomain_mail'] = np.where(test['P_emaildomain'].isin(['mail.com']),1,0)

train['R_emaildomain_icloud'] = np.where(train['R_emaildomain'].isin(['icloud.com']),1,0)
test['R_emaildomain_icloud'] = np.where(test['R_emaildomain'].isin(['icloud.com']),1,0)

train['addr1_251'] = np.where(train['addr1'].isin([251]),1,0)
test['addr1_251'] = np.where(test['addr1'].isin([251]),1,0)

train['addr2_65'] = np.where(train['addr2'].isin([65]),1,0)
test['addr2_65'] = np.where(test['addr2'].isin([65]),1,0)

train['C1_0'] = np.where(train['C1'].isin([0]),1,0)
test['C1_0'] = np.where(test['C1'].isin([0]),1,0)

train['C1_19'] = np.where(train['C1'].isin([19]),1,0)
test['C1_19'] = np.where(test['C1'].isin([19]),1,0)


# ### FE : feature aggregation 

# In[ ]:


i_cols = ['card1','card2','card3','card5','uid', 'uid2', 'uid3','uid4', 'uid5']
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_TransactionAmt_'+agg_type
        tmp = pd.concat([train[[col, 'TransactionAmt']], test[[col,'TransactionAmt']]])
        tmp = tmp.groupby([col])['TransactionAmt'].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})        
        tmp.index = list(tmp[col])
        tmp = tmp[new_col_name].to_dict()       
        train[new_col_name] = train[col].map(tmp)
        test[new_col_name]  = test[col].map(tmp)
        
i_cols = ['card1','card2','card3','card5','uid', 'uid2', 'uid3','uid4', 'uid5','addr1','addr2']
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_'+'D15'+'_'+agg_type
        tmp = pd.concat([train[[col, 'D15']], test[[col,'D15']]])
        tmp = tmp.groupby([col])['D15'].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})
        tmp.index = list(tmp[col])
        tmp = tmp[new_col_name].to_dict()       
        train[new_col_name] = train[col].map(tmp)
        test[new_col_name]  = test[col].map(tmp)
        
i_cols = ['card1','card4']
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_'+'id_02'+'_'+agg_type
        tmp = pd.concat([train[[col, 'id_02']], test[[col,'id_02']]])
        tmp = tmp.groupby([col])['id_02'].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})       
        tmp.index = list(tmp[col])
        tmp = tmp[new_col_name].to_dict()       
        train[new_col_name] = train[col].map(tmp)
        test[new_col_name]  = test[col].map(tmp)
        
i_cols = ['card1','card2','card3','card4','card5','uid', 'uid2', 'uid3','uid4', 'uid5']
for col in i_cols:
    for agg_type in ['mean', 'std']:
        new_col_name = col+'_'+'TransactionAmt_log'+'_'+agg_type
        tmp = pd.concat([train[[col, 'TransactionAmt_log']], test[[col,'TransactionAmt_log']]])
        tmp = tmp.groupby([col])['TransactionAmt_log'].agg([agg_type]).reset_index().rename(columns={agg_type: new_col_name})
        tmp.index = list(tmp[col])
        tmp = tmp[new_col_name].to_dict()       
        train[new_col_name] = train[col].map(tmp)
        test[new_col_name]  = test[col].map(tmp)


# In[ ]:


train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')

train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')

test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')
test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')

train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')

train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')
train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')

test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')
test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')


# ### FE : user count per period

# In[ ]:


for col in ['DT_M', 'DT_W', 'DT_D']:
    temp_df = pd.concat([train[[col]], test[[col]]])
    fq_encode = temp_df[col].value_counts().to_dict()

    train[col + '_total'] = train[col].map(fq_encode)
    test[col + '_total'] = test[col].map(fq_encode)

periods = ['DT_M', 'DT_W', 'DT_D']

i_cols = ['uid', 'uid2', 'uid3','uid4', 'uid5']
for period in periods:
    for col in i_cols:
        new_column = col + '_' + period

        temp_df = pd.concat([train[[col, period]], test[[col, period]]])
        temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)
        fq_encode = temp_df[new_column].value_counts().to_dict()

        train[new_column] = (train[col].astype(str) + '_' + train[period].astype(str)).map(fq_encode)
        test[new_column] = (test[col].astype(str) + '_' + test[period].astype(str)).map(fq_encode)

        train[new_column] /= train[period + '_total']
        test[new_column] /= test[period + '_total']


# ### FE : WoE

# In[ ]:


def cal_woe(train, test, train_target,i_cols):
    num_events = train_target.sum()
    num_non_events = train_target.shape[0] - train_target.sum()

    feature_list = []
    feature_iv_list = []
    for col in i_cols:
        with timer('cope with %s' % col):
            feature_list.append(col)

            woe_df = pd.DataFrame()
            woe_df[col] = train[col]
            woe_df['target'] = train_target
            events_df = woe_df.groupby(col)['target'].sum().reset_index().rename(columns={'target' : 'events'})
            events_df['non_events'] = woe_df.groupby(col).count().reset_index()['target'] - events_df['events']
            def cal_woe(x):
                return np.log( ((x['non_events']+0.5)/num_non_events) / ((x['events']+0.5)/num_events)  )
            events_df['WOE_'+col] = events_df.apply(cal_woe, axis=1)

            def cal_iv(x):
                return x['WOE_'+col]*(x['non_events'] / num_non_events - x['events'] / num_events)
            events_df['IV_'+col] = events_df.apply(cal_iv, axis=1)

            feature_iv = events_df['IV_'+col].sum()
            feature_iv_list.append(feature_iv)

            events_df = events_df.drop(['events', 'non_events', 'IV_'+col], axis=1)
            train = train.merge(events_df, how='left', on=col)
            test = test.merge(events_df, how='left', on=col)
            
    return train, test

i_cols = [
    'ProductCD',
    'M4', 
    'id_12', 'id_15', 'id_16', 'id_28', 'id_29', 'id_35', 'id_36', 'id_37', 'id_38',
    'DeviceType'    
]

# for col in i_cols:
#    train, test = cal_woe(train, test, train['isFraud'], [col])


# In[ ]:


prefix = ['C','D','Device','M','Transaction','V','addr','card','dist','id']
for i, p in enumerate(prefix):
    
    column_set = [x for x in train.columns.tolist() if x.startswith(prefix[i])]
    print(column_set)

    # Take NA count
    train[p + "group_nan_sum"] = train[column_set].isnull().sum(axis=1) / train[column_set].shape[1]
    test[p + "group_nan_sum"] = test[column_set].isnull().sum(axis=1) / test[column_set].shape[1]

    # Take SUM/Mean if numeric
    numeric_cols = [x for x in column_set if train[x].dtype != object]
    print(numeric_cols)
    if numeric_cols:
        train[p + "group_sum"] = train[column_set].sum(axis=1)
        test[p + "group_sum"] = test[column_set].sum(axis=1)
        train[p + "group_mean"] = train[column_set].mean(axis=1)
        test[p + "group_mean"] = test[column_set].mean(axis=1)
        # Zero Count
        train[p + "group_0_count"] = (train[column_set] == 0).astype(int).sum(axis=1) / (train[column_set].shape[1] - train[p + "group_nan_sum"])
        test[p + "group_0_count"] = (test[column_set] == 0).astype(int).sum(axis=1) / (test[column_set].shape[1] - test[p + "group_nan_sum"])


# ### FE : repalce missing values

# In[ ]:


# fill in mean for floats
# for c in train.columns:
#     if train[c].dtype=='float16' or  train[c].dtype=='float32' or  train[c].dtype=='float64':
#         train[c].fillna(train[c].mean())
#         train[c].fillna(train[c].mean())

# fill in -999 for categoricals
# train = train.fillna(-999)
# test = test.fillna(-999)


# ### FE : character feature encoding 

# In[ ]:


# Encode Str columns
for col in list(train):
    if train[col].dtype=='O':
        print(col)
        train[col] = train[col].fillna('unseen_before_label')
        test[col]  = test[col].fillna('unseen_before_label')
        
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)
        
        le = LabelEncoder()
        le.fit(list(train[col])+list(test[col]))
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])
        
        #train[col] = train[col].astype('category')
        #test[col] = test[col].astype('category')


# In[ ]:


"""
le = LabelEncoder()
for col in train.select_dtypes(include=['object', 'category']).columns:
    le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
    train[col] = le.transform(list(train[col].astype(str).values))
    test[col] = le.transform(list(test[col].astype(str).values))
"""


# ### FE : drop features 

# In[ ]:


many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]
big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]

print(">> many_null_cols :",many_null_cols)
print(">> big_top_value_cols :",big_top_value_cols)
print(">> one_value_cols :",one_value_cols)

cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols + one_value_cols_test))
cols_to_drop.remove('isFraud')

train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)

print(">> num of cols_to_drop :",len(cols_to_drop))


# In[ ]:


train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)


# ### Model

# In[ ]:


train = reduce_mem_usage(train)
test = reduce_mem_usage(test)


# In[ ]:


rm_cols = [
    'TransactionID',
    'uid', 'uid2', 'uid3','uid4', 'uid5',
    'bank_type',  
    'DT', 'DT_M', 'DT_W', 'DT_D',  # Temporary Variables
]

train = train.drop(rm_cols, axis=1)
test = test.drop(rm_cols, axis=1)


# In[ ]:


X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']
X_test = test.drop(['TransactionDT'], axis=1)


# In[ ]:


del train, test
gc.collect()


# In[ ]:


params = {
    'objective':'binary',
    'boosting_type':'gbdt',
    'metric':'auc',
    'n_jobs':-1,
    'learning_rate':0.01,
    'num_leaves': 2**8,
    'max_depth':-1,
    'tree_learner':'serial',
    'colsample_bytree': 0.85,
    'subsample_freq':1,
    'subsample':0.85,
    'max_bin':255,
    'verbose':-1,
    'seed': 47,
    'n_estimators':10000, # 800
    'early_stopping_rounds':500, # 100 
    'reg_alpha':0.3,
    'reg_lamdba':0.243
} 


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nNFOLDS = 5\nfolds = KFold(n_splits=NFOLDS)\n\ncolumns = X.columns\nsplits = folds.split(X, y)\ny_preds = np.zeros(X_test.shape[0])\ny_oof = np.zeros(X.shape[0])\nscore = 0\n\nfeature_importances = pd.DataFrame()\nfeature_importances[\'feature\'] = columns\n  \nfor fold_n, (train_index, valid_index) in enumerate(splits):\n    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]\n    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]\n    \n    dtrain = lgb.Dataset(X_train, label=y_train)\n    dvalid = lgb.Dataset(X_valid, label=y_valid)\n\n    clf = lgb.train(params, dtrain, valid_sets = [dtrain, dvalid], verbose_eval=200)\n    \n    feature_importances[f\'fold_{fold_n + 1}\'] = clf.feature_importance()\n    \n    y_pred_valid = clf.predict(X_valid)\n    y_oof[valid_index] = y_pred_valid\n    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")\n    \n    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS\n    y_preds += clf.predict(X_test) / NFOLDS\n    \n    del X_train, X_valid, y_train, y_valid\n    gc.collect()')


# In[ ]:


print(f"\nMean AUC = {score}")
print(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")


# In[ ]:


sub['isFraud'] = y_preds
sub.to_csv("submission.csv", index=False)


# In[ ]:


feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv('feature_importances.csv')
plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature')
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits))

