#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook
from sklearn.metrics import roc_auc_score
import gc

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:





# Ref Kernels
# * https://www.kaggle.com/davidcairuz/feature-engineering-lightgbm
# * https://www.kaggle.com/grazder/filling-card-nans
# * https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt
# * https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
# 

# In[ ]:


train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)


# In[ ]:


train.head()


# In[ ]:


# Number of unique classes in each object column
train_transaction.select_dtypes('object').apply(pd.Series.nunique, axis = 0)


# In[ ]:


train_transaction['TransactionDT'].plot(kind='hist',
                                        figsize=(15, 5),
                                        label='train',
                                        bins=50,
                                        title='Train vs Test TransactionDT distribution')
test_transaction['TransactionDT'].plot(kind='hist',
                                       label='test',
                                       bins=50)
plt.legend()
plt.show()


# In[ ]:


print('  {:.4f}% of Transactions that are fraud in train '.format(train_transaction['isFraud'].mean() * 100))


# In[ ]:


train_transaction.groupby('isFraud').count() .plot(kind='barh',
          title='Distribution of Target in Train',
          figsize=(15, 3),legend=None)
plt.show()


# In[ ]:


train_transaction['TransactionAmt'] .apply(np.log).plot(kind='hist',
          bins=100,
          figsize=(15, 5),
          title='Distribution of Log Transaction Amt')
plt.show()


# In[ ]:


train_transaction.groupby('ProductCD').count().sort_index().plot(kind='barh',
          figsize=(15, 3),legend=None,
         title='Count of Observations by ProductCD')
plt.show()


# 
# 1. ProductCD C has the most fraud with >11%
# 

# In[ ]:


train_transaction.groupby('ProductCD')['isFraud']     .mean()     .sort_index()     .plot(kind='barh',
          figsize=(15, 3),
         title='Percentage of Fraud by ProductCD')
plt.show()


# **Card**

# In[ ]:


card_cols = [c for c in train_transaction.columns if 'card' in c]
train_transaction[card_cols].head()


# In[ ]:


color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
color_idx = 0
for c in card_cols:
    if train_transaction[c].dtype in ['float64','int64']:
        train_transaction[c].plot(kind='hist',
                                      title=c,
                                      bins=50,
                                      figsize=(15, 2),
                                      color=color_pal[color_idx])
    color_idx += 1
    plt.show()


# In[ ]:


train_transaction_fr = train_transaction.loc[train_transaction['isFraud'] == 1]
train_transaction_nofr = train_transaction.loc[train_transaction['isFraud'] == 0]
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 8))
train_transaction_fr.groupby('card4')['card4'].count().plot(kind='barh', ax=ax1, title='Count of card4 fraud')
train_transaction_nofr.groupby('card4')['card4'].count().plot(kind='barh', ax=ax2, title='Count of card4 non-fraud')
train_transaction_fr.groupby('card6')['card6'].count().plot(kind='barh', ax=ax3, title='Count of card6 fraud')
train_transaction_nofr.groupby('card6')['card6'].count().plot(kind='barh', ax=ax4, title='Count of card6 non-fraud')
plt.show()


# In[ ]:





# **M1-M9**

# In[ ]:


m_cols = [c for c in train_transaction if c[0] == 'M']
train_transaction[m_cols].head()


# In[ ]:


(train_transaction[m_cols] == 'T').sum().plot(kind='bar',
                                              title='Count of T by M column',
                                              figsize=(15, 2),
                                              color=color_pal[3])
plt.show()
(train_transaction[m_cols] == 'F').sum().plot(kind='bar',
                                              title='Count of F by M column',
                                              figsize=(15, 2),
                                              color=color_pal[4])
plt.show()
(train_transaction[m_cols].isna()).sum().plot(kind='bar',
                                              title='Count of NaN by M column',
                                              figsize=(15, 2),
                                              color=color_pal[0])
plt.show()


# In[ ]:


import gc

del train_transaction, train_identity
del test_transaction, test_identity
gc.collect()


# In[ ]:


def reduce_mem_usage(df):
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
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = reduce_mem_usage(train)\ntest = reduce_mem_usage(test)')


# **Missing Values Analysis**

# In[ ]:


isna = train.isna().sum(axis=1)
isna_test = test.isna().sum(axis=1)


# This shows the distribution of the number of missing values for the training and test sets.

# In[ ]:


plt.hist(isna, normed=True, bins=30, alpha=0.4, label='train')
plt.hist(isna_test, normed=True, bins=30, alpha=0.4, label='test')
plt.xlabel('Number of features which are NaNs')
plt.legend()


# In[ ]:


training_missing = train.isna().sum(axis=0) / train.shape[0] 
test_missing = test.isna().sum(axis=0) / test.shape[0] 


# In[ ]:


change = (training_missing / test_missing).sort_values(ascending=False)
change = change[change<1e6] # remove the divide by zero errors


# In[ ]:


change


# *Looking at the distribution of values for a feature which changes significantly.*
# > We will look at D15.

# In[ ]:


fig, axs = plt.subplots(ncols=2)

train_vals = train["D15"].fillna(-999)
test_vals = test[test["TransactionDT"]>2.5e7]["D15"].fillna(-999) # values following the shift


axs[0].hist(train_vals, alpha=0.5, normed=True, bins=25)
    
axs[1].hist(test_vals, alpha=0.5, normed=True, bins=25)


fig.set_size_inches(7,3)
plt.tight_layout()


# **Does more missing data increase the chance of a Fradulent transaction**

# In[ ]:


isna_df = pd.DataFrame({'missing_count':isna,'isFraud':train['isFraud']})


# In[ ]:


plt.plot(isna_df.groupby('missing_count').mean(), 'k.')
plt.ylabel('Fraction of fradulent transactions')
plt.xlabel('Number of missing variables')
plt.axhline(0)


# Clearly, there are certain numbers of missing data which correlates with a an increased chance of the transaction being fradulent.

# **Baseline Model With Features Engineering**

# **Create a day Features**

# In[ ]:


def make_day_feature(df, offset=0, tname='TransactionDT'):
    """
    Creates a day of the week feature, encoded as 0-6. 
    
    Parameters:
    -----------
    df : pd.DataFrame
        df to manipulate.
    offset : float (default=0)
        offset (in days) to shift the start/end of a day.
    tname : str
        Name of the time column in df.
    """
    # found a good offset is 0.58
    days = df[tname] / (3600*24)        
    encoded_days = np.floor(days-1+offset) % 7
    return encoded_days

def make_hour_feature(df, tname='TransactionDT'):
    """
    Creates an hour of the day feature, encoded as 0-23. 
    
    Parameters:
    -----------
    df : pd.DataFrame
        df to manipulate.
    tname : str
        Name of the time column in df.
    """
    hours = df[tname] / (3600)        
    encoded_hours = np.floor(hours) % 24
    return encoded_hours


# In[ ]:


vals = plt.hist(train['TransactionDT'] / (3600*24), bins=1800)
plt.xlim(70, 78)
plt.xlabel('Days')
plt.ylabel('Number of transactions')
plt.ylim(0,1000)


# In[ ]:


train['weekday'] = make_day_feature(train, offset=0.58)
plt.plot(train.groupby('weekday').mean()['isFraud'])

plt.ylim(0, 0.04)
plt.xlabel('Encoded day')
plt.ylabel('Fraction of fraudulent transactions')
train['hours'] = make_hour_feature(train)
plt.plot(train.groupby('hours').mean()['isFraud'], color='k')
ax = plt.gca()
ax2 = ax.twinx()
_ = ax2.hist(train['hours'], alpha=0.3, bins=24)
ax.set_xlabel('Encoded hour')
ax.set_ylabel('Fraction of fraudulent transactions')

ax2.set_ylabel('Number of transactions')


# In[ ]:


train['hours'] = make_hour_feature(train)
plt.plot(train.groupby('hours').mean()['isFraud'], color='k')
ax = plt.gca()
ax2 = ax.twinx()
_ = ax2.hist(train['hours'], alpha=0.3, bins=24)
ax.set_xlabel('Encoded hour')
ax.set_ylabel('Fraction of fraudulent transactions')

ax2.set_ylabel('Number of transactions')


# In[ ]:


train.info()


# In[ ]:


useful_features = ['TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1',
                   'P_emaildomain', 'R_emaildomain', 'C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13',
                   'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M2', 'M3',
                   'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V17',
                   'V19', 'V20', 'V29', 'V30', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38', 'V40', 'V44', 'V45', 'V46', 'V47', 'V48',
                   'V49', 'V51', 'V52', 'V53', 'V54', 'V56', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V69', 'V70', 'V71',
                   'V72', 'V73', 'V74', 'V75', 'V76', 'V78', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V87', 'V90', 'V91', 'V92',
                   'V93', 'V94', 'V95', 'V96', 'V97', 'V99', 'V100', 'V126', 'V127', 'V128', 'V130', 'V131', 'V138', 'V139', 'V140',
                   'V143', 'V145', 'V146', 'V147', 'V149', 'V150', 'V151', 'V152', 'V154', 'V156', 'V158', 'V159', 'V160', 'V161',
                   'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V169', 'V170', 'V171', 'V172', 'V173', 'V175', 'V176', 'V177',
                   'V178', 'V180', 'V182', 'V184', 'V187', 'V188', 'V189', 'V195', 'V197', 'V200', 'V201', 'V202', 'V203', 'V204',
                   'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V219', 'V220',
                   'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228', 'V229', 'V231', 'V233', 'V234', 'V238', 'V239',
                   'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V249', 'V251', 'V253', 'V256', 'V257', 'V258', 'V259', 'V261',
                   'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V270', 'V271', 'V272', 'V273', 'V274', 'V275', 'V276',
                   'V277', 'V278', 'V279', 'V280', 'V282', 'V283', 'V285', 'V287', 'V288', 'V289', 'V291', 'V292', 'V294', 'V303',
                   'V304', 'V306', 'V307', 'V308', 'V310', 'V312', 'V313', 'V314', 'V315', 'V317', 'V322', 'V323', 'V324', 'V326',
                   'V329', 'V331', 'V332', 'V333', 'V335', 'V336', 'V338', 'id_01', 'id_02', 'id_03', 'id_05', 'id_06', 'id_09',
                   'id_11', 'id_12', 'id_13', 'id_14', 'id_15', 'id_17', 'id_19', 'id_20', 'id_30', 'id_31', 'id_32', 'id_33',
                   'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'device_name', 'device_version', 'OS_id_30', 'version_id_30',
                   'browser_id_31', 'version_id_31', 'screen_width', 'screen_height', 'had_id',"weekday","hours"]


# In[ ]:


cols_to_drop = [col for col in train.columns if col not in useful_features]
cols_to_drop.remove('isFraud')
cols_to_drop.remove('TransactionDT')


# In[ ]:


train = train.drop(cols_to_drop, axis=1)
test = test.drop(cols_to_drop, axis=1)


# In[ ]:


columns_a = ['TransactionAmt', 'id_02', 'D15']
columns_b = ['card1', 'card4', 'addr1']

for col_a in columns_a:
    for col_b in columns_b:
        for df in [train, test]:
            df[f'{col_a}_to_mean_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].transform('mean')
            df[f'{col_a}_to_std_{col_b}'] = df[col_a] / df.groupby([col_b])[col_a].transform('std')


# In[ ]:


# New feature - log of transaction amount.
train['TransactionAmt_Log'] = np.log(train['TransactionAmt'])
test['TransactionAmt_Log'] = np.log(test['TransactionAmt'])

# New feature - decimal part of the transaction amount.
train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)
test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Some arbitrary features interaction
for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 
                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:

    f1, f2 = feature.split('__')
    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)
    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

    le = LabelEncoder()
    le.fit(list(train[feature].astype(str).values) + list(test[feature].astype(str).values))
    train[feature] = le.transform(list(train[feature].astype(str).values))
    test[feature] = le.transform(list(test[feature].astype(str).values))

# Encoding - count encoding for both train and test
for feature in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'id_36']:
    train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))
    test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

# Encoding - count encoding separately for train and test
for feature in ['id_01', 'id_31', 'id_33', 'id_36']:
    train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))
    test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))


# In[ ]:


emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 
          'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',
          'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',
          'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 
          'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
          'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',
          'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 
          'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 
          'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
          'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
          'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',
          'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 
          'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 
          'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 
          'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 
          'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 
          'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 
          'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',
          'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']

# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654
for c in ['P_emaildomain', 'R_emaildomain']:
    train[c + '_bin'] = train[c].map(emails)
    test[c + '_bin'] = test[c].map(emails)
    
    train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
    test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])
    
    train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
    test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')


# In[ ]:


train['P_emaildomain']


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfor col in train.columns:\n    if train[col].dtype == 'object':\n        le = LabelEncoder()\n        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))\n        train[col] = le.transform(list(train[col].astype(str).values))\n        test[col] = le.transform(list(test[col].astype(str).values))")


# In[ ]:


X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']

X_test = test.drop(['TransactionDT'], axis=1)

del train, test
gc.collect()


# In[ ]:


from sklearn.model_selection import KFold
import lightgbm as lgb


# In[ ]:


params = {}
params['learning_rate']= 0.003
params['boosting_type']='gbdt'
params['objective']='binary'
params['metric']= 'auc'
params['sub_feature']=0.5
params['num_leaves']= 10
params['min_data']=50
params['max_depth']=10


# In[ ]:


X.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nNFOLDS = 2\nfolds = KFold(n_splits=NFOLDS)\n\ncolumns = X.columns\nsplits = folds.split(X, y)\ny_preds = np.zeros(X_test.shape[0])\ny_oof = np.zeros(X.shape[0])\nscore = 0\n\nfeature_importances = pd.DataFrame()\nfeature_importances[\'feature\'] = columns\n  \nfor fold_n, (train_index, valid_index) in enumerate(splits):\n    X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]\n    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]\n    \n    dtrain = lgb.Dataset(X_train, label=y_train)\n    dvalid = lgb.Dataset(X_valid, label=y_valid)\n\n    clf = lgb.train(params, dtrain, 100, valid_sets = [dtrain, dvalid], verbose_eval=200, early_stopping_rounds=500)\n    \n    feature_importances[f\'fold_{fold_n + 1}\'] = clf.feature_importance()\n    \n    y_pred_valid = clf.predict(X_valid)\n    y_oof[valid_index] = y_pred_valid\n    print(f"Fold {fold_n + 1} | AUC: {roc_auc_score(y_valid, y_pred_valid)}")\n    \n    score += roc_auc_score(y_valid, y_pred_valid) / NFOLDS\n    y_preds += clf.predict(X_test) / NFOLDS\n    \n    del X_train, X_valid, y_train, y_valid\n    gc.collect()\n    \nprint(f"\\nMean AUC = {score}")\nprint(f"Out of folds AUC = {roc_auc_score(y, y_oof)}")')


# In[ ]:


feature_importances['average'] = feature_importances[[f'fold_{fold_n + 1}' for fold_n in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv('feature_importances.csv')

plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));


# **Baseline Model With Fill NaNs in amount category with most frequent value **

# In[ ]:


train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')


# In[ ]:


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

import gc

del train_transaction, train_identity
del test_transaction, test_identity
gc.collect()


# In[ ]:


def fill_pairs(train, test, pairs):
    for pair in pairs:

        unique_train = []
        unique_test = []

        print(f'Pair: {pair}')
        print(f'In train{[pair[1]]} there are {train[pair[1]].isna().sum()} NaNs' )
        print(f'In test{[pair[1]]} there are {test[pair[1]].isna().sum()} NaNs' )

        for value in train[pair[0]].unique():
            unique_train.append(train[pair[1]][train[pair[0]] == value].value_counts().shape[0])

        for value in test[pair[0]].unique():
            unique_test.append(test[pair[1]][test[pair[0]] == value].value_counts().shape[0])

        pair_values_train = pd.Series(data=unique_train, index=train[pair[0]].unique())
        pair_values_test = pd.Series(data=unique_test, index=test[pair[0]].unique())
        
        print('Filling train...')

        for value in pair_values_train[pair_values_train == 1].index:
            train.loc[train[pair[0]] == value, pair[1]] = train.loc[train[pair[0]] == value, pair[1]].value_counts().index[0]

        print('Filling test...')

        for value in pair_values_test[pair_values_test == 1].index:
            test.loc[test[pair[0]] == value, pair[1]] = test.loc[test[pair[0]] == value, pair[1]].value_counts().index[0]

        print(f'In train{[pair[1]]} there are {train[pair[1]].isna().sum()} NaNs' )
        print(f'In test{[pair[1]]} there are {test[pair[1]].isna().sum()} NaNs' )
        
    return train, test


# In[ ]:


card_features = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
train[card_features].head()
pd.concat([train[card_features].isna().sum(), test[card_features].isna().sum()], axis=1).rename(columns={0: 'train_NaNs', 1: 'test_NaNs'})


# In[ ]:


pairs = [('card1', 'card2'), ('card1', 'card3')]

train, test = fill_pairs(train, test, pairs)


# In[ ]:


pd.concat([train[card_features].isna().sum(), test[card_features].isna().sum()], axis=1).rename(columns={0: 'train_NaNs', 1: 'test_NaNs'})


# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import train_test_split

X = train.loc[:, train.columns != 'isFraud']
y = train.iloc[:, train.columns == 'isFraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


d_train = lgb.Dataset(X_train, label= y_train)


# In[ ]:


params = {}
params['learning_rate']= 0.003
params['boosting_type']='gbdt'
params['objective']='binary'
params['metric']='auc'
params['sub_feature']=0.5
params['num_leaves']= 10
params['min_data']=50
params['max_depth']=10


# In[ ]:


from sklearn import preprocessing

# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values)) 


# In[ ]:


clf = lgb.train(params, d_train, 100)


# In[ ]:


clf = lgb.train(params, d_train, 100)
y_pred = clf.predict(X_test)


# In[ ]:


#convert into binary values

for i in range(0,len(X_test.index)):
    if (y_pred[i] >= 0.04):
        y_pred[i] = 1
    else:
        y_pred[i] =0
len(y_pred)   


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_pred, y_test)
accuracy


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.001, 1])
plt.ylim([0, 1.001])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# sorted(zip(clf.feature_importances_, X.columns), reverse=True)
feature_imp = pd.DataFrame(sorted(zip(clf.feature_importance(),X_train.columns)), columns=['Value','Feature'])


# In[ ]:


feature_imp = feature_imp.sort_values(by="Value", ascending=False)


# In[ ]:


plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp.iloc[0:30,:])
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.show()
plt.savefig('lgbm_importances-01.png')


# In[ ]:




