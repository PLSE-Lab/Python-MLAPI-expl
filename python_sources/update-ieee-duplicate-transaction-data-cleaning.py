#!/usr/bin/env python
# coding: utf-8

# ## The purpose of this kernel is to identify duplicate transactions so that everyone can clean them in their own way

# ![Jacky](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRcix14P8VsjOUfPoXeRvK7qqqLMTe51nNJSa_3Ul4Y8UpdpCRXkA)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold
from bayes_opt import BayesianOptimization
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from sklearn import metrics
from sklearn import preprocessing
import itertools
from scipy import interp
# Lgbm
import lightgbm as lgb
import seaborn as sns


import matplotlib.pylab as plt


import os
import gc

import datetime

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


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



train_transaction = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID'))
test_transaction = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID'))

train_identity = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID'))
test_identity = reduce_mem_usage(pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID'))

sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')


# In[ ]:


# merge 
df_train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
df_test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print("Train shape : "+str(df_train.shape))
print("Test shape  : "+str(df_test.shape))


del train_transaction, train_identity, test_transaction, test_identity
gc.collect()


# In[ ]:


comb = pd.concat([df_train,df_test],axis=0,sort=True)


# In[ ]:


comb['duplicate'] = (comb['TransactionAmt'].astype('str')+comb['C1'].astype('str')+comb['C2'].astype('str')+
                     comb['C4'].astype('str')+comb['C5'].astype('str')+comb['C6'].astype('str')+
                     comb['C7'].astype('str')+comb['C8'].astype('str')+comb['C9'].astype('str')+
                     comb['C10'].astype('str')+comb['C11'].astype('str')+comb['C12'].astype('str')+
                     comb['C13'].astype('str')+comb['TransactionAmt'].astype('str')+comb['card1'].astype('str')+
                     comb['V101'].astype('str')+comb['V126'].astype('str')+comb['V128'].astype('str')+
                     comb['V127'].astype('str')+comb['V129'].astype('str')+comb['V130'].astype('str')+
                     comb['V131'].astype('str')+comb['id_31'].astype('str') +comb['id_33'].astype('str')+
                     comb['id_34'].astype('str')+comb['id_35'].astype('str')+comb['id_36'].astype('str')+
                     comb['id_37'].astype('str') +comb['id_38'].astype('str')+comb['M1'].astype('str')+
                     comb['M2'].astype('str')+comb['M3'].astype('str')+comb['M4'].astype('str')+
                     comb['M5'].astype('str')+comb['M6'].astype('str')+comb['M7'].astype('str')+
                     comb['M8'].astype('str')+comb['M9'].astype('str')+comb['P_emaildomain'].astype('str')+
                     comb['ProductCD'].astype('str')+comb['R_emaildomain'].astype('str')+comb['addr1'].astype('str')+
                     comb['D1'].astype('str')+comb['D2'].astype('str')+comb['D3'].astype('str')+
                     comb['D4'].astype('str')+comb['D5'].astype('str')+comb['D6'].astype('str')+
                     comb['D7'].astype('str')+comb['D8'].astype('str')+comb['D10'].astype('str')+
                     comb['D11'].astype('str')+comb['D12'].astype('str')+comb['D13'].astype('str')+
                     comb['D14'].astype('str')+comb['D15'].astype('str')+comb['C3'].astype('str')+
                     comb['V103'].astype('str'))


# In[ ]:


gc.collect()


# In[ ]:


lbl = preprocessing.LabelEncoder()
lbl.fit(list(comb['duplicate'].values))
comb['duplicate'] = lbl.transform(list(comb['duplicate'].values))


# In[ ]:


comb['duplicate'].value_counts()


# In[ ]:


print('Number duplicate transactions :',(comb['duplicate'].value_counts().values > 1).sum())


# In[ ]:


duplicate = comb['duplicate'].value_counts()[(comb['duplicate'].value_counts() > 1)].index


#  I believe these transactions are the online game or subscription

# In[ ]:


subscription = comb
tmp = comb.index
subscription.index = subscription['duplicate'].values


# In[ ]:


subscription = comb.loc[duplicate]


# In[ ]:


print("Number not Fraud : ",len(subscription.loc[subscription['isFraud'] == 0]),"Number Fraud : ", len(subscription.loc[subscription['isFraud'] == 1]))
sns.countplot(subscription['isFraud'], palette='Set3')


# In[ ]:


print("Subscription in the test data :", subscription['isFraud'].isna().sum())


# In[ ]:


subscription.loc[subscription['isFraud'] == 1]


# In[ ]:


isFraud_subscription = list(subscription.loc[subscription['isFraud'] == 1].index)


# In[ ]:


print('Users who Fraud from time to time :',(subscription.loc[isFraud_subscription]['isFraud'] == 0).sum())


# In[ ]:


is_Fraud_test = []
for i in duplicate :
    if (len(subscription.loc[i]['isFraud'].value_counts()) == 1 and subscription.loc[i]['isFraud'].isna().sum() >0) :
        comb.loc[comb.duplicate == i,'isFraud'] = subscription['isFraud'].value_counts().index[0]


# In[ ]:


comb.drop('duplicate',axis = 1,inplace = True)


# In[ ]:


df_train = comb[:len(df_train)]
df_test = comb[len(df_train):]
del comb
gc.collect()


# In[ ]:


df_test.loc[df_test['isFraud'] >=0]


# ### Upvote if you find this kernel useful     :)
# 
# ### Thank
