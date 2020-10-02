#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

plt.style.use('seaborn')
sns.set(font_scale=1)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test_bqCt9Pv.csv')


# In[ ]:


train.head()


# In[ ]:


# Mapping categorical data into numerical data

a=train['Employment.Type'].map({'Salaried': 1, 'Self employed': 0})
b=test['Employment.Type'].map({'Salaried': 1, 'Self employed': 0})


# In[ ]:


train['Employment.Type']=a
test['Employment.Type']=b


# In[ ]:


# Converting DoB into date-time format

train['Date.of.Birth']=pd.to_datetime(train['Date.of.Birth'],format='%d-%m-%y')
test['Date.of.Birth']=pd.to_datetime(test['Date.of.Birth'],format='%d-%m-%y')


# In[ ]:


# Saving only year of birth as a features

train['Year of Birth'] = train['Date.of.Birth'].dt.year 
test['Year of Birth'] = test['Date.of.Birth'].dt.year 
train=train.drop(['Date.of.Birth'],axis=1)
test=test.drop(['Date.of.Birth'],axis=1)


# In[ ]:


#

act_months=[]
for i in train['AVERAGE.ACCT.AGE']:
    a=list(map(int, re.findall(r'\d+', i)))
    act_months.append(12*a[0]+a[1])


# In[ ]:


credit_length=[]
for i in train['CREDIT.HISTORY.LENGTH']:
    a=list(map(int, re.findall(r'\d+', i)))
    credit_length.append(12*a[0]+a[1])


# In[ ]:


train['ACCOUNT AGE (IN MONTHS)']=act_months
train['CREDIT HISTORY DURATION (IN MONTHS)']= credit_length


# In[ ]:


test_act_months=[]
for i in test['AVERAGE.ACCT.AGE']:
    a=list(map(int, re.findall(r'\d+', i)))
    test_act_months.append(12*a[0]+a[1])
test['ACCOUNT AGE (IN MONTHS)']=test_act_months


# In[ ]:


test_credit_length=[]
for i in test['CREDIT.HISTORY.LENGTH']:
    a=list(map(int, re.findall(r'\d+', i)))
    test_credit_length.append(12*a[0]+a[1])
test['CREDIT HISTORY DURATION (IN MONTHS)']= test_credit_length


# In[ ]:


train=train.drop(['AVERAGE.ACCT.AGE'],axis=1)
train=train.drop(['CREDIT.HISTORY.LENGTH'],axis=1)
test=test.drop(['AVERAGE.ACCT.AGE'],axis=1)
test=test.drop(['CREDIT.HISTORY.LENGTH'],axis=1)


# In[ ]:


train=train.drop(['DisbursalDate'],axis=1)
test=test.drop(['DisbursalDate'],axis=1)


# In[ ]:


le = preprocessing.LabelEncoder()
train['PERFORM_CNS.SCORE.DESCRIPTION']=le.fit_transform(train['PERFORM_CNS.SCORE.DESCRIPTION'])
test['PERFORM_CNS.SCORE.DESCRIPTION']=le.fit_transform(test['PERFORM_CNS.SCORE.DESCRIPTION'])


# In[ ]:


train['PERFORM_CNS.SCORE.DESCRIPTION'].value_counts()


# In[ ]:


ohe = OneHotEncoder()
train_ohe = ohe.fit_transform(train['PERFORM_CNS.SCORE.DESCRIPTION'].values.reshape(-1,1)).toarray()
test_ohe = ohe.fit_transform(test['PERFORM_CNS.SCORE.DESCRIPTION'].values.reshape(-1,1)).toarray()


# In[ ]:


trainOneHot = pd.DataFrame(train_ohe, columns = ["CNS_SCORE_"+str(int(i)) for i in range(train_ohe.shape[1])])
train = pd.concat([train, trainOneHot], axis=1)
testOneHot = pd.DataFrame(test_ohe, columns = ["CNS_SCORE_"+str(int(i)) for i in range(test_ohe.shape[1])])
test = pd.concat([test, testOneHot], axis=1)


# In[ ]:


train=train.fillna(value=0)
test=test.fillna(value=0)


# In[ ]:


train=train.drop(['CNS_SCORE_14'],axis=1)


# In[ ]:


X=train.drop(['loan_default'],axis=1)
Y=train['loan_default']


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.33, random_state=42)


# In[ ]:


random_state=42
lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
}


# In[ ]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
predictions = test[['UniqueID']]
val_aucs = []


# In[ ]:


for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    
    N = 5
    p_valid,yp = 0,0
    for i in range(N):
#         X_t, y_t = (X_train, y_train)
#         X_t = pd.DataFrame(X_t)
#         X_t = X_t.add_prefix('var_')
    
        trn_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val)
      
        evals_result = {}
        lgb_clf = lgb.train(lgb_params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result
                       )
        p_valid += lgb_clf.predict(X_val)
        yp += lgb_clf.predict(test)
    val_score = roc_auc_score(y_val, p_valid)
    val_aucs.append(val_score)
    
#     predictions['fold{}'.format(fold+1)] = yp/N


# In[ ]:


predictions['loan_status']=yp/5


# In[ ]:


predictions.to_csv('submission.csv',index=False)


# In[ ]:




