#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train='../input/hr-analytics/train.csv'
test='../input/hr-analytics/test.csv'
sub='../input/hr-analytics/ss.csv'


# In[ ]:


df_train=pd.read_csv(train)
df_test=pd.read_csv(test)
df_sub=pd.read_csv(sub)


# In[ ]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import lightgbm as lgb
from sklearn.metrics import f1_score


# In[ ]:


df_train.head()


# In[ ]:





# In[ ]:


df_train.fillna(-1,inplace=True)


# In[ ]:


df_train['department']= le.fit_transform(df_train['department']) 
df_train['region']= le.fit_transform(df_train['region']) 
df_train['gender']= le.fit_transform(df_train['gender'])
df_train['recruitment_channel']= le.fit_transform(df_train['recruitment_channel']) 
df_train['education']= le.fit_transform(df_train['education'].astype('str') ) 
 


# In[ ]:


df_train.head()


# In[ ]:


X=df_train.loc[:, df_train.columns != 'is_promoted']


# In[ ]:


y=df_train['is_promoted']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial').fit(X_train, y_train)
clf.score(X_test, y_test)


# In[ ]:



lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_eval,
                early_stopping_rounds=50)


# In[ ]:





# In[ ]:


f1_score(y_test, clf.predict(X_test), average='macro')  


# In[ ]:


f1_score(y_test,gbm.predict(X_test), average='macro')  

