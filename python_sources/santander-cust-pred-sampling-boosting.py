#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


df_train = pd.read_csv("../input/train.csv")


# In[ ]:


df_train.head()


# In[ ]:


len(df_train)


# In[ ]:


len(df_train[df_train['target'] == 1])


# In[ ]:


len(df_train[df_train['target'] == 0])


# In[ ]:


from sklearn.utils import resample

# Separate majority and minority classes
df_train_1 = df_train[df_train['target']==1]
df_train_0 = df_train[df_train['target']==0]


# In[ ]:


# Upsample minority class
df_train_1_upsampled = resample(df_train_1, 
                                 replace=True,     # sample with replacement
                                 n_samples=179902,    # to match majority class
                                 random_state=123456) # reproducible results


# In[ ]:


df_train = pd.concat([df_train_0, df_train_1_upsampled])


# In[ ]:


len(df_train)


# In[ ]:


len(df_train[df_train['target'] == 1])


# In[ ]:


len(df_train[df_train['target'] == 0])


# In[ ]:


y_train = df_train['target']


# In[ ]:


train_id = df_train['ID_code']


# In[ ]:


df_train = df_train.drop(['ID_code','target'], axis=1)


# In[ ]:


df_train.head()


# In[ ]:


#from sklearn.ensemble import AdaBoostClassifier
#model = AdaBoostClassifier(n_estimators=300)


# In[ ]:


# Gradient Boosting 
#from sklearn.ensemble import GradientBoostingClassifier
#model = GradientBoostingClassifier(learning_rate=0.25,n_estimators=300,max_features=150)
# n_estimators = 100 (default)
# loss function = deviance(default) used in Logistic Regression


# In[ ]:


#from xgboost import XGBClassifier
#model = XGBClassifier(learning_rate=0.35,n_estimators = 300)
#  (default)
#  (default)


# In[ ]:


from sklearn.linear_model import SGDClassifier
model = SGDClassifier(loss='log', shuffle=True, random_state=101, n_jobs=8, learning_rate='optimal', 
                      alpha=0.1, l1_ratio=0.3)


# In[ ]:


model.fit(df_train, y_train)


# In[ ]:


#from scipy.stats import pearsonr
#df_train['var_0'].corr(df_train['var_6'])


# In[ ]:


df_test = pd.read_csv("../input/test.csv")


# In[ ]:


df_test.head()


# In[ ]:


test_id = df_test['ID_code']


# In[ ]:


df_test = df_test.drop(['ID_code'], axis=1)


# In[ ]:


df_test.head()


# In[ ]:


predictions = model.predict(df_test)


# In[ ]:


test = test_id.to_frame(name='ID_code')


# In[ ]:


test['target'] = pd.Series(predictions)


# In[ ]:


test.head()


# In[ ]:


test.to_csv("submission.csv", columns = test.columns, index=False)

