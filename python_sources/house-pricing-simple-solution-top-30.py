#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV , train_test_split , cross_val_score
from sklearn.metrics import classification_report , confusion_matrix


from sklearn.linear_model import LogisticRegression


from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import os
import warnings
warnings.filterwarnings('ignore')

print(os.listdir("../input/"))


# In[2]:


train = pd.read_csv('../input/train.csv' , index_col= 'Id')
test = pd.read_csv('../input/test.csv'  , index_col= 'Id')
train = train[train.GrLivArea < 4500]
train.reset_index(drop=True, inplace=True)
label = train[['SalePrice']]
train.drop('SalePrice' , axis = 1 , inplace=True)
train.head(3)


# In[3]:


label.head(3)


# In[4]:


train.info()


# In[5]:


numerical_col = []
cat_col = []
for x in train.columns:
    if train[x].dtype == 'object':
        cat_col.append(x)
        print(x+': ' + str(len(train[x].unique())))
    else:
        numerical_col.append(x)
        
print('CAT col \n', cat_col)
print('Numerical col\n')
print(numerical_col)


# In[6]:


numerical_col.remove('MSSubClass')
cat_col.append('MSSubClass')


# In[7]:


train_num = train[numerical_col]
train_num.head()


# In[8]:


imputer = Imputer(missing_values='NaN' , strategy='median' , axis = 0)
imputer = imputer.fit(train_num)
train_num = imputer.transform(train_num)


# In[9]:


test_num = imputer.transform(test[numerical_col])


# In[10]:


print(train_num.shape)
print(test_num.shape)


# In[11]:


X_train , X_test , y_train , y_test=  train_test_split(train_num , label , test_size= 0.2 , random_state=123)


# In[12]:


from sklearn.linear_model import LinearRegression
clf = LinearRegression(normalize=True)
scores = cross_val_score(clf, X_train, y_train, cv=5).mean()
scores


# In[13]:


from sklearn.linear_model import Lasso
clf = Lasso(alpha=0.3, normalize=True)
scores = cross_val_score(clf, X_train, y_train, cv=5).mean()
scores


# In[14]:


from sklearn.linear_model import ElasticNet
clf = ElasticNet(alpha=1, l1_ratio=0.5, normalize=False)
scores = cross_val_score(clf, X_train, y_train, cv=5).mean()
scores


# In[15]:


import xgboost
clf=xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
colsample_bytree=1, max_depth=7)
scores = cross_val_score(clf, X_train, y_train, cv=5).mean()
scores


# In[16]:


train_cat = train[cat_col]
test_cat = test[cat_col]
print(train_cat.info())
print(test_cat.info())


# In[17]:


dropp = ['MiscFeature' , 'PoolQC' , 'Fence' ,'Alley' ]
train_cat.drop(columns=dropp , axis=1, inplace=True)


# In[18]:


train_cat = train_cat.astype('category')
print(train_cat.info())


# In[19]:


test_cat.drop(columns=dropp , axis=1, inplace=True)
test_cat = test_cat.astype('category')
test_cat.info()


# In[20]:


most_freq = {}
for col in train_cat.columns:
    p = train_cat[col].mode()[0] 
    train_cat[col].fillna(p, inplace=True)
    most_freq[col] = p


# In[21]:


for col in train_cat.columns:
    test_cat[col].fillna(most_freq[col], inplace=True)


# In[22]:


print(test_cat.info())
print(train_cat.info())


# In[23]:


train_cat.head(2)


# In[24]:


test_cat.head(2)


# In[25]:


train_num =pd.DataFrame(train_num)
train_num.head(2)


# In[26]:


test_num =pd.DataFrame(test_num)
test_num.head(2)


# In[27]:


for col in train_cat:
    train_cat[col] = train_cat[col].cat.codes
for col in test_cat:
    test_cat[col] = test_cat[col].cat.codes


# In[28]:


train_cat.head(2)


# In[29]:


train_num.index = train_cat.index


# In[30]:


test_num.index = test_cat.index


# In[31]:


train_cat = pd.get_dummies(train_cat)
test_cat = pd.get_dummies(test_cat)


# In[32]:


train_ = train_num.join(train_cat)


# In[33]:


test_ = test_num.join(test_cat)


# In[34]:


scalar = MinMaxScaler()
train_ = scalar.fit_transform(train_)
test_ = scalar.transform(test_)


# In[35]:


# import xgboost
# clf=xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.07, gamma=0, subsample=0.75,
# colsample_bytree=1, max_depth=7)
# scores = cross_val_score(clf, train_, label, cv=5).mean()
# scores


# In[36]:


import lightgbm as lgb
lightgbm = lgb.LGBMRegressor(objective='regression', 
                                       num_leaves=8,
                                       learning_rate=0.03, 
                                       n_estimators=4000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )
scores = cross_val_score(lightgbm, train_, label, cv=5).mean()
scores


# In[37]:


clf.fit(train_ , label)
pre = clf.predict(test_)
submit = pd.read_csv('../input/sample_submission.csv')
submit.head()


submit.SalePrice = pre

submit.to_csv('submit.csv', index = False)


# In[38]:


submit.head()


# In[ ]:





# In[ ]:




