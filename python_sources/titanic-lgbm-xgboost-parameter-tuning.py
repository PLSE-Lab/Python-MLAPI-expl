#!/usr/bin/env python
# coding: utf-8

# <h1>Hi, Kagglers!</h1>
# 
# <br>
# For first time I made a notebook.<br>
# If you have any questions and advices, please comment :)<br>
# 
# 
# I used LGBM and XGBoost to predict Survived.<br>
# They have much parameters affecting prediction.<br>
# I think **Parameter Tuning** is neccesary for this.<br>
# 
# Some of Tuning code I write down this kernel. please check.
# (It's really long time to tune parameters. I recommend to run on Kaggle Notebook.)
# 
# 
# 
# I enjoy the Kaggle Life !!<br>
# Thank you.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# If you are not installed modules, please install first.<br>
# 
# pip install scikit-learn<br>
# pip install xgboost<br>
# pip install lightgbm
# 

# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import re
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_rows', 10000)


# In[ ]:


df_train = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')

df_survive = df_train['Survived']
Id = df_test['PassengerId']


# <h2>Check data</h2>

# In[ ]:


df_train.info()


# In[ ]:


df_test.info()


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# <h2>Data Cleaning</h2>

# In[ ]:


df_train['Age'].describe()


# In[ ]:


df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())


# In[ ]:


df_train['Sex'] = df_train['Sex'].apply(lambda x: 1 if x == 'male' else 0)
df_test['Sex'] = df_test['Sex'].apply(lambda x: 1 if x == 'male' else 0)


# In[ ]:


df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1


# In[ ]:


def family_group(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif (s > 7):
        return 0


# In[ ]:


df_train['FamilyGroup'] = df_train['FamilySize'].apply(family_group)
df_test['FamilyGroup'] = df_test['FamilySize'].apply(family_group)


# In[ ]:


def age_group(s):
    if (s > 0) & (s < 10):
        return 1
    elif (s > 10) & (s <= 20):
        return 2
    elif (s > 20) & (s <= 30):
        return 3
    elif (s > 30) & (s <= 40):
        return 4
    elif (s > 40) & (s <= 50):
        return 5
    elif (s > 50) & (s <= 60):
        return 6
    elif (s > 60) & (s <= 70):
        return 7
    elif (s > 70) & (s <= 80):
        return 8


# In[ ]:


df_train['AgeGroup'] = df_train['Age'].apply(age_group)
df_test['AgeGroup'] = df_test['Age'].apply(age_group)


# In[ ]:


def fare_group(s):
    if (s <= 8):
        return 1
    elif (s > 8) & (s <= 15):
        return 2
    elif (s > 15) & (s <= 31):
        return 3
    elif s > 31:
        return 4


# In[ ]:


df_train['FareGroup'] = df_train['Fare'].apply(fare_group)
df_test['FareGroup'] = df_test['Fare'].apply(fare_group)


# In[ ]:


df_dummy = pd.get_dummies(df_train['Pclass'],prefix='Pclass')
df_train = pd.concat([df_train,df_dummy],axis=1)
del df_train['Pclass']
df_dummy = pd.get_dummies(df_test['Pclass'],prefix='Pclass')
df_test = pd.concat([df_test,df_dummy],axis=1)
del df_test['Pclass']


# In[ ]:


df = df_train.drop(['Survived', 'PassengerId','Name', 'Ticket', 'Age', 'Fare','Embarked','Cabin'], axis = 1)


# In[ ]:


df_test = df_test.drop(['Name', 'Ticket','Age', 'Fare','Embarked','Cabin','PassengerId'],axis = 1)


# <h2>Prepare training</h2>

# In[ ]:


X_train = df
y_train = df_survive
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2)


# <h2>Grid Search</h2>
# I did GridSearch (It's hidden input/output. It costs time.).

# In[ ]:


search_xgb = {'n_estimators': [1, 10, 50, 100, 1000],
              'max_features': [1, 5, 10, 15, 20],
              'random_state'      : [0, 10, 50],
              'min_samples_split' : [3, 5, 10, 15, 20],
              'max_depth'         : [1, 5, 10, 20, 100]}


# In[ ]:


gs_xgb = GridSearchCV(XGBClassifier(),
                      search_xgb,
                      cv=5, #Cross Validation
                      verbose=True, #Display logs
                      n_jobs=-1) #Multi Tasking
gs_xgb.fit(X_train, y_train)
 
print(gs_xgb.best_estimator_)


# In[ ]:


search_lgb = {"max_depth": [10, 25, 50, 75, 100, 1000],
              "learning_rate" : [0.001,0.01,0.05,0.1],
              "num_leaves": [100,300,900,1200],
              "n_estimators": [100,200,500,1000]
             }


# In[ ]:


gs_lgb = GridSearchCV(LGBMClassifier(),
                      search_lgb,
                      cv=5, #Cross Validation
                      verbose=True, #Display logs
                      n_jobs=-1) #Multi Tasking
gs_lgb.fit(X_train, y_train)
 
print(gs_lgb.best_estimator_)


# In[ ]:


XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                    importance_type='gain', interaction_constraints='',
                    learning_rate=0.300000012, max_delta_step=0, max_depth=5,
                    min_child_weight=1,monotone_constraints='()', n_estimators=10,
                    n_jobs=0,num_parallel_tree=1, random_state=0, reg_alpha=0,
                    reg_lambda=1,scale_pos_weight=1, subsample=1, tree_method='exact',
                    validate_parameters=1,verbosity=None)
XGB.fit(X_train,y_train)


# In[ ]:


LGB = LGBMClassifier(learning_rate=0.05, max_depth=10, num_leaves=100)
LGB.fit(X_train , y_train)


# <h2>Check Score</h2>

# In[ ]:


print ("Training score:",XGB.score(X_train,y_train),"Test Score:",XGB.score(X_test,y_test))
print ("Training score:",LGB.score(X_train,y_train),"Test Score:",LGB.score(X_test,y_test))


# <h2>Submit prediction</h2>

# In[ ]:


y_test = LGB.predict(df_test)

submission = pd.DataFrame({'PassengerId':Id, 'Survived':y_test})
submission.to_csv('submission.csv', index = False)


# <h3>Thank you for reading. If you like, please upvote!</h3>
