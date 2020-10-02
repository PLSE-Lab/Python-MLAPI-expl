#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold

import xgboost as xgb
import lightgbm as lgbm
import catboost as cb

np.random.seed(42)


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

train_df.shape, test_df.shape


# In[ ]:


train_df.head()


# In[ ]:


used_columns = ['Pclass', 'Name', 'Sex', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Embarked']


# In[ ]:


y_train = train_df[['Survived']]
y_train.shape


# In[ ]:


X_all = train_df[used_columns].append(test_df[used_columns])
X_all.shape


# In[ ]:


X_all.isna().sum()


# In[ ]:


X_all['Embarked'].value_counts()


# In[ ]:


X_all['Embarked'].fillna('S', inplace=True)
X_all['Fare'].fillna(X_all['Fare'].median(), inplace=True)


# In[ ]:


X_all['Title'] = X_all['Name'].str.extract(' ([A-Za-z]+)\.')
X_all['Title'] = X_all['Title'].replace(['Ms', 'Mlle'], 'Miss')
X_all['Title'] = X_all['Title'].replace(['Mme', 'Countess', 'Lady', 'Dona'], 'Mrs')
X_all['Title'] = X_all['Title'].replace(['Dr', 'Major', 'Col', 'Sir', 'Rev', 'Jonkheer', 'Capt', 'Don'], 'Mr')


# In[ ]:


# X_all = pd.concat([X_all, pd.get_dummies(X_all[['Sex', 'Embarked']])], axis=1)
X_all["Sex"] = X_all["Sex"].map({"male": 1, "female": 0}).astype(int)    
X_all["Embarked"] = X_all["Embarked"].map({"S": 1, "C": 2, "Q": 3}).astype(int)    
X_all['Title'] = X_all['Title'].map({'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3}).astype(int)   


# In[ ]:


X_all['TicketNumber'] = X_all['Ticket'].str.split()
X_all['TicketNumber'] = X_all['TicketNumber'].str[-1]
X_all['TicketNumber'] = LabelEncoder().fit_transform(X_all['TicketNumber'])


# In[ ]:


X_all.head()


# In[ ]:


X_all.drop(['Name', 'Ticket'], axis=1, inplace=True)
X_all.head()


# In[ ]:


X_all['FamilySize'] = X_all['SibSp'] + X_all['Parch'] + 1
X_all['IsAlone'] = X_all['FamilySize'].apply(lambda x: 1 if x == 1 else 0)
X_all.head()


# In[ ]:


X_train = X_all[0:y_train.shape[0]]
X_test = X_all[y_train.shape[0]:]
X_train.shape, y_train.shape, X_test.shape


# In[ ]:


y_train = np.ravel(y_train)


# ### XGBoost

# In[ ]:


get_ipython().run_cell_magic('time', '', "\npars = {\n    'colsample_bytree': 1,                 \n    'learning_rate': 0.05,\n    'max_depth': 5,\n    'subsample': 1,\n    'objective': 'binary:logistic',\n}\n\nkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n\nxgb_models = []\nfor train_index, val_index in kf.split(X_train, y_train):\n    train_X = X_train.iloc[train_index]\n    val_X = X_train.iloc[val_index]\n    train_y = y_train[train_index]\n    val_y = y_train[val_index]\n    xgb_train = xgb.DMatrix(train_X, train_y)\n    xgb_eval = xgb.DMatrix(val_X, val_y)\n    xgb_model = xgb.train(pars,\n                  xgb_train,\n                  num_boost_round=200,\n                  evals=[(xgb_train, 'train'), (xgb_eval, 'val')],\n                  verbose_eval=10,\n                  early_stopping_rounds=20\n                 )\n    xgb_models.append(xgb_model)\n\n    \naccuracy_score(y_train, np.round(np.sum([xgb_model.predict(xgb.DMatrix(X_train), ntree_limit=xgb_model.best_iteration) for xgb_model in xgb_models],axis=0) / 5).astype(int))")


# In[ ]:


submission = pd.DataFrame(
    {
        'PassengerId': test_df['PassengerId'], 
        'Survived': np.round(np.sum([xgb_model.predict(xgb.DMatrix(X_test), ntree_limit=xgb_model.best_iteration) for xgb_model in xgb_models],axis=0) / 5).astype(int)
    }
)
submission.to_csv("submission_xgboost.csv", index=False)


# In[ ]:





# In[ ]:




