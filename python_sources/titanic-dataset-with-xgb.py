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


train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train_df.head()


# In[ ]:


train_df.shape


# In[ ]:


test_df.shape


# In[ ]:


train_df['grp'] ='train'
test_df['grp'] ='test'


# In[ ]:


titanic_df = pd.concat([train_df, test_df])


# In[ ]:


titanic_df.head()


# See what percentage of the observations are missing, for each field

# In[ ]:


titanic_df.isnull().sum()/titanic_df.shape[0]


# Impute age

# In[ ]:


titanic_df['Age'].fillna(np.mean(titanic_df['Age']),inplace=True)


# Make dummies

# In[ ]:


Emb_dum = pd.get_dummies(titanic_df['Embarked'], prefix='Emb', drop_first=True)
Sex_dum = pd.get_dummies(titanic_df['Sex'], prefix='Sex', drop_first=True)


# Create model

# In[ ]:


titanic_df.columns


# In[ ]:


titanic_df=pd.concat([titanic_df, Emb_dum, Sex_dum], axis=1)


# In[ ]:


titanic_df['Name_len']=titanic_df['Name'].str.len()
titanic_df['Has_cab']=titanic_df['Cabin'].isnull()*1


# In[ ]:


columns = ['Survived', 'Age', 'Fare', 'Parch', 'Pclass', 'SibSp', 'Name_len', 'Has_cab', 'grp',  'PassengerId'] + list(Emb_dum.columns) + list(Sex_dum.columns)


# In[ ]:


titanic_df_mod = titanic_df[columns]


# In[ ]:


train_mod = titanic_df_mod[titanic_df_mod['grp'] == 'train']


# In[ ]:


del train_mod['grp']
del train_mod['PassengerId']


# In[ ]:


train_mod.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
y = train_mod['Survived'].values
X = train_mod.iloc[:, 1:].values
X_s = scaler.fit_transform(X)

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test =     train_test_split(X_s, y, test_size=.20, random_state=5)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, f1_score, roc_auc_score


# Baseline LR

# In[ ]:


clf_lr = LogisticRegression()


# In[ ]:


clf_lr.fit(X_train, y_train)


# In[ ]:


yh_test_lr = clf_lr.predict_proba(X_test)


# In[ ]:


log_loss_lr_holdout = log_loss(y_test, yh_test_lr[:, 1], eps=1e-7)
f1_score_lr_holdout = f1_score(y_test, yh_test_lr[:, 1] > .5)
print("Test log loss Regularized Logistic Regression {0:,.4f}".
      format(log_loss_lr_holdout))
print("Test F1 score Regularized Logistic Regression {0:,.4f}".
      format(f1_score_lr_holdout))


# Grid search LR

# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


params = {
    'penalty':['l1', 'l2'],        # l1 is Lasso, l2 is Ridge
    'solver':['liblinear'],
    'C': np.linspace(0.00002,1,100)
}

lr = LogisticRegression()
lr_gs = GridSearchCV(lr, params, cv=5, verbose=0, scoring='neg_log_loss').fit(X_train, y_train)


# In[ ]:


lr_gs.best_score_


# In[ ]:


lr_gs.best_params_


# In[ ]:


lr_best = lr_gs.best_estimator_
yh_test_lr = lr_best.predict_proba(X_test)


# In[ ]:


log_loss_lr_holdout = log_loss(y_test, yh_test_lr[:, 1], eps=1e-7)
f1_score_lr_holdout = f1_score(y_test, yh_test_lr[:, 1] > .5)
print("Test log loss Regularized Logistic Regression {0:,.4f}".
      format(log_loss_lr_holdout))
print("Test F1 score Regularized Logistic Regression {0:,.4f}".
      format(f1_score_lr_holdout))


# Baseline XGB

# In[ ]:


import xgboost as xgb


# In[ ]:


gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
yh_test_gbm = gbm.predict_proba(X_test)


# In[ ]:


log_loss_gbm_holdout = log_loss(y_test, yh_test_gbm[:, 1], eps=1e-7)
f1_score_gbm_holdout = f1_score(y_test, yh_test_gbm[:, 1] > .5)
print("Test log loss Regularized Logistic Regression {0:,.4f}".
      format(log_loss_gbm_holdout))
print("Test F1 score Regularized Logistic Regression {0:,.4f}".
      format(f1_score_gbm_holdout))


# In[ ]:


np.linspace(0.05,10,10)


# In[ ]:


params = {
    'max_depth':[3, 5, 7],        # l1 is Lasso, l2 is Ridge
    'n_estimators':[100, 300, 500],
    'learning_rate': np.linspace(0.05,10,10)
}

gbm = xgb.XGBClassifier()
gbm_gs = GridSearchCV(gbm, params, cv=5, verbose=1, scoring='neg_log_loss').fit(X_train, y_train)


# In[ ]:


gbm_best = gbm_gs.best_estimator_
yh_test_gbm = gbm_gs.predict_proba(X_test)


# In[ ]:


log_loss_gbm_holdout = log_loss(y_test, yh_test_gbm[:, 1], eps=1e-7)
f1_score_gbm_holdout = f1_score(y_test, yh_test_gbm[:, 1] > .5)
print("Test log loss Regularized Logistic Regression {0:,.4f}".
      format(log_loss_gbm_holdout))
print("Test F1 score Regularized Logistic Regression {0:,.4f}".
      format(f1_score_gbm_holdout))


# Prepare for submission

# In[ ]:


submit_mod = titanic_df_mod[titanic_df_mod['grp'] == 'test']


# In[ ]:


submit_mod.shape


# In[ ]:


submit_mod.head()


# In[ ]:


PassengerId_col = submit_mod['PassengerId']


# In[ ]:


del submit_mod['grp']
del submit_mod['PassengerId']


# In[ ]:


X_sub = submit_mod.iloc[:, 1:].values
X_sub_s = scaler.fit_transform(X_sub)


# In[ ]:


Survived_col = (gbm_gs.predict_proba(X_sub_s)>.5)[:,1]*1


# In[ ]:


S_df=pd.DataFrame({'PassengerId' : PassengerId_col, 'Survived' : Survived_col})


# In[ ]:


S_df.head()


# In[ ]:


S_df.to_csv('submission.csv', index=False)


# In[ ]:


import os
print(os.listdir("../working"))


# In[ ]:


# kaggle competitions submit -c digit-recognizer -f submission.csv -m "My fist submission"

