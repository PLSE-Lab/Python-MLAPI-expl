#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')

train_final = pd.read_csv('/kaggle/input/titanic/train.csv')
test_final = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train


# In[ ]:


test


# In[ ]:


train = train.drop(columns= ['Name','Ticket','Cabin'])
test = test.drop(columns= ['Name','Ticket','Cabin'])


# In[ ]:


train['Embarked_S'] = (train['Embarked'] == 'S').astype(int)
train['Embarked_C'] = (train['Embarked'] == 'C').astype(int)
train['Embarked_Q'] = (train['Embarked'] == 'Q').astype(int)
train['Gender'] = (train['Sex'] == 'male').astype(int)


# In[ ]:


test['Embarked_S'] = (test['Embarked'] == 'S').astype(int)
test['Embarked_C'] = (test['Embarked'] == 'C').astype(int)
test['Embarked_Q'] = (test['Embarked'] == 'Q').astype(int)
test['Gender'] = (test['Sex'] == 'male').astype(int)


# In[ ]:


train = train.drop(columns = ['Sex'])
test = test.drop(columns = ['Sex'])


# In[ ]:


train = train.drop(columns = ['Embarked'])
test = test.drop(columns = ['Embarked'])


# In[ ]:


train.isnull().sum()


# In[ ]:


train.fillna(0,inplace=True)
test.fillna(0,inplace=True)


# In[ ]:


X = train.drop(columns = ['Survived'])
y = train['Survived']


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_X = scaler.fit_transform(X)


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


# In[ ]:


scores = []
best_svc = SVC(kernel='rbf')
cv = StratifiedKFold(n_splits =10, random_state=42, shuffle=True)

for train_index, test_index in cv.split(scaled_X, y):
    print("Train Index: ",train_index)
    print("Test Index: ",test_index)
    X_train, X_test, y_train, y_test = scaled_X[train_index], scaled_X[test_index], y[train_index], y[test_index]
    best_svc.fit(X_train, y_train)
    scores.append(best_svc.score(X_test, y_test))


# In[ ]:


print("Overall Score: ",np.mean(scores))


# # Alternative method for cross validation:

# In[ ]:


from sklearn.model_selection import cross_val_score

scores_alt = cross_val_score(best_svc, scaled_X, y, cv=10)

print("Overall Score: ",np.mean(scores_alt))


# # Using cross-validation for various models, to see which is best:

# In[ ]:


from sklearn.ensemble import RandomForestClassifier 
from catboost import CatBoostClassifier, cv, Pool
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

clf = RandomForestClassifier(n_estimators=1000)
cbc = CatBoostClassifier(eval_metric = 'Accuracy', random_seed = 42, learning_rate=0.01)
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.01, max_depth=3)
lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.01)

clf_scores = cross_val_score(clf, scaled_X, y, cv=10)
cbc_scores = cross_val_score(cbc, scaled_X, y, cv=10)
xgb_scores = cross_val_score(xgb, scaled_X, y, cv=10)
lgbm_scores = cross_val_score(lgbm, scaled_X, y, cv=10)


# In[ ]:


print("Random Forest: ",np.mean(clf_scores))
print("CatBoost: ",np.mean(cbc_scores))
print("XGBoost: ",np.mean(xgb_scores))
print("LightGBM: ",np.mean(lgbm_scores))


# # Using the best model to get our predictions:

# In[ ]:


train_final.fillna(-999,inplace=True)

X = train_final.drop(columns = ['Survived'])
y = train_final['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85, random_state=1234)

features = np.where(X.dtypes!=float)[0]

cbc = CatBoostClassifier(eval_metric = 'Accuracy', random_seed = 42, use_best_model=True)
cbc.fit(X_train, y_train, cat_features=features, eval_set= (X_test, y_test), early_stopping_rounds=100)


# In[ ]:


from sklearn.metrics import accuracy_score

pred = cbc.predict(X_test)

accuracy_score(y_test, pred)


# In[ ]:


test_final.fillna(-999,inplace=True)

predictions = cbc.predict(test_final)

predictions


# In[ ]:


result = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})


# In[ ]:


result.to_csv("submission.csv", index=False)

