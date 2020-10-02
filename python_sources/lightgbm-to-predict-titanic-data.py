#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn import metrics

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt


# In[ ]:


# Load data set
train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


train_y = train['Survived']
train_y


# In[ ]:


del train['Name'], train['Ticket'], train['Cabin'], train['Survived']
del test['Name'], test['Ticket'], test['Cabin']

train.head()


# In[ ]:


test.head()


# In[ ]:


type(test)


# In[ ]:


train['Sex'] = train['Sex'].replace({'male': 0, 'female':1})
train['Embarked'] = train['Embarked'].replace({'Q': 0, 'S':1, 'C':1}) 

test['Sex'] = test['Sex'].replace({'male': 0, 'female':1})
test['Embarked'] = test['Embarked'].replace({'Q': 0, 'S':1, 'C':1}) 


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


# Check null data
train[train['Pclass'].isnull()]
train[train['Sex'].isnull()]
train[train['Age'].isnull()]
train[train['SibSp'].isnull()]
train[train['Parch'].isnull()]
train[train['Fare'].isnull()]
train[train['Embarked'].isnull()]


# In[ ]:


test[test['Pclass'].isnull()]
test[test['Sex'].isnull()]
test[test['Age'].isnull()]
test[test['SibSp'].isnull()]
test[test['Parch'].isnull()]
test[test['Fare'].isnull()]
test[test['Embarked'].isnull()]


# In[ ]:


X, y = train, train_y

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

print("X_train len:{}".format(len(X_train)))
print("X_test len:{}".format(len(X_test)))
print("y_train len:{}".format(len(y_train)))
print("y_test len:{}".format(len(y_test)))


# In[ ]:


y_train.value_counts(normalize=True)


# In[ ]:


y_test.value_counts(normalize=True)


# In[ ]:


# generate dataset
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# LightGBM Hyper Parameter
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
}

# train model
model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)

# predict X_test data
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# AUC (Area Under the Curve)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
#print(fpr)
#print(tpr)
#print(thresholds)
auc = metrics.auc(fpr, tpr)
print(auc)

y_pred


# In[ ]:


# predict test data
y_test_pred = model.predict(test, num_iteration=model.best_iteration)
y_test_pred


# In[ ]:


preds = np.round(y_test_pred).astype(int)
preds


# In[ ]:


# print IDs and predict results 
for i in range(0, len(preds)):
    print("{0},{1}".format(test['PassengerId'][i], preds[i]))


# In[ ]:


gender_submission = pd.read_csv('../input/gender_submission.csv')
gender_submission.head()


# In[ ]:


gender_submission['Survived'] = preds
gender_submission.head()


# In[ ]:


len(preds)
len(gender_submission['PassengerId'])


# In[ ]:


# save to csv
gender_submission.to_csv("submit.csv", index=False)


# In[ ]:


# Plot Feature Importance
lgb.plot_importance(model, figsize=(12, 6))
plt.show()

