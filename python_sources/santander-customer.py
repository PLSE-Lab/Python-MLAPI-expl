#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv('../input/train.csv')


# In[3]:


data.head(5)


# In[4]:


# data


# In[5]:


data = data.iloc[:, 1:]


# In[6]:


data.head(5)


# In[7]:


y = data['target']
# x.head(5)
x = data.iloc[:, 1:]
x.head(5)


# In[8]:


y.head(5)


# In[9]:


# import seaborn as sns


# In[10]:


# sns.countplot(data['target'], label = 'Count')


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)


# Light GBM

# In[12]:


import lightgbm as lgb

from sklearn.metrics import auc, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV


# In[13]:



estimator = lgb.LGBMClassifier(learning_rate = 0.125, metric = 'l1', 
                        n_estimators = 20, num_leaves = 38)


param_grid = {
    'n_estimators': [x for x in range(20, 36, 2)],
    'learning_rate': [0.10, 0.125, 0.15, 0.175, 0.2]}
gridsearch = GridSearchCV(estimator, param_grid)

gridsearch.fit(X_train, y_train,
        eval_set = [(X_test, y_test)],
        eval_metric = ['auc', 'binary_logloss'],
        early_stopping_rounds = 5)


# In[14]:


# print('Best parameters found by grid search are:', gridsearch.best_params_)


# In[15]:


gbm = lgb.LGBMClassifier(learning_rate = 0.125, metric = 'l1', 
                        n_estimators = 20)


gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'],
early_stopping_rounds=5)


# In[16]:


import matplotlib.pyplot as plt
ax = lgb.plot_importance(gbm, height = 0.4, 
                         max_num_features = 25, 
                         xlim = (0,100), ylim = (0,23), 
                         figsize = (10,6))
plt.show()


# In[17]:


sorted(gbm.feature_importances_,reverse=True)


# In[18]:


temp = 0 
total = sum(gbm.feature_importances_)
for feature in sorted(gbm.feature_importances_, reverse=True):
    temp+=feature
    if temp/total >= 0.85:
        print(feature,temp/total) # stop when we 
        break


# In[19]:


from sklearn.metrics import auc, accuracy_score, roc_auc_score
from sklearn import metrics

y_pred_prob = gbm.predict_proba(X_test)[:, 1]
auc_roc_0=str(metrics.roc_auc_score(y_test, y_pred_prob)) # store AUC score without dimensionality reduction
print('AUC without dimensionality reduction: \n' + auc_roc_0)


# In[20]:


x = x.drop(['var_81','var_170','var_0','var_21','var_44','var_133'], axis=1)


# In[21]:



# Remake our test/train set with our reduced dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=21)

reduc_estimator = lgb.LGBMClassifier(learning_rate = 0.125, metric = 'l1', 
                        n_estimators = 20, num_leaves = 38)

# Parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [x for x in range(20, 36, 2)],
    'learning_rate': [0.10, 0.125, 0.15, 0.175, 0.2]}

gridsearch = GridSearchCV(reduc_estimator, param_grid)

gridsearch.fit(X_train, y_train,
        eval_set = [(X_test, y_test)],
        eval_metric = ['auc', 'binary_logloss'],
        early_stopping_rounds = 5)
print('Best parameters found by grid search are:', gridsearch.best_params_)


# In[22]:


gbm = lgb.LGBMClassifier(learning_rate = 0.1, metric = 'l1', 
                        n_estimators = 20)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['auc', 'binary_logloss'],
early_stopping_rounds=5)


# In[23]:


y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
print('The accuracy of prediction is:', accuracy_score(y_test, y_pred))
print('The roc_auc_score of prediction is:', roc_auc_score(y_test, y_pred))
print('The null acccuracy is:', max(y_test.mean(), 1 - y_test.mean()))


# In[24]:


# X_test[:,1]


# In[25]:


y_pred_prob = gbm.predict_proba(X_test)[:, 1]
y_pred_prob


# In[26]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for credit card defaulting classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.grid(True)


# In[27]:



auc_roc_1 = str(metrics.roc_auc_score(y_test, y_pred_prob))
print('AUC with dimensionality reduction: \n' + auc_roc_1)
print('AUC without dimensionality reduction: \n' + auc_roc_0)


# In[28]:


from sklearn import metrics
print(metrics.confusion_matrix(y_test, y_pred))


# In[29]:


test = pd.read_csv('../input/test.csv')
# test.head()

test = test.iloc[:, 1:]
# test.head(5)
# test.shape
test = test.drop(['var_81','var_170','var_0','var_21','var_44','var_133'], axis=1)
# test.shape


# In[30]:


# predictions = np.zeros(len(data["target"]))

predictions = gbm.predict_proba(test)[:,1]
new = pd.read_csv('../input/test.csv')
new = new['ID_code']
# new.head(5)
# predictions.head(5)


# In[31]:


# predictions


# In[32]:


submission = pd.DataFrame({"ID_code": new, "target": predictions})
# submission = pd.DataFrame({"ID_code": test.Id, "target": redictions})
# submission["target"] = predictions
submission.to_csv("submission.csv", index=False)


# In[33]:


submission.head(5)


# In[ ]:




