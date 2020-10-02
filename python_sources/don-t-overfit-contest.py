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


train=pd.read_csv("../input/train.csv")


# In[ ]:


test=pd.read_csv("../input/test.csv")


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.head(10)


# In[ ]:


train.isnull().sum()


# In[ ]:


train.pop("id")


# In[ ]:


test.pop("id")


# In[ ]:


y_train=train.iloc[:,0]


# In[ ]:


y_train


# In[ ]:


train.pop("target")


# In[ ]:


train.shape


# In[ ]:


train1=train.copy()
y_train1=y_train.copy()
test1=test.copy()


# In[ ]:


train.head(10)


# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


sc=StandardScaler()


# In[ ]:


train=sc.fit_transform(train)


# In[ ]:


test=sc.fit_transform(test)


# In[ ]:


from sklearn.linear_model import LogisticRegression,Lasso , Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score


# In[ ]:


random_state=42
log_clf=LogisticRegression(random_state=random_state)
param_grid={ 
                'class_weight': ['balanced', None],
                  'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],
                  'penalty':['l1','l2']
           }
grid=GridSearchCV(log_clf,param_grid = param_grid , scoring = 'roc_auc', verbose = 1, n_jobs = -1)
grid.fit(train,y_train)

print("Best Score:" + str(grid.best_score_))
print("Best Parameters: " + str(grid.best_params_))


# In[ ]:


log_clf = LogisticRegression(C=0.1,random_state=42,class_weight='balanced',penalty='l1')


# In[ ]:


log_clf.fit(train,y_train)


# In[ ]:


pred_log=log_clf.predict(test)


# In[ ]:


pred_log #prediction by logistic regression


# In[ ]:


submission3 = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


submission3["target"]=pred_log


# In[ ]:


submission3.to_csv('submissionlog.csv', index=False)


# In[ ]:


from sklearn.feature_selection import RFE


# In[ ]:


selector = RFE(log_clf, 25, step=1)
selector.fit(train,y_train)


# In[ ]:


submission1 = pd.read_csv('../input/sample_submission.csv')
submission1['target'] = selector.predict_proba(test) #prediction by rfe
submission1.to_csv('submissionRFE.csv', index=False)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


random_state=42
gbr=GradientBoostingRegressor(random_state=random_state)
param_grid={ 
     "learning_rate": [0.01, 0.05 , 0.1 , 1],
    "max_depth":[5, 11, 17, 25, 37, 53],
    "max_features":["log2","sqrt"],
    "criterion": ["friedman_mse",  "mae"],
    "n_estimators":[10, 50, 200, 600, 1000]
    }
           
grid=GridSearchCV(gbr,param_grid = param_grid , scoring = 'roc_auc', verbose = 1, n_jobs = -1)
grid.fit(train,y_train)

print("Best Score:" + str(grid.best_score_))
print("Best Parameters: " + str(grid.best_params_))


# In[ ]:


gbr=GradientBoostingRegressor( learning_rate=0.01, max_leaf_nodes=10, criterion='friedman_mse', max_depth=11, max_features='sqrt', n_estimators=1000)


# In[ ]:


gbr.fit(train,y_train)


# In[ ]:


pred5=gbr.predict(test)


# In[ ]:


pred5


# In[ ]:


submissionm = pd.read_csv('../input/sample_submission.csv')
submissionm["target"]=pred5


# In[ ]:


j=0
for i in submissionm["target"]:
    if i>1:
        submissionm["target"][j]=1
    j+=1    


# In[ ]:


submissionm.to_csv('submission_gbr.csv', index=False)


# In[ ]:


finalsub=(submissionm["target"]+submission1["target"])/2


# In[ ]:


finalsub


# In[ ]:


subf= pd.read_csv('../input/sample_submission.csv')
subf["target"]=finalsub


# In[ ]:


subf.to_csv('submission_fin.csv', index=False)


# In[ ]:




