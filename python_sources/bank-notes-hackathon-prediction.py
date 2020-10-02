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


# **This is part of a hackathon I participated. Consdering the timelimit, I tried with Random Forest Classification. I am yet to explore multiple models**

# In[ ]:


import pandas as pd
import numpy as np
import os

df_train=pd.read_csv('../input/bank-notes/train.csv')
df_test=pd.read_csv('../input/bank-notes/test.csv')
df_train.head()


# No null records. All the values are numeric. We can only perform scaling to increase effciency

# In[ ]:


df_test.isnull().sum(axis=0)


# In[ ]:


X_train=df_train[df_train.loc[:,df_train.columns!= 'Class'].columns]
y_train=df_train['Class']
X_test=df_test


# Applied MinMax scaler

# In[ ]:


from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler()
X_train_scaled=scaler.fit_transform(X_train.values)
X_test_scaled=scaler.fit_transform(X_test.values)


# In[ ]:


X_train=X_train_scaled
X_test=X_test_scaled


# Implementing Random Forest Classifier

# In[ ]:


from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
clf=ensemble.RandomForestClassifier()

params = {
    'bootstrap': [False],
    'max_depth': [5, 10],
    'max_features': ['auto'],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 4],
    'n_estimators': [100, 250]}

gsv = GridSearchCV(clf, params, cv=3, n_jobs=-1, scoring='accuracy')
gsv.fit(X_train, y_train)


# In[ ]:


predictions=gsv.best_estimator_.predict(X_test)


# In[ ]:


gsv.best_estimator_


# In[ ]:


predictions


# In[ ]:


len(predictions)


# Rescaling the scaled values

# In[ ]:


X_test_rescaled=scaler.inverse_transform(X_test)
X_test_rescaled


# Merge the output with rescaled values and output the vlaues to a csv file

# In[ ]:


df_preds = pd.DataFrame(data=X_test_rescaled, columns=["Variance", "Skewness", "Curtosis","Entropy"])
df_preds_class=pd.DataFrame(data=predictions, columns=["Class"])


# In[ ]:


df_final_preds=df_preds.join(df_preds_class)


# In[ ]:


df_final_preds.to_csv('predictions.csv',header=False, index=False)


# #implementing XGBoost for prediction

# In[ ]:


import xgboost as xgb
from sklearn import metrics, model_selection


# In[ ]:


# function to run the xgboost model #
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0):
        params = {}
        params["objective"] = "binary:logistic"
        params['eval_metric'] = 'logloss'
        params["eta"] = 0.05
        params["subsample"] = 0.7
        params["min_child_weight"] = 10
        params["colsample_bytree"] = 0.7
        params["max_depth"] = 8
#        params["silent"] = 1
        params["seed"] = seed_val
        num_rounds = 100
        plst = list(params.items())
        xgtrain = xgb.DMatrix(train_X, label=train_y)

        if test_y is not None:
                xgtest = xgb.DMatrix(test_X, label=test_y)
                watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
                model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=10)
        else:
                xgtest = xgb.DMatrix(test_X)
                model = xgb.train(plst, xgtrain, num_rounds)

        pred_test_y = model.predict(xgtest)
        return pred_test_y


# In[ ]:


pred = runXGB(X_train, y_train, X_test)


# In[ ]:


cutoff = 0.2
pred[pred>=cutoff] = 1
pred[pred<cutoff] = 0


# In[ ]:


pred


# I cannot find the accuracy since we do not have a y_test dataframe.
# 

# In[ ]:


#from sklearn.metrics import accuracy_score
#accuracy = accuracy_score(y_test, pred)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))

