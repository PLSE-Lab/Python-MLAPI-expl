#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score,confusion_matrix,recall_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, RandomizedSearchCV
from xgboost import XGBClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
print(data.shape)
data.head()


# In[ ]:


data.info()


# ## We have 303 rows and 14 columns.
# - Our Independent column is 'Target'

# In[ ]:


sns.distplot(data['target'],kde= True)
data['target'].value_counts()


# ## we can see that our independent variable is Balanced.

# In[ ]:


data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
       'exercise_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']


# ## EDA
# 

# In[ ]:


data.columns


# In[ ]:


fig=plt.figure(figsize = (18,16))
for index,col in enumerate(data):
    plt.subplot(6,3,index+1)
    sns.distplot(data.loc[:,col],kde =False)
fig.tight_layout(pad=1.0)


# ## We can clearly see that few columns have categorical data
# - sex - Done
# - thal -  Done
# - ca
# - slope - Done
# - rest_ecg -Done
# - max_heart_rate - Done
# - exercise_angina - Done
# - fasting_blood_sugar - Done
# - serum_cholesterol- Done
# - resting_blood_pressure- Done
# - chest_pain_type-  Done

# In[ ]:


data['thalassemia'].value_counts()


# In[ ]:


data['thalassemia'] = data['thalassemia'].replace(0,2)


# In[ ]:


data['st_slope'].value_counts()


# In[ ]:


data['rest_ecg'].value_counts()


# In[ ]:


data['rest_ecg'] =data['rest_ecg'].replace(2,0)


# In[ ]:


sns.boxplot(y =data['max_heart_rate'])


# In[ ]:


data =data.drop(data[data['max_heart_rate']<80].index)


# In[ ]:


sns.boxplot(y=data['serum_cholesterol'])


# In[ ]:


data = data.drop(data[data['serum_cholesterol']>500].index)


# In[ ]:


sns.boxplot(y =data['resting_blood_pressure'])


# In[ ]:


data = data.drop(data[data['resting_blood_pressure']>179].index)


# In[ ]:


data.shape


# In[ ]:


correlation = data.corr()
plt.figure(figsize=(12,10))
sns.heatmap(correlation,annot=True,cmap = 'Blues')


# In[ ]:


pd.DataFrame(correlation['target']).sort_values(by = 'target', ascending = False).tail()


# In[ ]:


data = data.drop(['exercise_angina'],axis = 1)


# In[ ]:


data.shape


# Modeling

# In[ ]:


X= data.drop(['target'],axis = 1)
y = data['target']


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val,y_train,y_val = train_test_split(X,y,test_size = 0.2, random_state = 100)


# In[ ]:


lr = LogisticRegression()
lr.fit(X_train, y_train)
pred = lr.predict_proba(X_val)[:,1]
score = roc_auc_score(y_val,pred)
score


# In[ ]:


xgb = XGBClassifier(booster ='gbtree')

param_lst = {
            'eta' : [0.01,0.015,0.025,0.05,0.1,0.15,0.2,0.25],
            'lambda' : [0.01,0.05,0.07,0.1,0.2,0.3,0.4,1.0],
            'alpha' : [0,0.1,0.5,1.0],
            'gamma' : [0.05,0.07,0.1,0.3,0.5,0.7,0.9,1.0],
            'max_depth' : [3,5,7,9,12,15,17,25],
            'min_child_weight' : [1,3,5,7],
            'subsample' : [0.6,0.7,0.8,0.9,1.0],
            'colsample_bytree' : [0.6,0.7,0.8,0.9,1.0],
            'n_estimators': [100,120,130,140,150,160]
}

xgb_tune = RandomizedSearchCV(estimator=xgb, param_distributions= param_lst,
                             n_iter = 20,cv = 5)

       
xgb_search = xgb_tune.fit(X_train,y_train,
                          early_stopping_rounds = 5,
                           eval_set=[(X_val,y_val)],
                           verbose = False)

best_param = xgb_search.best_params_
xgb = XGBClassifier(**best_param)
print(best_param)


# In[ ]:


xgb_search.best_estimator_


# In[ ]:


y_pred = xgb_search.predict(X_val)
score0 = accuracy_score(y_pred,y_val)
print('Score: {}%'.format(round(score0*100,4)))


# In[ ]:


#cross validation for XGBoost
acc_scores1 =  cross_val_score(xgb,X,y,
                                 cv = 163,
                                 scoring = 'accuracy')


# In[ ]:


acc_scores1.mean()


# In[ ]:


acc_scores =  cross_val_score(lr,X,y,n_jobs=5,
                                 cv =5,
                                 scoring = 'accuracy')
acc_scores


# In[ ]:


import sklearn
sklearn.metrics.SCORERS.keys()


# In[ ]:


from sklearn.model_selection import cross_val_score, StratifiedKFold

cross_validation = cross_val_score(model_name, X, y, cv = StratifiedKFold(n_splits=5), scoring = 'accuracy')

