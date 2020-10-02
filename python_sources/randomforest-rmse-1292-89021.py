#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors


# In[ ]:


from warnings import filterwarnings
filterwarnings('ignore')


# In[ ]:


test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')
sample = pd.read_csv('sampleSubmission.csv')


# In[ ]:


y_train = train[['ISLEM_TUTARI']]


# In[ ]:


dms = pd.get_dummies(train[['ISLEM_TURU','SEKTOR']])
X_ = train.drop(['ISLEM_TURU', 'SEKTOR', 'YIL_AY', 'ISLEM_TUTARI','Record_Count'], axis=1)
X_train = pd.concat([X_, dms], axis=1)


# In[ ]:


dms = pd.get_dummies(test[['ISLEM_TURU','SEKTOR']])
X_ = test.drop(['ISLEM_TURU', 'SEKTOR', 'YIL_AY', 'ISLEM_TUTARI','Record_Count','ID'], axis=1)
X_test = pd.concat([X_, dms], axis=1)


# In[ ]:


rf_model = RandomForestRegressor()


# In[ ]:


rf_params = {'max_depth': [6,8,10,20], 
            'max_features': [2,5,10, 'auto'],
            'n_estimators': [200, 100, 500, 1000, 2000]}


# In[ ]:


rf_cv_model = GridSearchCV(rf_model, rf_params, cv = 10, n_jobs = -1, verbose = 2).fit(X_train,y_train)


# In[ ]:


rf_cv_model.best_params_


# In[ ]:


rf_tuned = RandomForestRegressor(max_depth = rf_cv_model.best_params_['max_depth'], 
                                     max_features = rf_cv_model.best_params_['max_features'],
                                     n_estimators = rf_cv_model.best_params_['n_estimators'].fit(X_train,y_train)


# In[ ]:


y_pred = rf_tuned.predict(X_test)


# In[ ]:


sample['Predicted'] = y_pred


# In[ ]:


sample.to_csv('RandomForest.csv', index = False)


# In[ ]:




