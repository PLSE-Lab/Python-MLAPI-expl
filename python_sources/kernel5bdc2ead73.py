#!/usr/bin/env python
# coding: utf-8

# # Roosevelt National Forest Classification
# Classifying forest types based on information about the area

# ## Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.datasets import make_classification
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load data

# In[ ]:


train_path = '../input/learn-together/train.csv'
test_path = '../input/learn-together/test.csv'
train_df = pd.read_csv(train_path, index_col='Id')
test_df = pd.read_csv(test_path, index_col='Id')
train_df.head()


# In[ ]:


test_df.head()


# 1. ## Get the list of variables and data types

# In[ ]:


train_df.dtypes


# In[ ]:


# Print columns name
train_df.columns


# ## Ramdom Forest Classifier

# ### Objective:
# The goal is to predict an integer classification for the forest cover type.

# *First, I need to define my X and y variables*

# In[ ]:


X = train_df.drop('Cover_Type', axis = 1)
y = train_df['Cover_Type']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 0)


# In[ ]:


rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)


# In[ ]:


print(classification_report(y_test,rfc_pred))


# In[ ]:


print(mean_absolute_error(rfc_pred, y_test))


# In[ ]:


no_estimators = [20,30,40,50,60,80,100,120,140,160,180,200,210,240,260,280,300,350,400, 420]


# In[ ]:


def getmae(X,y,K,v):
    rf = RandomForestClassifier(n_estimators=i)
    rf.fit(X, y)
    rf_pred = rf.predict(K)
    mae = mean_absolute_error(rf_pred, v)
    print('With no of estimators =' + str(i) + ',' + ' mae =' + str(mae))
    
    
    
    


# In[ ]:


for i in no_estimators:
    getmae(X_train, y_train, X_test, y_test)


# In[ ]:


# It appears using 350 estimators offers a slightly better mean absolute error, but not highly significant.

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X, y)


# **Using Support Vector Machines (SVM)**

# In[ ]:


from sklearn.svm import SVC


# In[ ]:


svm_model = SVC()


# In[ ]:


svm_model.fit(X_train, y_train)


# In[ ]:


svm_pred = svm_model.predict(X_test)


# In[ ]:


print(mean_absolute_error(svm_pred, y_test))


# **With Grid Search**

# In[ ]:


param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)


# In[ ]:


grid.fit(X_train, y_train)


# In[ ]:


grid.best_params_


# In[ ]:


grid.best_estimator_


# In[ ]:


grid_predictions = grid.predict(X_test)


# In[ ]:


print(mean_absolute_error(grid_predictions, y_test))


# *Since the error value is very close to that obtained with a Random Forest model, we can check if there would be an improvement with the entire data set*

# In[ ]:


svm_model2 = SVC()


# In[ ]:


svm_model2.fit(X, y)


# In[ ]:


param_grid2 = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 


# In[ ]:


grid2 = GridSearchCV(SVC(),param_grid2,refit=True,verbose=3)


# In[ ]:


grid2.fit(X, y)


# In[ ]:


grid2.best_params_


# In[ ]:


grid2.best_estimator_


# **Getting predictions based on the best parameters for SVM**

# In[ ]:


preds2 = grid2.predict(test_df)


# In[ ]:


# Get predictions
preds = rfc.predict(test_df)


# # Output

# In[ ]:


test_ids = test_df.index

output = pd.DataFrame({'Id': test_ids,
                       'Cover_Type': preds2})
output.to_csv('submission.csv', index=False)

output.head()

