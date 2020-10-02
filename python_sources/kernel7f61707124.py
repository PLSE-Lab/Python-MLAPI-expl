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


import pandas as pd
import numpy as np
import lightgbm as lgb
import re
from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import pickle
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score


from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,
                             roc_curve, recall_score, classification_report, f1_score,
                             precision_recall_fscore_support, roc_auc_score)

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_training_v2 = pd.read_csv("/kaggle/input/widsdatathon2020/training_v2.csv")
data_test = pd.read_csv("/kaggle/input/widsdatathon2020/unlabeled.csv")


# In[ ]:


y = data_training_v2['hospital_death']
X = data_training_v2.copy()
X = X.drop('hospital_death', axis=1)

X_test = data_test;


# In[ ]:


# threshold for removing correlated variables
threshold = 0.9

# Absolute value correlation matrix
corr_matrix = X.corr().abs()
corr_matrix.head()


# In[ ]:


# Upper triangle of correlations
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
upper.head()

# Select columns with correlations above threshold
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
print(to_drop)
print('There are %d columns to remove.' % (len(to_drop)))
#Drop the columns with high correlations
X = X.drop(columns = to_drop)


# In[ ]:


# Train missing values (in percent)
train_missing = (X.isnull().sum() / len(X)).sort_values(ascending = False)
train_missing.head()
train_missing = train_missing.index[train_missing > 0.75]
print('There are %d columns with more than 75%% missing values' % len(train_missing))
X = X.drop(columns = train_missing)


# In[ ]:


#Convert categorical variable into dummy/indicator variables.
X = pd.get_dummies(X)


# In[ ]:


# Initialize an empty array to hold feature importances
feature_importances = np.zeros(X.shape[1])

# Create the model with several hyperparameters
model = lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced')


# In[ ]:



   for i in range(2):

# Split into training and validation set
train_features, valid_features, train_y, valid_y = train_test_split(X, y, test_size = 0.25, random_state = i)

# Train using early stopping
model.fit(train_features, train_y, early_stopping_rounds=100, eval_set = [(valid_features, valid_y)],eval_metric = 'auc', verbose = 200)

# Record the feature importances
feature_importances += model.feature_importances_


# In[ ]:


# Make sure to average feature importances! 
feature_importances = feature_importances / 2
feature_importances = pd.DataFrame({'feature': list(X.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)
feature_importances.head()


# In[ ]:


# Find the features with zero importance
zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])
print('There are %d features with 0.0 importance' % len(zero_features))
feature_importances.tail()
# Drop features with zero importance
X = X.drop(columns = zero_features)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)

##model = XGBClassifier(tree_method = "exact", predictor = "cpu_predictor", verbosity = 1, objective = "multi:softmax", num_class= 3)

#print('***********')
# Create parameter grid
##parameters = {"learning_rate": [0.1, 0.01, 0.001],
##               "gamma" : [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
##               "max_depth": [2, 4, 7, 10],
##               "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
##               "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
##               "reg_alpha": [0, 0.5, 1],
##               "reg_lambda": [1, 1.5, 2, 3, 4.5],
##               "min_child_weight": [1, 3, 5, 7],
##               "n_estimators": [100, 250, 500, 1000]}

##from sklearn.model_selection import RandomizedSearchCV

# Create RandomizedSearchCV Object
##model_rscv = RandomizedSearchCV(model, param_distributions = parameters, scoring = "f1_micro", cv = 10, verbose = 3, random_state = 40 )

# Fit the model
##model_rscv.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_test)
###y_pred = model_rscv.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


y_prob_pred = model.predict_proba(X_test)
###y_prob_pred = model_rscv.predict_proba(X_test)
#print(y_prob_pred)
y_prob_pred = y_prob_pred[:, 1]
#print(y_prob_pred)


# In[ ]:



print(len(X.columns))

final_features = X.columns

test_X = data_test.copy()

#print(data_test.columns)
#print(len(data_test.columns))

#Convert categorical variable into dummy/indicator variables.
test_X = pd.get_dummies(test_X)

columns_to_drop =  list(set(test_X.columns) - set(X.columns))

test_X = test_X.drop(columns = columns_to_drop)

print(len(test_X.columns))
    
#actual_pred = model.predict(test_X)
#print(actual_pred)

actual_pred_prob = model.predict_proba(test_X)
print(actual_pred_prob)
actual_pred_prob = actual_pred_prob[:, 1]
print(actual_pred_prob)


output = pd.DataFrame({'encounter_id': test_X.encounter_id, 'hospital_death': actual_pred_prob })
output.to_csv('submission.csv', index=False)

