#!/usr/bin/env python
# coding: utf-8

# This Kernel uses [TPOT AutoML](https://epistasislab.github.io/tpot/) tool(open source & based on Scikit-learn)  to stochastically produces a pipeline using Genetic programming with most accuracy based on training data.
# TPOT generates a python file that has recommended model, the I run the model code on the training data and then predict the target for the testing data and export the submission file. 
# 

# In[ ]:


#TPOT AutoML tool
from tpot import TPOTClassifier


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np # linear algebra
np.set_printoptions(precision=2)

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame


# Preprocessing, modelling and evaluating
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
from sklearn.preprocessing import Imputer,MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error, mean_absolute_error
from sklearn import svm

## Hyperopt modules
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


#     Credit for: 
# *  [@artgor - Andrew Lukyanenko](https://www.kaggle.com/artgor/eda-and-models)
# 
# *  [My paper using DNA Sequence Alignment](http://www.wseas.us/e-library/transactions/systems/2008/27-535.pdf)
# 

# In[ ]:


df_trans = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
df_test_trans = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

df_id = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
df_test_id = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

df_train = df_trans.merge(df_id, how='left', left_index=True, right_index=True)
df_test = df_test_trans.merge(df_test_id, how='left', left_index=True, right_index=True)


# In[ ]:


one_value_cols = [col for col in df_train.columns if df_train[col].nunique() <= 1]
one_value_cols_test = [col for col in df_test.columns if df_test[col].nunique() <= 1]


# In[ ]:


many_null_cols = [col for col in df_train.columns if df_train[col].isnull().sum() / df_train.shape[0] > 0.9]
many_null_cols_test = [col for col in df_test.columns if df_test[col].isnull().sum() / df_test.shape[0] > 0.9]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'big_top_value_cols = [col for col in df_train.columns if df_train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]\nbig_top_value_cols_test = [col for col in df_test.columns if df_test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]\ncols_to_drop = list(set(many_null_cols + many_null_cols_test +\n                        big_top_value_cols +\n                        big_top_value_cols_test +\n                        one_value_cols+ one_value_cols_test))\nlen(cols_to_drop)')


# Removing confusing columns (One value, Many Nulls, etc..)

# In[ ]:


cols_to_drop.remove('isFraud')

df_train = df_train.drop(cols_to_drop, axis=1)
df_test = df_test.drop(cols_to_drop, axis=1)


# Categorical columns

# In[ ]:


cat_cols = ['id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29',
            'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo', 'ProductCD', 'card4', 'card6', 'M4','P_emaildomain',
            'R_emaildomain', 'card1', 'card2', 'card3',  'card5', 'addr1', 'addr2', 'M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']


# Enconding categorical features

# In[ ]:


get_ipython().run_cell_magic('time', '', 'for col in cat_cols:\n    if col in df_train.columns:\n        le = preprocessing.LabelEncoder()\n        le.fit(list(df_train[col].astype(str).values) + list(df_test[col].astype(str).values))\n        df_train[col] = le.transform(list(df_train[col].astype(str).values))\n        df_test[col] = le.transform(list(df_test[col].astype(str).values))   ')


# Removing high correlated features

# In[ ]:


get_ipython().run_cell_magic('time', '', "threshold = 0.97\n    \n# Absolute value correlation matrix\ncorr_matrix = df_train[df_train['isFraud'].notnull()].corr().abs()\n\n# Getting the upper triangle of correlations\nupper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))\n\n# Select columns with correlations above threshold\nto_drop = [column for column in upper.columns if any(upper[column] > threshold)]\n\nprint('There are %d columns to remove.' % (len(to_drop)))\ndf_train = df_train.drop(columns = to_drop)\ndf_test = df_test.drop(columns = to_drop)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "X_train = df_train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT'], axis=1)\ny_train = df_train.sort_values('TransactionDT')['isFraud']\nX_test = df_test.sort_values('TransactionDT').drop(['TransactionDT'], axis=1)")


# In[ ]:


del df_train
df_test = df_test[["TransactionDT"]]

import gc
gc.collect()


# The following code used in the first run to generate TPOT AutoML python code which represents the recommended model from the tool
# to be used in training and prediction for submission to generate best accuracy, so after I got the model python file, I commented the use of TPOT code and implace the model to generate the result.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.25)\npipeline_optimizer = TPOTClassifier()\npipeline_optimizer = TPOTClassifier(generations=33, population_size=99, cv=10,\n                                    random_state=30, verbosity=2,config_dict="TPOT light")\npipeline_optimizer.fit(X_train, y_train)\nprint(pipeline_optimizer.score(X_val, y_val))\npipeline_optimizer.export(\'IEEE_Frauds_tpot_exported_pipeline_light.py\')')


# In[ ]:


#%%time
#features = X_train
#training_features, testing_features, training_target, testing_target = \
#            train_test_split(features, y_train, random_state=30)
#
#imputer = SimpleImputer(strategy="median", missing_values=np.NaN)
#imputer.fit(training_features)
#training_features = training_features.fillna(training_features.median())#imputer.transform(training_features)
#testing_features = testing_features.fillna(testing_features.median())#imputer.transform(testing_features)
#
# Average CV score on the training set was:0.9739357203229231
#exported_pipeline = make_pipeline(
#    MaxAbsScaler(),
#    DecisionTreeClassifier(criterion="entropy", max_depth=10, min_samples_leaf=8, min_samples_split=11)
#)
#
#exported_pipeline.fit(training_features, training_target)


# In[ ]:


score_from_training = exported_pipeline.score(testing_features,testing_target)
print("Score= " + str(score_from_training))


# In[ ]:


testing_feature = X_test
testing_features = testing_feature.fillna(testing_feature.median())#imputer.transform(testing_features)


# In[ ]:


#------------------ Predict for Submission ---------------------------
results = exported_pipeline.predict(testing_features)


# In[ ]:


dfIsFraud = pd.DataFrame(data={'TransactionID':X_test.index.values, 'isFraud':results})


# In[ ]:


dfIsFraud.to_csv('submission.csv',index = False)

