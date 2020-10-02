#!/usr/bin/env python
# coding: utf-8

# **Hello, this is my second Kernel on Kaggle...**
# 
# *I tried to add an Outlier feature for each numerical column of the dataset. I thought that could improve the accuracy of the model, but not. Appreciate your thoughts and insights.*
# 

# Based on https://www.kaggle.com/inversion/ieee-simple-xgboost
# *thanks to @inversion*

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
from collections import Counter


# In[ ]:


from sklearn import preprocessing
import lightgbm as lgb
import xgboost as xgb


# In[ ]:


print( "\nReading data from disk ...")
train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')
train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')
sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

del train_transaction, train_identity, test_transaction, test_identity


# **Adds feature hours**
# from https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
# thanks to @FChimel

# In[ ]:


def make_hour_feature(df, tname='TransactionDT'):
    """
    Creates an hour of the day feature, encoded as 0-23. 
    
    Parameters:
    -----------
    df : pd.DataFrame
        df to manipulate.
    tname : str
        Name of the time column in df.
    """
    hours = df[tname] / (3600)        
    encoded_hours = np.floor(hours) % 24
    return encoded_hours

train['hours'] = make_hour_feature(train)
test['hours'] = make_hour_feature(test)


# **Find Outliers**

# In[ ]:


Features_train = train.select_dtypes(include=[np.number])
Features_train = Features_train.columns.values

Features_test = test.select_dtypes(include=[np.number])
Features_test = Features_test.columns.values


# In[ ]:


def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   

# detect outliers from Features
Outliers_train = detect_outliers(train,2,Features_train)
Outliers_test = detect_outliers(test,2,Features_test)


# In[ ]:


# Add Outliers Feature
Outliers_train = pd.DataFrame(Outliers_train)
Outliers_train['Outliars'] = '1'
Outliers_train.columns.values[0] = "Id"
Outliers_train = Outliers_train.set_index('Id')
train = train.merge(Outliers_train, how='left', left_index=True, right_index=True)
train['Outliars'] = train['Outliars'].fillna(0)

Outliers_test = pd.DataFrame(Outliers_test)
Outliers_test['Outliars'] = '1'
Outliers_test.columns.values[0] = "Id"
Outliers_test = Outliers_test.set_index('Id')
test = test.merge(Outliers_test, how='left', left_index=True, right_index=True)
test['Outliars'] = test['Outliars'].fillna(0)

del Outliers_train, Outliers_test, Features_train, Features_test


# In[ ]:


y_train = train['isFraud'].copy()
X_train = train.drop('isFraud', axis=1)
del train
X_train = X_train.fillna(-999)

X_test = test.copy()
del test
X_test = X_test.fillna(-999)


# In[ ]:


# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))


# In[ ]:


clf = xgb.XGBClassifier(n_estimators=500,
                        n_jobs=4,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        missing=-999)

clf.fit(X_train, y_train)


# In[ ]:


sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
sample_submission.to_csv('simple_xgboost.csv')


# **SANDBOX**
