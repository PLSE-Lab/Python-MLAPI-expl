#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The aim of this kernel is to be able to perform Lasso & Ridge Regularization as part of the chapter - 6 of "Introduction to Statistical Learning" book. 
# 
# Regularization is a process of reducing the predictors with the aim of reducing the variance while improving the bias of the model. The aim is to reduce overfitting of the models.
# 
# There are two types of regularization techniques specified in the book:
# 1. Ridge Regularization (L2)
# 2. Lasso Regularization (L1) 
# 
# Let's see how these two perform both in non-cross validated data set and with cross validation.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Without Cross Validation

# In[ ]:


df = pd.read_csv("../input/train.csv")
df.shape


# In[ ]:


df.info()


# In[ ]:


# Let's separate out the data set into train and validation data set

X_train, X_val, y_train, y_val = train_test_split(df.drop(labels=['TARGET'], axis=1), df['TARGET'], test_size=0.25, random_state=0)

X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)

# shapes
X_train.shape, X_val.shape


# In[ ]:


# Lets scale the data so that is ready to be used for Lasso regularization
scaler = StandardScaler()
scaler.fit(X_train)


# ### Lasso Regularization 
# 
# Let's perform Lasso on the LR model.
# - C = Inverse of regularization strength, smaller values specify stronger regularization.
# - [](http://)penalty = specify the norm used in the penalization. 

# In[ ]:


sel = SelectFromModel(LogisticRegression(C=1, penalty = 'l1'))
sel.fit(scaler.transform(X_train), y_train)


# In[ ]:


print('Total features', X_train.shape[1])
print('Selected features', sum(sel.get_support()))
print('Removed features', np.sum(sel.estimator_.coef_ == 0))


# As we can see, we have used Lasso regularization to remove non-important features from the dataset. One question to ask is how conservative do you want to be when performing regularization?
# 
# Do we want to remove more features and potentially not give enough features to the model?

# In[ ]:


# let's create a function to build random forests and compare the performance in training and test set:

def RandomForest(X_train, X_val, y_train, y_val):
    rf = RandomForestClassifier(n_estimators = 200, random_state = 1, max_depth = 4)
    rf.fit(X_train, y_train)
    print("Training set")
    
    pred = rf.predict_proba(X_train)
    print("Random forest roc-auc: {}".format(roc_auc_score(y_train, pred[:,1])))
    
    print("Validation set")
    pred = rf.predict_proba(X_val)
    print("Random forest roc-auc: {}".format(roc_auc_score(y_val, pred[:,1])))


# In[ ]:


# Transforming the training set and test set.
X_train_lasso = sel.transform(X_train)
X_val_lasso   = sel.transform(X_val)

RandomForest(X_train_lasso, X_val_lasso, y_train, y_val)


# ### Ridge Regularization 
# 
# - Also called L2 regression is the most commonly used method of regularization for the problems which do not have a unique solution. It adds penalty equivalent to square of the magnitude of coefficients. Unlike L1, it does not shrink coefficients to zero, but near to zero. 

# In[ ]:


sfm = SelectFromModel(LogisticRegression(C=1, penalty = 'l2'))
sfm.fit(scaler.transform(X_train), y_train)


# In[ ]:


print('Total features-->',X_train.shape[1])
print('Selected featurs-->',sum(sfm.get_support()))
print('Removed featurs-->',np.sum(sfm.estimator_.coef_==0))


# How did it came up with selected features list? It used those coefficients whose absolute value is greater than absolute coefficient mean. 

# In[ ]:


np.sum(np.abs(sfm.estimator_.coef_) > np.abs(sfm.estimator_.coef_).mean())


# In[ ]:


# transforming the training set and test set
X_train_l2 = sfm.transform(X_train)
X_val_l2   = sfm.transform(X_val)

RandomForest(X_train_l2, X_val_l2, y_train, y_val)


# Let's compare Lasso with Ridge regularization:
#        ROC AUC Training    Features used     Features removed   ROC AUC Test
# Lasso  0.769               166               203                0.7982 
# Ridge  0.8039              108               38                 0.8063
# 
# As we can see, the results are almost similar. 

# ## Conclusion:
# 
# Learnings from this exercise are:
# 1. How to implement L1 regularization with Logistic Regression?
# 2. How to implement L2 regularization with Logisitc Regression?
# 3. How to print the number of selected features via regularization process?

# Reference: 
# 1. [https://www.kaggle.com/raviprakash438/lasso-and-ridge-regularisation]

# In[ ]:




