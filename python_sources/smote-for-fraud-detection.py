#!/usr/bin/env python
# coding: utf-8

# # Investigate SMOTE for a simple classifier for Fraud Detection   
# ### This kernel is higly inspired from [Khyati Mahendru post on Medium](https://medium.com/analytics-vidhya/balance-your-data-using-smote-98e4d79fcddb)

# **Fraud Detection** is a dataset higly imbalanced as the vast majority of samples refer to non-fraud transactions.
# 

# # SMOTE   
# **S**ynthetic **M**inority **O**versampling **TE**chnique   
#  >This technique generates synthetic data for the minority class.
#  SMOTE proceeds by joining the points of the minority class with line segments and then places artificial points on these lines.
#  
# The SMOTE algorithm works in 4 simple steps:
# 
# 1. Choose a minority class input vector
# 2. Find its k nearest neighbors (k_neighbors is specified as an argument in the SMOTE() function)
# 3. Choose one of these neighbors and place a synthetic point anywhere on the line joining the point under consideration and its chosen neighbor
# 4. Repeat the steps until data is balanced
# 
# SMOTE is implemented in Python using the [imblearn](https://imbalanced-learn.readthedocs.io/en/stable/install.html) library   
# (to install use: `pip install -U imbalanced-learn`).   
# 
# Additional resources on SMOTE and related tasks:   
# 
# [SMOTE oversampling](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)   
# 
# [SMOTE docs & examples](https://imbalanced-learn.readthedocs.io/en/stable/auto_examples/index.html)   
# 
# [Tips for advanced feature engineering](https://towardsdatascience.com/4-tips-for-advanced-feature-engineering-and-preprocessing-ec11575c09ea)
# 

# In[ ]:


# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# import logistic regression model and accuracy_score metric
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, cohen_kappa_score
from imblearn.over_sampling import SMOTE, SVMSMOTE


# In[ ]:


# Helper functions to compute and print metrics for classifier
def confusion_mat(y_true,y_pred, label='Confusion Matrix - Training Dataset'):
    print(label)
    cm = pd.crosstab(y_true, y_pred, rownames = ['True'],
                  colnames = ['Predicted'], margins = True)
    print(pd.crosstab(y_true, y_pred, rownames = ['True'],
                  colnames = ['Predicted'], margins = True))
    return cm

def metrics_clf(y_pred,y_true, print_metrics=True):
    acc=accuracy_score(y_true, y_pred)
    bal_acc=balanced_accuracy_score(y_true, y_pred)
    f1 =f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    if print_metrics:
        print(f'Accuracy score = {acc:.3f}\n')
        print(f'Balanced Accuracy score = {bal_acc:.3f}\n')
        print(f'F1 Accuracy score = {f1:.3f}\n')
        print(f'Cohen Kappa score = {kappa:.3f}\n')
    return (acc,bal_acc,f1, kappa)


# # Load data

# In[ ]:


# Show full output in cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'


# In[ ]:


# Load data
data = pd.read_csv('../input/creditcardfraud/creditcard.csv')


# In[ ]:


# Show five sampled records
data.sample(5)


# In[ ]:


# Show proportion of Classes
# 1 means Fraud, 0 Normal
_= data['Class'].value_counts().plot.bar(color=['coral', 'deepskyblue'])
data['Class'].value_counts()
print('Proportion of the classes in the data:\n')
print(data['Class'].value_counts() / len(data))


# 

# In[ ]:


# Remove Time from data
data = data.drop(['Time'], axis = 1)
# create X and y array for model split
X = np.array(data[data.columns.difference(['Class'])])
y = np.array(data['Class']).reshape(-1, 1)
X
y


# ## Scale data

# > Split data into Training and Test using stratify = Class return arrays with the same proportion of classes   
# although these are highly imbalanced (0.998 for "0" class and 0.002 for "1" class)!!

# In[ ]:


# split into training and testing datasets using stratify, i.e. same proportion class labels (0/1) in training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 2, shuffle = True, stratify = y)


# In[ ]:


print('Proportion of the classes in training data:\n')
unique, counts = np.unique(y_train, return_counts=True)
print(f'"{unique[0]}": {counts[0]/len(y_train):.3f}')
print(f'"{unique[1]}": {counts[1]/len(y_train):.3f}')


# In[ ]:


print('Proportion of the classes in test data:\n')
unique, counts = np.unique(y_test, return_counts=True)
print(f'"{unique[0]}": {counts[0]/len(y_test):.3f}')
print(f'"{unique[1]}": {counts[1]/len(y_test):.3f}')


# In[ ]:


# standardize the data
# fit only on training data (to avoid data leakage)
scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


# ## Logistic regression model

# In[ ]:


# Fit a simple Logistic regression model
model_LR = LogisticRegression(solver = 'lbfgs')


# ## Model without SMOTE

# In[ ]:


# fit the model
model_LR.fit(X_train, y_train.ravel())

# prediction for training dataset
train_pred = model_LR .predict(X_train)

# prediction for testing dataset
test_pred = model_LR.predict(X_test)


# ## Metrics on Training

# In[ ]:


(acc_train, b_acc_train, f1_train, k_train)=metrics_clf(y_train,train_pred)
cm_train=confusion_mat(y_train.ravel(),train_pred,'Confusion Matrix - Train Dataset (NO SMOTE)')


# ## Metrics on Test

# In[ ]:


(acc_test_sm, b_acc_test_sm, f1_test_sm, k_test_sm)=metrics_clf(y_test,test_pred)
cm_test=confusion_mat(y_test.ravel(),test_pred,'Confusion Matrix - Test Dataset (NO SMOTE)')


# # Metrics analysis
# This simple classifier show very high accuracy but this is not due to correct classification. 
# The model has predicted the majority class for almost all the examples (see confusion matrix),and being the majority class ("0" i.e. not fraud transaction) about 99.8% of total samples this leads to such high accuracy scores.
# More significative metrics for imbalanced dataset are:
# 1. F1 score
# 2. Cohen Kappa
# 3. Balanced accuracy
# 
# For a detailed article/discussion to this metrics refer to [Which Evaluation Metric Should You Choose](https://neptune.ai/blog/f1-score-accuracy-roc-auc-pr-auc)

# ## Model with SMOTE (Synthetic Minority Oversampling Technique)
# [SMOTE parameters](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html#imblearn.over_sampling.SMOTE)

# In[ ]:


sm = SMOTE(random_state = 42, n_jobs=-1, sampling_strategy='minority')
#sm= SVMSMOTE(random_state=42, k_neighbors=20, n_jobs=-1)


# In[ ]:


# generate balanced training data
# test data is left untouched
X_train_new, y_train_new = sm.fit_sample(X_train, y_train.ravel())

# observe that data has been balanced
ax = pd.Series(y_train_new).value_counts().plot.bar(title='Class distribution', y='Count',color=['coral', 'deepskyblue'])
_= ax.set_ylabel('Count')


# In[ ]:


# fit the model on balanced training data
_= model_LR.fit(X_train_new, y_train_new)

# prediction for Training data
train_pred_sm = model_LR.predict(X_train_new)

# prediction for Testing data
test_pred_sm = model_LR.predict(X_test, )


# ## Metrics on Training (SMOTE)
# > ### NOTE how Accuracy is now almost equal to Balanced Accuracy, F1 and Cohen Kappa both improved

# In[ ]:


(acc_test_sm, b_acc_test_sm, f1_test_sm, k_test_sm)=metrics_clf(y_train_new,train_pred_sm)
cm_test=confusion_mat(y_train_new.ravel(),train_pred_sm,'Confusion Matrix - Train Dataset (SMOTE)')


# * ## Metrics on Test (SMOTE)
# > ### On Test set metrics have worsen !!

# In[ ]:


(acc_test_sm, b_acc_test_sm, f1_test_sm, k_test_sm)=metrics_clf(y_test,test_pred_sm)
cm_test_sm=confusion_mat(y_test.ravel(),test_pred_sm,'Confusion Matrix - Test Dataset')

