#!/usr/bin/env python
# coding: utf-8

# ## This Note Book describes the names, formulas, simple points, and Python implementation code of typical metrics for the model.

# In[ ]:


import numpy as np
import pandas as pd


# # RMSE (Root Mean Squared Error)
# $$RMSE = \sqrt{ \frac 1N \sum ^{N}_{i=1} (y_{i} - \hat{y}_{i})^2}$$
# 
# - Statistically meaningful indicators
# - Outliers must be removed beforehand because they are greatly affected by outliers

# In[ ]:


# RMSE (Root Mean Squared Error)
from sklearn.metrics import mean_squared_error

y_true = [1.0, 1.5, 2.0, 1.2, 1.8]
y_pred = [0.8, 1.5, 1.8, 1.3, 3.0]

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
# 0.55317


# # RMSLE (Root Mean Squared Logarithmic Error)
# $$RMSLE = \sqrt{ \frac 1N \sum ^{N}_{i=1} (log(1 + y_{i}) - log(1 + \hat{y}_{i}))^2}$$
# 
# - Use when the effect of a large value is strong if the objective variable is not converted
# - This indicator focuses on the ratio of the true value to the predicted value

# In[ ]:


# RMSLE (Root Mean Squared Logarithmic Error)
from sklearn.metrics import mean_squared_log_error

y_true = [100, 0, 400]
y_pred = [200, 10, 200]

rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
print(rmsle)
# 1.49449


# # MAE (Mean Absolute Error)
# $$MAE = \frac 1N \sum^{N}_{i=1} |y_{i} - \hat{y}_{i}|$$
# - Indicators that reduce the effects of outliers
# - Difficult to handle when differentiating

# In[ ]:


# MAE (Mean Absolute Error)
from sklearn.metrics import mean_absolute_error

y_true = [100, 160, 60]
y_pred = [80, 100, 100]

mae = mean_absolute_error(y_true, y_pred)
print(mae)
# 40.0


# # accuracy, error rate
# $$accuracy = \frac {TP + TN}{TP + TN + FP + FN}$$
# $$$$
# $$error\ rate = 1 - accuracy$$
# - Unbalanced data makes it difficult to evaluate model performance
# - Not often used in analytical competitions

# In[ ]:


# accuracy, error rate
from sklearn.metrics import accuracy_score

# Binary classification of 0 and 1
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

accuracy = accuracy_score(y_true, y_pred)
print(accuracy)
# 0.625


# # precision, recall
# $$precision = \frac {TP} {TP + FP}$$
# $$$$
# $$recall = \frac {TP} {TP+ FN}$$
# - precision and recall are in a trade-off relationship with each other
# - Focus on precision if you want to reduce false positives
# - Focus on recall if you want to avoid missing a positive example

# In[ ]:


# precision, recall
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# Binary classification of 0 and 1
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(precision, recall)
# 0.75 0.6


# # logloss (cross entropy)
# $$logloss = - \frac 1N \sum^{N}_{i=1} (y_{i}\log{p_{i}} + (1-y_{i})\log(1-p_{i}))$$
# - Representative metrics for classification tasks
# - y is a label indicating whether it is a positive example, p is a probability that it is a positive example

# In[ ]:


# logloss
from sklearn.metrics import log_loss

# True value and predicted probability of binary classification of 0 and 1
y_true = [1, 0, 1, 1, 0, 1]
y_prob = [0.1, 0.2, 0.8, 0.8, 0.1, 0.3]

logloss = log_loss(y_true, y_prob)
print(logloss)
# 0.71356


# ## Extra

# In[ ]:


# confusion matrix
from sklearn.metrics import confusion_matrix

# Binary classification of 0 and 1
y_true = [1, 0, 1, 1, 0, 1, 1, 0]
y_pred = [0, 0, 1, 1, 0, 0, 1, 1]


# TP(True Positive), TN(True Negative), FP(False Positive), FN(False Negative)
tp = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 1))
tn = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 0))
fp = np.sum((np.array(y_true) == 0) & (np.array(y_pred) == 1))
fn = np.sum((np.array(y_true) == 1) & (np.array(y_pred) == 0))

confusion_matrix1 = np.array([[tp, fp], [fn, tn]])
print(confusion_matrix1)

# array([[TP, FP]
#        [FN, TN]])


confusion_matrix2 = confusion_matrix(y_true, y_pred)
print(confusion_matrix2)

# array([[TN, FP]
#        [FN, TP]]) 


# In[ ]:




