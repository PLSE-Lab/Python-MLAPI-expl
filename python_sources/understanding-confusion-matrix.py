#!/usr/bin/env python
# coding: utf-8

# **Confusion Matrix**
# 
# Confusion matrix is an important tool to evaluate our classifier performance. It provides us a clear picture of the performance of our classifier. It creates a matrix where we can find the frequency of hits and misses of each labels.
# To compute the confusion matrix, you need to have a set of predictions, so they can be compared to actual targets.
# 
# Confusion matrix can be used to compute following parameters:
# * Precission
# * Recall
# * F1-Score
# 
# Note: This kernel intend to help newbies to understand confusion metrics and make sense of Precision, Recall and F1 Score.
# 
# **For demonstrating the confusion matrix we will make use of the Diabetes Classification dataset and will perform following steps.**
# * Split the data set into training and test sets.
# * Train the model using RandomForestClassifier.
# * Predict on the test set.
# * Generate confusion Matrix using sklearns confusion_matrix.
# *  Calculate Precision, Recall and F1-Score

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/train.csv")


# In[ ]:


#Lets have a look into the matadata 
data.info()


# In[ ]:


data.describe()


# In[ ]:


#Lets have a look into some sample data
data.head()


# In[ ]:


#Provide features for X and label for y
X = data.drop('diabetes',axis=1)
y = data['diabetes']
# Split the data into traing and test sets
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)
#Initialize Random Forest Classifier
rfc = RandomForestClassifier()
#Fit model on the training Data
rfc.fit(X_train,y_train)
#Make prediction
predictions = rfc.predict(X_test)


# In[ ]:


#Generate Confusion Matrix
conf_matrix = confusion_matrix(predictions,y_test)
print(conf_matrix)


# ![](https://cdncontribute.geeksforgeeks.org/wp-content/uploads/Confusion_Matrix1_1.png)
# 
# We can visualize the confusion matrix that is generated for our model's prediction as below: 
# 
#            0       1
#     0     117     34
#     1     14      38
#     
#    Here in this matrix we have:
#    
#    For 0:
#    True Positive(TP)     = 117,
#    False Positive(FP)    = 14,
#    False Negative(FN)  = 34
#    
#    For 1:
#    True Positive(TP) = 38,
#    False Positive(FP) = 34,
#    False Negative(FN) = 14

# Precision, Recall and F1-Score can be calculated as fellow:
# * Precision  = TP / (TP + FP)
# * Recall       = TP / (TP + FN)
# * F1-Score  = 2 ( Precision * Recall)/(Precision + Recall)
# 

# In[ ]:


#Lets calculate Precision, Recall and F1 score for label 0 and 1
#For Label 0
tp = conf_matrix[0,0]
fp = conf_matrix[1,0]
fn = conf_matrix[0,1]

precision  = tp / (tp + fp)
recall     = tp / (tp + fn)
f1_score   = 2*( precision * recall)/(precision + recall)

print('precision, recall and f1-score for label 0')
print('The precision for label 0 is: {0:.2f}'.format(precision))
print('The recall for label 0 is: {0:.2f}'.format(recall))
print('The f1-score for label 0 is: {0:.2f}'.format(f1_score))
print('\n')

#For Label 1 

tp = conf_matrix[1,1]
fp = conf_matrix[0,1]
fn = conf_matrix[1,0]

precision  = tp / (tp + fp)
recall     = tp / (tp + fn)
f1_score   = 2*( precision * recall)/(precision + recall)

print('precision, recall and f1-score for label 1')
print('The precision for label 1 is: {0:.2f}'.format(precision))
print('The recall for label 1 is: {0:.2f}'.format(recall))
print('The f1-score for label 1 is: {0:.2f}'.format(f1_score))


# We can Validate our calculation by comparing it with sklearn Metrics classification_report.

# In[ ]:


print(classification_report(predictions,y_test))


# In[ ]:




