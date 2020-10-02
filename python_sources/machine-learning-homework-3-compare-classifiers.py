#!/usr/bin/env python
# coding: utf-8

# <a id="0"> <a/>
# # Compare Classifiers <br>
# In this program, we will read file "column-2C-weka.csv"  . This file has class property and 6 numeric features  . Class property has 2 values named "Abnormal" and "Normal" . We will fit 3 different classfiers for this file and compare 3 models and find which model is most suitable for the data <br>
# 
# 1. [Read Data](#1)<br>
# 2. [Train Test Split](#2)<br>
# 3. [Fit Classifiers](#3)<br>
#     3.1. [K Neighbors Classifier ](#3.1)<br>
#     3.2. [Random Forest Classifier](#3.2)<br>
#     3.3. [Logistic Regression ](#3.3)<br>
# 4. [Confusion matrixes ](#4)<br>
# 5. [ROC Curve](#5)<br>
# 6. [Conclusion](#6)<br>

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import models 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# metrics 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve

# graphs 
import matplotlib.pyplot as plt 

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## <div id="1">1. Read Data <div/>

# In[ ]:


data = pd.read_csv('../input/column_2C_weka.csv')
print ( 'distinct class values : ' , data['class'].unique())

data['class'] = [1 if each == 'Normal' else 0 for each in data['class'] ]
x = data.loc[:,data.columns != 'class'] # veya data.drop(['class'], axis = 1) 
y = data.loc[:,'class']

x.head()


# ## <div id="2">2. Train Test Split<div/>

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)

print ('x_train shape: {} '.format(x_train.shape))
print ('y_train shape: {} '.format(y_train.shape))
print ('x_test shape: {} '.format(x_test.shape))
print ('y_test shape: {} '.format(y_test.shape))


# ## <div id="3">3. Fit Classifiers<div/>
# ### <div id="3.1">3.1.K Neighbors Classifier<div/>

# In[ ]:



# find  best k value 
knn_accuracy_list =[]
for k in  range (1,25):
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(x_train, y_train)
    knn_accuracy_list.append(knn.score(x_test, y_test))
    
print ('best k is :{} , best acccuracy is :{} '.format(  knn_accuracy_list.index (np.max(knn_accuracy_list))+1, np.max(knn_accuracy_list)))

#knn classifier 
knn = KNeighborsClassifier(n_neighbors = knn_accuracy_list.index (np.max(knn_accuracy_list))+1 )
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
print ('KNeighborsClassifier test accuracy ' , knn.score(x_test, y_test) )


# ### <div id="3.2">3.2. Random Forest Classifier<div/>

# In[ ]:


rfc_accuracy_list =[]
for r in  range (1,25):
    rfc = RandomForestClassifier(random_state = r)
    rfc.fit(x_train, y_train)
    rfc_accuracy_list.append(rfc.score(x_test, y_test))
    
print ('best r is :{} , best acccuracy is :{} '.format(  rfc_accuracy_list.index (np.max(rfc_accuracy_list))+1, np.max(rfc_accuracy_list)))
    
rfc = RandomForestClassifier(random_state = rfc_accuracy_list.index (np.max(rfc_accuracy_list))+1)
rfc.fit(x_train,y_train)
y_pred_rfc = rfc.predict(x_test)
print ('RandomForestClassifier test accuracy ' , rfc.score(x_test, y_test) )


# ### <div id="3.3">3.3. Logistic Regression<div/>

# In[ ]:


logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred_logreg = logreg.predict(x_test)
print ('LogisticRegression test accuracy ' , logreg.score(x_test, y_test) )


# KNeighborsClassifier test accuracy  **0.8817204301075269**<br>
# RandomForestClassifier test accuracy  **0.8924731182795699**<br>
# LogisticRegression test accuracy  **0.8602150537634409**<br>
# 
# According to accuracy RandomForestClassifier > KNeighborsClassifier > LogisticRegression

# ## <div id="4">4. Confusion matrixes<div/>
# 
# A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known<br><br>
# 
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**Pred No** &nbsp;&nbsp;&nbsp;&nbsp;**Pred Yes**<br>
# **Actual No**  &nbsp;TN&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FP <br>
# **Actual Yes** &nbsp;FN&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;TP<br>
# 
# This is a list of rates that are often computed from a confusion matrix for a binary classifier: <br>
# 
# **Accuracy:** Overall, how often is the classifier correct?<br>
#     * (TP+TN)/total
# **Misclassification Rate: **Overall, how often is it wrong?<br>
#     * (FP+FN)/total
#     * 1 - Accuracy
#     * also known as "Error Rate"
# **True Positive Rate: **When it's actually yes, how often does it predict yes?<br>
#     * TP/actual yes
#     * also known as "Sensitivity" or "Recall"
# **False Positive Rate:** When it's actually no, how often does it predict yes?<br>
#     * FP/actual no 
# **True Negative Rate: **When it's actually no, how often does it predict no?<br>
#     * TN/actual no
#     * 1 - False Positive Rate
#     * also known as "Specificity"
# **Precision: **When it predicts yes, how often is it correct?<br>
#     * TP/predicted yes
# **Prevalence:** How often does the yes condition actually occur in our sample?<br>
#     * Actual yes/total

# In[ ]:


cm = confusion_matrix(y_test,y_pred_knn)
print('KNN Confusion matrix: \n',cm)
print('KNN Classification report: \n',classification_report(y_test,y_pred_knn))

cm = confusion_matrix(y_test,y_pred_rfc)
print('RandomForestClassifier Confusion matrix: \n',cm)
print('RandomForestClassifier Classification report: \n',classification_report(y_test,y_pred_rfc))

cm = confusion_matrix(y_test,y_pred_logreg)
print('LogisticRegression Confusion matrix: \n',cm)
print('LogisticRegression Classification report: \n',classification_report(y_test,y_pred_logreg))


# ## <div id="5">5.ROC Curve<div/>
#     
# The ROC curve is created by plotting the **true positive rate (TPR) ** against the ** false positive rate (FPR)** at various threshold settings. The** true-positive rate** is also known as **sensitivity, recall or probability of detection **in machine learning. 

# In[ ]:



y_pred_knn_prob = knn.predict_proba(x_test)[:,1]
y_pred_rfc_prob = rfc.predict_proba(x_test)[:,1]
y_pred_lr_prob = logreg.predict_proba(x_test)[:,1]

fpr_knn, tpr_knn, thresholds = roc_curve(y_test, y_pred_knn_prob) 
fpr_rfc, tpr_rfc, thresholds = roc_curve(y_test, y_pred_rfc_prob) 
fpr_lr, tpr_lr, thresholds = roc_curve(y_test, y_pred_lr_prob) 

plt.figure (figsize=[13 ,8])
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_knn, tpr_knn, label='KNN')
plt.plot(fpr_rfc, tpr_rfc, label='Random Forest')
plt.plot(fpr_lr, tpr_lr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC')
plt.show()


# ## <div id="6">6. Conclusion<div/>
# 
# According to accurancy and precision , for this data , below are classifiers from best to worst :
# 
# 1. Random Forest Classifier
# 2. K Neighbors Classifier
# 3. Logistic Regression

# 
