#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing Data
data=pd.read_csv("../input/creditcard.csv")


# Exploratory Data Analysis

# In[ ]:


data.info()


# In[ ]:


data.head(5)


# In[ ]:


#Checking subcount of class variable
data['Class'].value_counts()


# In[ ]:


#Checking Total missing value
data.isnull().sum(axis=0)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


#Checking count of Target variable using countplot
sns.countplot('Class',data=data)


# In[ ]:



#sns.distplot(data['Class'],kde=True)


# In[ ]:


#checking Corelation between independent feature
plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(),annot=True, fmt=".1g")


# In[ ]:


#Corelation matrix
data.corr()


# Dividing data in dependent(y) and independent(X) features

# In[ ]:


X=data.iloc[:,0:-1]


# In[ ]:


y=data.iloc[:,-1]


# In[ ]:


X.head(2)


# In[ ]:


y.head(2)


# In[ ]:


from sklearn.model_selection import train_test_split


# Spliting training data using train test split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Fitting decision tree classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy',random_state=0)


# In[ ]:


#Fitting model
classifier.fit(X_train,y_train)


# In[ ]:


#predicting the value
y_pred=classifier.predict(X_test)


# Model Evaluation

# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


#Building confusion matrix
cm=confusion_matrix(y_test,y_pred)


# In[ ]:


cm


# In[ ]:


from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,f1_score


# In[ ]:


probs=classifier.predict_proba(X_test)


# In[ ]:


probs=probs[:,1]


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test,probs)


# In[ ]:


#ROC Curve
plt.plot([0, 1], [0, 1], linestyle='--')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='*')


# In[ ]:


# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, probs)


# In[ ]:


#Calculate f1 score
f1 = f1_score(y_test, y_pred)


# In[ ]:


f1


# Random Forest classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


Rclassifier=RandomForestClassifier(n_estimators=500,criterion='entropy')


# In[ ]:


Rclassifier.fit(X_train,y_train)


# In[ ]:


y_pred1=Rclassifier.predict(X_test)


# In[ ]:


Rcm=confusion_matrix(y_test,y_pred1)


# In[ ]:


Rcm


# In[ ]:


#Calculate f1 score
R_f1 = f1_score(y_test, y_pred1)


# In[ ]:


R_f1

