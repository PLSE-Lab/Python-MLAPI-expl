#!/usr/bin/env python
# coding: utf-8

# # **Spam Filtering using NLP**
# To train 3 different models that classify messages as spam or ham and choose the model with highest precision.
# 
# * SVM Classifier
# * Naive Bayes Classifier
# * Random Forest Classifier

# ## **Importing some basic libraries**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## **Reading the data**

# In[ ]:


df = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')
df.shape


# In[ ]:


df.head()


# In[ ]:


df= df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)
df= df.rename(columns={'v2':'text','v1':'label'})


# In[ ]:


df.head()


# In[ ]:


df.isnull().any()        #check if there is any null data


# ## **Creating bag of words**

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

cv=CountVectorizer(stop_words='english')

feature_vectors= cv.fit_transform(df['text'])


# In[ ]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(feature_vectors,df['label'],test_size=0.3, random_state=42)
print(np.shape(x_train), np.shape(x_test),np.shape(y_train))
print('There are {} samples in the training set and {} samples in the test set'.format(
x_train.shape[0], x_test.shape[0]))


# ## **Modelling**
# 
# **Support Vector Machine**

# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

model_SVM = SVC()
model_SVM.fit(x_train, y_train)

y_pred_SVM = model_SVM.predict(x_test)

print("Training Accuracy using SVM :", model_SVM.score(x_train, y_train))
print("Testing Accuracy using SVM:", model_SVM.score(x_test, y_test))

print('Confusion matrix')
cm1 = confusion_matrix(y_test, y_pred_SVM)
print(cm1)

print("Accuracy Score for Test Set using SVM:", accuracy_score(y_test, y_pred_SVM))


# **Naive Bayes**

# In[ ]:


from sklearn.naive_bayes import MultinomialNB

model_NB= MultinomialNB()
model_NB.fit(x_train,y_train)

y_pred_NB=model_NB.predict(x_test)

print("Training Accuracy using Naive Bayes:", model_NB.score(x_train, y_train))
print("Testing Accuracy using Naive Bayes:", model_NB.score(x_test, y_test))

print('Confusion matrix')
cm2 = confusion_matrix(y_test, y_pred_NB)
print(cm2)

from sklearn.metrics import accuracy_score
print("Accuracy Score for Test Set using Naive Bayes:", accuracy_score(y_test, y_pred_NB))


# **Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

model_RF=RandomForestClassifier(n_estimators=31, random_state=111)
model_RF.fit(x_train,y_train)

y_pred_RF=model_RF.predict(x_test)

print("Training Accuracy using Random Forest:", model_RF.score(x_train, y_train))
print("Testing Accuracy using Random Forest:", model_RF.score(x_test, y_test))

print('Confusion matrix')
cm3 = confusion_matrix(y_test, y_pred_RF)
print(cm3)

from sklearn.metrics import accuracy_score
print("Accuracy Score for Test Set using Random Forest:", accuracy_score(y_test, y_pred_RF))


# ## Performance Metrics
# 
# The Confusion matrix is one of the most intuitive metrics used for finding the correctness and accuracy of the model.
# 
# TN FP
# 
# FN TP
# 
# The precision is the ratio TP / (TP + FP) where TP is the number of true positives and FP the number of false positives.
# 
# For this use case, the precision should be higher i.e. less False Positives(FP) because we don't want ham messages to be classified as spam and hence, missing out on any important message.
# 
# **Comparing the results**
# 
# Support Vector Machine(SVM) performs the best with zero False Positives and highest accuracy.
