#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# all imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.


# # Import Data using Pandas

# In[ ]:


data = pd.read_csv("../input/column_2C_weka.csv")


# # Check for null/missing values and correlation

# In[ ]:


data.isnull().values.any()
#no such null or missing values


# In[ ]:


data.corr()
# TODO decide if co-relation between sacral_Scope and pelvic incidence is significant


# In[ ]:


data.head(5)


# In[ ]:


features = data.values
labels = features[:,6]
features = features[:,0:6]


# # Split data in training and testing set with test set having 30% data

# In[ ]:


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3)


# # Check for skewedness in split

# In[ ]:


print(len(labels_train[labels_train=="Abnormal"])*100/len(labels_train))
print(len(labels_test[labels_test=="Abnormal"])*100/len(labels_test))
print(len(labels_train[labels_train=="Normal"])*100/len(labels_train))
print(len(labels_test[labels_test=="Normal"])*100/len(labels_test))
# we have  66% abnormal samples in training set and 70% in test set.
# consistent with the overall data


# # Using RandomForest Classifier for classification

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf = RandomForestClassifier(n_estimators=25,min_samples_split=2)


# In[ ]:


from sklearn.naive_bayes import GaussianNB


# In[ ]:


clf.fit(features_train, labels_train)


# In[ ]:


pred = clf.predict(features_test)


# In[ ]:


print(accuracy_score(pred, labels_test))
print(classification_report(pred, labels_test))


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


parameters = {'n_estimators' : np.arange(10,100), 'min_samples_split' : np.arange(2,100)}


# In[ ]:


gridCV = GridSearchCV(clf,parameters)
gridCV.fit(features,labels)
print(gridCV.best_score_)
print(gridCV.best_estimator_)


# # We are getting around 84% accuracy with good precision and recall score. Parameters tuned using GridCV.
# Run GridCV snipped with caution. Takes time.

# In[ ]:




