#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Pima indian diabetes dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 
# The **objective** of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

# The datasets consists of several medical predictor variables and one target variable, Outcome. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

# In[ ]:


import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.model_selection import (KFold, StratifiedKFold,
                                     cross_val_predict, cross_val_score,
                                     train_test_split)

from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB


# In[ ]:


input_data = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")
input_data


# ### EDA

# In[ ]:


input_data.isnull()


# In[ ]:


input_data.describe()


# In[ ]:


input_data.isin([0]).any()


# Find out zeros values in dataset:
# isin([0]) function gives us the features having '0' as a value, which is doesn't make sense. All the features are medically critical, so cannot have zero readings in a healthy human being.  
# 
# We need to impute zeroes with np.nan so that in one go we can impute nan with possibles values, maybe mean or median. 

# In[ ]:


from fancyimpute.knn import KNN


features_with_zero_values = ['BMI', 'BloodPressure', 'Glucose', 'Insulin', 'SkinThickness']
input_data[features_with_zero_values] = input_data[features_with_zero_values].replace(0, np.nan)
data = KNN(k=5).fit_transform(input_data.values)
inputdata = pd.DataFrame(data, columns=input_data.columns)


# In[ ]:


inputdata.isin([0]).any()  


# No zero values after imputation of zeros 

# In[ ]:


inputdata.isna().any()  


# No NA values 

# In[ ]:


inputdata.isnull().any()  


# No Null values. 

# In[ ]:


inputdata.hist(figsize=(12, 12))
plt.show()


# From the above histograms on various features, we observes that range of DiabetesPedigreefunction is very small in comparison to range of insulin. 
# 
# This meant that all the features are not at the same scale. It is preferred to scale the features down to the same magnitude by a process known as feature scaling.S tandardization is one method of feature scaling and does so by replacing the values with their z scores. We will apply standardization to our features, but before doing that, we will split our df into feature (X) and outcome (y) then scale the features.

# ### Model Fitting and Analysis 

# In[ ]:


X = inputdata.drop('Outcome', axis=1)   # input feature vector
y = inputdata['Outcome']                # labelled target vector

scaler = StandardScaler()                # scaling 
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)
X.head()


# Now, all our input features are at the same scale. 

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=10, stratify=y)
X_train.head()


# In[ ]:


y_train.head()


# ### KNN

# In[ ]:


# cross validation kFold

kfold = StratifiedKFold(n_splits=10, random_state = 10)


# In[ ]:


# KNN model
clf = KNeighborsClassifier(n_neighbors=3)

# model fitting
clf.fit(X_train, y_train)


# In[ ]:


scores = cross_val_score(clf, X_train, y_train, cv=kfold)
scores.mean()


# In[ ]:


# prediction on validation data

y_pred = clf.predict(X_valid)
y_pred


# In[ ]:


# confusion matrix

cfm = confusion_matrix(y_pred, y_valid)
cfm


# In[ ]:


# accuracy score
print('accuracy of KNN: ',accuracy_score(y_pred,y_valid))


# ### Naive Bayes
# 
# * Using Gaussian NB on the same dataset to compare with KNN.

# In[ ]:


# model

clf_g = GaussianNB()
clf_g


# In[ ]:


# validation score

scores = cross_val_score(clf_g, X_train, y_train, cv=kfold)
scores.mean()


# In[ ]:


# model fitting
clf_g.fit(X_train, y_train)


# In[ ]:


# prediction on validation data

y_pred = clf_g.predict(X_valid)
y_pred


# In[ ]:


# confusion matrix

cfm = confusion_matrix(y_pred, y_valid)
cfm


# In[ ]:


# accuracy score
print('accuracy of GaussianNB: ',accuracy_score(y_pred,y_valid))


# ### Conclusion
# 
# Gaussian Naive Bayes performed better than the KNearestNeighbours(KNN) on the same Pima Indian Diabetes dataset. 
# 
# 
