#!/usr/bin/env python
# coding: utf-8

# Hi, 
# My Objective: Try to predict if the mushroom is poisionous or not. I have considered all 22 variables. I did not eliminate any variable maybe I should have eliminate some. I have tried KNN and Keras before on this dataset all of my predictions were very accurate. First I thought that the model is overfitted. But I have checked other Kernels  and it seems that they have also very accurate predictions. I think the reason for that may be the existence of a certain rule for classifying mushrooms. Or the model is really overfitted and we are all doing sth wrong.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


#packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#dataset

dataset = pd.read_csv('../input/mushrooms.csv')


# In[ ]:


#explore dataset

dataset.head(10)
dataset.describe()
dataset.info()
#dataset.loc[dataset["EDIBLE"].isnull()]


# In[ ]:


#assign X and y
X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]].values
y = dataset.iloc[:, [0]].values
#X = pd.DataFrame(X)
#y = pd.DataFrame(y)


# In[ ]:


# Encoding categorical data for X dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_0 = LabelEncoder()
X[:, 0] = labelencoder_X_2.fit_transform(X[:, 0])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_2.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_2.fit_transform(X[:, 4])
labelencoder_X_5 = LabelEncoder()
X[:, 5] = labelencoder_X_2.fit_transform(X[:, 5])
labelencoder_X_6 = LabelEncoder()
X[:, 6] = labelencoder_X_2.fit_transform(X[:, 6])
labelencoder_X_7 = LabelEncoder()
X[:, 7] = labelencoder_X_2.fit_transform(X[:, 7])
labelencoder_X_8 = LabelEncoder()
X[:, 8] = labelencoder_X_2.fit_transform(X[:, 8])
labelencoder_X_9 = LabelEncoder()
X[:, 9] = labelencoder_X_2.fit_transform(X[:, 9])
labelencoder_X_10 = LabelEncoder()
X[:, 10] = labelencoder_X_2.fit_transform(X[:, 10])
labelencoder_X_11 = LabelEncoder()
X[:, 11] = labelencoder_X_2.fit_transform(X[:, 11])
labelencoder_X_12 = LabelEncoder()
X[:, 12] = labelencoder_X_2.fit_transform(X[:, 12])
labelencoder_X_13 = LabelEncoder()
X[:, 13] = labelencoder_X_2.fit_transform(X[:, 13])
labelencoder_X_14 = LabelEncoder()
X[:, 14] = labelencoder_X_2.fit_transform(X[:, 14])
labelencoder_X_15 = LabelEncoder()
X[:, 15] = labelencoder_X_2.fit_transform(X[:, 15])
labelencoder_X_16 = LabelEncoder()
X[:, 16] = labelencoder_X_2.fit_transform(X[:, 16])
labelencoder_X_17 = LabelEncoder()
X[:, 17] = labelencoder_X_2.fit_transform(X[:, 17])
labelencoder_X_18 = LabelEncoder()
X[:, 18] = labelencoder_X_2.fit_transform(X[:, 18])
labelencoder_X_19 = LabelEncoder()
X[:, 19] = labelencoder_X_2.fit_transform(X[:, 19])
labelencoder_X_20 = LabelEncoder()
X[:, 20] = labelencoder_X_2.fit_transform(X[:, 20])
labelencoder_X_21 = LabelEncoder()
X[:, 21] = labelencoder_X_2.fit_transform(X[:, 21])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
#encoding for y
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y_0 = LabelEncoder()
y[:, 0] = labelencoder_y_0.fit_transform(y[:, 0])


# In[ ]:


#split data -try the model with random state 42 and also it is chwcked with random state 0
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
y_train = y_train.astype(int)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = y_pred.round().astype(int)
y_test = y_test.astype(int)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

