#!/usr/bin/env python
# coding: utf-8

# First is exploration on the data using matplotlib and seaborn. Then, there are  different classifier models to predict the quality of the wine.
# 
# 1. Stochastic Gradient Descent Classifier
# 
# 2. Support Vector Classifier(SVC)
# 
# Then there is cross validation evaluation technique to optimize the model performance.
# 
# 1. Grid Search CV
# 
# 2. Cross Validation Score

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
get_ipython().run_line_magic('matplotlib', 'inline')


# Load data

# In[ ]:


wine = pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')


# Lets look into the data

# In[ ]:


wine.head()


# In[ ]:


wine.info()


# Next we see that fixed acidity does not give any specification to classify the quality.

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)


# Next we see that volatile acidity decreases as quality increases.

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)


# Next we see that citric acid increases as quality increases.

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'citric acid', data = wine)


# Next we see that residual suagr does not give any specification to classify the quality.

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'residual sugar', data = wine)


# Composition of chloride also go down as we go higher in the quality of the wine
# 

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'chlorides', data = wine)


# Next we see that free sulphur dioxide does not give any specification to classify the quality.

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)


# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine)


# Sulphates level goes higher with the quality of wine.

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'sulphates', data = wine)


# Alcohol level also goes higher as te quality of wine increases.

# In[ ]:


fig = plt.figure(figsize = (10,6))
sns.barplot(x = 'quality', y = 'alcohol', data = wine)


# Dividing wine as good and bad by giving the limit for the quality.

# In[ ]:


bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])
wine['quality'].value_counts()


# Splitting data.

# In[ ]:


X = wine.drop('quality', axis = 1)
y = wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# **1. SGD CLASSIFIER**

# In[ ]:


sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)


# 88% accuracy using stochastic gradient descent classifier

# In[ ]:


print(classification_report(y_test, pred_sgd))


# **2. Support Vector Classifier**

# In[ ]:


svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)


# Support vector classifier gets 86% accuracy.

# In[ ]:


print(classification_report(y_test, pred_svc))


# **Grid Search CV.**

# Finding best parameters for our SVC model

# In[ ]:


param = { 'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],'kernel':['linear', 'poly','rbf'],'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4] }
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)


# Now lets see what are the best parameters for SVC as returned by the grid search.

# In[ ]:


grid_svc.best_params_


# 1. Now running SVC with the best parameters.
# 2. We see SVC improves from 86% to 90% using Grid Search CV

# In[ ]:


svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))

