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


# Let's set the column for the analysis

# In[ ]:


dataset = pd.read_csv(os.path.join(dirname, filename))
X = dataset.iloc[:, 0:30].values
y = dataset.iloc[:, 30].values


# In[ ]:


dataset.head()

Preparing Training & Test set
# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[ ]:


dataset['class'].sum()


# This is important step to feature scale the data set

# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# We are using K-NN algorithm to perform the Fraud detection

# In[ ]:



# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2, algorithm='auto')
classifier.fit(X_train, y_train)


# This is the step where the algo will predict the fraud

# In[ ]:


y_pred = classifier.predict(X_test)


# The system is already predict the fraud, let see how accurate the results from confusion matrix

# In[ ]:



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Wow That's awesome, we got almost 99.99% correct predictions

# In[ ]:


#Accuracy
accuracy =  (cm[0,0]+cm[1,1])/len(X_test)
print(accuracy)


# Here are the results 
# Bingo!
# 

# Now Let's start with SVM

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection  import GridSearchCV
def svc_param_selection(X, y, nfolds):
    Cs = [0.001, 0.01, 0.1, 5, 10,15]
    gammas = [0.001, 0.01, 0.1,0.5, 0.10, 1]
    param_grid = {'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
    grid_search.fit(X_train, y_train)
    grid_search.best_params_
    return grid_search.best_params_






# In[ ]:


svc_param_selection(X_train,y_train,5)


# In[ ]:


classifier = SVC(kernel = 'rbf', random_state = 3, gamma=0.1, degree=3, C=10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

