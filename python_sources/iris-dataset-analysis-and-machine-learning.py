#!/usr/bin/env python
# coding: utf-8

# ## Iris Dataset Analysis And Machine Learning

# ## Part 1- Data Preprocessing

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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# ### Importing the Dataset

# In[ ]:


data = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')


# In[ ]:


data.head()


# ### Summary of The Dataset

# In[ ]:


data.info()


# In[ ]:


data.shape


# In[ ]:


data.describe()


# In[ ]:


data.groupby('species').size()


# In[ ]:


data.groupby('species').mean()


# ## Part 2 - Data Visualization

# Box and whisker plots

# In[ ]:


data.plot(kind='box' , sharex = False , sharey = False, figsize=(15,10))


# Histogram

# In[ ]:


data.hist(edgecolor = 'black', linewidth=1.2, figsize=(15,5))


# Boxplot on each feature split out by species
# 

# In[ ]:


data.boxplot(by="species",figsize=(15,10))


# Scatter plot matrix

# In[ ]:


# Using seaborn pairplot to see the bivariate relation between each pair of features
sns.pairplot(data, hue="species")


# From the above, 
# 
# We can see that **Iris-Setosa** is separated from both other species in all the features.
# 

# ## Applying different Classification Models

# In[ ]:


# Importing metrics for evaluation
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[ ]:


# Seperating the data into dependent and independent variables
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# ### Logistic Regression

# In[ ]:


# LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# In[ ]:


# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:


# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# ### Naive Bayes

# In[ ]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# In[ ]:


# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:


# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# ### Support Vector Machine

# In[ ]:


# Support Vector Machine's 
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# In[ ]:


# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:


# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# ### K-Nearest Neighbours

# In[ ]:


# K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# In[ ]:


# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:


# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# ### Decision tree

# In[ ]:


# Decision Tree's
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


# In[ ]:


# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[ ]:


# Accuracy score
from sklearn.metrics import accuracy_score
print('accuracy is',accuracy_score(y_pred,y_test))


# If you have reached till here, So i hope you liked my Analysis.
# 
# If you learned anything new from this dataset then do give a upvote.
# 
# I'm a rookie and any suggestion in the comment box is highly appreciated.
# 
# If you have any doubt reagrding any part of the notebook, feel free to comment your doubt in the comment box.
# 
# Thank you!!
