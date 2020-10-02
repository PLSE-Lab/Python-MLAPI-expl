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


# Importing the libraries

# In[ ]:


import pandas as pd


# Importing dataset

# In[ ]:


df = pd.read_csv("/kaggle/input/breast_cancer.csv")


# In[ ]:


df.head()

Creating X and y
# In[ ]:


X = df.iloc[:, 1:-1].values
y = df["Class"].values


# In[ ]:


y


# Splitting the dataset into training and testing

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Training the logistic regression model using train set

# In[ ]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state = 0)
model.fit(X_train, y_train)


# Predicting test set result

# In[ ]:


y_predicted = model.predict(X_test)


# In[ ]:


model.score(X_test, y_predicted)


# Making the confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print(cm)


# Accuracy

# In[ ]:


(84+47)/(84+47+3+3)


# Computing the accuracy with K-Fold cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
print("Accuracy {:.2f}%".format(accuracies.mean() * 100))
print("Standard Deviation {:.2f}%".format(accuracies.std() * 100))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




