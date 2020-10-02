#!/usr/bin/env python
# coding: utf-8

# <center><b><h3>Introduction to Descision Tree <h3><Center>

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


# Import all the needed libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


# Loading the dataset.
dataset = "../input/drug200.csv"
df_data = pd.read_csv(dataset)
df_data.head() # Checking the top 5 rows of the dataset.


# In[ ]:


df_data["Drug"].value_counts() # Check the count of each type of Drug in the dataset.


# In[ ]:


# Converting pandas dataframe to numpy array to use Scikit learn library and removing the target variable.
X = df_data[["Age","Sex","BP","Cholesterol","Na_to_K"]].values
print(X[0:5])


# In[ ]:


# Converting the catrgorical values into continous(int) as sklearn doesnot support categorical variables.
# Using pandas.get_dummies().
from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]


# In[ ]:


y = df_data["Drug"].values
print(y[0:5])


# In[ ]:


# Create a Train\Test Split.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 3)
print('Train set : ', X_train.shape, y_train.shape)
print('Test set : ', X_test.shape, y_test.shape)


# In[ ]:


# Train the model using the training dataset.
model = DecisionTreeClassifier(criterion = "entropy", max_depth=4)
model.fit(X_train, y_train)


# In[ ]:


y_hat = model.predict(X_test)
print(y_hat)


# In[ ]:


# Compare the predicited and the actual values.
print(y_hat[0:10])
print(y_test[0:10])


# In[ ]:


# Evaluate the model.
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test, y_hat)
print('Accuracy of the model is : ', accuracy)


# In[ ]:




