#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In this notebook we will go through the following:
# 1. Importing the dataset and inferring basic details from the data
# 2. Performing EDA
# 3. Implementing ML models

# ## 1. Importing the dataset and inferring basic details from the data

# Let's start by reading the data to a dataframe.

# In[27]:


# using pandas to read data
data = pd.read_csv('../input/Iris.csv')


# Now we will see the content of the data to get some idea about the data.

# In[28]:


# .head() method by defalut return first 5 records
# one can pass any number in the method to see more/less records
# example: data.head(x) where x can be any number
data.head()


# In[29]:


# .info method returns the info related to kind of data in the dataframe
data.info()


# Here the *Id* column has no prominent role to play. So let's just take it out from the dataframe.

# In[30]:


# axis=1 to tell pandas to take the column out (axis=1 signifies rows) 
# inplace=True to save some memory of using another variable to store new modified dataframe
data.drop('Id', axis=1, inplace=True)


# In[31]:


# .describe() is used to get some statistical insight from the data
# one thing to note here is that this method basically deals with numeric data
data.describe()


# ## 2. Performing EDA

# Let's start by importing visualization libraries.

# In[32]:


import matplotlib.pyplot as plt
import seaborn as sns

# to get the plots inline with the code
get_ipython().run_line_magic('matplotlib', 'inline')


# It would be a good idea to see how all the features relates with each other.<br>
# Let's do a pair plot to visualize the same.

# In[33]:


sns.pairplot(data)


# From the above plot we can see that PetalLengthCm and PetalWidthCm are quite correlated.
# 
# Let's get correlation of all the features to see how they are correlated to each other.

# In[34]:


# .corr is used to find the correlation of features
# this method basically deals with numeric data only
data.corr()


# From the above table we can infer that the following pairs are closely related to each other:
# * SepalLengthCm and PetalLengthCm
# * SepalLengthCm and PetalWidthCm
# * PetalLengthCm and PetalWidthCm
# 
# We will plot a heatmap to visualize the correlation of the features.

# In[35]:


sns.heatmap(data.corr(), annot=True)


# Before going further we will be converting the species feature to numerical values. This is important because machines understand numbers preety easilt compared to text.
# We will do so by implementing a function which will assign dummy values 0,1,2 to Iris-setosa, Iris-versicolor, Iris-virginica respectively.

# In[36]:


print(data['Species'].unique())


# In[40]:


# assigning dummy values 0,1,2 to Iris-setosa, Iris-versicolor, Iris-virginica respectively.

def getDummies(species):
    if species == 'Iris-setosa':
        return 0
    elif species == 'Iris-versicolor':
        return 1
    else:
        return 2


# In[41]:


data['Species'] = data['Species'].apply(getDummies)


# In[46]:


print(data['Species'].unique())


# ## 3. Implementing ML models
# 
# This is a classification problem, so we will be using the following algorithms:
# * Linear SVC
# * KNeighbours Classifier
# * SVC
# * Ensemble Classifier
# 
# Before we go ahead and implement ML models it would be essential to split our dataset into training and testing sets.
# 
# For doing so we will use train_test_split from scikit-learn

# In[52]:


#for performance metrics
from sklearn.metrics import accuracy_score

#for splitting data
from sklearn.model_selection import train_test_split


# Our target is to predict *Species*. So, other columns are our features. <br>
# We'll be having two datasets X and y with featues and target respectively.

# In[56]:


X = data.drop('Species', axis=1)
y = data['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# **Linear SVC**

# In[57]:


#importing LinearSVC
from sklearn.svm import LinearSVC


# In[59]:


LSVCModel = LinearSVC()  #creating LinearSVC model
LSVCModel.fit(X_train, y_train)  #fitting training data to the model
predictions = LSVCModel.predict(X_test) #predicting values from model
print("Linear SVC Accuracy:", accuracy_score(y_test, predictions)) #accuracy score


# **KNeighbours Classifier**

# In[60]:


#importing KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier


# In[61]:


KNCModel = KNeighborsClassifier()  #creating KNeighborsClassifier model
KNCModel.fit(X_train, y_train)  #fitting training data to the model
predictions = KNCModel.predict(X_test) #predicting values from model
print("Linear SVC Accuracy:", accuracy_score(y_test, predictions)) #accuracy score


# **SVC**

# In[62]:


#importing SVC
from sklearn.svm import SVC


# In[63]:


SVCModel = SVC()  #creating SVC model
SVCModel.fit(X_train, y_train)  #fitting training data to the model
predictions = SVCModel.predict(X_test) #predicting values from model
print("Linear SVC Accuracy:", accuracy_score(y_test, predictions)) #accuracy score


# **Ensemble Classifier**
# 
# *Random Forest Classifier*

# In[64]:


#importing RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier


# In[72]:


RFCModel = RandomForestClassifier(n_estimators=100)  #creating RandomForestClassifier model
RFCModel.fit(X_train, y_train)  #fitting training data to the model
predictions = RFCModel.predict(X_test) #predicting values from model
print("Random Forest Classifier Accuracy:", accuracy_score(y_test, predictions)) #accuracy score


# ### Observations

# From the above results, we can say that all the models performed quite well.
# The **SVC** performed better from the lot.
# 

# In[ ]:




