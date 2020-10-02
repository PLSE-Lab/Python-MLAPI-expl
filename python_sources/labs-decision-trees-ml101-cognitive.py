#!/usr/bin/env python
# coding: utf-8

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


# About dataset
# Imagine that you are a medical researcher compiling data for a study. You have collected data about a set of patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug c, Drug x and y.
# 
# Part of your job is to build a model to find out which drug might be appropriate for a future patient with the same illness. The feature sets of this dataset are Age, Sex, Blood Pressure, and Cholesterol of patients, and the target is the drug that each patient responded to.
# 
# It is a sample of binary classifier, and you can use the training part of the dataset to build a decision tree, and then use it to predict the class of a unknown patient, or to prescribe it to a new patient.

# In[ ]:


df=pd.read_csv("../input/drug200.csv", delimiter=",")


# Droping the unknown values

# In[ ]:


df.dropna


# In[ ]:


df.head()


# 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
X=df[["Age","Sex","BP","Cholesterol","Na_to_K"]].values
X


# In[ ]:





# Sklearn Decision Trees do not handle categorical variables. But still we can convert these features to numerical values using different options available...

# In[ ]:


from sklearn import preprocessing
le_sex=preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1]=le_sex.transform(X[:,1])


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 


# In[ ]:


X[:5]


# In[ ]:


y=df["Drug"]


# ## Importing train_test_split method from sklearn for splitting the dataset into training and testing values
# 

# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=3)


# 

# ## Checking wether the dimensions of y_test,x_test are same or not. Also, checking for the same observation for x_train and y_train too

# In[ ]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# We will first create an instance of the <b>DecisionTreeClassifier</b> called <b>drugTree</b>.<br>
# Inside of the classifier, specify <i> criterion="entropy" </i> so we can see the information gain of each node.

# In[ ]:



drugTree=DecisionTreeClassifier(criterion="entropy",max_depth=4)


# In[ ]:


drugTree.fit(x_train,y_train)


# In[ ]:


y_predict=drugTree.predict(x_test)
y_predict


# ## Checking for accuracy score

# In[ ]:


from sklearn import metrics
import matplotlib.pyplot as plt
metrics.accuracy_score(y_test,y_predict)


# # The accuracy score is approx 0.9833 out of 1 which means that our Decision Tree Model is 98.33% accurate.

# In[ ]:




