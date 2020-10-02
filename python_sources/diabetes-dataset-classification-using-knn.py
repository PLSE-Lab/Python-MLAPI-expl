#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# In[2]:


x_test = pd.read_csv("../input/Diabetes_Xtest.csv")
X_Train = pd.read_csv("../input/Diabetes_XTrain.csv")
Y_Train = pd.read_csv("../input/Diabetes_YTrain.csv")


# In[3]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=17)


# In[4]:


X = X_Train.values
Y = Y_Train.values
print(X.shape)
print(Y.shape)
Y = Y.reshape(576, )
print(Y.shape)


# In[5]:


classifier.fit(X,Y)


# In[6]:


xt = x_test.values
y_pred = classifier.predict(xt)


# In[7]:


x_test = x_test.drop(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'], axis = 1)


# In[8]:


x_test['Outcome'] = y_pred


# In[9]:


x_test.to_csv('diabetes.csv', index=True)

