#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# * [Reading Data](#1)
# * [Normalization](#2)
# * [Train Test Split](#3)
# * [Parameter Initializing and Sigmoid Function](#4)
# * [Forward and Backward Propogation](#5)
# * [Updating Parameters](#6)
# * [Prediction](#7)
# * [Logistic Regression](#8)
# * [Logistic Regression with Sklearn](#9)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# <a id="1"></a> <br>
# # Reading Data

# In[ ]:


df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")


# In[ ]:


df.info()


# # Data Description
# **age**: The person's age in years
# 
# **sex**: The person's sex (1 = male, 0 = female)
# 
# **cp**: The chest pain experienced (Value 1: typical angina, Value 2: atypical angina, Value 3: non-anginal pain, Value 4: asymptomatic)
# 
# **trestbps**: The person's resting blood pressure (mm Hg on admission to the hospital)
# 
# **chol**: The person's cholesterol measurement in mg/dl
# 
# **fbs**: The person's fasting blood sugar (> 120 mg/dl, 1 = true; 0 = false)
# 
# **restecg**: Resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy by Estes' criteria)
# 
# **thalach**: The person's maximum heart rate achieved
# 
# **exang**: Exercise induced angina (1 = yes; 0 = no)
# 
# **oldpeak**: ST depression induced by exercise relative to rest ('ST' relates to positions on the ECG plot. See more here)
# 
# **slope**: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3: downsloping)
# 
# **ca**: The number of major vessels (0-3)
# 
# **thal**: A blood disorder called thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect)
# 
# **target**: Heart disease (0 = no, 1 = yes)

# In[ ]:


df.head()


# In[ ]:


y = df.target.values
x_data = df.drop(["target"],axis =  1)


# <a id="2"></a> <br>
# # Normalization

# In[ ]:


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


# <a id="3"></a> <br>
# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
# by taking transpose of data we switch the location of rows and colums
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


# <a id="4"></a> <br>
# # Parameter Initializing and Sigmoid Function

# In[ ]:


def initialize_weight_and_bias(dimension):
   
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

# w,b = initialize_weight_and_bias(30)

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


# <a id="5"></a> <br>
# # Forward and Backward Propogation

# In[ ]:


def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    
    return cost,gradients


# <a id="6"></a> <br>
# # Updating Parameters

# In[ ]:


def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iterarion):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    # we update(learn) parameters weights and bias
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


# <a id="7"></a> <br>
# # Prediction

# In[ ]:


def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction


# <a id="8"></a> <br>
# # Logistic Regression

# In[ ]:


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weight_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 15)


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 2, num_iterations = 30)


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, num_iterations = 50)


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, num_iterations = 100)


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 300)


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, num_iterations = 300)


# <a id="9"></a> <br>
# # Linear Regression with Sklearn

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))

