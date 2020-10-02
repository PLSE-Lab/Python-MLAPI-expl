#!/usr/bin/env python
# coding: utf-8

# # INTRODUCTION
# * In this kernel, we will investigate Indian diabete data and try to apply logistic regression.
# 
# <br>Content:
# 1. [Import Libraries](#1)
# 1. [Reading Data](#2)
# 1. [Normalization](#4)
# 1. [Train/Test Split](#5)
# 1. [Parameter initialize and sigmoid function](#6)
# 1. [Updating Parameters](#7)
# 1. [Prediction](#8)
# 1. [Logistic Regression](#9)
# 1. [Test the Model](#10)

# <a id="1"></a> <br>
# # Import Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


from IPython.display import Image
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id="2"></a> <br>
# # Reading Data

# In[ ]:


data = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
data.head()


# In[ ]:


data.columns


# # =============================================================================
# # Means of Columns
# 
# * Pregnanices: Number of times pregnant
# * Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
# * BloodPressure: Diastolic blood pressure (mm Hg)
# * SkinThickness: Triceps skin fold thickness (mm)
# * Insulin: 2-Hour serum insulin (mu U/ml)
# * BMI: Body mass index (weight in kg/(height in m)^2)
# * DiabetesPedigreeFunction: Diabetes pedigree function
# * Age: Age (years)
# * Outcome: Class variable (0 or 1) 268 of 768 are 1, the others are 0
# 
# # =============================================================================

# In[ ]:


x_data = data.drop(["Outcome"],axis=1)
y = data.Outcome.values

# we seperate the result( Outcome column ) and other variables from each other.


# <a id="4"></a> <br>
# # Normalization

# * Features which have too much numeric value can be dominated less numeric value features
# * so, we normalize our data to predict most truth machine learning model.
# * Normalizing is compress the data values between 0-1 as proportionally.

# In[ ]:


x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
x.head()


# <a id="5"></a> <br>
# # Train/Test Split

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

# we will do matrice product, for matrice product first matrix's column and second matrix's row must be same so we will transpose our train/test data

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


# #### now, features are row and values are column

# <a id="6"></a> <br>
# # Parameter Initialize and Sigmoid Function

# In[ ]:


def fill_weights_and_bias(sizeofcolumn):
    w = np.full((sizeofcolumn,1),0.01)
    b = 0.00
    return w,b

# w = weights, b = bias 


# In[ ]:


def sigmoid(z):
    #sigmoid function returns y_head value
    y_head = (1 / ( 1 + np.exp(-z))) # its formula of sigmoid func.
    return y_head


# In[ ]:


def forward_backward_propagation(w,b,x_train,y_train):
    # we must use weights and bias for training model
    # we must change w and b for appropriate shape to matrice product
    
    z = np.dot(w.T,x_train) + b
    
    y_head = sigmoid(z)
    
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head) #thats formula for our wrong predictions
    cost = (np.sum(loss))/x_train.shape[1] # thats average of loss 
    # forward propagation is completed
    
    # backward propagation
    
    derivative_weight = (np.dot(x_train,((y_head-y_train).T))) / x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train) / x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    
    return cost,gradients

    #this func loop 1 times but we want to update our data as we learn new datas.


# <a id="7"></a> <br>
# # Updating Parameters

# In[ ]:


def update(w,b,x_train,y_train,learning_rate,loopnumber):
    
    cost_list = []
    cost_list2  =[]
    index = []
    
    # updating(learning) parameters is loopnumber times
    for i in range(1,loopnumber):
        # make forward and backward propagation and find cost and gradients
        cost,gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)
        
        # updating 
        w = w - learning_rate*gradients["derivative_weight"]
        b = b - learning_rate*gradients["derivative_bias"]
        
        # we may want information about progress
        if( i % 10 == 0):
            cost_list2.append(cost)
            index.append(i)
            print("Cost after {} times loop: {}".format(i,cost))
        
    # showing progress as visual is important
    parameters = {"weights" : w,"bias" : b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Loop")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


# <a id="8"></a> <br>
# # Prediction

# In[ ]:


def predict(w,b,x_test):
    
    z = sigmoid(np.dot(w.T,x_test)+b)
    y_prediction = np.zeros((1,x_test.shape[1]))
            
    # if z is bigger than 0.5, our prediction is sign one (y_head = 1)
    # if z is smaller than 0.5, our prediction is sign zero ( y_head = 0)
    
    for i in range(1,x_test.shape[1]):
        if (z[0,i] <= 0.5):
            y_prediction[0,i] == 0
        else:
            y_prediction[0,i] == 1
            
    return y_prediction


# <a id="9"></a> <br>
# # Logistic Regression

# In[ ]:


def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,loopnumber):
    # initialize
    sizeofcolumn = x_train.shape[0]
    w,b = fill_weights_and_bias(sizeofcolumn)
    
    # forward and backward propagation
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,loopnumber)
    
    y_prediction_test = predict(parameters["weights"], parameters["bias"], x_test)
    # y_prediction_test our y values for test data now we will comparise each other
    
    #print test erros
    print("Test accuracy is: {}".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100 ))


# <a id="10"></a> <br>
# # Testing our Logistic Regression Model

# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 1, loopnumber=600)


# In[ ]:




