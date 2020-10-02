#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression
# 

# ## Suad Emre UMAR
# 
# 

# ## Table of Contents
# ***
# * [1. Introduction](#c) <br>
# * [2. Importing dataset and data preprocessing](#2) <br>
# * [3. Normalization ](#3) <br>
# * [4. Logistic Regression](#4) <br>
#   * [4.1. Train Test Split](#4.1) <br>
#   * [4.2. Parameter Initialize and Sigmoid Function](#4.2) <br>
#   * [4.3. Forward & Backward Propagation](#4.3) <br>
#   * [4.4. Updating(learning) Parameters](#4.4) <br>
#   * [4.5. Predict](#4.5) <br>
#   * [4.6. Logistic Regression](#4.6) <br>
#   * [4.7. Logistic Regression with sklearn](#4.7) <br>
# * [5.Conclusion](#5) <br>    

# ## 1.Introduction
# <a id="c"></a>
# In this kernel, i will work on Logistic Regression.
# ### What is the Logistic Regression
# * When data have binary classification (outputs : 0 or 1), we can use logistic regression.
# * Logistic regression is a predictive analysis.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## 2.Importing dataset and data preprocessing
# <a id="2"></a>

# * First of all i am looking to data.
# * I am cheching the colums and i must drop the columns which is not relating with my predictions.
# * If i use the unusefuly columns on my Logistics model,it negatively affects on model.
# * Then i will determinate to x,y values
# 

# In[ ]:


data = pd.read_csv("../input/voice.csv")


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.label.value_counts()


# * I want to put boundry like female : 0 , male : 1
# * Weil The model can determine which data belongs to which class (0 or 1).

# In[ ]:


data.label = [0 if each=='female' else 1 for each in data.label]


# In[ ]:


data.head()


# In[ ]:


# I determined x and y
# y : outputs 
# x : features
y = data.label.values
x_data = data.drop(["label"],axis=1) 


# In[ ]:


y


# In[ ]:


x_data.head()


# In[ ]:


data.head()


# ## 3.Normalization
# <a id="3"></a>

# In[ ]:


#. I muss make all my data's values between 0 and 1. Because no one data should be affected by the size of other data. 
#normalization =(x-min(x))/(max(x)-min(x))
x = (x_data - np.min(x_data)) / (np.max(x_data)-np.min(x_data)).values


# In[ ]:


x.head()


# ## 4. Logistic Regression
# <a id="4"></a>

# ### 4.1. Train and Test Split
# <a id="4.1"></a>

# In[ ]:


# we need x_train,x_test,y_train,y_test 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train = x_train.T
x_test  = x_test.T
y_train = y_train.T
y_test  = y_test.T

print("x_train: ", x_train.shape)
print("x_test: ", x_test.shape)
print("y_train: ", y_train.shape)
print("y_test: ", y_test.shape) 


# ### 4.2. Parameter Initialize and Sigmoid Funtion
# <a id="4.3"></a>

# In[ ]:


def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

#sigmoid function
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head 


# ### 4.3. Forward & Backward Propagation
# <a id="4.3"></a>

# In[ ]:


def forward_backward_propagation(w,b,x_train,y_train):
    #forward propagation
    z=np.dot(w.T,x_train)+b
    y_head=sigmoid(z)
    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost=(np.sum(loss))/x_train.shape[1] # x_train.shape[1] is for scaling
    
    #backward propagation
    # In backward propagation we will use y_head that found in forward propagation
    derivative_weight=(np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1] is for scaling
    derivative_bias=np.sum(y_head-y_train)/x_train.shape[1]                   # x_train.shape[1] is for scaling
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    
    return cost,gradients 


# ### 4.4. Updating(learning) Parameters
# <a id="4.4"></a>

# In[ ]:


# Updating(learning) parameters
def update(w, b, x_train, y_train, learning_rate,number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    # updating(learning) parameters is number_of_iterarion times
    for i in range(number_of_iteration):
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
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list 


# ### 4.5. Predict
# <a id="4.5"></a>

# In[ ]:


#prediction
def predict(w,b,x_test):
    # x_test is an input for forward propagation
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


# ### 4.6. Logsitic Regression
# <a id="4.6"></a>

# In[ ]:


#Logistic Regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 20
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print train/test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 50) 


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 200) 


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 300)  


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 500)  


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 2, num_iterations = 500)  


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.5, num_iterations = 500) 


# ### 4.7. Logisic Regression With Sklearn
# <a id="4.7"></a>

# In[ ]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T))) 


# ## 5. Conclusion
# <a id="5"></a>

# We are following those steps for ML.
