#!/usr/bin/env python
# coding: utf-8

# Hi everyone
# In this chapter we will learn using logistic regression.
# First we will import required libraries and add data

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


# In[ ]:


# read csv file
data=pd.read_csv('../input/voice.csv')
data.head(7)


# In[ ]:


print(data.info())


# In[ ]:


data.label.unique


# As you can see only 'label' values are object. All others are float64.
# 
# In logistic regression we will train and determine label values. We will use label values as 0 or 1.

# In[ ]:


data.label=[1 if each =="female" else 0 for each in data.label]
data.label.values


# After changing label values to 1 or 0 let see last and first 6 data rows

# In[ ]:


data.tail(6)#last six


# In[ ]:


data.head(6)#first six


# lets see data info again

# In[ ]:


data.info()


# for logistic regression we will determine x and y values

# In[ ]:


y=data.label.values
x_data=data.drop(['label'],axis=1)


# In[ ]:


np.min(x_data)


# In[ ]:


np.max(x_data)


# we need normalize x values

# In[ ]:


#normalization
#(x-min(x))/(max(x)-min(x))
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values


# we will determine x_train, x_test, y_train, y_test for LR

# In[ ]:


#train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2, random_state=42)
#find transpose
x_train=x_train.T
x_test=x_test.T
y_train=y_train.T
y_test=y_test.T
print("x_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)


# we will create initialize_weights_and_bias and sigmoid  functions for LR

# In[ ]:


#parameter initialize and sigmoid function
#dimention=30
def initialize_weights_and_bias(dimension):
    w=np.full((dimension,1),0.01)
    b=0.0
    return w,b
def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head
#sigmoid(0)


# than we will create forward_backward_propagation function

# In[ ]:


#forward propagation
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


# In[ ]:


# Updating(learning) parameters
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


# In[ ]:


# prediction
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


# After creating all required funcs lets define logistic_regression func then check.
# learning_rate = 1, num_iterations = 300

# In[ ]:


#logistic_regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 30
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 300)


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 3, num_iterations = 200)


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 2, num_iterations = 300)


# we tried different learning_rate and  num_iterations values for finding best test accuracy.
# The best one we found test accuracy:97.94952681388013 % for learning_rate = 2, num_iterations = 300.
# 
# 
# Lest use sklearn library for our project.

# In[ ]:


#sklearn with logistic regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("Test Accuracy :{}",format(lr.score(x_test.T,y_test.T)))


# As you can see this is the best one. Test accuracy is 98.11 %.
# 
# I hope you enjoy my tutorial.
# 
# See you soon.
