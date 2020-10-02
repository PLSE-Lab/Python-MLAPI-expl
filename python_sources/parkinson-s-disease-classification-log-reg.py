#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Data Set Information:**
# 
# This dataset is composed of a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). Each column in the table is a particular voice measure, and each row corresponds one of 195 voice recording. The main aim of the data is to discriminate healthy people from those with PD, according to "status" column which is set to 0 for healthy and 1 for PD.

# **Attribute Information:**
# 
# Matrix column entries (attributes):
# MDVP:Fo(Hz) - Average vocal fundamental frequency
# MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
# MDVP:Flo(Hz) - Minimum vocal fundamental frequency
# MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP - Several 
# measures of variation in fundamental frequency
# MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
# NHR,HNR - Two measures of ratio of noise to tonal components in the voice
# status - Health status of the subject (one) - Parkinson's, (zero) - healthy
# RPDE,D2 - Two nonlinear dynamical complexity measures
# DFA - Signal fractal scaling exponent
# spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation 

# In[ ]:


data=pd.read_csv("../input/parkinsons2.csv")


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


#correlation data
f,ax=plt.subplots(figsize=(14,14))
sns.heatmap(data.corr(),annot=True,ax=ax,fmt=".2f")
plt.xticks(rotation=90)
plt.show()


# In[ ]:


#prepare of data
y=data.status.values
x_data=data.drop(["status"],axis=1)
#normlizasyon
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[ ]:


#split data for test%train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)


# In[ ]:


#transpose of each 
x_train=x_train.T
x_test=x_test.T
y_train=y_train.T
y_test=y_test.T


# In[ ]:


#determination of initial values of weights and bias
def initialize_weights_and_bias(dimension):
    b=0.0
    w=np.full((dimension,1),0.01)
    return w,b


# In[ ]:


#the formula of the activation function
def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head


# In[ ]:


#calculate of z, forward and backward propagation
def forward_backward_propagation(w,b,x_train,y_train):
#forward  propagation   
    z=np.dot(w.T,x_train)+b
    y_head=sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost=(np.sum(loss))/x_train.shape[1]    
#backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost,gradients


# In[ ]:


#update parameter values to reduce cost value
def update(w,b,x_train,y_train,learning_rate,number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    for i in range(number_of_iteration):
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


# In[ ]:


#prediction of test values
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
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    dimension =  x_train.shape[0] 
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    # Print test Errors
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1.2, num_iterations = 400)


# **There is also a short version of the above codes.**

# In[ ]:


#Logistic regression with sklearn 
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(x_train.T,y_train.T)
print("test-accuracy : {} ".format(log.score(x_test.T,y_test.T)*100))


# 
# **Logistic Regression  predicted correctly  the class of data at 89.74 percent.  I hope you like. Waiting for comments and suggestions.**
# 
