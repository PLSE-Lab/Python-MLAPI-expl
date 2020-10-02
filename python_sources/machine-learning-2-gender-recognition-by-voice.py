#!/usr/bin/env python
# coding: utf-8

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


# <a id="1"></a> <br>
# # Logistic Regression
# * When we have binary classification( 0 and 1 outputs) we can use logistic regression

# * Computation graph of logistic regression
# <a href="http://ibb.co/c574qx"><img src="http://preview.ibb.co/cxP63H/5.jpg" alt="5" border="0"></a>
#     * Parameters to be found are weights and bias
#     * Initial values of weight and bias parameters can be chosen arbitrarily
#     * For every iteration, we are going to calculate loss function
#     * Sum of the loss function will be our cost function
#     * We are going to update weight and bias parameters using derivative of cost function and a learning rate
#     * Learning rate is a hyperparameter that is chosen randomly and tuned afterward.
#     * After many iteratios, the cost wil be minimized and we will obtain final weight and bias parameters to be used (our machine will learn them)
#     * Using these final weight and bias parameters we are going to predict a given test data

# * Mathematical expression of log loss(error) function is: 
#     <a href="https://imgbb.com/"><img src="https://image.ibb.co/eC0JCK/duzeltme.jpg" alt="duzeltme" border="0"></a>

#   * Example of a cost function vs weight
#    <a href="http://imgbb.com/"><img src="http://image.ibb.co/dAaYJH/7.jpg" alt="7" border="0"></a>

# **Updating Weight and Bias**
# 
# * alpha = learning rate
# * J: cost function
# * w: weight
# * b: bias
#     <a href="http://imgbb.com/"><img src="http://image.ibb.co/hYTTJH/8.jpg" alt="8" border="0"></a>
# *  Using similar way, we update bias

# **Derivatives of cost function wrt w and b**
# 
# $$ \frac{\partial J}{\partial w} = \frac{1}{m}x(  y_head - y)^T$$
# $$ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (y_head-y)$$

# * For sigmoid function please visit
# https://en.wikipedia.org/wiki/Sigmoid_function

# In[ ]:


data = pd.read_csv('../input/voice.csv')


# In[ ]:


data.info()


# In[ ]:


data.label.value_counts()


# In[ ]:


# Convert label feature: female = 0 male = 1
data['label'] = [1 if i=='male' else 0 for i in data.label]
data.label.value_counts()


# In[ ]:


# data selection
x_data = data.drop(['label'], axis=1) # it is a matrix excluding label feature
y = data.label.values # it is a vector wich contains only label feature


# In[ ]:


x_data.head()


# In[ ]:


y


# In[ ]:


# normalization of x_data and obtaining x
x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
x.head()


# In[ ]:


# train test split (we split our data into 2 parts: train and test. Test part is 20% of all data)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42) # 0.2=20%


# In[ ]:


# take transpose of all these partial data
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T 


# In[ ]:


# initialize w: weight and b: bias
dimension = 20
def initialize(dimension):
    w = np.full((dimension,1), 0.01)
    b = 0.0
    return w,b


# In[ ]:


# sigmoid function
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

# check sigmoid function
sigmoid(0)


# In[ ]:


def cost(y_head, y_train):
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost_value = np.sum(loss)/x_train.shape[1] # for scaling
    return cost_value


# In[ ]:


def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train)+b
    y_head = sigmoid(z)
    cost_value = cost(y_head, y_train)
    
    # backward propagation
    derivative_weight = (np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = (np.sum(y_head-y_train))/x_train.shape[1]
    
    return cost_value, derivative_weight, derivative_bias


# In[ ]:


def logistic_regression(x_train, x_test, y_train, y_test, learning_rate, num_iteration):
    w,b = initialize(dimension)
    cost_list = []
    index = []
    for i in range(num_iteration):
        cost_value, derivative_weight, derivative_bias = forward_backward_propagation(w,b,x_train,y_train)
        
        # updating weight and bias
        w = w-learning_rate*derivative_weight
        b = b-learning_rate*derivative_bias

        if i % 10 == 0:
            index.append(i)
            cost_list.append(cost_value)
            print('cost after iteration {}: {}'.format(i,cost_value))
    
    # in for loop above, we have obtained final values of parameters(weight and bias): machine has learnt them 
           
    z_final = np.dot(w.T,x_test)+b
    z_final_sigmoid = sigmoid(z_final) #z_final value after sigmoid function
    
    # prediction
    y_prediction = np.zeros((1,x_test.shape[1]))
    # if z_final_sigmoid is bigger than 0.5, our prediction is sign 1 (y_head_=1)
    # if z_final_sigmoid is smaller than 0.5, our prediction is sign 0 (y_head_=0)
    for i in range(z_final_sigmoid.shape[1]):
        if z_final_sigmoid[0,i]<= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
            
    # print test errors
    print('test accuracy: {} %'.format(100-np.mean(np.abs(y_prediction-y_test))*100))
    
    # plot iteration vs cost function
    plt.figure(figsize=(15,10))
    plt.plot(index, cost_list)
    plt.xticks(index, rotation='vertical')
    plt.xlabel('number of iteration', fontsize=14)
    plt.ylabel('cost', fontsize=14)
    plt.show()          


# In[ ]:


# run the program
# Firstly, learning_rate and num_iteration are chosen randomly. Then it is tuned accordingly
logistic_regression(x_train, x_test, y_train, y_test, learning_rate=1.5, num_iteration=200)


# # LOGISTIC REGRESSION USING SKLEAR

# In[ ]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train.T,y_train.T)


# In[ ]:


# prediction of test data
log_reg.predict(x_test.T)


# In[ ]:


# actual values
y_test


# At first glance, we see that 21th value of predicted data is 1, however it is 0 in actual data (y_test). There can be also other wrong predictions

# In[ ]:


print('test_accuracy: {}'.format(log_reg.score(x_train.T,y_train.T)))

