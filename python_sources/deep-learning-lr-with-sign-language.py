#!/usr/bin/env python
# coding: utf-8

# # Cihan Yatbaz
# ###  29 / 11 / 2018
# 
# 
# 
# 1.  [Introduction:](#0)
# 2. [Preparing Dataset :](#1)
# 3. [Creating Parameters :](#2)
# 4. [Forward and Backward Propagation  :](#3)
# 5. [Updating Parameter :](#4)
# 6. [Prediction Parameter :](#5)
# 7. [ Logistic Regression :](#6)
# 8. [Logistec Regression with Sklearn  :](#7)
# 9. [CONCLUSION :](#8)

# <a id="0"></a> <br>
# ## 1) Introduction
# * We will be working on this kernel Sign Language data. We'll introduce 80% of the sign language we have, and we will try to predict the remaining 20%.
# * In this Kernel we will do the Logistic Regression step by step. Then we will learn how to do it very easily with Sklearn.
# * Let's start by creating our libraries
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# <a id="1"></a> <br>
# ## 2) Preparing Dataset
# ---
# * Now we'll upload our library and then let's see the 0 and 1 signs we'll work on.

# In[ ]:


# Load data set
x_l = np.load('../input/Sign-language-digits-dataset/X.npy')
y_l = np.load('../input/Sign-language-digits-dataset/Y.npy')
img_size = 64  # pixel size

 # for sign zero
plt.subplot(1,2,1)  
plt.imshow(x_l[260])  # Get 260th index
plt.axis("off")

# for sign one
plt.subplot(1,2,2)
plt.imshow(x_l[900])  # Get 900th index
plt.axis("off")


# * Now we will concatenate our pictures consisting of 0 and 1.
# * We have image of 255 one sign, 255 zero sign

# In[ ]:


# From 0 to 204 zero sign, from 205 to 410 is one sign
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0)

# We will create their labels. After that, we will concatenate on the Y.
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z,o), axis=0).reshape(X.shape[0],1)
print("X shape: ", X.shape)
print("Y shape: ", Y.shape)


# * The shape of the X is (410, 64, 64)
#     * 410 means that we have 410 images (zero and one signs)
#     * 64 means that our image size is 64x64 (64x64 pixels)
# * The shape of the Y is (410,1)
#     * 410 means that we have 410 labels (0 and 1)

# * Now we reserve 80% of the values as 'train' and 20% as 'test'.
# * Then let's create x_train, y_train, x_test, y_test arrays
# 
# 

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state=42)
# random_state = Use same seed while randomizing
print(x_train.shape)
print(y_train.shape)


#   * Since our data in X is 3D, we need to flatten it to 2D to use Deep Learning.
#   * Since our data in Y is 2D, we don't need to flatten.

# In[ ]:


x_train_flatten = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test_flatten = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print('x_train_flatten: {} \nx_test_flatten: {} '.format(x_train_flatten.shape, x_test_flatten.shape))


# * Now x and y 2D
# * 4096 = 64 * 64

# In[ ]:


# Here we will change the location of our samples and features. '(328,4096) -> (4096,328)' 
x_train = x_train_flatten.T
x_test = x_test_flatten.T
y_train = y_train.T
y_test = y_test.T

print("x train: ", x_train.shape)
print("x test: ", x_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


# <a id="2"></a> <br>
# ## 3) Creating Parameters
# 
# * Parameters are weight and bias.
# * Our parameters are "w" and "b" standing for "Weights" and "Bias"
# * z = (w.t)x + b => z equals to (transpose of weights times input x) + bias
# * In an other saying => z = b + px1w1 + px2w2 + ... + px4096*w4096
# * y_head = sigmoid(z)
# * Sigmoid function makes z between zero and one so that is probability.
# 

# In[ ]:


# Now let's create the parameter and sigmoid function. 
# So what we need is dimension 4096 that is number of pixel as a parameter for our initialize method(def)

def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01)
    b = 0.0
    return w,b

# Sigmoid function
# z = np.dot(w.T, x_train) +b
def sigmoid(z):
    y_head = 1/(1 + np.exp(-z))  # sigmoid function finding formula
    return y_head
sigmoid(0)  # o should result in 0.5


# In[ ]:


w,b = initialize_weights_and_bias(4096)
print(w)
print("----------")
print(b)


# <a id="3"></a> <br>
# ## 4) Forward and Backward Propagation
# * To reduce our cost function, we create parameters w, b.
#     * b = b - learning_rate(gradient of b) 
#     * w = w - learning_rate(gradient of w) 
# * Our cost function is equal to the sum of the losses of each image.
# * To reduce losses, we need to update our cost with the Gradient Descent Method.
# * To do this, we'll create our Forward propagation and Backward propagation parameters.
# 
#    
# 

# In[ ]:


# In backward propagation we will use y_head that found in forward progation
# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation
def forward_backward_propagation(w, b, x_train, y_train):
    # Forward propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -(1 - y_train) * np.log(1 - y_head) - y_train * np.log(y_head)
    cost = (np.sum(loss)) / x_train.shape[1]   # x_train.shape[1] is for scaling
    
    # Backward propagation
    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1] # x_train.shape[1] is for 
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1] # x_train.shape[1] is for     
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost, gradients


# <a id="4"></a> <br>
# ## 5) Updating(Learning) Parameters
# ---
# * Now let's apply Updating Parameter 
# 

# In[ ]:


def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    # updating(learning) parameters is number_of_iteration times
    for i in range(number_of_iteration):
        # make forward and backward propagation and find cost gradients
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)  # adding costs to cost_list
        # lets update
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]        
        if i % 10 == 0:
            cost_list2.append(cost)   # adding costs to cost_list2
            index.append(i) # Adds a cost to the index in every 10 steps
            print("Cost after iteration %i: %f" %(i, cost))
        
        # we update (learn) parameters weights and bias
    
    parameters = {"weight": w, "bias":b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


# <a id="5"></a> <br>
# ## 6) Prediction Parameter

# * We need our model to be able to prediction
# * We need x_test to make prediction.

# In[ ]:


# Let's create prediction parameter

def predict(w,b,x_test):
    # x_test is a input for forward prapagation
    z = sigmoid(np.dot(w.T, x_test) +b)
    y_prediction = np.zeros((1, x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head = 1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head = 0),      
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction


# <a id="6"></a> <br>
# ##  7) Logistic Regression
# 

# Now lets put them all together.

# In[ ]:


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, n_iterations):
    # Initialize
    dimension = x_train.shape[0]   # that is 4096
    w, b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, n_iterations)
    
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train) 
    
    # print train / test errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))    

logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 0.01, n_iterations = 170)


# <a id="7"></a> <br>
# ## 8) Logistec Regression with Sklearn

# * With the Sklearn library, we can find the result you found above in a much easier way.

# In[ ]:


from sklearn import linear_model
lr_sl = linear_model.LogisticRegression(random_state=42, max_iter = 150)

print("test accuracy: {}".format(lr_sl.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))


# In[ ]:





# <a id="8"></a> <br>
# > # CONCLUSION 
# * If you want a more detailed kernel. Check out DATAI TEAM's Deep Learning Tutorial for Beginners Kernel.  https://www.kaggle.com/kanncaa1/deep-learning-tutorial-for-beginners
# ---
# <br> **Thank you for your votes and comments.**                                                                                                                                             
# <br>**If you have any suggest, May you write for me, I will be happy to hear it.**
