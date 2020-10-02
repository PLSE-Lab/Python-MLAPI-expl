#!/usr/bin/env python
# coding: utf-8

# In this tutorial, I am going to work on Cancer Dataset and implement some ML techniques for training and testing data.

# ## Content:
# 
# 1. [Introduction](#1)
# 1. [Overview the Data Set](#2)
# 1. [Code Part](#3)
#     * [Logistic Regression](#4)
#         * [Initializing parameters](#5)
#         * [Forward and Barckward Propagation](#6)
#         * [Updating parameters](#7)
#     * [Logistic Regression with Sklearn](#8)

# <a id = "1"></a>
# # 1. Introduction

# In the dataset, we have different features and numerical variables. We can detect if a person is cancer or not by looking at this data.

# ![](https://static.independent.co.uk/s3fs-public/thumbnails/image/2020/02/04/09/cancer.jpg)

# <a id = "2"></a>
# # 2. Overview the Data Set

# Dataset information:
# 
# * Dataset Characteristics: Multivariate
# * Attribute Characteristics: Real
# * Attribute Characteristics: Classification
# * Number of Instances: 569
# * Number of Attributes: 32
# * Missing Values: No

# Column names and meanings:
# * id: ID number
# * diagnosis: The diagnosis of breast tissues (M = malignant, B = benign)
# * radius_mean: mean of distances from center to points on the perimeter
# * texture_mean: standard deviation of gray-scale values
# * perimeter_mean: mean size of the core tumor
# * area_mean: area of the tumor
# * smoothness_mean: mean of local variation in radius lengths
# * compactness_mean: mean of perimeter^2 / area - 1.0
# * concavity_mean: mean of severity of concave portions of the contour
# * concave_points_mean: mean for number of concave portions of the contour
# * symmetry_mean
# * fractal_dimension_mean: mean for "coastline approximation" - 1
# * radius_se: standard error for the mean of distances from center to points on the perimeter
# * texture_se: standard error for standard deviation of gray-scale values
# * perimeter_se
# * area_se
# * smoothness_se: standard error for local variation in radius lengths
# * compactness_se: standard error for perimeter^2 / area - 1.0
# * concavity_se: standard error for severity of concave portions of the contour
# * concave_points_se: standard error for number of concave portions of the contour
# * symmetry_se
# * fractal_dimension_se: standard error for "coastline approximation" - 1
# * radius_worst: "worst" or largest mean value for mean of distances from center to points on the perimeter
# * texture_worst: "worst" or largest mean value for standard deviation of gray-scale values
# * perimeter_worst
# * area_worst
# * smoothness_worst: "worst" or largest mean value for local variation in radius lengths
# * compactness_worst: "worst" or largest mean value for perimeter^2 / area - 1.0
# * concavity_worst: "worst" or largest mean value for severity of concave portions of the contour
# * concave_points_worst: "worst" or largest mean value for number of concave portions of the contour
# * symmetry_worst
# * fractal_dimension_worst: "worst" or largest mean value for "coastline approximation" - 1

# <a id = "3"></a>
# # 3. Code Part

# * We use pandas for reading the dataset
# * Numpy is used for numerical data handling
# * Matplotlib is used for plotting the data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Reading csv file:

# In[ ]:


dataset = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")


# For getting general information about our data, we use "info()" function.

# In[ ]:


dataset.info()


# In[ ]:


dataset.head()


# * While diagnosing whether a person has cancer or not, we don't need to use "id" and "Unnamed32" columns.  

# In[ ]:


dataset = dataset.drop(["id"], axis = 1)


# Let's check if we can drop our "id" column.

# In[ ]:


dataset.head()


# In[ ]:


dataset = dataset.drop(["Unnamed: 32"], axis = 1)


# Let's check if we can drop our "Unnamed: 32" column.

# In[ ]:


dataset.head()


# Now, I am going to Convert diagnosis variables into numerical variables.(0s and 1s)

# In[ ]:


dataset.diagnosis = [1 if i == "M" else 0 for i in dataset.diagnosis]


# In[ ]:


dataset.head(2)


# It's okay now.

# In[ ]:


x = dataset.drop(["diagnosis"], axis = 1)
y = dataset.diagnosis.values


# In[ ]:


x.head()


# <a id = "4"></a>
# ## Logistic Regression

# I am going to normalize my variables in order to study with them.( I am going to scale them between 0 and 1)

# In[ ]:


x = (x - np.min(x))/(np.max(x) - np.min(x)).values
# (x - min(x))/ (max(x) - min(x))


# In[ ]:


x.head()


# ## Train-Test-Split

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T


# In[ ]:


print("x train after taking transpose: ",x_train.shape)
print("x test after taking transpose: ",x_test.shape)
print("y train after taking transpose: ",y_train.shape)
print("y test after taking transpose: ",y_test.shape)


# <a id = "5"></a>
# ## Parameter Initialize and Sigmoid Function

# In[ ]:


# dimension = 30
def initialize_weights_and_bias(dimension):
    weights = np.full((dimension,1),0.01)
    bias = 0.0
    return weights, bias


# In[ ]:


weights, bias = initialize_weights_and_bias(30)


# In[ ]:


weights


# In[ ]:


bias


# Sigmoid function formula:
# ![](https://www.aliozcan.org/wp-content/uploads/2019/11/sigmoid-Fonksiyonu.png)

# In[ ]:


def sigmoid(z):
    y_head = 1 / ( 1 + np.exp(-z)) 
    return y_head


# Our y_head will be a probabilistic value.

# Also we know(you can check from the graph) if sigmoid function takes 0, it gives 0.5 as an output.

# In[ ]:


sigmoid(0)


# So, our method works.

# * Dimension of x_train: (30*455)
# * Dimension of weights: (30*1) 
# * In order to do a matrix multiplication, we need to take transpose of weights(or you can also take transpose of x_train instead of weights.)

# <a id = "6"></a>
# ## Forward and Backward Propagation:

# ![](https://www.cs.iusb.edu/~danav/teach/c463/neuron.gif)

# In[ ]:


def forward_backward_propagation(w,b,x_train,y_train):
    
    # forward propagation
    
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1] = 455 -> for scaling
    
    # backward propagation
    
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients


# Gradients are dictionaries, parameters are stored inside it.

# ![](https://cdn-images-1.medium.com/max/800/1*EJPT0utTkQ2qrHfjDID5RA.png)

# <a id = "7"></a>
# ## Update Part:

# ![](https://miro.medium.com/max/2124/1*WGHn1L4NveQ85nn3o7Dd2g.png)

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
        if i % 25 == 0:
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


# * learning rate: How fast we learn.
# * number_of_iteration: How many times we are gonne do forward and backward propagation.
# * cost_list is used for containing cost values.
# * cost_list2 is used for containing cost values after every 25 steps.

# ### Prediction:

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


# Logistic regression:

# In[ ]:


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 4096
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 10, num_iterations = 500)   


# <a id = "8"></a>
# # Sklearn with Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


x_train.shape


# In[ ]:


lr = LogisticRegression()
lr.fit(x_train.T, y_train.T)


# We use "score()" function. It preans, predict my values and then find the accuracy.

# In[ ]:


print("Test accuracy: ", lr.score(x_test.T, y_test.T))

