#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression on Catvnoncat Dataset 

# > You can access and download the Catvnoncat Dataset: [https://www.kaggle.com/muhammeddalkran/catvnoncat](https://www.kaggle.com/muhammeddalkran/catvnoncat)

# # Needed Library 

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


# > Before starting implementing model, I want to show how to convert your images into gray. 

# # Reading Dataset with h5py Library 

# In[ ]:


import h5py
test = h5py.File('/kaggle/input/catvnoncat/catvnoncat/test_catvnoncat.h5', 'r')
x_test = test['test_set_x']
y_test = test['test_set_y']
train = h5py.File('/kaggle/input/catvnoncat/catvnoncat/train_catvnoncat.h5', 'r')
x_train = train['train_set_x']
y_train = train['train_set_y']


# * We use cv2 library to convert images into gray. 

# In[ ]:


import cv2
x_train_grayscale = np.zeros(x_train.shape[:-1])
for i in range(x_train.shape[0]): 
    x_train_grayscale[i] = cv2.cvtColor(x_train[i], cv2.COLOR_BGR2GRAY) 
x_test_grayscale = np.zeros(x_test.shape[:-1])
for i in range(x_test.shape[0]): 
    x_test_grayscale[i] = cv2.cvtColor(x_test[i], cv2.COLOR_BGR2GRAY) 
print("x_train shape: " ,x_train_grayscale.shape)
print("x_test shape: " ,x_test_grayscale.shape)
plt.subplot(1,2,1)

plt.text(25, 70, "Cat", fontsize=16)
plt.imshow(x_test_grayscale[20].reshape(64,64))
plt.axis('off')
plt.subplot(1,2,2)
plt.text(18, 70, "Non-Cat", fontsize=16)
plt.imshow(x_test_grayscale[49].reshape(64,64))
plt.axis('off')


# # Implementing Logistic Regression 

# # Reding Dataset
# * Our dataset includes train, test, and class parts.

# In[ ]:


train_set_x_orig = np.array(train["train_set_x"][:]) # your train set features
train_set_y_orig = np.array(train["train_set_y"][:]) # your train set labels
test_dataset = h5py.File('/kaggle/input/catvnoncat/catvnoncat/test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels
classes = np.array(test_dataset["list_classes"][:]) # the list of classes


# * Lets look at shape of catvnoncat dataset, namely x_test, y_test, x_train, y_train.
# * Also, we can look at how many classes are there and what are those.

# In[ ]:



print("Shape of x_test : " + str(test_set_x_orig.shape))
print("Shape of y_test : " + str(test_set_y_orig.shape))
print("Shape of x_train : " + str(train_set_x_orig.shape))
print("Shape of y_train : " + str(train_set_y_orig.shape))
print("Classes : ", classes)
img_size = 64
rgb = 3;
plt.subplot(1,2,1)
plt.text(25, 70, "Cat", fontsize=16)
plt.imshow(x_test[20].reshape(img_size,img_size,rgb))
plt.axis('off')
plt.subplot(1,2,2)
plt.text(18, 70, "Non-Cat", fontsize=16)
plt.imshow(x_test[49].reshape(img_size,img_size,rgb))
plt.axis('off')


# * Our train_set_y_orig and test_set_y_orig are 1 d array and their shape is like (x,).
# * We should reshape them and make them 2 d array like (x,1), (1,x)

# In[ ]:


train_set_y = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
test_set_y = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

print("y_train shape: " ,train_set_y.shape)
print("y_test shape: " ,test_set_y.shape)


# # Flatten Our Original x_train and x_test Arrays 
# * Shape of original x_train and x_test dataset arrays is like (x, 64, 64, 3).
# * It means that we have x images and they are 64x64 colorful(rgb) images.
# * To make operation on them, we should flatten them. Actually, we make it 2d array like a long line. 
# * 64 * 64 * 3 = 12288 

# In[ ]:


number_of_train = x_train.shape[0]
number_of_test = x_test.shape[0]
number_of_pixel = train_set_x_orig.shape[1]
train_set_x_flatten = train_set_x_orig.reshape(number_of_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(number_of_test, -1).T
print("X train flatten",train_set_x_flatten.shape)
print("X test flatten",test_set_x_flatten.shape)


# * As it is known, one pixel value of rgb image is in range 0 - 255
# * Therefore, to standardize it, we should divide all pixels by 255.

# In[ ]:


train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

print("x train: ",train_set_x.shape)
print("x test: ",test_set_x.shape)
print("y train: ",train_set_y.shape)
print("y test: ",test_set_y.shape)


# # Weights and Bias Initialization

# In[ ]:


def parameter_initialization(size):
    w = np.zeros((size,1))
    print(w.shape)
    b = 0.0
    assert(w.shape==(size,1))
    assert(isinstance(b,float)or isinstance(b,int))
    return w,b


# * "size" is dimension of your image or data. Our image size is 64 * 64 * 3 = 12288
# * Shape of our weight array is (12288, 1) and we full it with ones as default.
# * "b" is our bias and it is just variable and its value is zero as default.

# # Forward Propagation Initialization
# * Firstly, we find our z result by using z = (w.T)x + b equation, namely z = np.dot(w.T,x_train) + b.
# * Then, we evaluate our z result by using sigmoid funtion which is sigmaod(z) = 1/(1+ e^-z).
# * Lastly, we calculate loss function which is -(1-y)log(1-y_head) - ylog(y_head) and cost function which is summation of loss function results.
# 

# # Sigmoid Function

# In[ ]:


def sigmoid(z):
    y_head = 1.0/(np.exp(-z)+1)
    return y_head
#y_head = sigmoid(3)
#print(y_head) # for example as z = 3 result is ~0.952574; as z = 0 result is 0.5


# # Forward Propagation Function

# In[ ]:


def forward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z) # probabilistic 0-1
    loss = y_train*np.log(y_head)+(1-y_train)*np.log(1-y_head)
    cost = -((np.sum(loss))/x_train.shape[1])    # x_train.shape[1]  is for scaling
    cost= np.squeeze(cost)
    assert(cost.shape == ())
    return cost, y_head 


# # Backward Propagation Initialization
# * In Backward Propagation, we update weights and bias with loss derivative according to weights and bias.
# * For weights, we use (1/sample_size)* x(y_head - y_train)T
# * For bais, we use (1/sample_size)* (sum from 1 to sample_size (y_head - y_train))

# In[ ]:


def backward_propagation(w,b,x_train,y_train):
    #forward propagation and return result cost and y_head
    #we use y_head in derivative of bias and weights.
    #we return cost and gradients, actually it is dictionary as data type, to update weight and bias

    cost, y_head = forward_propagation(w,b,x_train,y_train)
    sample_size = x_train.shape[1]
    weights_derivative = (np.dot(x_train,(y_head-y_train).T))/sample_size
    bias_derivative = np.sum(y_head-y_train)/sample_size
    assert(weights_derivative.shape == w.shape)
    assert(bias_derivative.dtype == float)
    gradients = {"weights_derivative": weights_derivative, "bias_derivative":bias_derivative}
    return cost, gradients


# # Update Weights and bias
# * To update weights, we use w:= w - learning_rate * weights_derivative
# * To update bias, we use b:= b - learning_rate * bias_derivative
# 

# In[ ]:


def update_weights_and_bias(w,b,x_train,y_train,learning_rate,number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    for i in range(number_of_iteration):
        #backward and forward propagation
        cost,gradients = backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        w = w - learning_rate*gradients["weights_derivative"]
        b = b - learning_rate*gradients["bias_derivative"]
        if i % 100 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i,cost))
        #update weights and bias
    parameters_dictionary = {"weights":w, "bias" : b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation = 'vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters_dictionary, gradients, cost_list


# # Prediction
# * If the result of sigmoid function is more than 0.5, we assume that It is true; otherwise false.

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
# predict(parameters["weight"],parameters["bias"],x_test)


# # Start Testing
# > We implemented everything that we need. Now, we are ready to train our data.

# In[ ]:


def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,number_of_iteration):
    size = x_train.shape[0]
    print(size)
    w,b = parameter_initialization(size)
    parameters,gradients,cost_list = update_weights_and_bias(w,b,x_train,y_train,learning_rate,number_of_iteration)
    y_prediction_test = predict(parameters["weights"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weights"],parameters["bias"],x_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
logistic_regression(train_set_x, train_set_y, test_set_x, test_set_y,learning_rate = 0.015,number_of_iteration = 1000)


# > Note: We can also use Sklearn Library to do what we did above
# 
# # Sklearn Library
# * Library provides coders with opportunity to implement Logistic Regression
# * To use this library, you can check : [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

# In[ ]:


from sklearn import linear_model as lm
logistic_regression = lm.LogisticRegression( max_iter = 1000)
print("train accuracy: {} %".format(logistic_regression.fit(train_set_x.T,train_set_y_orig.T).score(test_set_x.T,test_set_y_orig.T)))
print("test accuracy: {} %".format(logistic_regression.fit(train_set_x.T,train_set_y_orig.T).score(train_set_x.T,train_set_y_orig.T)))

