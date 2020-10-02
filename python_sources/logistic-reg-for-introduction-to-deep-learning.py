#!/usr/bin/env python
# coding: utf-8

# # Orhan SERTKAYA
# <br>Content:
# * [Introduction](#1):
# * [Preparing Dataset](#2):
# * [Train-Test Split Data](#3):
# * [Computation Graph](#4):
# * [Initializing Parameters(weights and bias) Method](#5):
# * [Forward Propagation](#6):
# * [Sigmoid Function Method](#7):
# * [Forward and Backward Propagation Method](#8):
# * [Updating(Learning) Parameters Method](#9):
# * [Prediction Method](#10):
# * [Logistic Regression Method](#11):
# * [Logistic Regression with Scikit-Learn](#12):
# * [Confusion Matrix](#13):
# * [Conclusion](#14):

# <a id="1"></a> <br>
# # INTRODUCTION
# * In this kernel,we will learn how to use Logistic Regression Algorithm step by step.This kernel is also an introduction to deep learning.I will use the same dataset for introduction to deep learning(ANN).If you understand this topic comfortably, you will not have difficulty in the introduction to deep learning.
# * Logistic regression is actually a very simple neural network.
# * You can look at my another kernel that I wrote about logistic regression in detail.<br>
# * ==> <a href="https://www.kaggle.com/orhansertkaya/mush-classification-logistic-regression-algorithm">Mushroom Classification and Logistic Regression Algorithm</a>
# * Let's start.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


import seaborn as sns

# plotly
import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import warnings
import warnings
# filter warnings
warnings.filterwarnings('ignore')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.


# <a id="2"></a> <br>
# ## Preparing Dataset
# 
# * In this data there are 2062 sign language digits images.
# * we will use only sign 0 and 1 for simplicity. 
# * In data, sign zero is between indexes 204 and 408. Number of zero sign is 205.
# * Also sign one is between indexes 822 and 1027. Number of one sign is 206. Therefore, we will use 205 samples from each classes(labels).
# * Lets prepare our X and Y arrays. X is image array (zero and one signs) and Y is label array (0 and 1).

# In[ ]:


# load data set
x_l = np.load('../input/Sign-language-digits-dataset/X.npy')
y_l = np.load('../input/Sign-language-digits-dataset/Y.npy')
img_size = 64
plt.subplot(1,2,1)
plt.imshow(x_l[260].reshape(img_size,img_size))
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(x_l[900].reshape(img_size,img_size))
plt.axis('off')


# * In order to create image array, I concatenate zero sign and one sign arrays
# * Then I create label array 0 for zero sign images and 1 for one sign images.

# In[ ]:


# from 0 to 204 is zero sign and from 205 to 410 is one sign 
X = np.concatenate((x_l[204:409],x_l[822:1027]),axis=0)
# Now,we need to create label of zeros and ones.After that we concatenate them.
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z,o),axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)


# * The shape of the X is (410, 64, 64)
#     * 410 means that we have 410 images (zero and one signs)
#     * 64 means that our image size is 64x64 (64x64 pixels)
# * The shape of the Y is (410,1)
#     *  410 means that we have 410 labels (0 and 1) 
# * Lets split X and Y into train and test sets.
#     * test_size = percentage of test size. test = 15% and train = 75%
#     * random_state = use same seed while randomizing. It means that if we call train_test_split repeatedly, it always creates same train and test distribution because we have same random_state.

# <a id="3"></a> <br>
# ## Train-Test Split Data

# In[ ]:


# Now,lets create x_train, y_train, x_test, y_test arrays
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.15,random_state=42)
number_of_train = x_train.shape[0]
number_of_test = y_test.shape[0]
print(x_train.shape)
print(y_train.shape)


# * Now we have 3 dimensional input array (X) so we need to make it flatten (2D) in order to use as input for our first deep learning model.
# * Our label array (Y) is already flatten(2D) so we leave it like that.
# * Lets flatten X array(images array).
# 

# In[ ]:


x_train_flatten = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])
x_test_flatten = x_test.reshape(number_of_test,x_test.shape[1]*x_test.shape[2])
print("x train flatten",x_train_flatten.shape)
print("x test flatten",x_test_flatten.shape)


# * As you can see, we have 348 images and each image has 4096 pixels in image train array.
# * Also, we have 62 images and each image has 4096 pixels in image test array.
# * Then lets take transpose.(it depends on you but I use like that.)

# In[ ]:


x_train = x_train_flatten.T
x_test = x_test_flatten.T
y_train = y_train.T
y_test = y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


# <a id="4"></a> <br>
# ##  Computation Graph
# * Now lets look at computation graph of logistic regression
# <a href="http://ibb.co/c574qx"><img src="http://preview.ibb.co/cxP63H/5.jpg" alt="5" border="0"></a>
#     * Parameters are weight and bias.
#     * Weights: coefficients of each pixels
#     * Bias: intercept
#     * z = (w.t)x + b  => z equals to (transpose of weights times input x) + bias 
#     * In an other saying => z = b + px1*w1 + px2*w2 + ... + px4096*w4096
#     * y_head = sigmoid(z)
#     * Sigmoid function makes z between zero and one so that is probability. You can see sigmoid function in computation graph.
# * Why we use sigmoid function?
#     * It gives probabilistic result
#     * It is derivative so we can use it in gradient descent algorithm (we will see as soon.)
# * Lets make example:
#     * Lets say we find z = 4 and put z into sigmoid function. The result(y_head) is almost 0.9. It means that our classification result is 1 with 90% probability.

# <a id="5"></a> <br>
# ## Initializing Parameters(weights and bias) Method
# * As you know input is our images that has 4096 pixels(each image in x_train).
# * Each pixels have own weights.
# * The first step is multiplying each pixels with their own weights.
# * The question is that what is the initial value of weights?
#     * There are some techniques that I will explain at artificial neural network but for this time initial weights are 0.01.
#     * Okey, weights are 0.01 but what is the weight array shape? As you understand from computation graph of logistic regression, it is (4096,1)
#     * Also initial bias is 0.

# In[ ]:


# lets initialize parameters
# So what we need is dimension 4096 that is number of pixels as a parameter for our initialize method(def)
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b= 0.0
    return w,b
#w,b = initialize_weights_and_bias(4096)


# <a id="6"></a> <br>
# ## Forward Propagation
# * The all steps from pixels to cost is called forward propagation
#     * z = (w.T)x + b => in this equation we know x that is pixel array, we know w (weights) and b (bias) so the rest is calculation. (T is transpose)
#     * Then we put z into sigmoid function that returns y_head(probability). When your mind is confused go and look at computation graph. Also equation of sigmoid function is in computation graph.
#     * Then we calculate loss(error) function. 
#     * Cost function is summation of all loss(error).
#     * Lets start with z and the write sigmoid definition(method) that takes z as input parameter and returns y_head(probability)

# <a id="7"></a> <br>
# ## Sigmoid Function Method

# In[ ]:


# calculation of z
#z = np.dot(w.T,x_train)+b
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


# * Mathematical expression of log loss(error) function is that: 
#     <a href="https://imgbb.com/"><img src="https://image.ibb.co/eC0JCK/duzeltme.jpg" alt="duzeltme" border="0"></a>
# * It says that if you make wrong prediction, loss(error) becomes big. 
# * The cost function is summation of loss function. Each image creates loss function. Cost function is summation of loss functions that is created by each input image.
# * Lets implement forward propagation.

# <a id="8"></a> <br>
# ## Forward and Backward Propagation Method

# In[ ]:


# In backward propagation we will use y_head that found in forward progation
# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    
    return cost,gradients


# <a id="9"></a> <br>
# ## Updating(Learning) Parameters Method

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


# <a id="10"></a> <br>
# ## Prediction Method

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


# * Now lets put them all together.

# <a id="11"></a> <br>
# ## Logistic Regression Method

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


# In[ ]:


logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 150)


# <a id="12"></a> <br>
# ## Logistic Regression with Scikit-Learn
# * In sklearn library, there is a logistic regression method that ease implementing logistic regression.
# * The accuracies are different from what we find. Because logistic regression method use a lot of different feature that we do not use like different optimization parameters or regularization.

# In[ ]:


from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))


# <a id="13"></a> <br>
# ## Confusion Matrix

# In[ ]:


y_pred = logreg.predict(x_test.T)
y_pred


# In[ ]:


y_pred = logreg.predict(x_test.T)
y_true = y_test.T

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true,y_pred)

f, ax = plt.subplots(figsize=(10,10))
sns.heatmap(cm, annot=True, linewidth=0.5, fmt=".0f",  cmap='RdPu', ax = ax)
plt.xlabel = ("y_pred")
plt.ylabel = ("y_true")
plt.show()


# ## Summary
# What we did at this part:
# * Initialize parameters weight and bias
# * Forward propagation
# * Loss function
# * Cost function
# * Backward propagation (gradient descent)
# * Prediction with learnt parameters weight and bias
# * Logistic regression with sklearn
# 

# <a id="14"></a> <br>
# # Conclusion
# * If you like it, please upvote :)
# * If you have any question, I will be appreciate to hear it.
