#!/usr/bin/env python
# coding: utf-8

# # **Flower Recognition with ANN Implementation**
# 
# 
# ### **Content**
# * [Introduction](#1)
# * [Data Preparation](#2)
# * [Logistic Regression for Deep Learning](#4)
#     * [Forward Propagation](#5)
#         * [Sigmoid Function](#7)
#         * [Loss and Cost Function](#6)
#     * [Backward Propagation](#8)
#     * [Updating Parameters](#9)
#     * [Prediction](#10)
#     * [Model Creation](#11)    
# * [Artificial Neural Network with Keras](#12)  
# * [Conclusion](#3)

# <a id="1"></a> 
# ## **Introduction**
# 
# * The following notebook which I created below is the sample of the my "deep learning" learning phase. 
# * Deep learning is the part of machine learning. 
# * It extracts features from given dataset. 
# * It has hidden layers unlike machine learning techniques. Below is the example of how it works.

# ![sample.png](http://i68.tinypic.com/6oipao.png)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import cv2 # For reading images
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir('../input/flowers/flowers'))


# <a id="2"></a> 
# ### **Data Preparation**
# * We'll pick two kind of flowers for classification. These are Daisies and tulips.

# In[ ]:


# Daisies path from Flower Recognation folder
daisy_path = "../input/flowers/flowers/daisy/"


#  Tulip path from Flower Recognation folder
tulip_path = "../input/flowers/flowers/tulip/" 


# * Method for reading images from data folder. We will use cv2 library for images.
# [For more information about cv2](https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_image_display/py_image_display.html)

# In[ ]:


trainLabels = [] # For labels. Daisy and tulip
data = [] # All image array

# Dimensions of the images are not fixed. They have various sizes and we will fix tham to 128x128
size = 128,128

def readImages(flowerPath, folder):
    
    imagePaths = []
    for file in os.listdir(flowerPath):
        if file.endswith("jpg"):  # use only .jpg extensions
            imagePaths.append(flowerPath + file)
            trainLabels.append(folder)
            img = cv2.imread((flowerPath + file), 0)
            im = cv2.resize(img, size)
            data.append(im)            
            
    return imagePaths


# * Method for showing sample images

# In[ ]:


def showImage(imgPath):
    img = cv2.imread(imgPath)
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.axis('off')
    plt.show()


# * Load daisies and tulips

# In[ ]:


daisyPaths = readImages(daisy_path, 'daisy')
tulipPaths = readImages(tulip_path, 'tulip')


# * Let's look at some flowers

# In[ ]:


showImage(daisyPaths[np.random.randint(0,500)])
showImage(tulipPaths[np.random.randint(0,500)])


# * Converting images to numpy array for classification

# In[ ]:


rawData = np.array(data)
rawData.shape


# * Now we have 1821 samples with 128x128 size. 
# * We will normalize data for binary classification.

# In[ ]:


rawData = rawData.astype('float32') / 255.0


# * We will create X and Y for our classification. X -> our binary flower date, Y -> label data 

# In[ ]:


X = rawData
z = np.zeros(877)
o = np.ones(876)
Y = np.concatenate((z, o), axis = 0).reshape(X.shape[0], 1)

print("X shape: " , X.shape)
print("Y shape: " , Y.shape)


# <a id="4"></a>
# ### **Logistic Regression for Deep Learning**
# 
# * It is the basis of deep learning. 
# * In next steps we will make a binary classification. Then we'll create the neural network model.
# 
# * Next step is the train - test split operation

# In[ ]:


# Let's create train and test data
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.15, random_state = 42)
numberOfTrain = xTrain.shape[0]
numberOfTest = xTest.shape[0]


# In[ ]:


# Transforming data to 2D.

xTrainFlatten = xTrain.reshape(numberOfTrain, xTrain.shape[1] * xTrain.shape[2])
xTestFlatten = xTest.reshape(numberOfTest, xTest.shape[1] * xTest.shape[2])

print("X train flatten", xTrainFlatten.shape)
print("X test flatten", xTestFlatten.shape)


# In[ ]:


x_train = xTrainFlatten.T
x_test = xTestFlatten.T
y_train = yTrain.T
y_test = yTest.T
print("x train: ",xTrain.shape)
print("x test: ",xTest.shape)
print("y train: ",yTrain.shape)
print("y test: ",yTest.shape)


# Intialize parameters and layer sizes.
# * 3 is the layer size. 
# * We have 3 layers for our ANN model.

# In[ ]:


def initializeParametersAndLayerSizesNN(x_train, y_train):
    
    parameters = {"weight1": np.random.randn(3, x_train.shape[0]) * 0.1,
                  "bias1": np.zeros((3, 1)),
                  "weight2": np.random.randn(y_train.shape[0], 3) * 0.1,
                  "bias2": np.zeros((y_train.shape[0], 1))}
    
    return parameters


# <a id="5"></a> 
# **Forward propagation**
# * Multiplication of weights and features and addition of bias values. 
# * Z1 is the result of first process. Then we will use tanh() activation function with Z1 and we get A1
# * After getting A1 value we'll multiply weights and A1 values for Z2. Then we'll use the sigmoid function for getting the A2 value. 
# * tanh() function compresses data  [-1, 1] range
# * sigmoid function compresses data [0, 1] range.

# <a id="7"></a> 
# * Method for sigmoid function

# In[ ]:


# Method for sigmoid function
# z = np.dot(w.T, x_train) + b
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


# * Method for forward propagation

# In[ ]:


def forwardPropagationNN(x_train, parameters):

    Z1 = np.dot(parameters["weight1"], x_train) + parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"], A1) + parameters["bias2"]
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


# <a id="6"></a> 
# * Method for compute cost.

# In[ ]:


# Compute cost
def computeCostNN(A2, Y, parameters):
    
    logprobs = np.multiply(np.log(A2),Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    
    return cost


# <a id="8"></a> 
# **Backward Propagation**
# 
# * It means derivative.
# * Function which is shown below derivative of according to the our parameters (weights and bias).

# In[ ]:


def backwardPropagationNN(parameters, cache, X, Y):

    dZ2 = cache["A2"]-Y
    dW2 = np.dot(dZ2,cache["A1"].T)/X.shape[1]
    db2 = np.sum(dZ2,axis =1,keepdims=True)/X.shape[1]
    dZ1 = np.dot(parameters["weight2"].T,dZ2)*(1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1,X.T)/X.shape[1]
    db1 = np.sum(dZ1,axis =1,keepdims=True)/X.shape[1]
    grads = {"dweight1": dW1,
             "dbias1": db1,
             "dweight2": dW2,
             "dbias2": db2}
    
    return grads


# <a id="9"></a> 
# **Updating Parameters**
# * We need update parameters for complexity of our model. We'll use learning rate and derivatized parameters.

# In[ ]:


def updateParametersNN(parameters, grads, learning_rate):
    
    parameters = {"weight1": parameters["weight1"] - learning_rate * grads["dweight1"],
                  "bias1": parameters["bias1"] - learning_rate * grads["dbias1"],
                  "weight2": parameters["weight2"] - learning_rate * grads["dweight2"],
                  "bias2": parameters["bias2"] - learning_rate * grads["dbias2"]}
    
    return parameters


# <a id="10"></a> 
# **Prediction**
# * Definition of prediction method is shown below.

# In[ ]:


# prediction
# x_test is the input of forward propagation.
def predictNN(parameters, x_test):

    A2, cache = forwardPropagationNN(x_test, parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(A2.shape[1]):
        if A2[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


# <a id="11"></a> 
# **Model Creation**
# * Parameter initialization
# * Forward propagation and cost value calculation
# * Backward propagation
# * Updating parameters
# * The above steps will be repeated until the given iteration count. 
# * After iterations end, prediction process will be started.
# * Now we will create 2 - Layer neural network

# In[ ]:


# 2 - Layer neural network
def two_layer_neural_network(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    
    cost_list = []
    index_list = []
    
    # Initialize parameters
    parameters = initializeParametersAndLayerSizesNN(x_train, y_train)

    for i in range(0, num_iterations):
        # Forward propagation
        A2, cache = forwardPropagationNN(x_train, parameters)
        # Calculation of cost value
        cost = computeCostNN(A2, y_train, parameters)
         # Backward propagation
        grads = backwardPropagationNN(parameters, cache, x_train, y_train)
         # Updating parameters
        parameters = updateParametersNN(parameters, grads, learning_rate)
        
        if i % 10 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    plt.plot(index_list,cost_list)
    plt.xticks(index_list,rotation = 'vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    
    # Prediction
    y_prediction_test = predictNN(parameters, x_test)
    y_prediction_train = predictNN(parameters, x_train)

    # Print results
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters

parameters = two_layer_neural_network(x_train, y_train, x_test, y_test, learning_rate = 0.01, num_iterations = 500)


# <a id="12"></a> 
# ### **Artificial Neural Network with Keras**

# Implementing keras library
# * Each classifier is a layer definition (Each Dense)
#     * units -> node count of the layer
#     * kernel_initializer -> initialization form of weights    
#     * activation -> activation function choice
#     * input_dim -> dimension of inputs (128 x 128)
# * In compilation;
#     * optimizer -> adam is the algorithm for training neural networks
#     * loss -> cost function is the same as logistic regression. Sum of cost values.
#     * metrics -> for accuracy
#     * cross_val_score ->  library for calculating accuracy values
#     * epoch -> iteration count

# In[ ]:


# Reshaping for keras
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T


# In[ ]:


# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library

def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 500)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()

print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))


# <a id="3"></a> 
# ## **Conclusion**
# * We used logistic regression for creating artificial neural network model. Then we did the same operation with keras library. 
# * *If you have a suggestion, I'd be happy to read it.*
