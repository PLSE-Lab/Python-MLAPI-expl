#!/usr/bin/env python
# coding: utf-8

# # Content
# * Introduction
# * Import library
# * Load data
# * Sample data representation
# * Data split train and test
# * Converting the 3D matrix to 2D matrix
# * Manually writing Neural Network functions
#     * Intialize Parameters And Layer Sizes
#     * Sigmoid Function
#     * Forward Propagation
#     * Loss Function And Cost Function
#     * Backward Propagation
#     * Update Parameters Function
#     * Prediction Function
#     * Create Model
# * Neural Network With Keras Library
# * Conclusion

# # INTRODUCTION
# * I wouldn't try my first deep learning experience.
# * I first tried to create the neural network model by hand. Then I did the same with the keras library.
# * I wish you good readings. The conclusion is about to meet again in the chapter.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Load data
# * In this data there are 2062 sign language digits images.
# * At the beginning of tutorial we will use only sign 0 and 1 for simplicity. 
# * In data, sign zero is between indexes 204 and 408. Number of zero sign is 205.
# * Also sign one is between indexes 822 and 1027. Number of one sign is 206. Therefore, we will use 205 samples from each classes(labels).
# * Lets prepare our X and Y arrays. X is image array (zero and one signs) and Y is label array (0 and 1).

# In[ ]:


# load data
x_l = np.load('../input/Sign-language-digits-dataset/X.npy')
Y_l = np.load('../input/Sign-language-digits-dataset/Y.npy')
img_size = 64


# # Sample data representation
# * We are plotting our sample data.

# In[ ]:


# sample data representation
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')
plt.show()


# * In order to create image array, we concatenate zero sign and one sign arrays
# * Then we create label array 0 for zero sign images and 1 for one sign images.

# In[ ]:


X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)


# # Data split train and test
# * With the train_test_split method we import from sklearn library, we divide the data into two as train and test.
# * We will train our model with the train part of the data we have divided and we will test our model with the test part.
# * We split the data to 80% train and 20% test.

# In[ ]:


# data division train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
print('X_train: ',X_train.shape)
print('X_test: ',X_test.shape)
print('Y_train: ',Y_train.shape)
print('Y_test: ',Y_test.shape)


# # Converting the 3D matrix to 2D matrix
# * We have 3 dimensional input array (X) so we need to make it flatten (2D) in order to use as input for our first.
# * Our label array (Y) is already flatten(2D) so we leave it like that.

# In[ ]:


# Converting the 3D matrix to 2D matrix
print('X_train: ',X_train.shape)
print('X_test: ',X_test.shape)

number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])

print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)

x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T

print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


# * x_train and y_train parameter as a parameter and layer sizes to initialize the function of writing.

# In[ ]:


# initialize parameters and layer sizes
def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    parameters = {
        'weight1':np.random.randn(3,x_train.shape[0]) * 0.1,
        'bias1':np.zeros((3,1)),
        'weight2':np.random.randn(y_train.shape[0],3) * 0.1,
        'bias2':np.zeros((y_train.shape[0],1))
    }
    return parameters


# * We write to sigmoid function

# In[ ]:


def sigmoid_func(z):
    """  
    Sigmoid function f(z) = 1 / 1 + e^-z
    """
    y_head = 1 / (1 + np.exp(-z))
    return y_head


# * We write to forward propagation function

# In[ ]:


# Forward Propagation
def forward_propagation_NN(x_train, parameters):
    Z1 = np.dot(parameters['weight1'],x_train) + parameters['bias1']
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters['weight2'],A1) + parameters['bias2']
    A2 = sigmoid_func(Z2)
    
    cache = {
        'Z1':Z1,
        'A1':A1,
        'Z2':Z2,
        'A2':A2
    }
    
    return A2, cache


# * We write to cost funtion

# In[ ]:


def compute_cost_NN(A2, Y, parameters):
    logprobs = np.multiply(np.log(A2),Y)
    cost = - np.sum(logprobs) / Y.shape[1]
    return cost


# * We write to backward propagation function

# In[ ]:


# Backward Propagations
def backward_propagation_NN(parameters, cache, X, Y):
    dZ2 = cache['A2'] - Y
    dW2 = np.dot(dZ2, cache['A1'].T) / X.shape[1]
    db2 = np.sum(dZ2, axis = 1, keepdims = True) / X.shape[1]
    dZ1 = np.dot(parameters['weight2'].T, dZ2) * (1 - np.power(cache['A1'], 2))
    dW1 = np.dot(dZ1, X.T) / X.shape[1]
    db1 = np.sum(dZ1, axis = 1, keepdims = True) / X.shape[1]
    grads = {
        'dweight1':dW1,
        'dbias1':db1,
        'dweight2':dW2,
        'dbias2':db2,
    }
    return grads


# We write to update function for weight and bias update.

# In[ ]:


# Update Function
def update_parameters_NN(parameters, grads, learning_rate = 0.03):
    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],
                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],
                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],
                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}
    return parameters


# * We write to prediction function for to test our model

# In[ ]:


# Prediction
def predict_NN(parameters,x_test):
    A2, cache = forward_propagation_NN(x_test, parameters)
    Y_prediction = np.zeros((1, x_test.shape[1]))
    for i in range(A2.shape[1]):
        if A2[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
    
    return Y_prediction


# * Write to create model function.

# In[ ]:


def two_layer_neural_network(x_train, y_train, x_test, y_test, num_iterations):
    cost_list = []
    index_list = []
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)
    
    for i in range(0, num_iterations):
        A2, cache = forward_propagation_NN(x_train, parameters)
        cost = compute_cost_NN(A2, y_train, parameters)
        grads = backward_propagation_NN(parameters, cache, x_train, y_train)
        parameters = update_parameters_NN(parameters, grads)
        
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
        
    plt.plot(index_list,cost_list)
    plt.xticks(index_list,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    
    y_prediction_test = predict_NN(parameters, x_test)
    y_prediction_train = predict_NN(parameters, x_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters


# * Create Model
#         * Since the learning rate was 0.01 when it was too slow, my model didn't learn very well and accuracy remained at 54%.
#         * When I increased the learning rate to 0.03, my model was well learned and accuracy increased to 96%.

# In[ ]:


parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)


# * Train and test data reshaping

# In[ ]:


x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T


# * Import keras library
# * From sklearn.model_selection import cross_val_score method 

# In[ ]:


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library


# * Using the keras library, we create our Neural Network model and put it into the cross_val_score method.
# * When I set the cv parameter of cross_val_score method to 4, accuracy is 74%, and 5 is 54%.

# In[ ]:


def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: ", mean)
print("Accuracy variance: ", variance)


# # Conclusion
# * A notebook I've made to try what I've learned more about deep learning and try to better understand deep learning.
# * I'm new to programming. I'm even more new in data science, machine learning, deep learning and artificial intelligence. But I am working. And I'il be an artificial intelligence developer. Your comments are very important to me.
# * Thank you for reading my notebook. Waiting for your criticism.
