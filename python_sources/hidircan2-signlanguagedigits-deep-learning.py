#!/usr/bin/env python
# coding: utf-8

# <font color='orange'>
# <br>Content:
# * [Introduction](#1)
# * [Overview the Data Set](#2)
# * [Logistic Regression](#3)
#     * [Computation Graph](#4)
#     * [Initializing parameters](#5)
#     * [Forward Propagation](#6)
#         * Sigmoid Function
#         * Loss(error) Function
#         * Cost Function
#     * [Optimization Algorithm with Gradient Descent](#7)
#         * Backward Propagation
#         * Updating parameters
#     * [Logistic Regression with Sklearn](#8)
#     * [Summary and Questions in Minds](#9)
#     
# * [Artificial Neural Network](#10)
#     * [2-Layer Neural Network](#11)
#         * [Size of layers and initializing parameters weights and bias](#12)
#         * [Forward propagation](#13)
#         * [Loss function and Cost function](#14)
#         * [Backward propagation](#15)
#         * [Update Parameters](#16)
#         * [Prediction with learnt parameters weight and bias](#17)
#         * [Create Model](#18)
#     * [L-Layer Neural Network](#19)
#         * [Implementing with keras library](#22)
# 
# 
# 

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


# <a id="Overview the Data Set"></a> <br>
# # Overview the Data Set
# 

# In[ ]:


# load data set
x_l = np.load('/kaggle/input/signlanguagedigitdatauploaded-pc/X.npy')
Y_l = np.load('/kaggle/input/signlanguagedigitdatauploaded-pc/Y.npy')
img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')


# * In order to create image array, I concatenate zero sign and one sign arrays
# * Then I create label array 0 for zero sign images and 1 for one sign images.

# In[ ]:


# Join a sequence of arrays along an row axis.
X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
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

# In[ ]:


# Then lets create x_train, y_train, x_test, y_test arrays
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]


# * Now we have 3 dimensional input array (X) so we need to make it flatten (2D) in order to use as input for our first deep learning model.
# * Our label array (Y) is already flatten(2D) so we leave it like that.
# * Lets flatten X array(images array).
# 

# In[ ]:


X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)


# * As you can see, we have 348 images and each image has 4096 pixels in image train array.
# * Also, we have 62 images and each image has 4096 pixels in image test array.
# * Then lets take transpose. You can say that WHYY, actually there is no technical answer. I just write the code(code that you will see oncoming parts) according to it :)

# In[ ]:


x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)


# <a id="3"></a> <br>
# # Logistic Regression
# * When we talk about binary classification( 0 and 1 outputs) what comes to mind first is logistic regression.
# * However, in deep learning tutorial what to do with logistic regression there??
# * The answer is that  logistic regression is actually a very simple neural network. 
# * By the way neural network and deep learning are same thing. When we will come artificial neural network, I will explain detailed the terms like "deep".
# * In order to understand logistic regression (simple deep learning) lets first learn computation graph.

# <a id="5"></a> <br>
# ## Initializing parameters
# * As you know input is our images that has 4096 pixels(each image in x_train).
# * Each pixels have own weights.
# * The first step is multiplying each pixels with their own weights.
# * The question is that what is the initial value of weights?
#     * There are some techniques that I will explain at artificial neural network but for this time initial weights are 0.01.
#     * Okey, weights are 0.01 but what is the weight array shape? As you understand from computation graph of logistic regression, it is (4096,1)
#     * Also initial bias is 0.
# * Lets write some code. In order to use at coming topics like artificial neural network (ANN), I make definition(method).

# In[ ]:


# short description and example of definition (def)
def dummy(parameter):
    dummy_parameter = parameter + 5
    return dummy_parameter
result = dummy(3)     # result = 8

# lets initialize parameters
# So what we need is dimension 4096 that is number of pixels as a parameter for our initialize method(def)
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b


# In[ ]:


#w,b = initialize_weights_and_bias(4096)


# In[ ]:


def initialize_weight_and_bias(dimension):
    
    w=np.full((dimension,1),0.01)
    b = 0.0
    return w,b


# <a id="6"></a> <br>
# ## Forward Propagation
# * The all steps from pixels to cost is called forward propagation
# 

# In[ ]:


# calculation of z
#z = np.dot(w.T,x_train)+b
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


# In[ ]:


y_head = sigmoid(0)
y_head


# In[ ]:


# Forward propagation steps:
# find z = w.T*x+b
# y_head = sigmoid(z)
# loss(error) = loss(y,y_head)
# cost = sum(loss)
def forward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z) # probabilistic 0-1
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling
    return cost 


# <a id="7"></a> <br>
# ##  Optimization Algorithm with Gradient Descent
# 
# 
# 

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


# * Up to this point we learn 
#     * Initializing parameters (implemented)
#     * Finding cost with forward propagation and cost function (implemented)
#     * Updating(learning) parameters (weight and bias). Now lets implement it.

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
#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)


# * Woow, I get tired :) Up to this point we learn our parameters. It means we fit the data. 
# * In order to predict we have parameters. Therefore, lets predict.
# * In prediction step we have x_test as a input and while using it, we make forward prediction.

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
# predict(parameters["weight"],parameters["bias"],x_test)


# * We make prediction.
# * Now lets put them all together.

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
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 150)


# In[ ]:


def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iterations):
    dimension = x_train.shape[0]
    w,b = initialize_weight_and_bias(dimension)
    parameters,gradients,cost_list=update=(w,b,x_train,y_train,learning)


# * We learn logic behind simple neural network(logistic regression) and how to implement it.
# * Now that we have learned logic, we can use sklearn library which is easier than implementing all steps with hand for logistic regression.
# 
# 
# 

# <a id="8"></a> <br>
# ## Logistic Regression with Sklearn
# * In sklearn library, there is a logistic regression method that ease implementing logistic regression.
# 

# In[ ]:


from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)
print("test accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_test.T, y_test.T)))
print("train accuracy: {} ".format(logreg.fit(x_train.T, y_train.T).score(x_train.T, y_train.T)))


# In[ ]:


from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state = 42,max_iter=100)
print("Test Accuracy is : {}".format(logreg.fit(x_train.T,y_train.T).score(x_test.T,y_test.T)))


# <a id="10"></a> <br>
# # Artificial Neural Network (ANN)
# * It is also called deep neural network or deep learning.
# * **What is neural network:** It is basically taking logistic regression and repeating it at least 2 times.
# * In logistic regression, there are input and output layers. However, in neural network, there is at least one hidden layer between input and output layer.
# * **What is deep, in order to say "deep" how many layer do I need to have:** When I ask this question to my teacher, he said that ""Deep" is a relative term; it of course refers to the "depth" of a network, meaning how many hidden layers it has. "How deep is your swimming pool?" could be 12 feet or it might be two feet; nevertheless, it still has a depth--it has the quality of "deepness". 32 years ago, I used two or three hidden layers. That was the limit for the specialized hardware of the day. Just a few years ago, 20 layers was considered pretty deep. In October, Andrew Ng mentioned 152 layers was (one of?) the biggest commercial networks he knew of. Last week, I talked to someone at a big, famous company who said he was using "thousands". So I prefer to just stick with "How deep?""
# * **Why it is called hidden:** Because hidden layer does not see inputs(training set)
# * For example you have input, one hidden and output layers. When someone ask you "hey my friend how many layers do your neural network have?" The answer is "I have 2 layer neural network". Because while computing layer number input layer is ignored. 
#     

# <a id="11"></a> <br>
# ## 2-Layer Neural Network
# * Size of layers and initializing parameters weights and bias
# * Forward propagation
# * Loss function and Cost function
# * Backward propagation
# * Update Parameters
# * Prediction with learnt parameters weight and bias
# * Create Model

# <a id="12"></a> <br>
# ## Size of layers and initializing parameters weights and bias
# * For x_train that has 348 sample $x^{(348)}$:
# $$z^{[1] (348)} =  W^{[1]} x^{(348)} + b^{[1] (348)}$$ 
# $$a^{[1] (348)} = \tanh(z^{[1] (348)})$$
# $$z^{[2] (348)} = W^{[2]} a^{[1] (348)} + b^{[2] (348)}$$
# $$\hat{y}^{(348)} = a^{[2] (348)} = \sigma(z^{ [2] (348)})$$
# 
# * At logistic regression, we initialize weights 0.01 and bias 0. At this time, we initialize weights randomly. Because if we initialize parameters zero each neuron in the first hidden layer will perform the same comptation. Therefore, even after multiple iterartion of gradiet descent each neuron in the layer will be computing same things as other neurons. Therefore we initialize randomly. Also initial weights will be small. If they are very large initially, this will cause the inputs of the tanh to be very large, thus causing gradients to be close to zero. The optimization algorithm will be slow.
# * Bias can be zero initially.

# In[ ]:


# intialize parameters and layer sizes
def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    parameters = {"weight1": np.random.randn(3,x_train.shape[0]) * 0.1,
                  "bias1": np.zeros((3,1)),
                  "weight2": np.random.randn(y_train.shape[0],3) * 0.1,
                  "bias2": np.zeros((y_train.shape[0],1))}
    return parameters


# <a id="13"></a> <br>
# ## Forward propagation
# * Forward propagation is almost same with logistic regression.
# * The only difference is we use tanh function and we make all process twice.
# * Also numpy has tanh function. So we do not need to implement it.

# In[ ]:



def forward_propagation_NN(x_train, parameters):

    Z1 = np.dot(parameters["weight1"],x_train) +parameters["bias1"]
    A1 = np.tanh(Z1)
    Z2 = np.dot(parameters["weight2"],A1) + parameters["bias2"]
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


# <a id="14"></a> <br>
# ## Loss function and Cost function
# * Loss and cost functions are same with logistic regression
# * Cross entropy function
# 

# In[ ]:


# Compute cost
def compute_cost_NN(A2, Y, parameters):
    logprobs = np.multiply(np.log(A2),Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost


# <a id="15"></a> <br>
# ## Backward propagation
# * As you know backward propagation means derivative.
# * If you want to learn (as I said I cannot explain without talking bc it is little confusing), please watch video in youtube.
# * However the logic is same, lets write code.

# In[ ]:


# Backward Propagation
def backward_propagation_NN(parameters, cache, X, Y):

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


# <a id="16"></a> <br>
# ## Update Parameters 
# * Updating parameters also same with logistic regression.
# * We actually do alot of work with logistic regression

# In[ ]:


# update parameters
def update_parameters_NN(parameters, grads, learning_rate = 0.01):
    parameters = {"weight1": parameters["weight1"]-learning_rate*grads["dweight1"],
                  "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],
                  "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],
                  "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]}
    
    return parameters


# <a id="17"></a> <br>
# ## Prediction with learnt parameters weight and bias
# * Lets write predict method that is like logistic regression.

# In[ ]:


# prediction
def predict_NN(parameters,x_test):
    # x_test is a input for forward propagation
    A2, cache = forward_propagation_NN(x_test,parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(A2.shape[1]):
        if A2[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction


# <a id="18"></a> <br>
# ## Create Model
# * Lets put them all together

# In[ ]:


# 2 - Layer neural network
def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):
    cost_list = []
    index_list = []
    #initialize parameters and layer sizes
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)

    for i in range(0, num_iterations):
         # forward propagation
        A2, cache = forward_propagation_NN(x_train,parameters)
        # compute cost
        cost = compute_cost_NN(A2, y_train, parameters)
         # backward propagation
        grads = backward_propagation_NN(parameters, cache, x_train, y_train)
         # update parameters
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
    
    # predict
    y_prediction_test = predict_NN(parameters,x_test)
    y_prediction_train = predict_NN(parameters,x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters

parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)


# <font color='purple'>
# Up to this point we create 2 layer neural network and learn how to implement
# * Size of layers and initializing parameters weights and bias
# * Forward propagation
# * Loss function and Cost function
# * Backward propagation
# * Update Parameters
# * Prediction with learnt parameters weight and bias
# * Create Model
# 
# <br> Now lets learn how to implement L layer neural network with keras.

# <a id="19"></a> <br>
# # L Layer Neural Network
# 
#     
#     

# In[ ]:


# reshaping
x_train, x_test, y_train, y_test = x_train.T, x_test.T, y_train.T, y_test.T


# <a id="22"></a> <br>
# ## Implementing with keras library
# 

# In[ ]:


# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[1]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))


# In[ ]:




