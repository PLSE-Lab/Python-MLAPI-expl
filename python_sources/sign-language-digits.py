#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 

# KERAS 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library


import warnings 
# Filter Warnings 
warnings.filterwarnings('ignore')

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Constants

# In[ ]:


IMG_HEIGHT = 64 
IMG_WIDTH = 64 
PIX = IMG_HEIGHT * IMG_WIDTH 


# ## At first 
# * At first, we will use just 0 and 1
# * After this, we improve our model for all classes 
# * In data, sign zero is between indexes 204 and 408. Number of zero sign is 205.
# * Also sign one is between indexes 822 and 1027. Number of one sign is 206. Therefore, we will use 205 samples from each classes(labels).

# In[ ]:


# Load data set 
x_l = np.load('../input/sign-language-digits-dataset/X.npy')
Y_l = np.load('../input/sign-language-digits-dataset/Y.npy')

plt.subplot(1,2,1)
plt.imshow(x_l[260].reshape(IMG_WIDTH, IMG_HEIGHT))
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(x_l[900].reshape(IMG_WIDTH, IMG_HEIGHT))
plt.axis('off')

plt.show()


# In[ ]:


# Prepare our X and Y 

X = np.concatenate((x_l[204:409], x_l[822:1027]), axis=0)
Y = np.concatenate((np.zeros(205), np.ones(205)), axis=0).reshape(X.shape[0], 1)

print(f"X's shape is {X.shape} and Y's shape is {Y.shape}")


# In[ ]:


# Then let's create our train and test sets 

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.15, random_state=42)

number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]


# * Now, we have 3d array for our X's and have 2d array our test. 
# * We need 2d array for training proccess, in this case we have already 2d array for test datasets 
# * So we need to reshape our train sets 

# In[ ]:


X_train_flatten = X_train.reshape(number_of_train, PIX) 
X_test_flatten = X_test.reshape(number_of_test, PIX)
print(f"X_train_flatten's shape is {X_train_flatten.shape} and X_test_flatten's shape is {X_test_flatten.shape}")


# * Like you see, we have 348 images and each image has 4096 pixels for train set 
# * we have 62 images and each image has 4096 pixels for test set 
# * so now we need to transpose our matrix because of multiplication process. We do this because we're gonna multiply our weights matrix with our features. 

# In[ ]:


x_train = X_train_flatten.T
x_test = X_test_flatten.T 
y_train = Y_train.T 
y_test = Y_test.T 
print('X train shape', x_train.shape)
print('X test shape', x_test.shape)
print('Y train shape', y_train.shape)
print('X test shape', y_test.shape)


# # Logistic Regression 
# * When we talk about binary classification, you can firstly think of Logistic Regression Algorithm. 
# * Actually, Logistic Regression is not a deep learning algorithm, but we can easily accept it is a very simple neural network. 
# * In order to understand what ANN is, we have to understand the Logistic Regr. 

# ### Initializing parameters
# * Our input that is our images are 4096, and every input has own weights. 
# * The first step is multiplying each pixels with their own weights. 
# * All rigth, What are these weights initial values ? 
# * First, we have to give them randomly.

# In[ ]:


def initialize_parameters(dimension): 
    w = np.full((dimension,1), 0.01)
    b = 0.0
    
    return w, b


# ### Forward Propagation 
# * The whole step from inputs to cost is called forward propagation
#     * z = (w.T).x + b => in this equation, we know that x is our pixel array,w(weights), b(bias) 
#     * Now, our result is z. After this, we're gonna put our equation into the sigmoid function, and it gives an output for it between 0 and 1. 
#     * We use sigmoid because of their y interval. its output has to be between 0 and 1. And we can use our output as a probability. 
#     * Then we calculate loss(error) function. Cost function is summation of all loss(error).
#     

# In[ ]:


def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head


# In[ ]:


# Example 
z = 15
y_head = sigmoid(z) 
y_head


# In[ ]:


def forward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T, x_train) + b 
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1-y_train) * np.log(1-y_head)
    cost = (np.sum(loss)) / x_train.shape[1]
    
    return cost


# ### Backward Propagation and Gradient Descent Algorithm 
# * We've covered initialization processes and forward propagation. 
# * Now, We know what our cost is, so we can start to train our model. 
# * Actually, our main goal is decreasing our cost, so if cost is high, our model don't work well. 
#     * Let's think first step, every thing starts with initializing weights and bias. Therefore cost is dependen on them. 
#     * In order to decrease cost, we need to update weights and bias.
#     * To do this, we have to use Gradient Descent Alg. and we will update our biasses and weights 
#     

# In[ ]:


def forward_backward_propagation(w,b,x_train,y_train):
    # Forward pro
    z = np.dot(w.T, x_train) + b 
    y_head = sigmoid(z)
   
    
    loss = -y_train * np.log(y_head) - (1-y_train) * np.log(1-y_head)
    cost = (np.sum(loss)) / x_train.shape[1]
    
    # Backward pro. 
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost, gradients


# Now, we can update our weights and biasses.[](http://)

# In[ ]:


def update(w,b,x_train,y_train, learning_rate, number_of_iteration): 
    cost_list = [] 
    cost_list2 = [] 
    index = [] 
    
    for i in range(number_of_iteration): 
        cost, gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        w = w - learning_rate * gradients['derivative_weight']
        b = b - learning_rate * gradients['derivative_bias']
        
        if i % 10 == 0: 
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i, cost))
            
    parameters = {'weight' : w , 'bias': b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list


# *  Up to this point we learn our parameters. It means we fit the data. 
# *  In order to predict we should have parameters. so, let's predict.

# In[ ]:


def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+b)
    y_predict = np.zeros((1,x_test.shape[1]))
    
    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            y_predict[0,i] = 0 
        else : 
            y_predict[0,i] = 1 
        
    return y_predict


# In[ ]:


def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations): 
    dimension = x_train.shape[0]
    w,b = initialize_parameters(dimension)
    
    parameters, gradients, cost_list = update(w,b,x_train,y_train,learning_rate, num_iterations)
    
    y_prediction_test = predict(parameters['weight'], parameters['bias'], x_test)    
    y_prediction_train = predict(parameters['weight'], parameters['bias'], x_train)
    
     # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 0.01, num_iterations = 150)


# In[ ]:


from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state=45, max_iter=150)
print('Test accuracy:' , logreg.fit(x_train.T, y_train.T).score(x_test.T,y_test.T))
#print('Train accuracy:' , logreg.fit(x_train.T, y_train.T).score(x_test.T,y_test.T))


# # Artifical Neural Network 
# * In logistic regression, there are just input and output layer. However, there is least one hidden layer between input and output layer in ANN. 
# 

# ## Size of Layers and Initializing Parameters weights and bias  

# In[ ]:


def initialize_parameters_and_layer_sizes_NN(x_train,y_train):
    parameters = {
        'weight1':np.random.randn(3, x_train.shape[0]) * 0.1 ,
        'bias1': np.zeros((3,1)),
        'weight2':np.random.randn(y_train.shape[0],3) * 0.1, 
        'bias2':np.zeros((y_train.shape[0],1))
                 } 
    return parameters


# ## Forward Propagation 
# * FP is almost the same with logistic regression.
# * The only difference is we use tanh funct instead of sigmoid func and we make the all proccess twice. 
# * Also, Numpy has the tanh function, we can easily implement it.

# In[ ]:


def forward_propagation_NN(x_train,parameters):
    Z1 = np.dot(parameters['weight1'],x_train ) + parameters['bias1']
    A1 = np.tanh(Z1) 
    Z2 = np.dot(parameters['weight2'],A1 ) + parameters['bias2'] 
    A2 = np.tanh(Z2)
    
    cache = {
        'Z1': Z1, 'A1':A1, 'Z2':Z2, 'A2':A2
    } 
    
    return A2, cache


# In[ ]:





# ## Loss and Cost Function 
# * We will use the Cross Entropy Function as a loss function 

# In[ ]:


def compute_cost_NN(A2,Y,parameters): 
    logprobs = np.multiply(np.log(A2), Y)
    cost = -np.sum(logprobs) / Y.shape[1]
    return cost 


# ## Backward Propagation 

# In[ ]:


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


# ## Update Parameters 

# In[ ]:


def update_parameters(parameters,grads,learning_rate=0.01):
    parameters = {
        'weight1' : parameters['weight1'] - learning_rate* grads['dweight1'],
        "bias1": parameters["bias1"]-learning_rate*grads["dbias1"],
        "weight2": parameters["weight2"]-learning_rate*grads["dweight2"],
        "bias2": parameters["bias2"]-learning_rate*grads["dbias2"]
    }
    
    return parameters


# ## Prediction 

# In[ ]:


def predict_NN(parameters,x_test):
    A2, cache = forward_propagation_NN(x_test, parameters)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(A2.shape[1]):
        if A2[0,i] <= 0.5:
            Y_prediction[0,i] = 0 
        else: 
            Y_prediction[0,i] = 1
            
    return Y_prediction


# # Create Model 
# * Let's put them all together 

# In[ ]:


def two_layer_neural_network(x_train,y_train,x_test,y_test,num_iterations):
    cost_list = [] 
    index_list = [] 
    
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)
    
    for i in range(0,num_iterations):
        A2,cache  = forward_propagation_NN(x_train, parameters)
        
        cost = compute_cost_NN(A2, y_train, parameters)
        
        grads = backward_propagation_NN(parameters,cache, x_train, y_train)
        
        parameters = update_parameters(parameters,grads, learning_rate=0.01) 
        
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


# # Neural Net with KERAS
# * We did it with 2 hidden layer because of simplicity and we did all of things by ourself because we had to understand what's going on under the hood. 
# * Now, we can use libraries to implement our neural network like Keras. 
# * If we use these libraries, we don't have to determine lots of things, we don't have to calculate anything, because Keras does it for us. 

# In[ ]:


def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = x_train.shape[0]))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train.T, y = y_train.T, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))


# In[ ]:





# In[ ]:




