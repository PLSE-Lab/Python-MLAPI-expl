#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_train.info()


# In[ ]:


Y_train_orig = df_train['label'].values
X_train_orig = df_train.drop('label', axis=1).values


# In[ ]:


print("X_train_orig shape:", X_train_orig.shape)
print("Y_train_orig shape:", Y_train_orig.shape)


# In[ ]:


X_train = X_train_orig.T
Y_train = Y_train_orig.reshape(1, Y_train_orig.shape[0])


# In[ ]:


print("X_train shape:", X_train.shape)
print("Y_train shape:", Y_train.shape)


# In[ ]:


print("Mean of X_train:", np.mean(X_train))


# In[ ]:


X_train = X_train / 255.


# In[ ]:


print("Mean of X_train after normalization:", np.mean(X_train))


# In[ ]:


Y_train_onehot = np.zeros((Y_train.max()+1, Y_train.shape[1]))
Y_train_onehot[Y_train, np.arange(Y_train.shape[1])] = 1

print("Shape of Y_train_onehot:", Y_train_onehot.shape)


# In[ ]:


def relu(z):
    z_relu = np.maximum(z, 0)
    
    activation_cache = (z)
    return z_relu, activation_cache


# In[ ]:


y = np.arange(-10, 10)
relu_of_y, _ = relu(y)
relu_of_y


# In[ ]:


def relu_derivative(z):
    z_derivative = np.zeros(z.shape)
    z_derivative[z > 0] = 1
    
    return z_derivative


# In[ ]:


derivative_relu_of_y = relu_derivative(relu_of_y)
derivative_relu_of_y


# In[ ]:


def softmax(z):
    z_exp = np.exp(z - np.max(z))
    z_softmax = z_exp / np.sum(z_exp, axis=0) 
    
    activation_cache = (z)
    return z_softmax, activation_cache


# In[ ]:


a = np.array([1, 5, 10])
b, _ = softmax(a)
print(b, np.sum(b))

a = np.array([1, 1, 1])
b, _ = softmax(a)
print(b, np.sum(b))


# In[ ]:


def initialize_parameters(layers_dims):
    L = len(layers_dims) - 1
    parameters = {}
    
    for l in range(1, L+1):
        parameters["W"+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * (1 / layers_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layers_dims[l], 1))
    
    return parameters


# In[ ]:


def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    
    cache = (A_prev, W, b)
    return Z, cache


def activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    
    if activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'softmax':
        A, activation_cache = softmax(Z)
    
    cache = (linear_cache, activation_cache)
    return A, cache


def L_forward_propagate(X, parameters):
    caches = []
    L = len(parameters) // 2
    
    A_prev = X
    for l in range(1, L):
        A, cache = activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], 'relu')
    
        A_prev = A
        caches.append(cache)
        
    AL, cache = activation_forward(A_prev, parameters["W"+str(L)], parameters["b"+str(L)], 'softmax')
    caches.append(cache)
    
    return AL, caches


# In[ ]:


def compute_cost(Y, AL):
    m = Y.shape[1]
    
    cost = -(1/m) * np.sum(Y * np.log(AL) + (1-Y) * np.log(1-AL))
    return cost


# In[ ]:


def linear_backward(dZ, linear_cache):
    (A_prev, W, b) = linear_cache
    m = A_prev.shape[1]
    
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dW, db, dA_prev

def activation_backward(Y, cache, activation, A=0, dA=0):
    (linear_cache, activation_cache) = cache
    
    (Z) = activation_cache
    if activation == 'relu':
        dZ = dA * relu_derivative(Z)
    elif activation == 'softmax':
        dZ = A - Y
    
    dW, db, dA_prev = linear_backward(dZ, linear_cache)
    
    return dW, db, dA_prev


def L_backward_propagate(X, Y, AL, parameters, caches):
    m = Y.shape[1]
    L = len(parameters) // 2
    grads = {}
    
    cache = caches[L-1]
    dW, db, dA_prev = activation_backward(Y, cache, 'softmax', A = AL)
    grads["dW"+str(L)] = dW
    grads["db"+str(L)] = db
    
    dA = dA_prev
    for l in range(L-1, 0, -1):
        cache = caches[l-1]
        
        dW, db, dA_prev = activation_backward(Y, cache, 'relu', dA = dA)
        
        grads["dW"+str(l)] = dW
        grads["db"+str(l)] = db
        dA = dA_prev
    
    return grads


# In[ ]:


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    params = {}
    
    # Need to make this parallel
    for l in range(1, L+1):
        params["W"+str(l)] = parameters["W"+str(l)] - learning_rate * grads["dW"+str(l)]
        params["b"+str(l)] = parameters["b"+str(l)] - learning_rate * grads["db"+str(l)]
    
    return params


# In[ ]:


def model(X, Y, epochs=300, mini_batch_size=16, learning_rate=0.01):
    layers_dims = [X.shape[0], 64, 32, 16, Y.shape[0]]
    costs = []
    beta = 0.9
    
    parameters = initialize_parameters(layers_dims)
    
    for i in range(0, epochs):
        print("Epoch", i+1, ":")
        
        m = X.shape[1]
        
        permutation = np.random.permutation(m)
        X_shuffle = X[:, permutation]
        Y_shuffle = Y[:, permutation]
        
        num_mini_batch = m // mini_batch_size
        
        avg_cost = 0
        for  j in range(0, num_mini_batch):
            X_mini_batch = X_shuffle[:, (j*mini_batch_size) : (j+1)*mini_batch_size]
            Y_mini_batch = Y_shuffle[:, (j*mini_batch_size) : (j+1)*mini_batch_size]
            
            AL, caches = L_forward_propagate(X_mini_batch, parameters)
            cost = compute_cost(Y_mini_batch, AL)
            grads = L_backward_propagate(X_mini_batch, Y_mini_batch, AL, parameters, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
        
            if j % 5 == 0:
                op = "Cost(%5d/%5d): %3.6f" % ((j+1) * mini_batch_size, m, cost)
                print(op, end='\r')
            
            avg_cost = beta * avg_cost + (1 - beta) * cost
        
        costs.append(avg_cost)
        
        if m % mini_batch_size != 0:
            X_mini_batch = X_shuffle[:, (num_mini_batch*mini_batch_size):]
            Y_mini_batch = Y_shuffle[:, (num_mini_batch*mini_batch_size):]
            
            AL, caches = L_forward_propagate(X_mini_batch, parameters)
            cost = compute_cost(Y_mini_batch, AL)
            grads = L_backward_propagate(X_mini_batch, Y_mini_batch, AL, parameters, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
            
            op = "Cost(%5d/%5d): %3.6f" % (m, m, cost)
            print(op, end='\r')
            
            avg_cost = beta * avg_cost + (1 - beta) * cost
            costs.append(avg_cost)
        
        print()
    return parameters, costs


# In[ ]:


parameters, costs = model(X_train, Y_train_onehot, epochs=75, mini_batch_size=64)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(range(0, len(costs)*5, 5), costs)


# In[ ]:


def predict(X, parameters):
    A2, _ = L_forward_propagate(X, parameters)
    
    Y_predict = np.argmax(A2, axis=0)
    Y_predict = Y_predict.reshape(1, Y_predict.shape[0])
    
    return Y_predict


# In[ ]:


Y_train_predict = predict(X_train, parameters)


# In[ ]:


def accuracy(Y, Y_pred):
    tp = np.sum((Y == Y_pred).astype(int))
    
    return tp / Y.shape[1]


# In[ ]:


acc = accuracy(Y_train, Y_train_predict)
print("Accuracy on training data:", acc)


# In[ ]:


df_test = pd.read_csv('../input/test.csv')
X_test = df_test.values.T
X_test = X_test / 255.

X_test.shape


# In[ ]:


Y_test_predict = predict(X_test, parameters)
Y_test_predict.shape


# In[ ]:


ids = range(1, Y_test_predict.shape[1]+1)

my_submission = pd.DataFrame({'ImageId': ids, 'Label': np.squeeze(Y_test_predict)})
my_submission.to_csv('submission.csv', index=False)


# In[ ]:




