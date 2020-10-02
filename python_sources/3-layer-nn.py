# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 19:02:21 2018

@author: Vince Liem
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def softmax(z):
    
    #Calculate exponent term first
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def softmax_loss(y,y_hat):
    
    # Clipping calue
    minval = 0.000000000001
    # Number of samples
    m = y.shape[0]
    # Loss formula, note that np.sum sums up the entire matrix and therefore does the job of two sums from the formula
    loss = -1/m * np.sum(y * np.log(y_hat.clip(min=minval)))
    return loss

# Log loss derivative, equal to softmax loss derivative
def loss_derivative(y,y_hat):
    
    return (y_hat-y)

def tanh_derivative(x):
    
    return (1 - np.power(x, 2))

def forward_prop(model, a0, output):
    
    # Load parameters from model
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    
    # Linear step
    z1 = a0.dot(W1) + b1
    
    # First activation function
    a1 = np.tanh(z1)
    
    # Second linear step
    z2 = a1.dot(W2) + b2
    
    # Second activation function
    a2 = np.tanh(z2)
    
    z3 = a2.dot(W3) + b3
    
    a3 = softmax(z3)
    
    cache = {'a0':a0,'z1':z1,'a1':a1,'z2':z2,'a2':a2, 'z3': z3, 'a3': a3}
    return a3 if output else cache

def backward_prop(model,cache,y):
    
    # Load parameters from model
    W2, W3 = model['W2'], model['W3']
    
    # Load forward propagation results
    a0, a1, a2, a3 = cache['a0'],cache['a1'],cache['a2'],cache['a3']
    
    # Get number of samples
    m = y.shape[0]
    
    # Backpropagation
    # Calculate loss derivative with respect to output
    dz3 = loss_derivative(y=y,y_hat=a3)
    
    # Calculate loss derivative with respect to second layer weights
    dW3 = 1/m*(a2.T).dot(dz3)
    
    # Calculate loss derivative with respect to second layer bias
    db3 = 1/m*np.sum(dz3, axis=0)
    
    # Calculate loss derivative with respect to third layer
    dz2 = np.multiply(dz3.dot(W3.T) ,tanh_derivative(a2))
    
    # Calculate loss derivative with respect to second layer weights
    dW2 = 1/m*(a1.T).dot(dz2)
    
    # Calculate loss derivative with respect to second layer bias
    db2 = 1/m*np.sum(dz2, axis=0)
    
    # Calculate loss derivative with respect to first layer
    dz1 = np.multiply(dz2.dot(W2.T) ,tanh_derivative(a1))
    
    # Calculate loss derivative with respect to first layer weights
    dW1 = 1/m*np.dot(a0.T, dz1)
    
    # Calculate loss derivative with respect to first layer bias
    db1 = 1/m*np.sum(dz1, axis=0)
    
    # Store gradients
    grads = {'dW3': dW3, 'db3': db3, 'dW2': dW2, 'db2': db2, 'dW1': dW1, 'db1': db1}
    return grads

def initialize_parameters(nn_input_dim,nn_hdim1,nn_hdim2,nn_output_dim):
    
    # First layer weights
    W1 = 2 *np.random.randn(nn_input_dim, nn_hdim1) - 1
    
    # First layer bias
    b1 = np.zeros((1, nn_hdim1))
    
    # Second layer weights
    W2 = 2 * np.random.randn(nn_hdim1, nn_hdim2) - 1
    
    # Second layer bias
    b2 = np.zeros((1, nn_hdim2))
    
    # Third layer weights
    W3 = 2 * np.random.randn(nn_hdim2, nn_output_dim) - 1
    
    # Second layer bias
    b3 = np.zeros((1, nn_output_dim))
    
    # Package and return model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
    return model

def update_parameters(model,grads,learning_rate):
    
    # Load parameters
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    
    # Update parameters
    W1 -= learning_rate * grads['dW1']
    b1 -= learning_rate * grads['db1']
    W2 -= learning_rate * grads['dW2']
    b2 -= learning_rate * grads['db2']
    W3 -= learning_rate * grads['dW3']
    b3 -= learning_rate * grads['db3']
    
    # Store and return parameters
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3 }
    return model

def predict(model, x):
    
    # Do forward pass
    c = forward_prop(model,x, False)
    #get y_hat
    y_hat = np.argmax(c['a3'], axis=1)
    return y_hat

def train(model,X_,y_,learning_rate, epochs=20000, print_loss=False):
    # Gradient descent. Loop over epochs
    for i in range(0, epochs):

        # Forward propagation
        cache = forward_prop(model,X_, False)
        #a1, probs = cache['a1'],cache['a2']
        # Backpropagation
        
        grads = backward_prop(model,cache,y_)
        # Gradient descent parameter update
        # Assign new parameters to the model
        model = update_parameters(model=model,grads=grads,learning_rate=learning_rate)
    
        # Pring loss & accuracy every 100 iterations
        if print_loss and i % 1000 == 0:
            a3 = cache['a3']
            print('Loss after iteration',i,':',softmax_loss(y_,a3))
            y_hat = predict(model,X_)
            y_true = y_.argmax(axis=1)
            print('Accuracy after iteration',i,':',accuracy_score(y_pred=y_hat,y_true=y_true)*100,'%')
    
    return model


df = pd.read_csv('../input/W1data.csv')
y = df[['Cultivar 1', 'Cultivar 2', 'Cultivar 3']].values
X = df.drop(['Cultivar 1', 'Cultivar 2', 'Cultivar 3'], axis = 1).values

# Hyper parameters
hiden_layer_size1 = 5
hiden_layer_size2 = 4
learning_rate = 0.01
epochs = 23001

np.random.seed(3)

# This is what we return at the end
model = initialize_parameters(nn_input_dim = X.shape[1], nn_hdim1 = hiden_layer_size1, nn_hdim2 = hiden_layer_size2, nn_output_dim = y.shape[1])
model = train(model, X, y, learning_rate = learning_rate, epochs = epochs, print_loss = True)

output = forward_prop(model, X, output = True)

print(output)