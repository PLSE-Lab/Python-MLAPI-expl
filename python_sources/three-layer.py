
# coding: utf-8

# Three layer network. Code credits to Jannes, we made use of much of the code in the notebooks.

# Package imports
# Matplotlib is a matlab like plotting library
import matplotlib
import matplotlib.pyplot as plt
# Numpy handles matrix operations
import numpy as np
import pandas as pd

# SciKitLearn is a useful machine learning utilities library
import sklearn
# The sklearn dataset module helps generating datasets
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score


def softmax(z):
    '''
    Calculates the softmax activation of a given input x
    See: https://en.wikipedia.org/wiki/Softmax_function
    '''
    #Calculate exponent term first
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def softmax_loss(y,y_hat):
    '''
    Calculates the generalized logistic loss between a prediction y_hat and the labels y
    See: http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/

    We need to clip values that get too close to zero to avoid zeroing out. 
    Zeroing out is when a number gets so small that the computer replaces it with 0.
    Therefore, we clip numbers to a minimum value.
    '''
    # Clipping calue
    minval = 0.000000000001
    # Number of samples
    m = y.shape[0]
    # Loss formula, note that np.sum sums up the entire matrix and therefore does the job of two sums from the formula
    loss = -1/m * np.sum(y * np.log(y_hat.clip(min=minval)))
    return loss


def loss_derivative(y,y_hat):
    '''
    Calculates the gradient (derivative) of the loss between point y and y_hat
    See: https://stats.stackexchange.com/questions/219241/gradient-for-logistic-loss-function
    '''
    return (y_hat-y)


def tanh_derivative(x):
    '''
    Calculates the derivative of the tanh function that is used as the first activation function
    See: https://socratic.org/questions/what-is-the-derivative-of-tanh-x
    '''
    return (1 - np.power(x, 2))


def forward_prop(model,a0):
    '''
    Forward propagates through the model, stores results in cache.
    See: https://stats.stackexchange.com/questions/147954/neural-network-forward-propagation
    A0 is the activation at layer zero, it is the same as X
    '''
    
    # Load parameters from model
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    
    # Linear step
    z1 = a0.dot(W1) + b1
    
    # First activation function
    a1 = np.tanh(z1)
    
    # Second linear step
    z2 = a1.dot(W2) + b2
    
    # Second activation function
    a2 = softmax(z2)
    
    # Third linear step
    z3 = a2.dot(W3) + b3
    
    # Second activation function
    a3 = softmax(z3)
    
    cache = {'a0':a0,'z1':z1,'a1':a1,'z2':z2,'a2':a2,'z3':z3,'a3':a3}
    return cache


def backward_prop(model,cache,y):
    '''
    Backward propagates through the model to calculate gradients.
    Stores gradients in grads dictionary.
    See: https://en.wikipedia.org/wiki/Backpropagation
    '''
    # Load parameters from model
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    
    # Load forward propagation results
    a0, a1, a2, a3 = cache['a0'],cache['a1'],cache['a2'],cache['a3']
    
    # Get number of samples
    m = y.shape[0]
    
    # Backpropagation
    # Calculate loss derivative with respect to output
    dz3 = loss_derivative(y=y,y_hat=a3)
   
    # Calculate loss derivative with respect to third layer weights
    dW3 = 1/m*(a2.T).dot(dz3)
    
    # Calculate loss derivative with respect to third layer bias
    db3 = 1/m*np.sum(dz3, axis=0)
    
    # Calculate loss derivative with respect to second layer
    dz2 = loss_derivative(y=y,y_hat=a2)
   
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
    grads = {'dW3':dW3,'db3':db3,'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
    return grads


def initialize_parameters(nn_input_dim,nn_hdim1,nn_hdim2,nn_output_dim):
    '''
    Initializes weights with random number between -1 and 1
    Initializes bias with 0
    Assigns weights and parameters to model
    '''
    # First layer weights
    W1 = 2 *np.random.randn(nn_input_dim, nn_hdim1) - 1
    
    # First layer bias
    b1 = np.zeros((1, nn_hdim1))
    
    # Second layer weights
    W2 = 2 * np.random.randn(nn_hdim1, nn_hdim2) - 1
    
    # Second layer bias
    b2 = np.zeros((1, nn_output_dim))
    
    # Second layer weights
    W3 = 2 * np.random.randn(nn_hdim2, nn_output_dim) - 1
    
    # Second layer bias
    b3 = np.zeros((1, nn_output_dim))
    
    # Package and return model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,'W3': W3, 'b3': b3}
    return model


def predict(model, x):
    '''
    Predicts y_hat as 1 or 0 for a given input X
    '''
    # Do forward pass
    c = forward_prop(model,x)
    #get y_hat
    y_hat = np.argmax(c['a3'], axis=1)
    return y_hat


def update_parameters(model,grads,learning_rate):
    '''
    Updates parameters accoarding to gradient descent algorithm
    See: https://en.wikipedia.org/wiki/Gradient_descent
    '''
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
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,'W3': W3, 'b3': b3}
    return model


def train(model,X_,y_,learning_rate, epochs=20000, print_loss=False):
    # Gradient descent. Loop over epochs
    for i in range(0, epochs):

        # Forward propagation
        cache = forward_prop(model,X_)
        #a1, probs = cache['a1'],cache['a2']
        # Backpropagation
        
        grads = backward_prop(model,cache,y_)
        # Gradient descent parameter update
        # Assign new parameters to the model
        model = update_parameters(model=model,grads=grads,learning_rate=learning_rate)
    
        # Pring loss & accuracy every 100 iterations
        if print_loss and i % (epochs/10) == 0:
            a3 = cache['a3']
            print('Loss after iteration',i,':',softmax_loss(y_,a3))
            y_hat = predict(model,X_)
            y_true = y_.argmax(axis=1)
            print('Accuracy after iteration',i,':',accuracy_score(y_pred=y_hat,y_true=y_true)*100,'%')
    
    return model
    

############################################################################################################
######################## RUN TEST  #########################################################################
############################################################################################################

# Read data
df = pd.read_csv('../input/W1data.csv')

# Make numpy array
np_df = np.array(df)

# Extract explanatory variables
X = np_df[:,range(0,13)]

# Extract dependent variable
y = np_df[:,range(13,16)]

# Hyper parameters
hiden_layer_size1 = 10
hiden_layer_size2 = 3

# I picked this value because it showed good results in my experiments
learning_rate = 0.05
# Initialize the parameters to random values. We need to learn these.
np.random.seed(10)
# This is what we return at the end
model = initialize_parameters(nn_input_dim=13, nn_hdim1= hiden_layer_size1, nn_hdim2= hiden_layer_size2, nn_output_dim= 3)
model = train(model,X,y,learning_rate=learning_rate,epochs=8000,print_loss=True)
