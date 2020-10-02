#!/usr/bin/env python
# coding: utf-8

# # Week 1 Challenge
# Congratulations! You made it through the first weeks content! The first week is the hardest because you have to learn about many new concepts. The first week is also the most math heavy. So it is all downhill from here. You deserve to kick your feed up and enjoy a glass of wine. But first, let's solve this weeks challenge.
# 
# The goal of the challenge is to deepen your understanding of how neural networks work. You will implement a multi class classifier in Excel and Python.

# ## The task
# You got a call from Italy. In a famous wine region, a disaster has happened. The labeling machine has mixed up the labels of three wine cultivars. Now there are thousands of bottles of which nobody knows who made them. Your first offer to distinguish the three wine makers by taste and then spend a couple of months drinking wine and labeling bottles as been refused. Instead, you are to build a classifier which recognizes the wine maker from 13 attributes of the wine.
# - Alcohol 
# - Malic acid 
# - Ash 
# - Alcalinity of ash 
# - Magnesium 
# - Total phenols 
# - Flavanoids 
# - Nonflavanoid phenols 
# - Proanthocyanins 
# - Color intensity 
# - Hue 
# - OD280/OD315 of diluted wines 
# - Proline 
# 
# The wine makers had 178 bottles left in their cellars. So we will use these bottles as our training data.

# ## The data
# You can find all data in the datasource connected to this notebook on kaggle. The Excel file contains everything you need to solve the challenge in Excel.
# The original data contains the 13 measurements for all 178 bottles. Because this is a multi class problem, the output $y$ has already been converted to one hot matrix. To make training easier, the data has also been normalized already. You will learn about normalization next week, but the basic goal is to ensure that all features of the data have the same mean and standard deviation. This makes it easier to deal with the data. The normalized data and one hot encodings have been copied over to the logistic regression and 2 Layer Wine net sheet.

# ### Loading the data into Python
# You can fork this notebook or simply create a new Kernel connected to this Kernels datasource in Kaggle. You can load the data like this:

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv('../input/W1data.csv')
df.head()


# In[ ]:


# Get labels
y = df[['Cultivar 1', 'Cultivar 2', 'Cultivar 3']].values
# Get inputs
X = df.drop(['Cultivar 1', 'Cultivar 2', 'Cultivar 3'], axis = 1)
X.shape, y.shape # Print shapes just to check


# ## Logistic regression
# Your first task it to implement logistic regression in Excel. To make the sheet easier to handle, Forward pass, Backward pass and weight update are ordered horizontally, not vertically. So scroll to the right to see the full sheet. The network has an input size of 13 and an output of 3. It uses a softmax activation for the output layer. To update parameters, copy over the ``New W1`` and New ``b1`` over into ``W1`` and ``b1``on the left of the sheet. **Make sure to use ``Paste values only`` when you copy over the weights!** Try to experiment with the learning rate while you train and get the loss as low as possible.

# ## 2 Layer network
# In the next sheet, you will implement a 2 layer neural network. The hidden unit has a size of 5. The activation function of the hidden layer is tanh, the activation function for the output layer is softmax again. You can train this network the same way as the logistic regression network. Note that you have to copy over two sets of weights and biases this time.

# In[ ]:


# Package imports
# Matplotlib is a matlab like plotting library
import matplotlib
import matplotlib.pyplot as plt
# Numpy handles matrix operations
import numpy as np
# SciKitLearn is a useful machine learning utilities library
import sklearn
# The sklearn dataset module helps generating datasets
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
# Display plots inline and change default figure size
get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)
import pandas as pd

# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates decision boundary plot
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Accent)
X = X.values
y = y.reshape(178,3)

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

def loss_derivative(y,y_hat):
    return (y_hat-y)

def tanh_derivative(x):
    return (1 - np.power(x, 2))

def forward_prop(model,a0):
    # Load parameters from model
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Linear step
    z1 = a0.dot(W1) + b1
    
    # First activation function
    a1 = np.tanh(z1)
    
    # Second linear step
    z2 = a1.dot(W2) + b2
    
    # Second activation function
    a2 = softmax(z2)
    cache = {'a0':a0,'z1':z1,'a1':a1,'z1':z1,'a2':a2}
    return cache

def backward_prop(model,cache,y):
    # Load parameters from model
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Load forward propagation results
    a0,a1, a2 = cache['a0'],cache['a1'],cache['a2']
    
    # Get number of samples
    m = y.shape[0]
    
    # Backpropagation
    # Calculate loss derivative with respect to output
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
    grads = {'dW2':dW2,'db2':db2,'dW1':dW1,'db1':db1}
    return grads

def initialize_parameters(nn_input_dim,nn_hdim,nn_output_dim):
    # First layer weights
    W1 = 2 *np.random.randn(nn_input_dim, nn_hdim) - 1
    
    # First layer bias
    b1 = np.zeros((1, nn_hdim))
    
    # Second layer weights
    W2 = 2 * np.random.randn(nn_hdim, nn_output_dim) - 1
    
    # Second layer bias
    b2 = np.zeros((1, nn_output_dim))
    
    # Package and return model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

def update_parameters(model,grads,learning_rate):
    # Load parameters
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    
    # Update parameters
    W1 -= learning_rate * grads['dW1']
    b1 -= learning_rate * grads['db1']
    W2 -= learning_rate * grads['dW2']
    b2 -= learning_rate * grads['db2']
    
    # Store and return parameters
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return model

def predict(model, x):
    # Do forward pass
    c = forward_prop(model,x)
    #get y_hat
    y_hat = np.argmax(c['a2'], axis=1)
    return y_hat

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
        if print_loss and i % 1 == 0:
            a2 = cache['a2']
            print('Loss after iteration',i,':',softmax_loss(y_,a2))
            y_hat = predict(model,X_)
            y_true = y_.argmax(axis=1)
            print('Accuracy after iteration',i,':',accuracy_score(y_pred=y_hat,y_true=y_true)*100,'%')
    
    return model

# Hyper parameters
hiden_layer_size = 3
# I picked this value because it showed good results in my experiments
learning_rate = 0.5
# Initialize the parameters to random values. We need to learn these.
np.random.seed(0)
# This is what we return at the end
model = initialize_parameters(nn_input_dim=13, nn_hdim= hiden_layer_size, nn_output_dim= 3)
model = train(model,X,y,learning_rate=learning_rate,epochs=500,print_loss=True)


# ## 3 Layer network
# After you have implemented a 2 layer network, open a new sheet and implement a 3 layer network. You may choose the size of the two hidden layers yourself. Just make sure that the output has a size of 3.

# In[ ]:


# Package imports
# Matplotlib is a matlab like plotting library
import matplotlib
import matplotlib.pyplot as plt
# Numpy handles matrix operations
import numpy as np
# SciKitLearn is a useful machine learning utilities library
import sklearn
# The sklearn dataset module helps generating datasets
import sklearn.datasets
import sklearn.linear_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
# Display plots inline and change default figure size
get_ipython().magic('matplotlib inline')
matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

import pandas as pd


# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates decision boundary plot
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Accent)

# Get labels
y = df[['Cultivar 1', 'Cultivar 2', 'Cultivar 3']].values
# Get inputs
X = df.drop(['Cultivar 1', 'Cultivar 2', 'Cultivar 3'], axis = 1)
X.shape, y.shape # Print shapes just to check
X = X.values

y = y.reshape(178,3)

type(y)

type(X)

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

def loss_derivative(y,y_hat):
    return (y_hat-y)

def tanh_derivative(x):
    return (1 - np.power(x, 2))

def forward_prop(model,a0):
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
    
    # Third linear step
    z3 = a2.dot(W3) + b3
    
    #Third activation function
    a3 = softmax(z3)
    
    cache = {'a0':a0,'z1':z1,'a1':a1,'z1':z1,'a2':a2, 'a3':a3, 'z3':z3}
    return cache

def backward_prop(model,cache,y):
    # Load parameters from model
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    
    # Load forward propagation results
    a0,a1, a2, a3 = cache['a0'],cache['a1'],cache['a2'], cache['a3']
    
    # Get number of samples
    m = y.shape[0]
    
    # Backpropagation
    # Calculate loss derivative with respect to output
    dz3 = loss_derivative(y=y,y_hat=a3)
   
    # Calculate loss derivative with respect to second layer weights
    dW3 = 1/m*(a2.T).dot(dz3)
    db3 = 1/m*np.sum(dz3, axis=0)
    
    
    # Calculate loss derivative with respect to second layer bias
    dz2 = np.multiply(dz3.dot(W3.T) ,tanh_derivative(a2))
    dW2 = 1/m*(a1.T).dot(dz2)
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

def initialize_parameters(nn_input_dim,nn_hdim,nn_output_dim):
    # First layer weights
    W1 = 2 *np.random.randn(nn_input_dim, nn_hdim) - 1
    
    # First layer bias
    b1 = np.zeros((1, nn_hdim))
    
    # Second layer weights
    W2 = 2 * np.random.randn(nn_hdim, nn_output_dim) - 1
    
    # Second layer bias
    b2 = np.zeros((1, nn_output_dim))
    
    # Third layer weights
    W3 = 2 * np.random.randn(nn_hdim, nn_output_dim) - 1
    
    # Third layer bias
    b3 = np.zeros((1, nn_output_dim))
    
    # Package and return model
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3':W3, 'b3':b3}
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
    model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3':W3, 'b3':b3}
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

def train(model,X_,y_,learning_rate, epochs=20000, print_loss=False):
    
    #empty array to store losses
    losses = []
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
        if print_loss and i % 100 == 0:
            
            a3 = cache['a3']
            
            loss = softmax_loss(y_,a3)
            
            print('Loss after iteration',i,':',loss)
            
            y_hat = predict(model,X_)
            
            y_true = y_.argmax(axis=1)
            
            print('Accuracy after iteration',i,':',accuracy_score(y_pred=y_hat,y_true=y_true)*100,'%')
            
            losses.append(loss)
    
    return model

def train_plot(model,X_,y_,learning_rate, epochs=20000, print_loss=False):
    
    #empty array to store losses
    losses = []
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
        if print_loss and i % 100 == 0:
            
            a3 = cache['a3']
            
            loss = softmax_loss(y_,a3)
            
            y_hat = predict(model,X_)
            
            y_true = y_.argmax(axis=1)
            
            losses.append(loss)
    
    plt.plot(losses)
    plt.xlabel("Epochs (x100)")
    plt.ylabel("Total Loss")
    plt.title("Learning rate = 0.05")
    
    return
# Hyper parameters
hiden_layer_size = 3
# I picked this value because it showed good results in my experiments
learning_rate = 0.05

# Initialize the parameters to random values. We need to learn these.
np.random.seed(0)
# This is what we return at the end
model = initialize_parameters(nn_input_dim=13, nn_hdim= hiden_layer_size, nn_output_dim= 3)
model = train(model,X,y,learning_rate=learning_rate,epochs=5000,print_loss=True)
plot = train_plot(model,X,y,learning_rate=learning_rate,epochs=5000,print_loss=True)


# ## Excel tips:
# ### Matrix multiplication
# The excel function for matrix multiplication is called [MMULT](https://support.office.com/en-us/article/MMULT-function-40593ed7-a3cd-4b6b-b9a3-e4ad3c7245eb). To multiply two matrices, select the output area where you want the output to be, enter the formula and hit CONTROL + SHIFT + ENTER. See this [youtube tutorial](https://www.youtube.com/watch?v=5bNooxRm960) if you have trouble. 
# 
# ### 'You cannot change part of an array.'
# Sometimes you might encounter a warning 'You cannot change part of an array.' In this case, either hit CONTROL + SHIFT + ENTER to apply the formula for the entire area or exit with ESC.
# 
# ### Transpose
# To transpose a matrix in excel you can use the [TRANSPOSE](https://support.office.com/en-us/article/TRANSPOSE-function-ed039415-ed8a-4a81-93e9-4b6dfac76027) function. Not that if you use transpose you *always* need to use CONTROL + SHIFT + ENTER, otherwise it will not do anything.
# 
# ### Exponents
# You can calculate $e^x$ using the EXP function. To calculate the exponent of multiple values element wise, like you have to do for softmax, you enter EXP and then the range of cells. For this you will have to hit CONTROL + SHIFT + ENTER again.
# 
# ### Softmax
# You will have to enter the formula for softmax element wise. That is, you enter the formula for one cell and can then expand it for the other cells. Note that softmax needs all cells of the example. So you need to make the reference to those fixed. Say you have a sample with 3 values you want to compute softmax for.
# 
# |0|A|B|C|
# |-|------|------|------|
# |1|0.3|0.4|0.1|
# 
# |$A_1$ 1|$A_1$ 2|$A_1$ 3|
# |-------|-------|-------|
# |EXP(A1)/SUM(EXP(\$A1:\$C1))|EXP(B1)/SUM(EXP(\$A1:\$C1))|EXP(C1)/SUM(EXP(\$A1:\$C1))|
# 
# ### Random initialization
# You can create random numbers in Excel using the [RAND](https://support.office.com/en-us/article/RAND-function-4cbfa695-8869-4788-8d90-021ea9f5be73) function. Note however, that the rand function create a new random number every time excel refreshes. So to create random numbers once and then freeze them, use RAND for all cells first, then copy it and paste the values using ``paste values only``on the same cells.

# ## Grading
# This weeks challenge is not a competition for the most accurate prediction. Instead, points are awarded to teams who submit the following implementations:
# Excel:
# - Logistic Regression: 2 points
# - 2 Layer Neural Net: 2 points
# - 3 Layer Neural Net: 4 points
# 
# Python
# - 2 Layer Neural Net: 2 points
# - 3 Layer Neural Net: 4 points
# 
# A total of 14 points can be won.
# 
# ## Submission
# For the excel challenges, submit your finished Excel notebook through slack.
# For the python notebook, create a public kernel and share the link via slack.
