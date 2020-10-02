#!/usr/bin/env python
# coding: utf-8

# ## Implementing the Gradient Descent Algorithm
# 
# Let's implement the basic functions of the Gradient Descent algorithm to find the boundary in a small dataset with two classes. 
# 
# 
# ### Instructions
# 
# Implement the functions that build the gradient descent algorithm, namely:
# 
# 1. The *sigmoid activation* function: `sigmoid`;
# 2. The formula for the *prediction*: `output_formula`;
# 3. The formula for the *error* at a point: `error_formula`;
# 4. The function that *updates the parameters* with one gradient descent step: `update_weights`.
# 
# **When you implement them**
#         
# * Run the `train` function and this will graph the several of the lines that are drawn in successive gradient descent steps. 

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# #### #1 Visualize the data
# 
# Let's start with some functions that will help us plot and visualize the data.

# In[18]:


# helper functions for plotting and drawing lines

def plot_points(X, y):
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s = 25, color = 'blue', edgecolor = 'k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s = 25, color = 'red', edgecolor = 'k')

def display(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)


# #### #2 Reading and plotting the data

# In[19]:


labels = ['grade1', 'grade2', 'accepted']
data = pd.read_csv('../input/admissions/data.csv', header=None, names=labels)

X = np.array(data[['grade1', 'grade2']])
y = np.array(data['accepted'])
plot_points(X,y)
plt.show()


# In[20]:


data.head()


# In[21]:


# dimensions of our dataset
data.shape


# We have 100 observations, 2 input features and the output variable

# #### #3 Implementing the basic functions
# 
# These are the formulas we'll nedd to implement.
# 
# 1. Sigmoid activation function
# 
# $$\sigma(x) = \frac{1}{1+e^{-x}}$$
# 
# 2. Output (prediction) formula
# 
# $$\hat{y} = \sigma(w_1 x_1 + w_2 x_2 + b)$$
# 
# 3. Error function
# 
# $$Error(y, \hat{y}) = - y \log(\hat{y}) - (1-y) \log(1-\hat{y})$$
# 
# 4. The function that updates the weights
# 
# $$ w_i \longrightarrow w_i + \alpha (y - \hat{y}) x_i$$
# 
# $$ b \longrightarrow b + \alpha (y - \hat{y})$$

# In[22]:


# Activation (sigmoid) function
def sigmoid(x):
    return 1/ ( 1 + np.exp(-x) )

# Output (prediction) formula
def output_formula(features, weights, bias):
    return sigmoid(np.matmul(features,weights) + bias)

# Error (log-loss) formula
def error_formula(y, output):
    return -y*np.log(output) - (1- y)*np.log(1-output)

# Gradient descent step
def update_weights(x, y, weights, bias, learnrate):
    output = output_formula(x, weights, bias)
    d_error = y - output
    weights += learnrate * d_error * x
    bias += learnrate * d_error
    return weights, bias


# ##### From Scores to Probabilities
# 
# Let's define the combination of two new perceptrons as `0.4*w1 + 0.6*w2 + b`. Which of the following values for the `weights` and the `bias` would result in the `final probability` of the point to be 0.88?
# * 2, 6, -2
# * 3, 5, -2.2 **[+]**
# * 5, 4, -3
# 
# First, we need to get the score for each of the inputs values. Then, we have to apply the `sigmoid` function to transform the score into a probability. 

# In[23]:


def eq_quiz(w1, w2, b):
    return w1*0.4 + w2*0.6 + b


# In[24]:


# probabilities
sigmoid(eq_quiz(2,6,-2)), sigmoid(eq_quiz(3,5,-2.2)), sigmoid(eq_quiz(5,4,-3))


# #### #4 Training function
# This function will help us iterate the gradient descent algorithm through all the data, for a number of epochs. It will also plot the data, and some of the boundary lines obtained as we run the algorithm.

# In[25]:


n_records, n_features = X.shape
np.random.normal(scale=1 / n_features**.5, size=n_features)


# In[26]:


np.random.seed(44)

epochs = 100
learnrate = 0.01

def train(features, targets, epochs, learnrate, graph_lines=False):
    
    # the error for each epoch
    errors = []
    # the number of records and the number of features
    n_records, n_features = features.shape
    last_loss = None
    # start with random weights
    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)
    print('Initial Random Weights: ', weights[0], weights[1])
    # start with a bias of 0
    bias = 0
    for e in range(epochs):
        del_w = np.zeros(weights.shape)
        for x, y in zip(features, targets):
            # calculate the output
            output = output_formula(x, weights, bias)
            # calculate the error
            error = error_formula(y, output)
            # update the weights and bias
            weights, bias = update_weights(x, y, weights, bias, learnrate)
        # printing out the log-loss error on the training set
        out = output_formula(features, weights, bias)
        loss = np.mean(error_formula(targets, out))
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e,"==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines and e % (epochs / 100) == 0:
            display(-weights[0]/weights[1], -bias/weights[1])
            

    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0]/weights[1], -bias/weights[1], 'black')

    # Plotting the data
    plot_points(features, targets)
    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()


# #### #5 Time to train the algorithm!
# When we run the function, we'll obtain the following:
# - 10 updates with the current training loss and accuracy
# - A plot of the data and some of the boundary lines obtained. The final one is in black. Notice how the lines get closer and closer to the best fit, as we go through more epochs.
# - A plot of the error function. Notice how it decreases as we go through more epochs.

# In[27]:


train(X, y, epochs, learnrate, True)

