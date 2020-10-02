#!/usr/bin/env python
# coding: utf-8

# Regularization will help you reduce overfitting.
#     
# Regularization will drive your weights to lower values.
# 
# L2 regularization and Dropout are two very effective regularization techniques.
# 
# ### Table of Contents
# 
# 1. Overview of the Problem set
# 
# 2. Load and prep the data
# 
# 3. Building the parts of our algorithm.
#     - ** 3.4 compute_cost_with_regularization() to compute the value of the cost  J_regularized instead of compute_cost()**
#     - ** 3.5 backward_propagation_with_regularization() to compute grads with L2 regularization **
# 
# 4. Merge all functions into a model
#     - ** 4.1 nn_model with L2 regularization**
# 
# 5. Visualizing and Analysis
# 
# 6. Lab

# In[ ]:


import os
import random
import sys
import datetime
## pip3 install opencv-python
import cv2
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


## constants
TRAIN_DIR = "./input/train/"
TEST_DIR = "./input/test/"
TRAIN_SIZE = 25000
TEST_SIZE = 12500
DEV_RATIO = 0.1
IMAGE_HEIGHT = IMAGE_WIDTH = 128

LEARNING_RATE = 0.0001
MINIBATCH_SIZE = 32
INPUT_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH * 3
OUTPUT_SIZE = 2


# ### 1. Overview of the Problem set
# A fun project to differentiate dogs from cats. Dataset is from Kaggle: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition.
# 
# The ./input/train/ dir contains 12500 cat images and 12500 dog images. Each filename contains "cat" or "dog" as label.
# 
# The ./input/test/ dir contains 12500 images to classify
# 

# ### 2. Load and prep the data

# In[ ]:


## data utility functions
def split_data(two_dims_datas, split_ratio=DEV_RATIO):
    left_count = int(two_dims_datas.shape[1] * split_ratio)
    left_datas = two_dims_datas[:, :left_count]
    right_datas = two_dims_datas[:, left_count:]
    print("input datas shape: {}, left datas shape:{},     right datas shape: {}".format(two_dims_datas.shape, left_datas.shape, right_datas.shape))
    return left_datas, right_datas


# In[ ]:


def load_data(dirname=TRAIN_DIR, file_count=1000, shuffle=True):
    all_filenames = os.listdir(dirname)
    random.shuffle(all_filenames)
    filenames = all_filenames[:file_count]
    
    ## images
    images = np.zeros((file_count, IMAGE_HEIGHT*IMAGE_WIDTH*3))
    for i in range(file_count):
        imgnd_origin = cv2.imread(dirname+filenames[i])
        imgnd_resized = cv2.resize(imgnd_origin, (IMAGE_HEIGHT, IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
        imgnd_flatten = imgnd_resized.reshape(1,-1)
        images[i] = imgnd_flatten
    
    ## labels from filenames
    labels_list = ["dog" in filename for filename in filenames]
    labels = np.array(labels_list, dtype='int8').reshape(file_count, 1)
    
    ## shuffle
    if shuffle:
        permutation = list(np.random.permutation(labels.shape[0]))
        labels = labels[permutation, :]
        images = images[permutation, :]

    ## normalization
    images = images/255.0
    
    return images.T, labels.T


# In[ ]:


images, labels = load_data(file_count=200)
dev_images, train_images = split_data(images)
dev_labels, train_labels = split_data(labels)


# ### 3. Building the parts of our algorithm.
# 
# **Points**
# 
# - Implement a 2-class classification neural network with a single hidden layer
# - Use units with a non-linear activation function, such as tanh 
# - Compute the cross entropy loss 
# - Implement forward and backward propagation
# 
# **Here is our model**:
# <img src="http://p1plx6n23.bkt.clouddn.com/classification_kiank.png" style="width:600px;height:300px;">
# 
# **Mathematically**:
# 
# For one example $x^{(i)}$:
# $$z^{[1] (i)} =  W^{[1]} x^{(i)} + b^{[1] (i)}\tag{1}$$ 
# $$a^{[1] (i)} = \tanh(z^{[1] (i)})\tag{2}$$
# $$z^{[2] (i)} = W^{[2]} a^{[1] (i)} + b^{[2] (i)}\tag{3}$$
# $$\hat{y}^{(i)} = a^{[2] (i)} = \sigma(z^{ [2] (i)})\tag{4}$$
# $$y^{(i)}_{prediction} = \begin{cases} 1 & \mbox{if } a^{[2](i)} > 0.5 \\ 0 & \mbox{otherwise } \end{cases}\tag{5}$$
# 
# Given the predictions on all the examples, you can also compute the cost $J$ as follows: 
# $$J = - \frac{1}{m} \sum\limits_{i = 0}^{m} \large\left(\small y^{(i)}\log\left(a^{[2] (i)}\right) + (1-y^{(i)})\log\left(1- a^{[2] (i)}\right)  \large  \right) \small \tag{6}$$
# 

# ** 3.1 Defining the neural network structure **

# In[ ]:


n_x = train_images.shape[0] # size of input layer
n_h = 4          # hard code the hidden layer size to be 4
n_y = train_labels.shape[0] # size of output layer

layer_sizes = (n_x, n_h, n_y)
print(layer_sizes)


# ** 3.2 - Initialize the model's parameters **

# In[ ]:


def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# ** 3.3 forward_propagation **
# 
# Values needed in the backpropagation are stored in "cache". The cache will be given as an input to the backpropagation function.

# In[ ]:


## SIGMOID

def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """
    
    s = 1.0/(1.0 + np.exp(-1.0 * z))
    
    return s


# In[ ]:


# forward_propagation

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
            W1 -- weight matrix of shape (n_h, n_x)
            b1 -- bias vector of shape (n_h, 1)
            W2 -- weight matrix of shape (n_y, n_h)
            b2 -- bias vector of shape (n_y, 1)
    
    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Implement Forward Propagation to calculate A2 (probabilities)
    #print(W1.shape, X.shape)
    #print(np.matmul(W1, X).shape, b1.shape)
    Z1 = np.add(np.matmul(W1, X), b1)
    A1 = np.tanh(Z1)
    Z2 = np.add(np.matmul(W2, A1), b2)
    A2 = sigmoid(Z2)
    
    assert(A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


# ** 3.4 compute_cost_with_regularization() to compute the value of the cost  J_regularized **

#  L2 Regularization
# 
# The standard way to avoid overfitting is called **L2 regularization**. It consists of appropriately modifying your cost function, from:
# $$J = -\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small  y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} \tag{1}$$
# To:
# $$J_{regularized} = \small \underbrace{-\frac{1}{m} \sum\limits_{i = 1}^{m} \large{(}\small y^{(i)}\log\left(a^{[L](i)}\right) + (1-y^{(i)})\log\left(1- a^{[L](i)}\right) \large{)} }_\text{cross-entropy cost} + \underbrace{\frac{1}{m} \frac{\lambda}{2} \sum\limits_l\sum\limits_k\sum\limits_j W_{k,j}^{[l]2} }_\text{L2 regularization cost} \tag{2}$$
# 

# In[ ]:


# compute_cost
def compute_cost_with_regularization(A2, Y, parameters, lambd):
    """
    Computes the cross-entropy cost given in equation (13)
    
    Arguments:
    A2 -- The sigmoid output of the second activation, of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    parameters -- python dictionary containing your parameters W1, b1, W2 and b2
    lambd -- regularization hyperparameter, scalar
    
    Returns:
    cost - value of the regularized loss function (formula (2))
    """
    
    m = Y.shape[1] # number of example
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Compute the cross-entropy cost
    logprobs = np.multiply(Y, np.log(A2)) + np.multiply((1 - Y), np.log(1 - A2))
    cross_entropy_cost = (-1.0/m) * np.sum(logprobs)
    
    # Compute L2 regularization cost
    L2_regularization_cost = (np.sum(np.square(W1)) + np.sum(np.square(W2)))*(lambd/(2*m))
    
    # add cross_entropy_cost and L2_regularization_cost
    cost = cross_entropy_cost + L2_regularization_cost
    
    # makes sure cost is the dimension we expect. 
    cost = np.squeeze(cost)
    assert(isinstance(cost, float))
    
    return cost


# ** 3.5 backward_propagation_with_regularization() to compute grads with L2 regularization **
# 
# **Instructions**:
# Backpropagation is usually the hardest (most mathematical) part in deep learning. To help you, here again is the slide from the lecture on backpropagation. You'll want to use the six equations on the right of this slide, since you are building a vectorized implementation.  
# 
# <img src="http://p1plx6n23.bkt.clouddn.com/grad_summary.png" style="width:600px;height:300px;">
# 
# - Tips:
#     - To compute dZ1 you'll need to compute $g^{[1]'}(Z^{[1]})$. Since $g^{[1]}(.)$ is the tanh activation function, if $a = g^{[1]}(z)$ then $g^{[1]'}(z) = 1-a^2$. So you can compute 
#     $g^{[1]'}(Z^{[1]})$ using `(1 - np.power(A1, 2))`.

# Of course, because you changed the cost, you have to change backward propagation as well! All the gradients have to be computed with respect to this new cost. 
# 
# Implement the changes needed in backward propagation to take into account regularization. 
# 
# The changes only concern dW1, dW2 and dW3. For each, you have to add the regularization term's gradient ($\frac{d}{dW} ( \frac{1}{2}\frac{\lambda}{m}  W^2) = \frac{\lambda}{m} W$).

# In[ ]:


# backward_propagation

def backward_propagation_with_regularization(X, Y, cache, parameters, lambd):
    """
    Implement the backward propagation using the instructions above.
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".cache output from forward_propagation()
    X -- input data of shape (2, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    lambd -- regularization hyperparameter, scalar
    
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]
        
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backward propagation: calculate dW1, db1, dW2, db2. Add the regularization term to dW2,dW1
    dZ2 = A2 - Y
    dW2 = (1.0/m) * np.matmul(dZ2, np.transpose(A1)) + (lambd/m)*W2 ## add the regularization term
    db2 = (1.0/m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dZ1 = np.matmul(np.transpose(W2), dZ2) * (1 - np.power(A1, 2))
    dW1 = (1.0/m) * np.matmul(dZ1, np.transpose(X)) + (lambd/m)*W1 ## add the regularization
    db1 = (1.0/m) * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


# ** 3.6 Implement the update rule.**
# 
# Use gradient descent. We need (dW1, db1, dW2, db2) in order to update (W1, b1, W2, b2).
# 
# **General gradient descent rule**: $ \theta = \theta - \alpha \frac{\partial J }{ \partial \theta }$ where $\alpha$ is the learning rate and $\theta$ represents a parameter.
# 
# **Illustration**: The gradient descent algorithm with a good learning rate (converging) and a bad learning rate (diverging). Images courtesy of Adam Harley.
# 
# <img src="http://p1plx6n23.bkt.clouddn.com/sgd.gif" style="width:400;height:400;">
# <img src="http://p1plx6n23.bkt.clouddn.com/sgd_bad.gif" style="width:400;height:400;">
# 

# In[ ]:


# update_parameters

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Updates parameters using the gradient descent update rule given above
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients 
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[ ]:





# In[ ]:





# ### 4. Merge all functions into a model

# ** 4.1 nn_model with L2 regularization**: The general methodology to build a Neural Network is to:
#     1. Define the neural network structure ( # of input units,  # of hidden units, etc). 
#     2. Initialize the model's parameters
#     3. Loop:
#         - Implement forward propagation
#         - Compute loss
#         - Implement backward propagation to get the gradients
#         - Update parameters (gradient descent)

# In[ ]:


# nn_model

def nn_model(X, Y, n_h, num_iterations = 10000, learning_rate=1.2, print_cost=False, lambd = 0.7):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (n_y, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    n_x = X.shape[0]
    n_y = Y.shape[0]
    
    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)
    costs = []
    for i in range(0, num_iterations):
         
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost_with_regularization(A2, Y, parameters, lambd)
        costs.append(cost)
 
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation_with_regularization(X, Y, cache, parameters, lambd)
 
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)
        
        # cache all about model
        trained_model = {
            "layer_sizes": (n_x, n_h, n_y),
            "learning_rate": learning_rate,
            "costs": costs,
            "parameters": parameters
        }

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return trained_model


# ** 4.2 Predictions **
# 
# Use forward propagation to predict results
# 
# Reminder: predictions = $y_{prediction} = \mathbb 1 \text{{activation > 0.5}} = \begin{cases}
#       1 & \text{if}\ activation > 0.5 \\
#       0 & \text{otherwise}
#     \end{cases}$  

# In[ ]:


# predict

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    
    return predictions


# ### 5. Visualizing and Analysis

# In[ ]:


def plot_costs(trained_model):
    # Plot learning curve (with costs)
    costs = np.squeeze(trained_model["costs"])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate ={}\n, layer_sizes={}\n, accuracy:{}, m:{}\n".format(
        trained_model["learning_rate"], trained_model["layer_sizes"],
        trained_model.get("accuracy"), trained_model.get("m")
    ))
    #plt.show()
    


# In[ ]:


# Example of a picture that was wrongly classified.
trained_model = nn_model(train_images, train_labels, 5, num_iterations = 50)

index = 11
plt.imshow(dev_images[:,index].reshape((IMAGE_HEIGHT, IMAGE_HEIGHT, 3)))
predictions = predict(trained_model["parameters"], dev_images[:,index:index+1])
print ("y = {}, you predicted that it is a {}".format(dev_labels[0,index],int(predictions)))


# <font color='blue'>
# **What you should remember** -- the implications of L2-regularization on:
# - The cost computation:
#     - A regularization term is added to the cost
# - The backpropagation function:
#     - There are extra terms in the gradients with respect to weight matrices
# - Weights end up smaller ("weight decay"): 
#     - Weights are pushed to smaller values.

# ### 6. Lab
