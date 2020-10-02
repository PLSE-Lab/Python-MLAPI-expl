#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Libraries

import os
import math
import h5py
import scipy

# numpy
import numpy as np

# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
from sklearn.model_selection import train_test_split

# Charts
import matplotlib.pyplot as plt

# Image IO
import skimage.io
import skimage.transform
from PIL import Image
from scipy import ndimage

# Machine Learning
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.preprocessing.image import ImageDataGenerator
# from cnn_utils import *   // See below

get_ipython().run_line_magic('matplotlib', 'inline')

# Set random seed to make results reproducable
np.random.seed(1)
tf.set_random_seed(1)

# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.


# ### This defines the cnn_utils codes

# In[ ]:


# from cnn_utils import *

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

#def predict(X, parameters):
#    
#    W1 = tf.convert_to_tensor(parameters["W1"])
#    b1 = tf.convert_to_tensor(parameters["b1"])
#    W2 = tf.convert_to_tensor(parameters["W2"])
#    b2 = tf.convert_to_tensor(parameters["b2"])
##    W3 = tf.convert_to_tensor(parameters["W3"])
##    b3 = tf.convert_to_tensor(parameters["b3"])
#    
##    params = {"W1": W1,
##              "b1": b1,
##              "W2": W2,
##              "b2": b2,
##              "W3": W3,
##              "b3": b3}
#
#    params = {"W1": W1,
#              "b1": b1,
#              "W2": W2,
#              "b2": b2}    
#    
#    x = tf.placeholder("float", [12288, 1])
#    
#    z3 = forward_propagation(x, params)
#    p = tf.argmax(z3)
#    
#    with tf.Session() as sess:
#        prediction = sess.run(p, feed_dict = {x: X})
#        
#    return prediction


# ## Set parameters and categories
# Gather the training and validation datasets, create the embedded categories to use, and define the height and width of images to process.

# In[ ]:


# Parameters
training_dataset_path = "../input/dataset/dataset_updated/training_set"
test_dataset_path = "../input/dataset/dataset_updated/validation_set"

# categories to use
categories = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']
n_categories = len(categories)
category_embeddings = {
    'drawings': 0,
    'engraving': 1,
    'iconography': 2,
    'painting': 3,
    'sculpture': 4
}

# After computing the mean image size, we can set a default width and a default height to resize the images
width = 64
height = 64
n_channels = 3


# ## Match the metadata to the training set
# Here well have a look a the size of each category

# In[ ]:


# training dataset metadata
n_imgs = []
for cat in categories:
    files = os.listdir(os.path.join(training_dataset_path, cat))
    n_imgs += [len(files)]
    
plt.bar([_ for _ in range(n_categories)], n_imgs, tick_label=categories)
plt.show()


# ## Look at some images

# In[ ]:


fig, axes = plt.subplots(nrows=1, ncols=n_categories, figsize=(15, 3))

cat_cpt=0
for cat in categories:
    category_path = os.path.join(training_dataset_path, cat)
    img_name = os.listdir(category_path)[0]
    img = skimage.io.imread(os.path.join(category_path, img_name))
    img = skimage.transform.resize(img, (width, height, n_channels), mode='reflect')
    axes[cat_cpt].imshow(img, resample=True)
    axes[cat_cpt].set_title(cat, fontsize=8)
    cat_cpt += 1

plt.show()


# ## Preprocessing of the training and test data sets

# In[ ]:


training_data = []
for cat in categories:
    files = os.listdir(os.path.join(training_dataset_path, cat))
    for file in files:
        training_data += [(os.path.join(cat, file), cat)]

test_data = []
for cat in categories:
    files = os.listdir(os.path.join(test_dataset_path, cat))
    for file in files:
        test_data += [(os.path.join(cat, file), cat)]


# In[ ]:


def load_dataset(tuples_list, dataset_path):
    indexes = np.arange(len(tuples_list))
    np.random.shuffle(indexes)
    
    X = []
    y = []
    n_samples = len(indexes)
    cpt = 0
    for i in range(n_samples):
        t = tuples_list[indexes[i]]
        try:
            img = skimage.io.imread(os.path.join(dataset_path, t[0]))
            img = skimage.transform.resize(img, (width, height,n_channels), mode='reflect')
            X += [img]
            y_tmp = [0 for _ in range(n_categories)]
            y_tmp[category_embeddings[t[1]]] = 1
            y += [y_tmp]
        except OSError:
            pass
        
        cpt += 1
        
        if cpt % 1000 == 0:
            print("Processed {} images".format(cpt))

    X = np.array(X)
    y = np.array(y)
    
    return X, y


# In[ ]:


# Load the training and test datasets
X_train, y_train = load_dataset(training_data, training_dataset_path)
X_val, y_val = load_dataset(test_data, test_dataset_path)


# ## Examine the processed data
# 
# Lets examine the shape of the data

# In[ ]:


print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_val.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("y_train shape: " + str(y_train.shape))
print ("X_val shape: " + str(X_val.shape))
print ("y_val shape: " + str(y_val.shape))
print ("Y categories in first row: " + str(y_val[0]))
conv_layers = {}


# ## Create the placeholders for Tensorflow

# In[ ]:


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes
        
    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """

    X = tf.placeholder(dtype="float", shape=[None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(dtype="float", shape=[None, n_y])
    
    return X, Y


# In[ ]:


## Test placehoider is working
X, Y = create_placeholders(height, width, n_channels, n_categories)
print ("X = " + str(X))
print ("Y = " + str(Y))


# ### Expected Output
# X = Tensor("Placeholder:0", shape=(?, 64, 64, 3), dtype=float32)
# Y = Tensor("Placeholder_1:0", shape=(?, 5), dtype=float32) 

# ## Initialize the Tensor Parameters

# In[ ]:


def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}
    
    return parameters


# In[ ]:


# Test the initializing parameters
tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1 = " + str(parameters["W1"].eval()[1,1,1]))
    print("W2 = " + str(parameters["W2"].eval()[1,1,1]))


# ### Expected output:
# W1 = [ 0.00131723  0.1417614  -0.04434952  0.09197326  0.14984085 -0.03514394
#  -0.06847463  0.05245192]
# W2 = [-0.08566415  0.17750949  0.11974221  0.16773748 -0.0830943  -0.08058
#  -0.00577033 -0.14643836  0.24162132 -0.05857408 -0.19055021  0.1345228
#  -0.22779644 -0.1601823  -0.16117483 -0.10286498]

# ## Forward propagation
# 
# In TensorFlow, there are built-in functions that carry out the convolution steps for you.
# 
# * **tf.nn.conv2d(X,W1, strides = [1,s,s,1], padding = 'SAME')**: given an input XX and a group of filters W1W1, this function convolves W1W1's filters on X. The third input ([1,f,f,1]) represents the strides for each dimension of the input (m, n_H_prev, n_W_prev, n_C_prev). You can read the full documentation [here](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)
# 
# * **tf.nn.max_pool(A, ksize = [1,f,f,1], strides = [1,s,s,1], padding = 'SAME')**: given an input A, this function uses a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window. You can read the full documentation [here](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool)
# 
# * **tf.nn.relu(Z1)**: computes the elementwise ReLU of Z1 (which can be any shape). You can read the full documentation [here](https://www.tensorflow.org/api_docs/python/tf/nn/relu).
# 
# * **tf.contrib.layers.flatten(P)**: given an input P, this function flattens each example into a 1D vector it while maintaining the batch-size. It returns a flattened tensor with shape [batch_size, k]. You can read the full documentation [here](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/flatten).
# 
# *  **tf.contrib.layers.fully_connected(F, num_outputs)**: given a the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation [here](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/fully_connected).
# 
# In the last function above (**tf.contrib.layers.fully_connected**), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you do not need to initialize those weights when initializing the parameters.
# 
# Implement the forward_propagation function below to build the following model: CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED.
# 
# In detail, we will use the following parameters for all the steps:
# 
#  - Conv2D: stride 1, padding is "SAME"
#  - ReLU
#  - Max pool: Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
#  - Conv2D: stride 1, padding is "SAME"
#  - ReLU
#  - Max pool: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
#  - Flatten the previous output.
#  - FULLYCONNECTED (FC) layer: Apply a fully connected layer without an non-linear activation function. Do not call the softmax here. This will result in 5 neurons in the output layer, which then get passed later to a softmax. In TensorFlow, the softmax and cost function are lumped together into a single function, which you'll call in a different function when computing the cost. 
# 

# In[ ]:


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1], padding='SAME')
    # RELU
    A1 = tf.nn.relu(Z1)
    # MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding='SAME')
    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1], padding='SAME')
    # RELU
    A2 = tf.nn.relu(Z2)
    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1,4,4,1], strides=[1,4,4,1], padding='SAME')
    # FLATTEN
    P = tf.contrib.layers.flatten(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 5 neurons in output layer. One of the arguments should be "activation_fn=None" 
    Z3 = tf.contrib.layers.fully_connected(P,n_categories,activation_fn=None)

    return Z3


# In[ ]:


# Test the forward propogation is working
tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(height, width, n_channels, n_categories)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3, {X: np.random.randn(2,height,width,n_channels), Y: np.random.randn(2,n_categories)})
    print("Z3 = " + str(a))


# ### Expected result:
# Z3 = [[ 1.5836598   0.75720584 -0.97342306 -0.14359426  0.32579383]
#  [ 1.4317374   1.0383936  -0.93356955 -0.11601364  0.27605867]]

# ## Compute the cost
# 
# 
# * **tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y)**: computes the softmax entropy loss. This function both computes the softmax activation function as well as the resulting loss. You can check the full documentation [here](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits).
# * **tf.reduce_mean**: computes the mean of elements across dimensions of a tensor. Use this to sum the losses over all the examples to get the overall cost. You can check the full documentation [here](https://www.tensorflow.org/api_docs/python/tf/reduce_mean).
# 

# In[ ]:


def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    
    return cost


# In[ ]:


# Test the cost function
tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(height, width, n_channels, n_categories)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4,height,width,n_channels), Y: np.random.randn(4,n_categories)})
    print("cost = " + str(a))


# ### Expected Output:
# cost = 0.044414163

# ## Create the model and test

# In[ ]:


def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    
    Arguments:
    X_train -- training set, of shape (None, 128, 128, 3)
    Y_train -- test set, of shape (None, n_y = 5)
    X_test -- training set, of shape (None, 128, 128, 3)
    Y_test -- test set, of shape (None, n_y = 5)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
     
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
        
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy, parameters


# In[ ]:


# Run the model now it is created
_, _, parameters = model(X_train, y_train, X_val, y_val)


# ### Expected Output
# 
# Cost after epoch 0: 1.040116
# Cost after epoch 5: 0.590800
# 
# Tensor("Mean_1:0", shape=(), dtype=float32)
# Train Accuracy: 0.8507965
# Test Accuracy: 0.78971964
