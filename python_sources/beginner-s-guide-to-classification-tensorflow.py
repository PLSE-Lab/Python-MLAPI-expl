#!/usr/bin/env python
# coding: utf-8

# If you're looking for the simplest way to get started with image classification, Keras, provides an interface to TensorFlow that allows you to create a neural network with only a few lines of code. (This kernel will help you get started: [Beginner's Guide to Image Classification (Keras)](https://www.kaggle.com/ndalziel/beginner-s-guide-to-image-classification-keras)). However, if you want the greater flexibility that you get from using TensorFlow directly, please read on....
# 
# To keep things simple we'll use a 4-layer perceptron (or feed-forward) architecture. However, since the success of the [Hinton team in the 2012 ImageNet competition](http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf), deep Convolutional Neural Network (CNN) architectures are typically used to achieve the best results on image classification. 
# 
# This notebook is split into five sections:
# 1. Intro to TensorFlow
# 2. Preprocessing the data
# 3. Setting up the network
# 4. Training the network, and
# 5. Making predictions

# ## 1. Introduction to TensorFlow
# 
# ### What is a tensor? 
# TensorFlow is a framework to run computations on tensors, which are  n-dimensional matrices. Although you can convert various Python objects to Tensor objects (using the tf.convert_to_tensor function), we'll see that you can pass in NumPy arrays of data.
# 
# ### How do you use TensorFlow?
# 
# TensorFlow separates the definition of computation from the execution of the computation. In practice, this means that you need to:
# 
# 1. Create a computational [graph](https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Directed_acyclic_graph.svg/1280px-Directed_acyclic_graph.svg.png)
# 
# 2. Run a session to execute the operations in the graph
# 
# More efficient computation is a major benefit of this approach. The graph structure facilitates computation across multiple CPUs or GPUs by allowing you to direct the computation of different parts of the graph to specific CPUs / GPUs. 
# 
# Let's take a look at a simple computation example...

# In[ ]:


import tensorflow as tf
# STEP 1 - Create a computational graph
op1 = tf.add(3,4)
op2 = tf.multiply(op1,5)

#STEP 2 - Run a session to execute the operations in the graph
with tf.Session() as sess:
    print (sess.run(op2))


# In the example above, the constants have been stored in the definition of the graph. If we want to define the graph in more general terms and pass in variables, we need to create placeholders...

# In[ ]:


# STEP 0 - Create placeholders
X = tf.placeholder(tf.int32,name = "X")
Y = tf.placeholder(tf.int32,name = "Y")
Z = tf.placeholder(tf.int32,name = "Z")

# STEP 1 - Create a computational graph
op1 = tf.add(X,Y)
op2 = tf.multiply(op1,Z)

#STEP 2 - Run a session to execute the operations in the graph
var1=3; var2 = 4; var3 = 5
with tf.Session() as sess:
    x = sess.run(op2, feed_dict={X: var1, Y: var2, Z: var3}) 
    print(x)


# Now that we understand how to do a simple operation in TensorFlow, let's turn to image classification...

# ## 2. Preprocessing the data

# In[ ]:


import math
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops


# Let's begin by reading in the training data, and taking a look at some sample images. Note that the data has already been partially pre-processed - the 28x28 data matrix for each image has been flattened into a 784-column vector. So, to display the images, we need to reshape into a 28x28 matrix. 

# In[ ]:


traindev = pd.read_csv('../input/train.csv')
X_traindev = traindev.loc[:,'pixel0':'pixel783']
Y_traindev = traindev.loc[:,'label']

for n in range(1,10):
    plt.subplot(1,10,n)
    plt.imshow(X_traindev.iloc[n].values.reshape((28,28)),cmap='gray')
    plt.title(Y_traindev.iloc[n])


# Next, we'll split the data into a training set and a cross-validation set. We'll use dummy variables to encode the labels - resulting in a 10-column label matrix (one for each digit). Note that we'll stack the training examples and labels in columns (following the convention in [Andrew Ng's Deep Learning course](https://www.coursera.org/learn/neural-networks-deep-learning)) by taking the transpose of X and Y - this will make the implementation easier later on..

# In[ ]:


# Create training data set
X_train = X_traindev[:40000].T.values
Y_train = Y_traindev[:40000]
Y_train = pd.get_dummies(Y_train).T.values

# Create cross-validation set
X_dev = X_traindev[40000:42000].T.values
Y_dev = Y_traindev[40000:42000]
Y_dev = pd.get_dummies(Y_dev).T.values

# read in test set
X_test = pd.read_csv('../input/test.csv').T.values

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of cross-validation examples = " + str(X_dev.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_dev shape: " + str(X_dev.shape))
print ("Y_dev shape: " + str(Y_dev.shape))
print ("X_test shape: " + str(X_test.shape))


# ## 3. Setting up the network
# We need to start off by initializing variables, creating placeholders and implementing the computation graph. At thre core of the function below are the **feed-foward equations**:
# 1.  Z(i) = W(i)*A(i-1) + b(i)
# 2. A(i) = activation_function * Z(i)
# 
# We use the rectified linear unit (relu) function as the activation function for the hidden layers. We need 784 units (28*28) as the number of units for the input layer, and we need 10 units for the output layer to correspond to the number of digits. We've chosen to add a 2nd and 3rd layer with 32 and 16 units respectively.

# In[ ]:


def create_graph(X_train,Y_train):
    #setup
    ops.reset_default_graph()                         # reset computation graph

    # initialize variables
    (n_x, training_examples) = X_train.shape                          
    n_y = Y_train.shape[0]                            
    costs = []

    # create placeholders
    X = tf.placeholder(tf.float32, shape=(n_x, None),name = "X")
    Y = tf.placeholder(tf.float32, shape=(n_y, None),name = "Y")
    
    # initialize weights
    W1 = tf.get_variable("W1", [32,784], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    W2 = tf.get_variable("W2", [16,32], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    W3 = tf.get_variable("W3", [10,16], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    
    # initialize biases
    b1 = tf.get_variable("b1", [32,1], initializer = tf.zeros_initializer())
    b2 = tf.get_variable("b2", [16,1], initializer = tf.zeros_initializer())
    b3 = tf.get_variable("b3", [10,1], initializer = tf.zeros_initializer())

    # create the graph for forward propagation
    Z1 = tf.add(tf.matmul(W1,X),b1)                                             
    A1 = tf.nn.relu(Z1)                                                         
    Z2 = tf.add(tf.matmul(W2,A1),b2)                                            
    A2 = tf.nn.relu(Z2)                                                         
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    return X, Y, Z3, training_examples


# Next, we'll specifiy how the model is optimized by choosing the **optimization algorithm** and the **cost (or loss) function.** The Adam optimization algorithm works well across a wide range of neural network architectures. (Adam essentially combined two other successful algorithms - gradient descent with momentum, and RMSProp.) For the loss function, 'softmax_cross_entropy_with_logits' is a good choice for multi-class classification. 

# In[ ]:


def define_optimization(Z3,Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) )
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
    
    return optimizer, cost


# Before we train the network, we need a function that creates randomized batches of training data so that we can implement mini-batch optimization (which will lead to faster optimization convergence)...

# In[ ]:


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# ## 4. Training the network
# Now, we'll open the TensorFlow session and execute the computation graph. Note that 1 epoch (or iteration) corresponds to a complete pass through the training set. For each epoch, we execute forward-prop and back-prop on all of the mini-batches. We'll print out the train and dev (or corss-validation) accuracy for each epoch, so can diagnose how well the neural network is performing...

# In[ ]:


def train_network(X_train,Y_train,X_dev, Y_dev, X_test, num_epochs,minibatch_size=64,print_n_epochs=1):
    
    tf.set_random_seed(1)                             
    X,Y,Z_final,training_examples = create_graph(X_train,Y_train)
    optimizer, cost = define_optimization(Z_final,Y)
    init = tf.global_variables_initializer() # set up variable initialization
    
    with tf.Session() as sess:
        sess.run(init) # initializes all the variables we've created
        for epoch in range(num_epochs):

            epoch_cost = 0.                       
            num_minibatches = int(training_examples / minibatch_size) 
            minibatches = random_mini_batches(X_train, Y_train)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], 
                                              feed_dict={X: minibatch_X, Y: minibatch_Y})                
                epoch_cost += minibatch_cost / num_minibatches
            
            print ("Cost after epoch %i: %.3f" % (epoch+1, epoch_cost), end = "") 
            correct_prediction = tf.equal(tf.argmax(Z_final), tf.argmax(Y))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print ("     Train Accuracy: %.3f" % (accuracy.eval({X: X_train, Y: Y_train})), end = "")
            print ("     Dev Accuracy: %.3f" % (accuracy.eval({X: X_dev, Y: Y_dev})))
        
        print ("Network has been trained")
        predict = tf.argmax(Z_final).eval({X: X_test})
        probs = tf.nn.softmax(Z_final).eval({X: X_test})
        
        return predict, probs
    
Y_predict, Y_probs = train_network(X_train, Y_train, X_dev, Y_dev, X_test, num_epochs = 20)


# ## 5. Making Predictions
# The function above returns the predictions, so we just need to create the submission file. Note that, although we're not using it here, the function above also uses the softmax function to return the probabilities associated with each digit.

# In[ ]:


Y_predict = Y_predict.reshape(-1,1)
predictions_df = pd.DataFrame (Y_predict,columns = ['Label'])
predictions_df['ImageID'] = predictions_df.index + 1
submission_df = predictions_df[predictions_df.columns[::-1]]
submission_df.to_csv("submission.csv", index=False, header=True)
submission_df.head()

