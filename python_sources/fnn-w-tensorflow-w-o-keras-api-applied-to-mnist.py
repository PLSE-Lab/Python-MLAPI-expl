#!/usr/bin/env python
# coding: utf-8

# This is a step-by-step tutorial on how to use Neural Networks with TensorFlow library, applied to handwitten digit renognizing. I will start with a basic data analysis, then go through the construction of the network's nodes with TensorFlow, then show how to perform classical train and test routines. I will rely on a simple FNN (feed-forward network), which is definitely far from being the world's best choice for this specific task (please refer to the competition's page for more accurate solutions). The purpose here is to learn the basics of TensorFlow while applying it to a real-world problem.
# 
# IMPORTANT: this tutorial assumes you know the theory and basics behind TensorFlow.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Some basic EDA

# In[ ]:


# load data
X_train = pd.read_csv('../input/train.csv')
X_test = pd.read_csv('../input/test.csv')
print(X_train.shape)


# There are 42K examples for training, each described along with 728 (ie, 28x28) pixels (grey-scale numeric values). Let's have a closer look at it.

# In[ ]:


# class distribution
X_train['label'].unique()


# In[ ]:


import matplotlib.pyplot as plt

# how are the grey values distributed?
d = X_train.describe().T

ax = d.plot(y=['mean'], figsize=(20, 5), color='Black', kind='line') # mean values in black
plt.fill_between(x=d.index, y1='min', y2='max', data=d, alpha=.3) # min/max values filled in with blue
plt.show()


# Some preprocessing might be done, for example to discard the non discriminant pixels that are always -or almost- zero. More generally, a feature selection may be useful in view of the significant number of variables. Here, I am not going to do neither.

# There is no ground truth that has been given for the test examples. Hence, I suggest to split the train data into train/validation subsets, with 80% data for train.
# Again, one may consider more reliable smpling, like the cross validation technique that allows for more robust results. Let's stick to the first solution here.

# In[ ]:


# train/ validation samples
msk = np.random.rand(len(X_train)) < 0.8
X_valid = X_train[~msk]
X_train = X_train[msk]
print ("{} samples for train, {} for validation".format(len(X_train), len(X_valid)))


# Some classification problems naturally produce imbalanced classes; which may be challenging for the learning algorithms.. Let's give it a check, just in case..

# In[ ]:


# class distribution
X_train['label'].value_counts().plot(kind='bar')


# Ok. Does not seem to be the case here.

# # A simple FNN, to start with..
# Let's get back to the main topic: neural networks with TensorFlow. For the sake of clarity, I'm going to start with a (very) simple network architecture: a feed-forward network (FNN) with 2 hidden layers (in addition to the output layer). Let's start by defining some global variables that will be used later in the training routine..

# In[ ]:


# network specific parameters (layers' input/output dimensions)
n_inputs = 28*28 # input size
n_hidden1 = 300 # this is the output size of hidden layer 1
n_hidden2 = 100 # this is the output size of hidden layer 2
n_outputs = 10 # this is the output (final) layer size (will output the probability of each class from the 10 classes of digits)


# In[ ]:


import tensorflow as tf

# i'm gonna put the gloabl constant definitions here/ To be used later in the training step
n_iterations = 50
n_batches = 33 # nb batches (subsets used for iterative training), ie each batch is approx. 1000 size
learn_rate = 0.00003 # learning rate parameter (fixed empirically. Other values might be tested)


# As you might have seen before, TensorFlow relies on symbolic references to tensors in order to manipulate the data flows from input to output, as well as the hidden layer's parameters (weights). For input data, we need to book a place in memory, this can be done based on the "palce holders" component, in this way:

# In[ ]:


# place holders for the very first (input) layer
X = tf.placeholder(dtype=tf.float32, shape=[None, n_inputs], name='X') # None means not-specified (varying) size
y = tf.placeholder(dtype=tf.int32, shape=[None], name='y')


# Ok, now, let me define a helper python function that we will use several times. It will create each of the layers with the specified parameters (number of neurons, activation functions, etc.). As for the fully-connected layers of our FNN, this operation simply performs some activation function on a weighted sum of input values. Each layer has a shape that depends on the previous layer and stores its own parameters (weights and bias). Also, we need to specify an initial value for the weights, here we choose a truncated normal distribution.

# In[ ]:


# a generic function used to create the different layers of the model
def create_layer(input_layer, n_neurons, layer_name="", activation_fun=None):
    with tf.name_scope(layer_name):
        n_inputs = int(input_layer.get_shape()[1])
        initial_value = tf.truncated_normal((n_inputs, n_neurons)) # initial value (will updated at each iteration)
        w = tf.Variable(initial_value, name="weight") # weight vector, initiazed to initial_value
        b = tf.Variable(tf.zeros([n_neurons]), name="bias") # bias vector
        op = tf.matmul(input_layer, w) + b
        if activation_fun:
            op = activation_fun(op)
        return op


# In[ ]:


# the overall model architecture
with tf.name_scope("myfnn"):
    h1 = create_layer(X, n_hidden1, layer_name='hl1', activation_fun=tf.nn.relu)
    h2 = create_layer(h1, n_hidden2, layer_name='hl2', activation_fun=tf.nn.relu)
    logits = create_layer(h2, n_outputs, layer_name='output') # output layer with no activation function


# Ok, we now have a simple model with 2 hidden layers, each is fully connected to the previous one. The output layer is a logit function that will ouput the predicted probability for each class. However, the model definition can not yet be considered achieved..
# 
# The second part is to define what to optimize within this network, and how. We start by definig a cost function, also called loss. Again, I'm not going to deep dive into the details, let's choose a cross entropy function and optimize it with a Gradient Descent Boosting algorithm.

# In[ ]:


# cost function
with tf.name_scope('loss'):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(entropy)

#  train operation
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    training_op = optimizer.minimize(loss)


# The following operation is not mandatory but is very useful. It will allow to asses the model performance on the validation (heldout) data, each several iterations:

# In[ ]:


#  eval node
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1) # predicted class is the most probable one
    acc = tf.reduce_mean(tf.cast(correct, tf.float32)) # a simple accuray measure


# The model definition part can now be considered done. Please note that TensorFlow has not performed any operation yet; not even booked the necessary memory space for executing the graph.
# The second part (runtime part) consists of feeding the graph with data and iteratively calling the computation nodes.
# Some points need to be clarified before going through the code:
# - 'res' is a python variable, used to store intermediate values of performance scores
# - we adopt a mini-batch strategy for learning, i.e., we need to generate a mini batch at each iteration, with the ability to rewind the dataset
# - mini batches are generated on-demand based on pandas slicing. Please note that TensorFlow implements an adapted api for dataset manipulation that offers a mini-batch generation capability (see https://www.tensorflow.org/datasets)
# - the loss value is necessarily computed on the train data, while we can compute accuracy both on train and validation (held-out) data. This could be useful to reveal overfitting problems.
# - the 'feed_dict' is a technique implemented by TensorFlow that allows feeding different sets of data each time the batch changes.

# In[ ]:


# result container
res = pd.DataFrame({'epoch':[], 'loss':[], 'acc_train':[], 'acc_valid':[]})

# we need to know the exact batch size to split data into batches
batch_size = int(len(X_train)/n_batches)
print ('batch size (auto computed) = {}'.format(str(batch_size)))
print ('n_iterations to go = {}'.format(str(n_iterations)))

# this operation is important before starting any training (initialize all variables with the value specified in definition)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print ('it\tloss_t\tacc_t\tacc_v')
    
    for iteration_id in range(n_iterations):
        for batch_id in range(n_batches):
            i = batch_id * batch_size # start/end of current batch
            j = (batch_id + 1) * batch_size
            xx_train, yy_train = X_train.iloc[i:j, 1:], X_train.iloc[i:j, 0]
            
            # train and update parameters from the current batch data
            sess.run(training_op, feed_dict={X:xx_train, y:yy_train})
            
        # perform an evaluation at the end of each iteration
        loss_val = sess.run([loss], feed_dict={X:X_train.iloc[:, 1:], y:X_train.iloc[:, 0]})
        # evaluate accuracy on train /eval data may reveal any overfitting effect
        acc_train_val = sess.run([acc], feed_dict={X:X_train.iloc[:, 1:], y:X_train.iloc[:, 0]})
        acc_valid_val = sess.run([acc], feed_dict={X:X_valid.iloc[:,1:], y:X_valid.iloc[:,0]})
        
        res = res.append({'epoch':iteration_id, 'loss':loss_val[0], 'acc_train':acc_train_val[0], 
                          'acc_valid':acc_valid_val[0]}, ignore_index=True)
        if iteration_id % 10 == 0:
            print('{}\t{}\t{}\t{}'.format(iteration_id, str(round(loss_val[0], 1)), 
                                          str(round(acc_train_val[0], 3)), str(round(acc_valid_val[0], 3))))


# The training step is now finished. Let's see how it went..

# In[ ]:


# plot loss values
res = res.set_index(['epoch'])
plt.figure(figsize=(20,5))
plt.subplot(1, 2, 1)
plt.plot(res.pop('loss'))
plt.subplot(1, 2, 2)

plt.plot(res)
plt.legend(res.columns)


# The plot at the left show how the loss amount decreases over iterations. At the right hand, we see the improvment in accuracy (coputed on either either train or validation sets) over time. The gat between two lines seems to increase, which may suggest an overfitting effect in the last iterations..
# 
# # That's it!
# Now that you have understood this basic example, you are encouraged to experiment different architectures, grid-search the best parameters, and optimizers. Good luck!
