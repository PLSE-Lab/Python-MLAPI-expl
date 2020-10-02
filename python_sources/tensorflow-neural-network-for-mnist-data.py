#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import pandas as pd
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


def load_dataset():
    train_set = pd.read_csv('../input/train.csv')
    test_set  = pd.read_csv('../input/test.csv' )
    
    train_set_x      = train_set.drop(columns=['label'])
    train_set_x_orig = np.array(train_set_x)
    train_set_y_orig = np.array(train_set['label'][:])
    
    #train_set_x_final, train_set_y_final, dev_set_x_final, dev_set_y_final = train_test_split(train_set_x_flat, train_set_y_orig, test_size=0.1, random_state=42)
    
    test_set_x_orig = np.array(test_set)
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig


# In[3]:


X_train_orig, Y_train_orig, X_test_orig = load_dataset()


# In[4]:


print(X_train_orig.shape)
print(Y_train_orig.shape)
print(X_test_orig.shape)


# In[5]:


X_train, X_dev, Y_train, Y_dev = train_test_split(X_train_orig, Y_train_orig, test_size = 0.1, random_state=42)


# In[6]:


def convert_to_one_hot(labels, C):
    C = tf.constant(C, name = 'C')    
    one_hot_matrix = tf.one_hot(labels, C, axis=0)

    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    
    return one_hot


# In[7]:


Y_train = convert_to_one_hot(Y_train, 10)

X_train = X_train.reshape(X_train.shape[0], -1).T
X_train = X_train/255

Y_dev = convert_to_one_hot(Y_dev, 10)

X_dev = X_dev.reshape(X_dev.shape[0], -1).T
X_dev = X_dev/255

X_test = X_test_orig.reshape(X_test_orig.shape[0], -1).T
X_test = X_test/255


# In[8]:


print(X_train.shape)
print(Y_train.shape)
print(X_dev.shape)
print(Y_dev.shape)
print(X_test .shape)


# In[9]:


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape=(n_x, None), name='X')
    Y = tf.placeholder(tf.float32, shape=(n_y, None), name='Y')

    return X, Y


# In[10]:


def initialize_parameters():
    W1 = tf.get_variable('W1', [25, 784], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable('b1', [25,   1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable('W2', [12,  25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable('b2', [12,   1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable('W3', [10,  12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable('b3', [10,   1], initializer = tf.zeros_initializer())
    
    parameters = { 'W1' : W1,
                   'b1' : b1,
                   'W2' : W2,
                   'b2' : b2,
                   'W3' : W3,
                   'b3' : b3 }
    
    return parameters


# In[11]:


def forward_propagation(X, parameters):
    
    W1 = parameters['W1'] 
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)

    return Z3


# In[12]:


def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = labels))
    
    return cost


# In[13]:


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    m = X.shape[1]
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    num_complete_minibatches = math.floor(m/mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


# In[14]:


def model(X_train, Y_train, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True):
    ops.reset_default_graph()
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    
    X, Y = create_placeholders(n_x, n_y) 
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y) 
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
            
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                
                _, mini_batch_cost = sess.run([optimizer, cost], feed_dict = { X : minibatch_X, Y : minibatch_Y})
                epoch_cost += mini_batch_cost / num_minibatches
                
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                plt.plot(np.squeeze(costs))
        
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_dev, Y: Y_dev}))
        
        return parameters


# In[15]:


parameters = model(X_train, Y_train, num_epochs = 700)


# In[16]:


def forward_propagation_for_predict(X, parameters):
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


# In[17]:


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
    
    x = tf.placeholder("float", [784, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction


# In[18]:


train_set_verify = pd.read_csv('../input/train.csv' )
test_set_submit  = pd.read_csv('../input/test.csv'  )


# In[19]:


index = 18

train_set_test = train_set_verify.iloc[index]
train_set_test = train_set_test.drop(['label'], axis=0)


# In[20]:


np_train_set_test = np.array(train_set_test)


# In[21]:


np_train_set_test = np_train_set_test.reshape(np_train_set_test.shape[0], -1)
np_train_set_test = np_train_set_test/255


# In[22]:


np_train_set_test.shape
my_image_prediction = predict(np_train_set_test, parameters)


# In[23]:


print("My algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))
print("Actual Value is: y = " + str(train_set_verify.iloc[index][0]))


# In[26]:


columns = ['ImageId', 'Label']
predictions = pd.DataFrame(columns = columns)


# In[28]:


W1 = tf.convert_to_tensor(parameters["W1"])
b1 = tf.convert_to_tensor(parameters["b1"])
W2 = tf.convert_to_tensor(parameters["W2"])
b2 = tf.convert_to_tensor(parameters["b2"])
W3 = tf.convert_to_tensor(parameters["W3"])
b3 = tf.convert_to_tensor(parameters["b3"])

x = tf.placeholder("float", [784, 1])
Z1 = tf.add(tf.matmul(W1, x), b1) 
A1 = tf.nn.relu(Z1)               
Z2 = tf.add(tf.matmul(W2, A1), b2)
A2 = tf.nn.relu(Z2)               
Z3 = tf.add(tf.matmul(W3, A2), b3)
p = tf.argmax(Z3)
sess = tf.Session()

for index in range(X_test.shape[1]):
    cur_test = X_test[:, index].reshape(X_test.shape[0], -1)
    prediction = sess.run(p, feed_dict = { x : cur_test })
    
    if index % 4000 == 0:
        print(index)
    #print("My algorithm predicts: y = " + str(np.squeeze(prediction))) 

    predictions = predictions.append({'ImageId' : index+1, 'Label' : str(np.squeeze(prediction))}, ignore_index=True)
    
sess.close()


# In[29]:


predictions = predictions.set_index('ImageId')


# In[30]:


predictions.to_csv('MNIST_SUBMISSION.csv')


# In[31]:


predictions.head()


# In[ ]:




