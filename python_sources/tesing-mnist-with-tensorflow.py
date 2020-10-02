# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

#lets visualize the data
import matplotlib.pyplot as plt
import seaborn as sns


#defining the required variables
learning_rate = 0.001
training_epochs = 15
batch_size = 100

num_classes = 10
n_samples = 42000

n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256

#function for multilayer perceptron
def multilayer_perceptron(x,weights,bias):
    
    #first layer with RELU
    layer_1 = tf.add(tf.matmul(x,weights['h1']),bias['b1'])
    #f(x) = max(0,x)
    layer_1 = tf.nn.relu(layer_1)
    
    #second hidden layer
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),bias['b2'])
    #f(x) = max(0,x)
    layer_2 = tf.nn.relu(layer_2)
    
    #output layer
    out_layer = tf.matmul(layer_2,weights['out']) + bias['out']
    
    return out_layer


weights = {
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,num_classes]))
}

bias = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([num_classes]))
}

x = tf.placeholder('float',[None,n_input])
y = tf.placeholder('float',[None,num_classes])

pred = multilayer_perceptron(x,weights,bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(pred,y))

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)


#train the model
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(n_samples/batch_size)
    
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
            avg_cost += c/total_batch
        
        print("Epoch:{} cost{:.4f}".format(epoch+1,avg_cost))
    
    
print("done")
    
    