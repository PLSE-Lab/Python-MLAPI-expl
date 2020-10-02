# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:50:50 2018

@author: andrewrona22
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as mt
import tensorflow as tf

#this function is useful in order to delete old graphs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

fashion_train = pd.read_csv("fashion-mnist_train.csv") #load the csv file
fashion_test = pd.read_csv("fashion-mnist_test.csv") #same
num_examples = 5000 #the total number of elements is 60000, without loss of
                    #generality we take less elements, for example 5000
num_test = 2000     #same reasoning for test set

#building 4 np arrays to use later
np_fashion_train = np.array(fashion_train[1:num_examples+1])
np_fashion_test = np.array(fashion_test[1:num_test+1])
test_images = np_fashion_test[:,1:]
test_labels = np_fashion_test[:,0]

#an example of a print of a random image from training set
firstrow = fashion_train[450:451]
x = np.array(firstrow)
new_x = np.delete(x,0)
y = new_x.reshape(28,28)
res = mt.imshow(y,cmap="gray") #so we can see the image as grayscale image

#starting CNN

height = 28
width = 28
channels = 1
n_inputs = height*width

#parameters of convolutional layer
conv1_fmaps = 32
conv1_ksize = 3
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

#parameters of pooling layer
pool2_fmaps = conv2_fmaps
#parameters of fully connected network and outputs
n_fc1 = 64
n_outputs = 10

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name = "X")
    X_reshaped = tf.reshape(X, shape=[-1,height,width, channels])
    y = tf.placeholder(tf.int32, shape = [None], name = "y")
    
conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size = conv1_ksize,
                         strides = conv1_stride, padding=conv1_pad,
                         activation = tf.nn.relu, name="conv1")
conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad,
                         activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool2"):
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool2_flat = tf.reshape(pool2, shape=[-1,pool2_fmaps*7*7])

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool2_flat, n_fc1, activation = tf.nn.relu,
                          name = "fc1")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name = "output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits,y,1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


#I decided to set the epochs to 10, but also 2 or 3 it's enough for good result,
#this because the train and the test sets are very similar
n_epochs = 10
batch_size = 100

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(num_examples // batch_size):
            #this cycle is for dividing step by step the heavy work of each neuron
            X_batch = np_fashion_train[iteration*batch_size:iteration*batch_size+batch_size,1:]
            y_batch = np_fashion_train[iteration*batch_size:iteration*batch_size+batch_size,0]
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict={X: test_images, y: test_labels})
        print("Epoch:",epoch+1, "Train accuracy:", acc_train, "test accuracy:", acc_test)
       
        save_path = saver.save(sess, "./my_fashion_model")