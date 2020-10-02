#!/usr/bin/env python
# coding: utf-8

# The goal for this Kernel is simple:
# Adapt the "original" [Tensorflow tutorial example][1] to produce a Kaggle submission.
# We will also try to add some explanation where this seems plausible. However, most of the explanation will be imported from the original tutorial. It kind of works now ;)
# 
#   [1]: https://www.tensorflow.org/get_started/mnist/beginners
# 
# This is Version 1.1 of the notebook. It adds evaluation against the test data and renames some variables.

# In[ ]:


# Start with reading and "preprocessing" some data

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Read in training data from train.csv
train = pd.read_csv('../input/train.csv')
trainfv = train.drop(['label'], axis=1).values.astype(dtype=np.float32)
# trainFeatureVectorsConvoFormat = trainFeatureVectors.reshape(42000, 28, 28, 1)

# Naming might be improveable... ;)
trainLabels = train['label'].tolist()
ohtrainLabels = tf.one_hot(trainLabels, depth=10)
ohtrainLabelsNdarray = tf.Session().run(ohtrainLabels).astype(dtype=np.float64)

# Read in testing data from test.csv
test = pd.read_csv('../input/test.csv')
testfv = test.drop(['label'], axis=1).values.astype(dtype=np.float32)
# testFeatureVectorsConvoFormat = testFeatureVectors.reshape(28000, 28, 28, 1)
testLabels = test['label'].tolist()
ohtestLabels = tf.one_hot(testLabels, depth=10)
ohtestLabelsNdarray = tf.Session().run(ohtestLabels).astype(dtype=np.float64)

print(trainfv[0],ohtrainLabelsNdarray[0])

# Show some training data and there labels
for i in range(5):
    pixels = trainfv[i].reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    print("Label: ",trainLabels[i]," - OneHot: ",ohtrainLabelsNdarray[i])

# Convert data from 0..255 to 0..1
# If you forget this (as I did), the model will not converge (well, it will converge, but to 
# something like 0.0921 accuracy - which is really, really BAD ;)
trainfv = np.multiply(trainfv, 1.0 / 255.0)    

print(trainfv[0],ohtrainLabelsNdarray[0])


# In[ ]:


# Now build the simple model: all 784 inputs are connected to 10 "decision nodes", which use 
# a thresholded and normalized softmax to compute a "probability" for each of the ten digits
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Define the way we evaluate the network
y = tf.nn.softmax(tf.matmul(x, W) + b)

# To implement cross-entropy we need to first add a new placeholder to input the correct answers:
y_ = tf.placeholder(tf.float32, [None, 10])

# numerically unstable implementation of cross entroy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Explaination of the cross entropy formula above:
# First, tf.log computes the logarithm of each element of y. 
# Next, we multiply each element of y_ with the corresponding element of tf.log(y). 
# Then tf.reduce_sum adds the elements in the second dimension of y, due to the 
# reduction_indices=[1] parameter. Finally, tf.reduce_mean computes the mean over 
# all the examples in the batch.

# No, define a training step 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# ![The weights learned by the model][1]
# 
# 
#   [1]: https://www.tensorflow.org/images/softmax-weights.png
# 
# This is a visualisation, embedded from the Tensorflow tutorial, of the weights learned by the model.

# In[ ]:


import math

# A mildly random version....simply pick one of the (consecutive) slices of size "size"
from random import randint

# This is used below, for simplicity and for faster convergence ;)
def random_batch(data,labels,size):
    value = math.floor(len(data) / size)    
    intervall = randint(0,value-1)
    return data[intervall*size:intervall*(size+1)],labels[intervall*size:intervall*(size+1)]

# print(random_batch(trainfv,ohtrainLabelsNdarray,3))

from random import sample

# of course, this could be done in a myriad of (better) ways...
# This "random" picking (not subarray-wise, as above) leads to far slower
# learning - which justifies the assumption that "similar" digits (from
# the same writer perhaps) which we want too learn to differentiate, 
# often are close together in the original data, making it useful to learn
# them together in batches
def really_random_batch(data,labels,size):
    res_data = np.empty((size,784))
    res_label = np.empty((size,10))
    
    l = len(data)
    s = sample(range(l), size)
    
    for i in range(size):
        res_data[i] = data[i]
        res_label[i] = labels[i]
        
    return np.array(res_data),np.array(res_label)

# A sample sample... ;)
# print(really_random_batch(trainfv,ohtrainLabelsNdarray,3))


# In[ ]:


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Training (go higher for better results, e.g. 1500)
for i in range(500):
  batch_xs, batch_ys = random_batch(trainfv,ohtrainLabelsNdarray,150)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  if not(i % 50): 
    print(i)
    
# Test our prediction 
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("\nAccuracy of the current model (against training samples): ",sess.run(accuracy, feed_dict={x: trainfv[0:10000],
                                    y_: ohtrainLabelsNdarray[0:10000]}))

print("\nAccuracy of the current model (against first 10000 test samples): ",sess.run(accuracy, feed_dict={x: testfv[0:10000],
                                    y_: ohtestLabelsNdarray[0:10000]}))


# With 1500 batches of 150 elements, you should reach an accuracy of about 0.92 (92% correct
# decisions...still a lot of errors, but, heh, it's just a start!)

# Re-running this code cell will produce a different accuracy, if you want to keep the result
# "fixed", you could set a random seed prior to calling random_batch


# Missing: Build a submission file from the model
# 
