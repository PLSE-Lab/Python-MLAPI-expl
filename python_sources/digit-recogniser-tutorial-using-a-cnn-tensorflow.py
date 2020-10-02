#!/usr/bin/env python
# coding: utf-8

# *March 2018*
# 
# **Introduction**
# 
# Previously I built a simple fully-connected neural network without any convolution [here.](https://www.kaggle.com/mauddib/digit-recogniser-with-r-and-h2o)
# I played around with the H2O framework a bit, but it was a bit slow. So much so that I could not run a tuning grid(Kaggle limits you to 21000 seconds on R notebook) to test a range of values for my hyper parameters, which is what machine learning is all about!
# 
# I ended up testing a few activation functions and adding a layer of extra nodes in an attempt to do some 'manual' empirical testing. Bit of a schlepp. At any rate I still managed to achieve a 0.97785 under those circumstances, not too shabby at all!
# 
# But now we get to test drive the more contemporary approach to machine learning, by building a Convolutional Neural Network or CNN.
# 
# Tensorflow allows you to build a flow for your model, which is called an architecture.
# 
# **Here is a typical CNN architecture:**
# 
# ![CNN](https://cdn-images-1.medium.com/max/1600/1*uUYc126RU4mnTWwckEbctw@2x.png)
# 
# 
# The convolution layers are the main powerhouse of a CNN model. Automatically detecting meaningful features given only an image and a label is not an easy task. The convolution layers learn such complex features by building on top of each other. The first layers detect edges, the next layers combine them to detect shapes, to following layers merge this information to infer that this is a car for example, or the number 9.
# 
# Without further ado, let us begin by loading the necessary libraries and also loading the training and test data:
# (The labels were given as integers between 0 and 9. We will convert these to one-hot encoding labels, ie, for the label 5, you will have an array that looks like this : [0,0,0,0,0,1,0,0,0,0])

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plot library

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#! /usr/bin/env python
# -*- coding: utf-8 -*- 
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
from keras.utils.np_utils import to_categorical   


train_data0 = pd.read_csv("../input/train.csv")
train_data = np.array(train_data0.iloc[:, 1:785])
train_label = np.array(train_data0.iloc[:,0])


# One-hot encoding
train_label = to_categorical(train_label, num_classes=10)

test_data0 = pd.read_csv("../input/test.csv")
test_data = np.array(test_data0)



# OK so lets see the first 10 labels and their one hot encoding equivalent:

# In[ ]:


print(np.array(train_data0.iloc[:,0])[:10])
print('is equivalent to:')
print(train_label[:10])


# Now let us print some examples of digits from the training data to get a better feel for the data we will be training on:

# In[ ]:



pixels = train_data[9].reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

pixels = train_data[7].reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()

pixels = train_data[6].reshape((28, 28))
plt.imshow(pixels, cmap='gray')
plt.show()


# At this point we will need to think about some seeds for our hyper parameters as well as start defining the convolution and pooling functions.
# 
# One of our hyperparameters is the number of epochs. Remember an epoch is, in Machine Learning, the processing by the learning algorithm of the entire training set once. 
# 
# Because we only have an hour on the kernel at Kaggle, so we are limited as to how many epochs we can  run. Normally you would run a few hundred, so if you are keen, fork this notebook and run it with epochs set to a few hundred overnight.
# 
# Let's see how it performs on 10 Epochs for now(play around with it and see what happens:

# In[ ]:


# Define hyperparameters
learning_rate = 0.0001
epoch = 7
batch_size = 20

# Define network parameters
n_input = 784
n_classes = 10

# Placeholder
X = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Convolution
def conv2d(name, x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    #return tf.nn.relu(x, name=name)
    return tf.nn.elu(x, name=name)

# Pooling
def maxpool2d(name, x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

weights = {
    'W1': tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1)),
    'W2': tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1)),
    'W4': tf.Variable(tf.truncated_normal([64 * 7 * 7, 784], stddev=0.1)),
    'Wo': tf.Variable(tf.truncated_normal([784, n_classes], stddev=0.1))
}

biases = {
    'b1': tf.Variable(tf.random_normal([32], stddev=0.1)),
    'b2': tf.Variable(tf.random_normal([64], stddev=0.1)),
    'b4': tf.Variable(tf.random_normal([784], stddev=0.1)),
    'bo': tf.Variable(tf.random_normal([n_classes], stddev=0.1))
}


# Next we need to define our model architecture. I am opting for one that looks as follows:
# 
# Input -> Conv+Maxpool -> Conv+Maxpool -> FC -> FC -> Output
# 
# So now we are ready to go through the architecture as described above, and then train the model and save the output.
# 
# We will apply standard initialization on the data.  The original data has pixel values between 0 and 255, but data between 0 and 1 should make the net converge faster.
# 
# Regarding cost optimization, we are going to use tf.nn.softmax_cross_entropy_with_logits, which computes the cross entropy of the result after applying the softmax function.
# 
# As for the optimizer, I have opted for AdamOptimizer which is faster than Stochastic Gradient Descent (SGD).
# 

# In[ ]:


def model(X, weights, biases):
    # Conv1
    x = tf.reshape(X, [-1, 28, 28, 1])
    
    # confine range to between 0 and 1
    x = x/255.
    
    #conv1 = tf.nn.relu(conv2d('conv1', x, weights['W1'], biases['b1']))
    conv1 = tf.nn.elu(conv2d('conv1', x, weights['W1'], biases['b1']))
    # Pool1
    pool1 = maxpool2d('pool1', conv1, k=2)

    # Conv2
    #conv2 = tf.nn.relu(conv2d('conv2', pool1, weights['W2'], biases['b2']))
    conv2 = tf.nn.relu(conv2d('conv2', pool1, weights['W2'], biases['b2']))

    # Pool2
    pool2 = maxpool2d('pool2', conv2, k=2)
    
    # Full connect layer
    fc = tf.reshape(pool2, [-1, weights['W4'].get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, weights['W4']), biases['b4'])
    #fc = tf.nn.relu(fc)
    fc = tf.nn.elu(fc)

    # output
    a = tf.add(tf.matmul(fc, weights['Wo']), biases['bo'])

    return a


# prediction
pred = model(X, weights, biases)

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
label = tf.argmax(pred, 1)
# evaluation
correct_pred = tf.equal(label, tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True  
with tf.Session(config=config) as sess:
    sess.run(init)  
    for e in range(epoch):
        step = 1
        while step*batch_size <= train_data.shape[0]:
            xs, ys = train_data[(step-1)*batch_size:step*batch_size, :], train_label[(step-1)*batch_size:step*batch_size, :]
            sess.run(optimizer, feed_dict={X:xs, y:ys})

            if step % 100 == 0:
                loss, acc = sess.run([cost, accuracy], feed_dict={X:xs, y:ys})

                print("Iter {0}, Minibatch Loss = {1}, Training accuracy = {2}".format(str(step),
                                                                                    loss, acc))
            step += 1
    print("Optimization Completed")
    test_labels = []
    for i in range(1000):
        xs, ys = test_data[i*28:(i+1)*28, :], test_data[i*28:(i+1)*28, 0:10]
        pred_ = sess.run(label, feed_dict={X:xs, y:ys})
        test_labels.extend(list(pred_))

f1 = open('label', 'wb')
pickle.dump(test_labels, f1)
f1.close()


# now we save the predictions to a csv ready for submission
df = pd.DataFrame({'Label': test_labels})
# Add 'ImageId' column
df1 = pd.concat([pd.Series(range(1,28001), name='ImageId'), 
                              df[['Label']]], axis=1)


df1.to_csv('ConvPool_X2.csv', index=False)
    


# **Outcome and Conclusion**
# 
# Your Best Entry 
# Your submission scored 0.98385, which is an improvement of your previous score of 0.97785. Great job! 
# 
# OK so with a 2 layer Conv+Maxpool CNN and only a few epochs we have managed to surpass our H2O model results!
# 
# I hope you had fun building a CNN architecture with Tensorflow. Try adding another convolution layer or two to your architecture, and let me know what you get!
# 
# Good luck!
# 
