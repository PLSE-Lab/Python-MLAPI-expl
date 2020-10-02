#!/usr/bin/env python
# coding: utf-8

# This is my first Notebook on Kaggle.
# What I actually did is to adapt the MNIST tutorial on tensorflow website to use the input of Kaggle. 

# In[ ]:


import numpy as np
import pandas as pd
import random 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import tensorflow as tf


# In[ ]:


data = pd.read_csv('../input/train.csv') 
data.reindex(np.random.permutation(data.index))
trainImages = data.drop('label', 1)
trainLabels = data['label']

testImages = pd.read_csv('../input/test.csv')

print('Data imported seccessfully ')


# In[ ]:


expInd = 724
plt.imshow(trainImages.iloc[expInd].values.reshape(28,28), cmap='Greys_r')
print(trainLabels.iloc[expInd].values.argmax())


# In[ ]:


# Create a TF interactive session 
sess = tf.InteractiveSession()


# In[ ]:


# Weight Initialization 
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# Convolution and Pooling 
def conv2d(x, W): 
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[ ]:


# Create the model
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

y_ = tf.placeholder(tf.float32, [None, 10])

# First Convolutional Layer 
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second Convolutional Layer 
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer 
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout 
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer 
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# In[ ]:


digits = np.matrix([[0,1,2,3,4,5,6,7,8,9],]*42000)

tl = np.matrix(trainLabels.values).transpose()
trainLabels = (tl == digits).astype(float)
trainLabels = pd.DataFrame(trainLabels)


# In[ ]:


# Train and Evaluate the Model 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv,1e-10,1.0)), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())
for i in range(400):
    iStart = i*100
    iEnd = (i+1)*100
    batch_xs, batch_ys = trainImages.iloc[iStart:iEnd].values, trainLabels.iloc[iStart:iEnd].values
    train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0}, session=sess)
    print("step %d, training accuracy %g"%(iStart, train_accuracy))
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5}, session=sess)


# In[ ]:


# Test trained model
correct_prediction = y_conv

predict = correct_prediction.eval(feed_dict={x: testImages.iloc[0:100].values, keep_prob: 1.0}, session=sess)

for ii in range(279): 
    i = ii + 1
    iStart = i*100
    iEnd = (i+1)*100
    predict = np.append(predict, correct_prediction.eval(feed_dict={x: testImages.iloc[iStart:iEnd].values, keep_prob: 1.0}, session=sess), axis=0)
    print("Prediction shape: ", predict.shape)


# In[ ]:


predict.argmax(1)


# In[ ]:


pred = pd.DataFrame(data = predict, columns = {'Label'}, dtype=int)
pred.index += 1 
pred.index.name = 'ImageId' 
pred


# In[ ]:


print(testImages.iloc[0].values.shape)pred.to_csv(path_or_buf='submission.csv')


# In[ ]:


plt.imshow(testImages.iloc[3].values.reshape(28,28), cmap='Greys_r')


# In[ ]:




