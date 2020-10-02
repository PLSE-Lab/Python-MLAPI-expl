#!/usr/bin/env python
# coding: utf-8

# # TensorFlow Approach
# ### Easy as 1-2-3... fourfivesixseveneightnineten... eleven.
# 
# * 1) Import libraries
# * 2) Test view the images
# * 3) Define a few helper functions
# * 4) Initialize placeholders for images and labels
# * 5) Set up two convolutional layers each with max pooling
# * 6) Flatten and add a fully connected layer with a dropout rate of 0.5
# * 7) Set up the optimizer
# * 8) Start a session and train the images in random batches of 300
# * 9) After 10 batches, calculate the accuracy and append to a list
# * 10) Repeat for a total of 500 steps
# * 11) Plot accuracy using matplotlib

# In[1]:


#Import libraries
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Import the data
f = open('../input/shipsnet.json')
dataset = json.load(f)
f.close()
data = np.array(dataset['data']).astype('uint8')
labels = np.array(dataset['labels']).astype('uint8')


# In[3]:


#View an image
index = 170 # Image to be reformed
pixel_vals = data[index]
arr = np.array(pixel_vals).astype('uint8')
im = arr.reshape((3, 6400)).T.reshape((80,80,3))
plt.imshow(im)


# In[4]:


#Initialize Weights
def initWeights(shape):
    weights = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(weights)

#Initialize Bias
def initBias(shape):
    bias = tf.constant(0.1,shape=shape)
    return tf.Variable(bias)

#Initialize Convolution Filter
def initPatch(input_x,weight_filter):
    return tf.nn.conv2d(input_x,weight_filter,strides=[1,1,1,1],padding='SAME')

#Initialize maxPooler
def maxPool(input_x):
    return tf.nn.max_pool(input_x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#Helper for creating convolutional layers
def convolutionalLayer(input_x,shape):
    bias = initBias([shape[3]])
    weight_filter = initWeights(shape)
    return tf.nn.relu(initPatch(input_x,weight_filter)+bias)

#Helper for creating fully connected layers
def fullyconnectedLayer(input_x,shape):
    input_size = int(input_x.get_shape()[1])
    bias = initBias([shape])
    weight_filter = initWeights([input_size,shape])
    return tf.matmul(input_x,weight_filter) + bias


# In[5]:


#Set up placeholders
x = tf.placeholder(tf.float32,shape=[None,19200])
y_label = tf.placeholder(tf.float32,shape=[None,2])
x_image = tf.reshape(x,[-1,80,80,3])


# In[6]:


#Define the network layers
conv_layer_1 = convolutionalLayer(x_image,[5,5,3,40])
pool_layer_1 = maxPool(conv_layer_1)
conv_layer_2 = convolutionalLayer(pool_layer_1,[5,5,40,80])
pool_layer_2 = maxPool(conv_layer_2)
flattened_layer = tf.reshape(pool_layer_2,[-1,20*20*80])
fully_connected_layer = tf.nn.relu(fullyconnectedLayer(flattened_layer,5000))


# In[7]:


#Set up a dropout rate
keep_prob = tf.placeholder(tf.float32)
dropout = tf.nn.dropout(fully_connected_layer,keep_prob=keep_prob)
y_pred = fullyconnectedLayer(dropout,2)


# In[8]:


#Adam Optimizer with learning rate of 0.001
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label,logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
fetch = optimizer.minimize(loss)


# In[9]:


#The meat and potatoes
init = tf.global_variables_initializer()
steps = 500
batch_size = 300
with tf.Session() as sess:
    sess.run(init)
    accuracy_list = []
    for i in range(steps):
        rand_list = np.random.randint(0,2799,batch_size)
        batch_x = data[rand_list]
        get_y = tf.one_hot(labels[rand_list],2)
        batch_y = np.array(sess.run(get_y))
        sess.run(fetch,feed_dict={x:batch_x,y_label:batch_y,keep_prob:0.5})
        if i%10 == 0:
            print("Results from step: {}".format(i))
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_label,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            acc2 = sess.run(acc,feed_dict={x:batch_x,y_label:batch_y,keep_prob:1.0})
            print(acc2)
            accuracy_list.append(acc2)


# In[10]:


#Plot out accuracy results for every 10th step
plt.plot(np.linspace(0,1,len(accuracy_list)),accuracy_list)

