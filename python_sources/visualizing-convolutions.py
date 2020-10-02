#!/usr/bin/env python
# coding: utf-8

# Ever wondered what a convolution actually looks like? Here is a simple way to visualize the outputs of a convolutional layer.

# **Setup and Import**

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

test = pd.read_csv('../input/test.csv')
test = StandardScaler().fit_transform(np.float32(test.values))
test = test.reshape(-1, 28, 28, 1)


# We can simply use the test set, since there is no need for labels. We only need to reshape it into 28x28 pictures. The StandardScaler() makes the pictures look prettier.

# **A first look**
# 
# Let's look at a single convolutional layer with a max-pooling operation. We can start by choosing any picture k from the test set. The parameters below allow us to choose what actually happens in our convolutional layer, e.g. how large the kernel is and how wide the strides are.  We can change them later in order to see how they affect the result.

# In[ ]:


k = 7 #choose which image to take from the test set
kernel = 4 #Kernel size
strides = 1 #Stride size
kernel_mp = 2 #Kernel size of the max pooling operation
strides_mp = 1 #Stride size in max pooling operation


# Note that changing the strides is what will reduce the dimensionality of our image. Changing the kernel size results in a more blurry image.
# 
# Before we start, let's check what picture we have chosen. We'll use imshow() for that.

# In[ ]:


plt.imshow(test[k,:,:,0])
plt.show()


# Okay, now let's implement a simple convolution depending on the parameters we have chosen earlier. Let x be the input, w the weight matrices, cl the convolutional layer and cl_mp the max pooling operation applied to the convolutional layer.

# In[ ]:


graph = tf.Graph()
with graph.as_default():
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    w = tf.Variable(tf.truncated_normal([kernel, kernel, 1, 16], stddev=0.1))
    cl = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    cl_mp = tf.nn.max_pool(cl, ksize=[1, kernel_mp, kernel_mp, 1], strides=[1, strides_mp, strides_mp, 1], padding='SAME')


# We can now visualize this data. We chose a depth (number of outputs) of 16, so we end up with 16 different images. Let's display them in a grid. The size of the images depends on the stride parameters.

# In[ ]:


with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    CL = sess.run(cl, feed_dict={x:[test[k]]})
    print("Image Shape =", CL.shape[1:3])
    f, ax = plt.subplots(4,4, figsize=(20,20))
    for i in range(16):
        ax[i//4,i%4].imshow(CL[0,:,:,i])
        ax[i//4,i%4].axis('off')
    plt.show()


# And after the max pooling?

# In[ ]:


with tf.Session(graph=graph) as sess: #start the session
    tf.global_variables_initializer().run() #initialize variables
    CL = sess.run(cl_mp, feed_dict={x:[test[k]]})
    print("Image Shape =", CL.shape[1:3])
    f, ax = plt.subplots(4,4, figsize=(20,20))
    for i in range(16):
        ax[i//4,i%4].imshow(CL[0,:,:,i])
        ax[i//4,i%4].axis('off')
    plt.show()


# **Multilayer Network**
# 
# So now let's take a look how our features change when we look at multiple layers, each with its own max pooling operation. We use 3 layers to give us an idea of the downsampling happening with each step. By choice of the stride of the max-pooling, the image size is reduced by half at each step, so we end up with 4x4 pictures at the end of layer 3. Also by choice of the depth (number of outputs) we'll have exactly 64 of those.

# In[ ]:


graph = tf.Graph()
with graph.as_default():

    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    w1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1))
    w2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1))
    w3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
    
    # Convolutional layer 1
    cl1 = tf.nn.conv2d(x, w1, strides=[1, 1, 1, 1], padding='SAME')
    cl1_mp = tf.nn.max_pool(cl1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')        
    # Convolutional layer 2
    cl2 = tf.nn.conv2d(cl1_mp, w2, strides=[1, 1, 1, 1], padding='SAME')
    cl2_mp = tf.nn.max_pool(cl2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Convolutional layer 3
    cl3 = tf.nn.conv2d(cl2_mp, w3, strides=[1, 1, 1, 1], padding='SAME')
    cl3_mp = tf.nn.max_pool(cl3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# Now take a look at the downsampling operations of a multilayer convolutional network.

# In[ ]:


#After Layer 1
with tf.Session(graph=graph) as sess: #start the session
    tf.global_variables_initializer().run() #initialize variables
    CL = sess.run(cl1_mp, feed_dict={x:[test[k]]})
    print("Image Shape =", CL.shape[1:3])
    print("Number of Images =", CL.shape[-1])
    f, ax = plt.subplots(4,4, figsize=(20,20))
    for i in range(16):
        ax[i//4,i%4].imshow(CL[0,:,:,i])
        ax[i//4,i%4].axis('off')
    plt.show()


# In[ ]:


#After Layer 2
with tf.Session(graph=graph) as sess: #start the session
    tf.global_variables_initializer().run() #initialize variables
    CL = sess.run(cl2_mp, feed_dict={x:[test[k]]})
    print("Image Shape =", CL.shape[1:3])
    print("Number of Images =", CL.shape[-1])
    f, ax = plt.subplots(4,8, figsize=(20,10))
    for i in range(32):
        ax[i//8,i%8].imshow(CL[0,:,:,i])
        ax[i//8,i%8].axis('off')
    plt.show()


# In[ ]:


#After Layer 3
with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    CL = sess.run(cl3_mp, feed_dict={x:[test[k]]})
    print("Image Shape =", CL.shape[1:3])
    print("Number of Images =", CL.shape[-1])
    f, ax = plt.subplots(8,8, figsize=(20,20))
    for i in range(64):
        ax[i//8,i%8].imshow(CL[0,:,:,i])
        ax[i//8,i%8].axis('off')
    plt.show()


# Looks cool, right? The images above are what will be fed into the fully connected layer in order to be classified.

# **Great Resources**
# 
# https://www.youtube.com/watch?v=FG7M9tWH2nQ
# 
# https://www.youtube.com/watch?v=BFdMrDOx_CM

# In[ ]:




