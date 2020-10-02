#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import tensorflow as tf


# In[ ]:


from tensorflow.examples.tutorials.mnist import input_data


# In[ ]:


mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


# In[ ]:


#INIT WEIGHT


# In[ ]:


def init_weight(shape):
    init_random_dist = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_dist)


# In[ ]:


#INIT BIAS
def init_bias(shape):
    init_bias_vals = tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias_vals)


# In[ ]:


#CONV2D
def conv2d(x,W):
    
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')


# In[ ]:


def max_pool_2by2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


# In[ ]:


#CONVOLUTIONAL LAYER
def convolution_layer(input_x,shape):
    W = init_weight(shape)
    b = init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W)+b)


# In[ ]:


def normal_full_layer(input_layer,size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weight([input_size,size])
    b = init_bias([size])
    return tf.matmul(input_layer,W)+b


# In[ ]:


#placeholder


# In[ ]:


x = tf.placeholder(tf.float32,shape=[None,784])


# In[ ]:


y_true = tf.placeholder(tf.float32,shape=[None,10])


# In[ ]:


#Layers
x_image = tf.reshape(x,[-1,28,28,1])


# In[ ]:


convo_1 = convolution_layer(x_image,shape=[5,5,1,32]) 
convo_1_pooling = max_pool_2by2(convo_1)


# In[ ]:


convo_2 = convolution_layer(convo_1_pooling,shape=[5,5,32,64])
convo_2_pooling = max_pool_2by2(convo_2)


# In[ ]:


convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))


# In[ ]:


#Dropout
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)


# In[ ]:


y_pred = normal_full_layer(full_one_dropout,10)


# In[ ]:


#LOSS FUNCTION
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))


# In[ ]:


#OPTIMIZER


# In[ ]:


optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)


# In[ ]:


init = tf.global_variables_initializer()


# In[ ]:


steps = 5000

with tf.Session() as sess:
    sess.run(init)
    
    for i in range(steps):
        batch_x , batch_y = mnist.train.next_batch(50)
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
        
        if i % 100 == 0:
            print("ON STEP: {}".format(i))
            print("ACCURACY:")
            matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:0.1}))
            print('\n')
                               
                               
                               
                               


# In[ ]:




