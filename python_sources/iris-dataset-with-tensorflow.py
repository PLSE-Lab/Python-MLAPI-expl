#!/usr/bin/env python
# coding: utf-8

# Import iris data set

# In[ ]:


from sklearn.datasets import load_iris


# In[ ]:


ds = load_iris()


# Let's see the features

# In[ ]:


ds.feature_names


# In[ ]:


ds.target_names


# In[ ]:


ds.data.shape


# Here are the labels

# In[ ]:


ds.target


# In[ ]:


import tensorflow as tf


# In[ ]:


x = tf.placeholder(shape=(None,4), dtype=tf.float32)
y = tf.placeholder(shape=(None,3), dtype=tf.float32)
w_xh = tf.get_variable(name='input-hidden', shape=(4,5), dtype=tf.float32, initializer=tf.random_normal_initializer)
w_hy = tf.get_variable(name='hidden-output', shape=(5,3), dtype=tf.float32, initializer=tf.random_normal_initializer)


# In[ ]:


b_xh = tf.get_variable(name='input-hidden-bias', shape=(1,5), dtype=tf.float32, initializer=tf.random_normal_initializer)


# In[ ]:


b_hy = tf.get_variable(name='hidden-output-bias', shape=(1,3), dtype=tf.float32, initializer=tf.random_normal_initializer)


# In[ ]:


def build_model(x):
    with tf.variable_scope('hidden'):
        h = tf.nn.sigmoid(tf.matmul(x, w_xh)+b_xh)
        print(h.shape)
    with tf.variable_scope('output'):    
        y_pred = tf.nn.softmax(tf.matmul(h,w_hy)+b_hy)
        print(y_pred.shape)
    return y_pred;


# In[ ]:


y_pred = build_model(x)


# In[ ]:


loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)


# In[ ]:


optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder


# In[ ]:


encoder = OneHotEncoder


# In[ ]:


import numpy as np


# In[ ]:


labels = encoder.fit_transform(np.array(ds.target).reshape(150,3))


# In[ ]:


import pandas as pd
labels = pd.get_dummies(np.array(ds.target))


# In[ ]:


labels


# In[ ]:


labels.shape


# In[ ]:


type(labels)


# In[ ]:


labels = labels.values


# In[ ]:


labels


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


trainx, testx, trainy, testy = train_test_split(ds.data, labels)


# In[ ]:


trainy.shape


# In[ ]:


accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,axis=1), tf.argmax(y_pred,axis=1)),tf.float32))


# In[ ]:


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for e in range(300):
        _, loss_val, train_acc = sess.run([optimizer, loss, accuracy], feed_dict={x:trainx, y:trainy})
        test_acc = accuracy.eval(feed_dict={x:testx, y:testy})
        #print(loss_val,train_acc)
        print(loss_val,test_acc)


# In[ ]:


writer = tf.summary.FileWriter(logdir='nn', graph=tf.get_default_graph())

