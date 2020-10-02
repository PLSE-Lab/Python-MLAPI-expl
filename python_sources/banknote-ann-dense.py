#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import tensorflow as tf
from sklearn.model_selection import train_test_split


# In[ ]:


df=pd.read_csv("../input/BankNote.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


X=df[df.columns[0:4]].values


# In[ ]:


X


# In[ ]:


# Z=df.iloc[:,:4].values


# In[ ]:


# Z


# In[ ]:


y=df[df.columns[4]]
#y


# In[ ]:


y.shape


# In[ ]:


Y=pd.get_dummies(y)
#Y


# In[ ]:


Y.shape


# In[ ]:


train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.2,random_state=420)


# In[ ]:


print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
n_dim=X.shape[1]
n_dim
n_class=2


# In[ ]:


training_epochs=500
learning_rate=0.04
tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None,n_dim])
y_ = tf.placeholder(tf.float32, [None, n_class])


# In[ ]:


def multi_layer_perceptron(x):
    layer_1=tf.layers.dense(x,10,tf.nn.relu)
    layer_2=tf.layers.dense(layer_1,10,tf.nn.relu)
    layer_3=tf.layers.dense(layer_2,10,tf.nn.relu)
    layer_4=tf.layers.dense(layer_3,10,tf.nn.relu)
    out_layer=tf.layers.dense(layer_4,2,tf.nn.sigmoid)
    return out_layer


# In[ ]:


y=multi_layer_perceptron(x)
init=tf.global_variables_initializer()


# In[ ]:


cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_steps = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)


# In[ ]:


sess = tf.Session()
sess.run(init)


# In[ ]:


for epoch in range(training_epochs):
    sess.run(training_steps, feed_dict={x:train_x, y_:train_y})
    cost = sess.run(cost_function, feed_dict={x:train_x, y_:train_y})
    print('epoch : ', epoch, '-', 'cost : ',cost)


# In[ ]:


y_test=sess.run(y, feed_dict={x:test_x})


# In[ ]:


p=np.argmax(y_test,1)
p


# In[ ]:


q=np.argmax(np.array(test_y),1)
q


# In[ ]:


sum(p==q)/len(test_x) *100


# In[ ]:




