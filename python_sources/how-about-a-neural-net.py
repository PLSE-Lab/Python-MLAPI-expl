#!/usr/bin/env python
# coding: utf-8

# This is an experiment I'm performing, a **Neural Net** on the **Titanic** data. It's silly, I know, but most people don't do this. For this experiment, I intend using Google Brain's **Tensorflow**.

# In[ ]:


import pandas as pd
import tensorflow as tf
import numpy as np
import statistics as stat
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')


# Let's import our dataset

# In[ ]:


train = pd.read_csv('../input/titanic//train.csv')
test = pd.read_csv('../input/titanic/test.csv')


# An overview of the data

# In[ ]:


print (train.describe())
print()
print (test.describe())


# In[ ]:


train.head()


# I will be using Astandri K's pre-processed data which I have imported. Nice feature engineering. Check out his kernel at [https://www.kaggle.com/astandrik/journey-from-statistics-eda-to-prediction]. 

# In[ ]:


dat = pd.read_csv('../input/titanic-preprocessed-data-by-astandri-k/titanic_dataset_preprocessed.csv')
dat.describe()


# Now that we have that covered, let's build our model.

# In[ ]:


# Layer 1
with tf.variable_scope('Layer_1', reuse=tf.AUTO_REUSE):
    data = tf.placeholder(tf.float32, [10, 18])
    weight = tf.get_variable('w', [18, 13], tf.float32, initializer=tf.random_normal_initializer())
    bias = tf.get_variable('b', [13], tf.float32, initializer=tf.constant_initializer(0.0))
    h_1 = tf.nn.xw_plus_b(data, weight, bias)


# In[ ]:


# Layer 2
with tf.variable_scope('Layer_2', reuse=tf.AUTO_REUSE):
    weight = tf.get_variable('w', [13, 5], tf.float32, initializer=tf.random_normal_initializer())
    bias = tf.get_variable('b', [5], tf.float32, initializer=tf.constant_initializer(0.0))
    h2 = tf.nn.xw_plus_b(h_1, weight, bias)


# In[ ]:


# Layer 3
with tf.variable_scope('Layer_3', reuse=tf.AUTO_REUSE):
    weight = tf.get_variable('w', [5, 2], tf.float32, initializer=tf.random_normal_initializer())
    bias = tf.get_variable('b', [2], tf.float32, initializer=tf.constant_initializer(0.0))
    h3 = tf.nn.xw_plus_b(h2, weight, bias)
    print(h3)
    


# In[ ]:


# Cross-entropy error and loss
labels = tf.placeholder(tf.float32, [10, None])
x_ent = tf.nn.softmax_cross_entropy_with_logits(logits=h3, labels=labels)
loss = tf.reduce_mean(x_ent)

vars = tf.trainable_variables()
for var in vars:
    print(var)


# In[ ]:


init = tf.global_variables_initializer()
sess = tf.Session()
print (init)
sess.run(init)


# Let's separate X from y and get complete training data 

# In[ ]:


X_train = dat.drop(['Survived'],axis=1)[:891]
y_train = dat['Survived'][:891].astype('int')
y1 = np.expand_dims(y_train, axis=1)
y2 = []
z = []
final_y = []
for i in range(0, 891):
    if y1[i][0] == 0:
        z.append(1)
    else:
        z.append(0)
for i in range(0, 891):
    temp = [y1[i], z[i]]
    final_y.append(temp)
final_y = np.array(final_y)
print(final_y.shape)


# Now, here's the tough part: ***Training the model***

# In[ ]:


print(X_train.shape)
final_y.shape


# In[ ]:


optim = tf.train.GradientDescentOptimizer(0.001).minimize(loss, var_list=vars)
with tf.Session() as sess:
    for i in range(0, 10):
        for j in range(0, 89):
            err = sess.run([optim],
                    feed_dict={
                        data: X_train.loc[j*10:(j+1)*9],
                        labels: final_y[j*10:(j+1)*10]
                    })
            print ('Batch {} error: {}'.format(batch_id+1, err))


# In[ ]:




