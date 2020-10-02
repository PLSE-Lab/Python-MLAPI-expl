#!/usr/bin/env python
# coding: utf-8

# This is my first notebook at Kaggle, it is really to get myself started on something.

# In[ ]:



import numpy as np
import pandas as pd 
from scipy import stats
import matplotlib.pyplot as plt

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# To load Kaggle enivornment

# In[ ]:


import kagglegym
# Create environment
env = kagglegym.make()
# Get first observation
observation = env.reset()


# In[ ]:


with pd.HDFStore('../input/train.h5') as train:
    df = train.get('train')
print(df.shape)


# Read in the value from train.h5 ile.

# Lets take a look at the data to have a feel of what we are actually having on hand.

# In[ ]:


print(df.shape)
df.head()

#newdata=df.ix[:,2:110]
#newdata=newdata.dropna()


# Lets look at the column list

# In[ ]:


for col in df.columns:
    print(col)


# get rid of rows with NAN in any column

# In[ ]:


newdata=df.dropna()
print(newdata.shape)


# observe the value

# In[ ]:


newdata["y"].hist(bins=100)


# observe another column

# In[ ]:



newdata["fundamental_11"].hist(bins=100)


# all seem to be zero mean, that would mean that the data had went through some scaling pre-process

# In[ ]:


newdata["fundamental_12"].hist(bins=100)


# In[ ]:


print(newdata.shape)


# In[ ]:



moments = df[['id', 'y']].groupby('id').agg([np.mean, np.std, stats.kurtosis, stats.skew]).reset_index()
moments.head()


# In[ ]:



moments = df[['timestamp','id']].groupby('id').agg([np.mean, np.std, stats.kurtosis, stats.skew]).reset_index()
moments.head()


# In[ ]:


nansum = df.isnull().sum()/len(df)
print(nansum)


# In[ ]:


print(nansum.shape)
plt.bar(np.arange(0,len(nansum),1),nansum)


# **
# 
# would continue to add more code or explaination.
# ------------------------------------------------
# 
# **

# In[ ]:


import tensorflow as tf
import math
FEATURE_SIZE=108   #total number of features also is the number of input to the network
batch_size=32 
LEARNING_RATE=0.01
hidden1_units=FEATURE_SIZE
hidden2_units=32
d = tf.placeholder(tf.float32, shape=(batch_size,FEATURE_SIZE))                                              
y = tf.placeholder(tf.float32, shape=(batch_size,1))
#hidden layer 1 that interface with the input
with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([FEATURE_SIZE, hidden1_units],
                        stddev=1.0 / math.sqrt(float(FEATURE_SIZE))),name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
    hidden1 = tf.nn.relu(tf.matmul(d, weights) + biases)
#hidden layer 2 
with tf.name_scope('hidden2'):
    weights = tf.Variable(tf.truncated_normal([FEATURE_SIZE, hidden2_units],
                        stddev=1.0 / math.sqrt(float(FEATURE_SIZE))),name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
#softmax output
with tf.name_scope('softmax_linear'):
    weights = tf.Variable(tf.truncated_normal([hidden2_units, 1],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),name='weights')
    biases = tf.Variable(tf.zeros([1]),name='biases')
    logits = tf.matmul(hidden2, weights) + biases    

#y = tf.to_int64(y)
#cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, y, name='xentropy')
#loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')



# In[ ]:




