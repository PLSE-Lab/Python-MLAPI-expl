#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
#import tensorflow as tf
from keras.datasets import mnist
import os
import pandas as pd
print(os.listdir("../input"))
import gc


# In[ ]:


data, data_ = mnist.load_data()


# In[ ]:


test_df = pd.read_csv('../input/test.csv')


# In[ ]:


trn_X = np.append(data[0], data_[0], axis=0).reshape(70000,784)
trn_Y = np.append(data[1], data_[1])


# In[ ]:


del  data, data_
gc.collect()


# In[ ]:


#sess = tf.Session()


# In[ ]:


#t = test_df.iloc[0].values


# In[ ]:


'''with tf.device('/gpu:0'):
    p = tf.placeholder(shape=[None, 784], dtype=tf.int32)
    x = tf.placeholder(shape=[None, 784], dtype=tf.int32)'''


# In[ ]:


#sess.run(tf.argmin(tf.reduce_sum(tf.abs(p-x), axis=1)), feed_dict={p:trn_X, x:t})


# In[ ]:


#%%timeit 
#with tf.device('/gpu:0'):
    #y = sess.run(tf.reduce_sum(tf.abs(p-x), axis=1), feed_dict={p:trn_X, x:trn_X[0:100]})
#trn_Y[sess.run(tf.argmin(y))]


# In[ ]:


#%%time 
t = []
for i in range(len(test_df)):
    t.append(trn_Y[np.argmin(np.sum(np.abs(trn_X-test_df.iloc[i].values), axis=1))])
    #if i % 100 == 0: print(i/28000)


# In[ ]:


#from multiprocessing import cpu_count
#print(cpu_count())
df = pd.DataFrame()
df['ImageId'] = test_df.index.values+1
df['Label'] = t


# In[ ]:


df.to_csv('sub.csv', index=False)
df.head()

