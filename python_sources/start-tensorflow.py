#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#tensorflow
import tensorflow as tf


# In[ ]:


nodel1 =tf.constant(3.0, tf.float32)
nodel2 =tf.constant(4.0)


# In[ ]:


nodel2


# In[ ]:


sess=tf.Session()
print(sess.run([nodel1, nodel2]))


# In[ ]:


sess.close()


# In[ ]:


with tf.Session() as sess:
    output = sess.run[nodel1, nodel2]
    print(output)


# In[ ]:


#tensorflow
import tensorflow as tf


# In[ ]:


a=tf.constant(5.0)
b=tf.constant(6.0)


# In[ ]:


c=a*b


# In[ ]:


sess=tf.Session()


# In[ ]:


print(sess.run(c))


# In[2]:


# create graph
File_Writer=tf.summary.File_Writer("C:\\Users\\tensorflow\\graph", sess.graph)


# In[11]:


#place holders
import tensorflow as tf
a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
adder_node=a+b
sess=tf.Session()
print(adder_node,(a:[1,3], b:[2,4]))


# In[ ]:




