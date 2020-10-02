#!/usr/bin/env python
# coding: utf-8

# <h1>TensorFlow Eager Mode</h1>

# By default Tensor Flow follows the lazy evaluation paradigm because the lazy evaluation paradigm is what allows distribution and deploy in ML Cloud (AWS/Google/Azure) and helps scale out the training. <b>tf.enable_eager_execution()</b> allows you to run the program code instantly which helps in debugging.
# 

# In[ ]:


import tensorflow as tf
tf.enable_eager_execution()

def compute_area(sides):
  # slice the input to get the sides
  a = sides[:,0]  # 5.0, 2.3
  b = sides[:,1]  # 3.0, 4.1
  c = sides[:,2]  # 7.1, 4.8
  
  # Heron's formula
  s = (a + b + c) * 0.5   # (a + b) is a short-cut to tf.add(a, b)
  areasq = s * (s - a) * (s - b) * (s - c) # (a * b) is a short-cut to tf.multiply(a, b), not tf.matmul(a, b)
  return tf.sqrt(areasq)

area = compute_area(tf.constant([
      [5.0, 3.0, 7.1],
      [2.3, 4.1, 4.8]
    ]))


print(area)


# 

# In[ ]:





# 
