#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

print(tf.__version__)


# In[ ]:


hello = tf.constant(name='hello_', value='Hello world!!!')
hello2 = tf.constant(value='Hello world 2!!! ')

with tf.Session() as sess:
    print (sess.run(hello))


# In[ ]:


a = tf.constant(34)
b= tf.constant(55)
c = a+b


# In[ ]:


print(c)

with tf.Session() as sess:
    print(sess.run(c))


# **Matrix multiplication**

# In[ ]:


mat = tf.constant([[3,6], [8,5], [1,4]])
vec = tf.constant([[3], [5]])


# In[ ]:


print(mat)


# In[ ]:


print(mat.shape)


# In[ ]:


print(mat.dtype)


# In[ ]:


out = tf.matmul(mat,vec)


# In[ ]:


print(out)


# In[ ]:


print(out.op)


# In[ ]:


with tf.Session() as sess:
    print(sess.run(out))


# **Graphs**

# In[ ]:


a=tf.constant(name='op', value=500)


# In[ ]:


print(a)


# In[ ]:


a=tf.constant(name='op', value=1000)


# In[ ]:


print(tf.get_default_graph())


# In[ ]:


graph = tf.get_default_graph()


# In[ ]:


b=graph.get_tensor_by_name('op:0')


# In[ ]:


with tf.Session() as sess:
    print(sess.run(b))


# In[ ]:


for op in graph.get_operations():
    print(op)


# **Executing tensors**

# In[ ]:


aa = tf.constant(43.45)
bb= tf.constant(23.67)
cc= aa + bb


# In[ ]:


with tf.Session() as sess:
    print (cc.eval())


# In[ ]:


with tf.Session() as sess:
    print (sess.run([cc,aa,bb]))


# **Variables**

# In[ ]:


v_a = tf.constant(5)
v_b = tf.placeholder(tf.float32)


# In[ ]:


v_c = tf.Variable(5)


# In[ ]:


with tf.Session() as sess:
    sess.run(v_c.initializer)
    print(sess.run(v_c))


# In[ ]:


var1 = tf.get_variable(name='myvar1', shape=(), dtype=tf.float32, initializer=tf.zeros_initializer())
var2 = tf.get_variable(name='myvar2', shape=(), dtype=tf.float32, initializer=tf.ones_initializer())
var3 = tf.get_variable(name='myvar3', shape=(), dtype=tf.float32, initializer=tf.random_uniform_initializer())


# In[ ]:


with tf.Session() as sess:
    for i in range(10):
        sess.run(var1.initializer)
        var2.initializer.run()
        var3.initializer.run()
        print(sess.run([var1, var2, var3]))
    


# In[ ]:


c_a = tf.constant(123.0)
op1 = var1.assign(3)
op2 = var2.assign(var2+42)
op3 = var3.assign(c_a+var3)


# In[ ]:


with tf.Session() as sess:
    sess.run(var1.initializer)
    var2.initializer.run()
    var3.initializer.run()
    for i in range(10):
        sess.run(op1)
        sess.run(op2)
        sess.run(op3)
        print(sess.run([var1, var2, var3]))
    


# **Gradients**

# In[ ]:


def func(x, y):
    return x*x + y*y


# In[ ]:


x0, y0 = 2, 3
import numpy as np


# Objective is to find angle where the change in function is maximum

# In[ ]:


for angle in np.linspace(0, 2*np.pi, 100):
    x, y = x0 + 0.1* np.cos(angle), y0 + 0.1*np.sin(angle)
    diff = func(x, y) - func(x0, y0)
    print('angle: ', angle, 'diff', diff)


# In[ ]:


np.tan(0.9519977738150889)


# In[ ]:


np.arctan(6/4)


# **Calculating gradient in Tensorflow**

# In[ ]:


g_x = tf.placeholder(tf.float32)
g_y = tf.placeholder(tf.float32)


# In[ ]:


fxy = g_x*g_x + g_y*g_y


# In[ ]:


grad = tf.gradients(fxy, [g_x, g_y])


# If having placeholders, we have to provide them as feeddict

# In[ ]:


with tf.Session() as sess:
    print(sess.run(grad, feed_dict={g_x:2, g_y:3}))


# **Fitting Line to points**

# In[ ]:


pts = [[2, 14], [3,12], [1,11], [3,15], [5,14], [4,12], [5,15], [2,11]]


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


pts = np.array(pts)
plt.scatter(pts[:,0], pts[:,1]) #all rows and column 0 .....


# In[ ]:


grid = [[m,c] for m in range(15) for c in range(15)]


# In[ ]:


grid


# In[ ]:


def get_loss(pts, m, c):
    loss = 0
    for pt in pts:
        diff = pt[1] - (m*pt[0] + c)
        loss += (diff) * (diff)
    return loss    

for (m,c) in grid:
    print('loss: ', get_loss(pts,m,c), ', m: ', m, ', c: ', c)


# **Linear Regression in Tensorflow**

# 

# In[ ]:


dummy_x = np.random.random(1000)


# In[ ]:


#y = 5x +3, m=5   c=3
dummy_y = 5 * dummy_x + 3+ 0.1*np.random.randn(1000)


# In[ ]:


plt.scatter(dummy_x, dummy_y, s=0.1)


# In[ ]:


r_x = tf.placeholder(shape=(1000,), dtype=tf.float32)
r_y = tf.placeholder(shape=(1000,), dtype=tf.float32)
m= tf.get_variable(name='slope',dtype=tf.float32,shape=(),initializer=tf.ones_initializer())
c= tf.get_variable(name='intercept',dtype=tf.float32,shape=(),initializer=tf.ones_initializer())


# In[ ]:


#objective function
yest = m*r_x+c
loss = tf.losses.mean_squared_error(r_y,yest)


# In[ ]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss)


# In[ ]:


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()
    for e in range(100):
        _, val_loss = sess.run([optimizer, loss], feed_dict={r_x:dummy_x, r_y:dummy_y})
        print('loss: ', val_loss, ', m: ', m.eval(), ', c: ', c.eval())


# **Saving graph for Tensorboard**

# In[ ]:


writer = tf.summary.FileWriter(logdir='log', graph=tf.get_default_graph())


# **Variable scoping**

# In[2]:


abc = tf.constant(3)
xyz = tf.Variable(5)
print('names ', abc.name, xyz.name)


# In[3]:


with tf.variable_scope('myscope'):
    abc = tf.constant(3)
    xyz = tf.Variable(5)
print('names ', abc.name, xyz.name)


# In[ ]:




