#!/usr/bin/env python
# coding: utf-8

# Recently I am diving into tensorflow and I had this idea of implementing some machine learning algorithms in tensorflow, and I came across this problem on stackoverflow in which the asker tried to apply linear regression on the Boston Housing dataset.
# 
# So I got the idea of the dataset and I decided to try it myself.

# In[15]:


import tensorflow as tf
import numpy as np
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt

boston=load_boston()
x_train, y_train = boston.data, boston.target.reshape(-1,1)


# I first loaded the dataset, and I got some weights that is supposed to be optimal with linear regression (from the internet), I wanted to know what is the minimal loss with linear regression

# In[16]:


weights = np.array([-1.08011358e-01, 4.64204584e-02,  2.05586264e-02,  2.68673382e+00,
                    -1.77666112e+01,  3.80986521e+00,  6.92224640e-04, -1.47556685e+00,
                    3.06049479e-01, -1.23345939e-02, -9.52747232e-01,  9.31168327e-03,
                    -5.24758378e-01])
bias = np.array([36.459488385090246])


# In[17]:


from sklearn import metrics
pred = np.matmul(x_train, weights) + bias
print(np.mean(np.square(y_train-pred)))
print(metrics.mean_squared_error(y_train, pred))


# The first strange thing is when I calculate the mean square error manually and use the function from sklearn, the results are different. So I print the array 'pred' and array y_train

# In[18]:


print(pred.shape)
print(y_train.shape)
print((y_train- pred).shape)
print((pred-y_train).shape)
print(np.ndim(pred))
print(np.ndim(y_train))


# Now it is clear that the error was caused by not understanding the broadcasting rule of numpy.
# 
# Broadcasting two arrays together follows these rules:
# 
# 1. If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
# 
# 2. The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
# 
# 3. The arrays can be broadcast together if they are compatible in all dimensions.
# 4. After broadcasting, each array behaves as if it had shape equal to the elementwise maximum of shapes of the two input arrays.
# 5. In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension
# 
# According to the first rule, `pred` will be prepended with 1 so it becomes shape \[1,506\] and then follows rule 5, the shape of pred becomes \[506,506\]

# In[19]:


print(np.mean(np.square(y_train-pred.reshape(-1,1))))
print(metrics.mean_squared_error(y_train, pred))


# Now it is the same. So get to the topic of feature scaling.

# In[21]:


tf.reset_default_graph()

W=tf.Variable(tf.zeros((13,1)), dtype=tf.float32)
b=tf.Variable(0.0)
X=tf.placeholder(tf.float32, shape=(None,x_train.shape[-1]), name='input')
Y=tf.placeholder(tf.float32, shape=(None,1), name='ouput')

Y_= tf.matmul(X, W) + b

loss=tf.reduce_mean(tf.square(Y_-Y))
optimizer = tf.train.GradientDescentOptimizer(0.000001)
train=optimizer.minimize(loss)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    epochs=1000
    sess.run(init)
    points=[ [],[] ]
    for i in range(epochs):
        if(i%100==0):
            print(i,sess.run(loss,feed_dict={X: x_train,Y:y_train}))
        sess.run(train,feed_dict={X: x_train,Y:y_train})
        if(i%2==0):
            points[0].append(1+i)
            points[1].append(sess.run(loss,feed_dict={X: x_train,Y:y_train})) 
    plt.plot(points[0],points[1],'r--')
    plt.axis([0,epochs,0,600])#
    plt.show()


# It can be seen from the figure above, GradientDescent has not reached the global minima yet but the loss already seemed to saturate. Also the learning rate is very small (hard to guess). With such small learning rate a lot more epochs are needed!

# In[23]:


from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)


# In[28]:


optimizer = tf.train.GradientDescentOptimizer(0.1)
train=optimizer.minimize(loss)

with tf.Session() as sess:
    epochs=300
    sess.run(init)
    points=[ [],[] ]
    for i in range(epochs):
        if(i%100==0):
            print(i,sess.run(loss,feed_dict={X: x_train,Y:y_train}))
        sess.run(train,feed_dict={X: x_train,Y:y_train})
        if(i%2==0):
            points[0].append(1+i)
            points[1].append(sess.run(loss,feed_dict={X: x_train,Y:y_train})) 
    plt.plot(points[0],points[1],'r--')
    plt.axis([0,epochs,0,600])#
    plt.show()


# Now we can see that GradientDescent succesfully finds the global minima and reasonably fast, the learning rate is also easier to guess.

# In[ ]:




