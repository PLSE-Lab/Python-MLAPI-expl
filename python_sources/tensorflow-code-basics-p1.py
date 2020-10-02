#This is a basic script to explain tensorflow framework

#In this script we create a simple model and evaluate it

#Imorting tensorflow library 
import tensorflow as tf
#Import below for the code to work in older version  of tensorflow : Note Session() will not work in tensorflow 2.0 we have to use tf.compat.v1.Session()

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

#Variables 
w = tf.Variable([.3],tf.float32)
b = tf.Variable([-.3],tf.float32)

x = tf.placeholder(tf.float32)

#Doing some operations
linearmodel = w*x + b

#Initalizing variables 
init = tf.global_variables_initializer()

#Running session 
sess = tf.Session()

#For giving desired values
y = tf.placeholder(tf.float32)

#Loss function for Linear regression 
squared_deltas = tf.square(linearmodel - y)
loss = tf.reduce_sum(squared_deltas)

#Reducing loss Using Gradient Descent
#Choosing learning rate=0.01 

optimizer =tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)


for i in range(1000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})
    
print(sess.run([w,b]))

#We will get output as [array([-0.9999969], dtype=float32), array([0.9999908], dtype=float32)] which means that
# value of w = -0.9999969 and b = 0.9999908 



