# In this script we would like to implement a simple AND Gate using Tensor Flow

 


#Imorting tensorflow library 
import tensorflow as tf
#Import below for the code to work in older version  of tensorflow : Note Session() will not work in tensorflow 2.0 we have to use tf.compat.v1.Session()

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

T,F = 1., -1.
bias  = 1. 

train_in = [
    [T,T,bias],
    [T,F,bias],
    [F,T,bias],
    [F,F,bias]
]

train_out = [
    [T],
    [F],
    [F],
    [F]
    
]

#Defining variables and assigning a random value
w = tf.Variable(tf.random_normal([3,1]))

#Defining Activation function (step function)

#Writing a custom function-We can use prefined function also

def step(x):
    is_greater = tf.greater(x,0)
    as_float = tf.to_float(is_greater)
    doubled = tf.multiply(as_float,2)
    return tf.subtract(doubled,1)

output = step(tf.matmul(train_in,w))
err = tf.subtract(train_out,output)
mse = tf.reduce_mean(tf.square(err))


delta = tf.matmul(train_in,err,transpose_a=True)
train = tf.assign(w,tf.add(w,delta))

sess = tf.Session()
sess.run(tf.initialize_all_variables())

#Iterating 

err, target = 1,0
#consider epoch as number of iterations
epoch,max_epoch = 0,10
while err> target and epoch < max_epoch:
    epoch +=1
    err,_ = sess.run([mse,train])
    print('epoch:',epoch , 'mse',err)
    
#Output 
#We can see that in 3 epochs we will get mse as zero.








