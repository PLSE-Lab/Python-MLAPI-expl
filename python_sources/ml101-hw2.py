#!/usr/bin/env python
# coding: utf-8

# **ML101 Home Work Week 2**

# In[ ]:


# coding=utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32)
    const = tf.constant([-3.,-1., 0., 1., 6.], dtype=tf.float32)
    # the subtraction of a vector by a scalar yeilds a vector by subtracting the scalar from each element of the vector
    f = -1. * tf.reduce_sum(tf.log(1. + tf.exp(-1. * tf.square(const - x))))
    grad = tf.gradients(f, x)

    x_val = np.arange(-20., 20., 0.1)
    f_val = []
    grad_val = []
    with tf.Session() as sess:
        for v in x_val:
            f_val.append(sess.run(f, feed_dict={x: v}))
            grad_val.append(sess.run(grad, feed_dict={x: v})[0])
    
    fig = plt.figure(0)
    # create a grid of sub-figures of 2 rows and 1 column. And first plot on the first sub-figure.
    ax = fig.add_subplot(211)
    ax.plot(x_val, f_val)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_xlim([-20, 20])
    
    learning_rates = [2,1.95,1.9,1.85,1.8,1.75,1.7,1.65,1.6,1.55,1.5,1.45,1.4,1.35,1.3,1.2,1.1,1]
    min_fx = None
    for learning_rate in learning_rates:
        #learning_rate = 0.1
        x0 = 1.
        steps = 0
        
        with tf.Session() as sess:
            x_current = x0
            f_current = sess.run(f, feed_dict={x: x_current})
            if min_fx is None:
                min_fx = f_current
            while True:
                # the value returned by sess.run(grad) is a vector with a single element
                grad_current = sess.run(grad, feed_dict={x: x_current})[0]
                x_next = x_current - learning_rate * grad_current
                f_next = sess.run(f, feed_dict={x: x_next})
                steps += 1
                
                if f_next < f_current:
                    f_current = f_next
                    x_current = x_next
                else:
                    #print ('for learning rate {l:.6f} best x after {n} steps: {x:.4f} and current f value is . {f:.20f}'.format(x=x_current, n=steps, f=f_current, l = learning_rate))
                    break
            if min_fx > f_current:
                min_fx = f_current
                result_x = x_current
                result_lr = learning_rate 

    print ('for learning rate {l:.6f} best x: {x:.4f} and  f(x) value is . {f:.20f}'.format(x=result_x, f=min_fx, l = result_lr))
    
    learning_rate = result_lr
    x0 = 1.
    steps = 0
    x_list = []
    f_list = []
    with tf.Session() as sess:
        x_current = x0
        f_current = sess.run(f, feed_dict={x: x_current})
        while True:
            # the value returned by sess.run(grad) is a vector with a single element
            grad_current = sess.run(grad, feed_dict={x: x_current})[0]
            x_next = x_current - learning_rate * grad_current
            f_next = sess.run(f, feed_dict={x: x_next})
            steps += 1

            if f_next < f_current:
                f_current = f_next
                x_current = x_next
                x_list.append(x_next)
                f_list.append(f_next)
            else:
                #print ('for learning rate {l:.6f} best x after {n} steps: {x:.4f} and current f value is . {f:.20f}'.format(x=x_current, n=steps, f=f_current, l = learning_rate))
                break
    
    # hold the sub-figure to overlap the scattered points on to the plot of function f

    ax.scatter(x_list[::10], f_list[::10], c='r')
    ax = fig.add_subplot(212)
    ax.plot(x_val, grad_val)
    ax.set_xlabel('x')
    ax.set_ylabel('gradient')
    ax.set_xlim([-20, 20])


# Using gradient descent algorithm, for learning rate 1.950000 the best value for  x: **-0.0015** and  the minimum f(x) value is . **-1.31979465484619140625**
# 
# I set initial value based on the graph, where the global minimum is near -1. I ran experiements with various learning rates ranging from -10 to 10 and found that values between 1.0 to 2.0 yield least f(x).
# 
# I set various values between 1.0 and 2.0 and obtained the best value for x for least f(x)
# 

# **See various values for different learning rates
# **
# 
# for learning rate 0.1000 best x after 115 steps: 0.0009 and current f value is . -1.31979322433471679688 
# for learning rate 0.0500 best x after 219 steps: 0.0018 and current f value is . -1.31979191303253173828 
# for learning rate 0.0200 best x after 461 steps: 0.0067 and current f value is . -1.31977796554565429688 
# for learning rate 0.0100 best x after 884 steps: 0.0085 and current f value is . -1.31976997852325439453 
# for learning rate 0.0050 best x after 1613 steps: 0.0132 and current f value is . -1.31974077224731445312
# for learning rate 0.0020 best x after 3612 steps: 0.0209 and current f value is . -1.31966948509216308594
# for learning rate 0.0010 best x after 6355 steps: 0.0331 and current f value is . -1.31949603557586669922
# for learning rate 0.0005 best x after 11115 steps: 0.0501 and current f value is . -1.31913137435913085938
# for learning rate 0.0002 best x after 23503 steps: 0.0777 and current f value is . -1.31822752952575683594
# for learning rate 0.0001 best x after 39152 steps: 0.1160 and current f value is . -1.31633162498474121094
# 
# 
# 
# for learning rate 0.10000 best x after 115 steps: 0.0009 and current f value is . -1.31979322433471679688 
# for learning rate 0.05000 best x after 219 steps: 0.0018 and current f value is . -1.31979191303253173828 
# for learning rate 0.02000 best x after 461 steps: 0.0067 and current f value is . -1.31977796554565429688 
# for learning rate 0.01000 best x after 884 steps: 0.0085 and current f value is . -1.31976997852325439453 
# for learning rate 0.00500 best x after 1613 steps: 0.0132 and current f value is . -1.31974077224731445312 
# for learning rate 0.00200 best x after 3612 steps: 0.0209 and current f value is . -1.31966948509216308594 
# for learning rate 0.00100 best x after 6355 steps: 0.0331 and current f value is . -1.31949603557586669922 
# for learning rate 0.00050 best x after 11115 steps: 0.0501 and current f value is . -1.31913137435913085938
# for learning rate 0.00020 best x after 23503 steps: 0.0777 and current f value is . -1.31822752952575683594
# for learning rate 0.00010 best x after 39152 steps: 0.1160 and current f value is . -1.31633162498474121094
# for learning rate 0.00005 best x after 65461 steps: 0.1612 and current f value is . -1.31312632560729980469
# for learning rate 0.00002 best x after 117842 steps: 0.2592 and current f value is . -1.30241298675537109375 
# for learning rate 0.00001	 best x after 167236 steps: 0.3739 and current f value is . -1.28293943405151367188
# 
# 
# for learning rate 1.800000 best x after 6  steps: -0.0015 and current f value is . -1.31979465484619140625 
# for learning rate 1.750000 best x after 6  steps: -0.0015 and current f value is . -1.31979465484619140625 
# for learning rate 1.700000 best x after 5  steps: -0.0016 and current f value is . -1.31979453563690185547 
# for learning rate 1.650000 best x after 5  steps: -0.0015 and current f value is . -1.31979465484619140625 
# for learning rate 1.600000 best x after 5  steps: -0.0013 and current f value is . -1.31979441642761230469 
# for learning rate 1.550000 best x after 6  steps: -0.0013 and current f value is . -1.31979453563690185547 
# for learning rate 1.500000 best x after 7  steps: -0.0014 and current f value is . -1.31979465484619140625 
# for learning rate 1.450000 best x after 6  steps: -0.0008 and current f value is . -1.31979441642761230469 
# for learning rate 1.400000 best x after 8  steps: -0.0014 and current f value is . -1.31979465484619140625 
# for learning rate 1.350000 best x after 7  steps: -0.0009 and current f value is . -1.31979441642761230469 
# for learning rate 1.300000 best x after 9  steps: -0.0014 and current f value is . -1.31979465484619140625 
# for learning rate 1.200000 best x after 10 steps: -0.0013 and current f value is . -1.31979453563690185547 
# for learning rate 1.100000 best x after 11 steps: -0.0013 and current f value is . -1.31979453563690185547 
# for learning rate 1.000000 best x after 11 steps: -0.0008 and current f value is . -1.31979441642761230469 
# 
# 
# 
# for learning rate 2.000000 best x after 4  steps: -0.0015 and current f value is . -1.31979441642761230469
# for learning rate 1.950000 best x after 5  steps: -0.0015 and current f value is . -1.31979465484619140625
# for learning rate 1.900000 best x after 6  steps: -0.0015 and current f value is . -1.31979465484619140625
# for learning rate 1.850000 best x after 5  steps: -0.0015 and current f value is . -1.31979453563690185547
# for learning rate 1.800000 best x after 6  steps: -0.0015 and current f value is . -1.31979465484619140625
# for learning rate 1.750000 best x after 6  steps: -0.0015 and current f value is . -1.31979465484619140625
# for learning rate 1.700000 best x after 5  steps: -0.0016 and current f value is . -1.31979453563690185547
# for learning rate 1.650000 best x after 5  steps: -0.0015 and current f value is . -1.31979465484619140625
# for learning rate 1.600000 best x after 5  steps: -0.0013 and current f value is . -1.31979441642761230469
# for learning rate 1.550000 best x after 6  steps: -0.0013 and current f value is . -1.31979453563690185547
# for learning rate 1.500000 best x after 7  steps: -0.0014 and current f value is . -1.31979465484619140625
# for learning rate 1.450000 best x after 6  steps: -0.0008 and current f value is . -1.31979441642761230469
# for learning rate 1.400000 best x after 8  steps: -0.0014 and current f value is . -1.31979465484619140625
# for learning rate 1.350000 best x after 7  steps: -0.0009 and current f value is . -1.31979441642761230469
# for learning rate 1.300000 best x after 9  steps: -0.0014 and current f value is . -1.31979465484619140625
# for learning rate 1.200000 best x after 10 steps: -0.0013 and current f value is . -1.31979453563690185547
# for learning rate 1.100000 best x after 11 steps: -0.0013 and current f value is . -1.31979453563690185547
# for learning rate 1.000000 best x after 11 steps: -0.0008 and current f value is . -1.31979441642761230469
# 

# In[ ]:


# coding=utf-8

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32)
    const = tf.constant([-3.,-1., 0., 1., 6.], dtype=tf.float32)
    # the subtraction of a vector by a scalar yeilds a vector by subtracting the scalar from each element of the vector
    f = -1. * tf.reduce_sum(tf.log(1. + tf.exp(-1. * tf.square(const - x))))
    grad = tf.gradients(f, x)

    x_val = np.arange(-20., 20., 0.1)
    f_val = []
    grad_val = []
    with tf.Session() as sess:
        for v in x_val:
            f_val.append(sess.run(f, feed_dict={x: v}))
            grad_val.append(sess.run(grad, feed_dict={x: v})[0])
    
    fig = plt.figure(0)
    # create a grid of sub-figures of 2 rows and 1 column. And first plot on the first sub-figure.
    ax = fig.add_subplot(211)
    ax.plot(x_val, f_val)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_xlim([-20, 20])
    
    learning_rate = 0.1
    x0 = -10.
    steps = 0
    x_list = []
    f_list = []
    with tf.Session() as sess:
        x_current = x0
        f_current = sess.run(f, feed_dict={x: x_current})
        while True:
            # the value returned by sess.run(grad) is a vector with a single element
            grad_current = sess.run(grad, feed_dict={x: x_current})[0]
            x_next = x_current - learning_rate * grad_current
            f_next = sess.run(f, feed_dict={x: x_next})
            steps += 1

            if f_next < f_current:
                f_current = f_next
                x_current = x_next
                x_list.append(x_next)
                f_list.append(f_next)
            else:
                print ('for learning rate {l:.6f} best x after {n} steps: {x:.4f} and current f value is . {f:.20f}'.format(x=x_current, n=steps, f=f_current, l = learning_rate))
                break
    
    # hold the sub-figure to overlap the scattered points on to the plot of function f

    ax.scatter(x_list[::10], f_list[::10], c='r')
    ax = fig.add_subplot(212)
    ax.plot(x_val, grad_val)
    ax.set_xlabel('x')
    ax.set_ylabel('gradient')
    ax.set_xlim([-20, 20])


# **Discuss the result if you use the initial x value as -10, and the learning rate as 0.1.**
# 
# With Initial vale as -10 and learining rate as 0.1, the best value for x is -10 because the gradient would be 0 and any learning rate would not make any difference
# 
