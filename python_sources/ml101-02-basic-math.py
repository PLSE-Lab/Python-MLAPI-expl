# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import tensorflow as tf
import matplotlib.pyplot as plt


def obj_func1(x):
    const = tf.constant([1., 3., 6., 10.], dtype=tf.float32)
    f = -1.0 * tf.reduce_sum(tf.log(1. + tf.exp(-0.5 * tf.square(const - x))))
    return f


def obj_func2(x):
    const = tf.constant([-3., -1., 0., 1., 6.], dtype=tf.float32)
    f = -1.0 * tf.reduce_sum(tf.log(1. + tf.exp(-1. * tf.square(const - x))))
    return f

domain = [-7.5, 8]
x_val = np.arange(*domain, 0.2)
x = tf.placeholder(dtype=tf.float32)
f = obj_func2(x)
grad = tf.gradients(f, x)

f_val = []
grad_val = []
with tf.Session() as sess:
    f_val = [sess.run(f, feed_dict={x: v}) for v in x_val]
    grad_val = [sess.run(grad, feed_dict={x: v})[0] for v in x_val]

fig = plt.figure(0)
ax = fig.add_subplot(211)
ax.plot(x_val, f_val)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_xlim(domain)
# ax = fig.add_subplot(212)
# ax.plot(x_val, grad_val)
# ax.set_xlabel('x')
# ax.set_ylabel('gradient')
# ax.set_xlim(domain)
fig.show()

# print(f_val)


learning_rate = 0.1
x0 = -1
steps = 0
x_list = []
f_list = []
f_0 = 0
with tf.Session() as sess:
    f_0 = sess.run(f, feed_dict={x: 0})
    print("{} = f(0)".format(f_0))
    x_current = x0
    f_current = sess.run(f, feed_dict={x: x_current})
    x_list.append(x_current)
    f_list.append(f_current)
    while(True):
        grad_current = sess.run(grad, feed_dict={x:x_current})[0]
        x_next = x_current - learning_rate * grad_current
        f_next = sess.run(f, feed_dict={x: x_next})
        steps += 1
        if f_next < f_current:
            f_current = f_next
            x_current = x_next
            x_list.append(x_current)
            f_list.append(f_current)
        else:
            print('Minimun value after {n} steps: {y:.4f} = f({x:.4f})'.format(n=steps, y=f_current, x=x_current))
            break
        
ax.hold(True)
ax.scatter(x_list[::10], f_list[::10], c='r')
ax.hold(False)