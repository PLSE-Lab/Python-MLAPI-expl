#!/usr/bin/env python
# coding: utf-8

# # Gradient descent optimization with TensorFlow 2.

# Last Updated on February 6, 2020

# Finding a global minimum of a given quantity is a common problem. When it is possible to acquire function gradients using automatic differentiation the gradient descent (GD) algorithm (https://en.wikipedia.org/wiki/Gradient_descent) could be a good solution. Here we will take a look at gradient-based algorithms and explore convergence. Many libraries provide automatic differentiation and GD algorithm implementation, we are going to use TensorFlow because it provides GPU support, works well with a big number of variables and large tensors.

# An example above is an artificial real-valued function of two variables func(x, y) that has one global and several local minimums.

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

print(tf.__version__)


# In[ ]:


def func(x, y):
    return - 5.5 * tf.exp(- 20.0 * (x - 0.3)**2 - 40.0 * (y - 0.3)**2) - 3.5 * tf.exp(- 15.0 * (x - 0.6)**2 - 10.0 * (y - 0.85)**2) - 2.0 * tf.sin(2.0 * (x - y))


x = np.linspace(0, 1, 400)
X, Y = np.meshgrid(x, x)
Z = func(X, Y)

plt.figure(figsize=(6, 4.7))
plt.contourf(X, Y, Z, 60, cmap='RdGy')
plt.xlabel('x', fontsize=19)
plt.ylabel('y', fontsize=19)
plt.tick_params(axis='both', which='major', labelsize=14)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=14) 


# In the code below, there is a function objective() that depends on two variables x, y which are constrained with constr(a, b) and take values from interval [0, 1]. A starting point "start" is chosen, then we create object opt = tf.keras.optimizers.SGD() and run opt.minimize() method several times. We explore 2 alghoritms, a simple gradient descent and "ADAM" alghoritm given by tf.keras.optimizers.SGD() and tf.keras.optimizers.Adam() respectively.

# In[ ]:


def constr(a, b):
    assert b > a
    return lambda x: tf.clip_by_value(x, a, b)


x = tf.Variable(0.0, trainable=True, dtype=tf.float64, name='x', constraint=constr(0, 1))
y = tf.Variable(0.0, trainable=True, dtype=tf.float64, name='y', constraint=constr(0, 1))


def objective():
    return - 5.5 * tf.exp(- 20.0 * (x - 0.3)**2 - 40.0 * (y - 0.3)**2) - 3.5 * tf.exp(- 15.0 * (x - 0.6)**2 - 10.0 * (y - 0.85)**2) - 2.0 * tf.sin(2.0 * (x - y))


def optimize(start, verbose=False, method='SGD'):
    x.assign(start[0])
    y.assign(start[1])

    if method == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=0.01)
 
    if method == 'ADAM':
        opt = tf.keras.optimizers.Adam(
            learning_rate=0.1,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
        )

    obj_vals = []
    coords = []

    for i in range(50):
        if verbose and i % 5 == 0:
            print(f'step: {i}, obj = {objective().numpy():.4f}, x = {x.numpy():.4f}, y = {y.numpy():.4f}')
        obj_vals.append(objective().numpy())
        coords.append((x.numpy(), y.numpy()))
        opt.minimize(objective, var_list=[x, y])
        
    return obj_vals, coords


# In[ ]:


def plot_res(obj_vals, coords):
    plt.figure(figsize=(16, 6))
    plt.subplot(121)
    plt.contourf(X, Y, Z, 60, cmap='RdGy')
    plt.xlabel('x', fontsize=19)
    plt.ylabel('y', fontsize=19)
    plt.tick_params(axis='both', which='major', labelsize=14)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14) 

    xcoord = [x[0] for x in coords]
    ycoord = [x[1] for x in coords]
    plt.plot(xcoord, ycoord, '.-')
    plt.plot(xcoord[-1], ycoord[-1], "y*", markersize=12)

    plt.subplot(122)
    plt.plot(obj_vals, '.-')
    plt.plot([len(obj_vals) - 1], obj_vals[-1], "y*", markersize=12)
    plt.xlabel('Step', fontsize=17)
    plt.ylabel('Objective', fontsize=17)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.show()


# Let's first use simple GD alghoritm.

# In[ ]:


obj_vals, coords = optimize([0.25, 0.65], verbose=True, method='SGD')
plot_res(obj_vals, coords)


# Here we can see that trajectory smoothly converges to the local minimum. Let's change the starting point.

# In[ ]:


obj_vals, coords = optimize([0.2, 0.65], method='SGD')  
plot_res(obj_vals, coords)


# It's descending to the global minimum now. However, we can see that having found the global minimum area the trajectory jumps inside it back and forth and seems nonconverging. This is a known problem with GD that can be on the one hand solved by carefully tunning learning_rate (trajectory update step). Another way to deal with that is to take into account previous values of the trajectory or so-called momentums (https://distill.pub/2017/momentum/). An algorithm that uses first and second momentums is "ADAM" (https://arxiv.org/abs/1412.6980). This algorithm is more than just GD with momentums and more details can be found in the article. Let's try it out.

# In[ ]:


obj_vals, coords = optimize([0.2, 0.65], method='ADAM')  
plot_res(obj_vals, coords)


# Here it is! Now we can see that trajectory is converging smoothly down into the global minimum point.

# # Applying gradients.

# In case you need to process gradients by hand before and then apply them explicitly, the example of the code above does so.

# In[ ]:


x.assign(0.25)
y.assign(0.65)
opt = tf.keras.optimizers.SGD(learning_rate=0.01)

for i in range(30):
    with tf.GradientTape() as tape:
        z = func(x, y)
    grads = tape.gradient(z, [x, y])
    processed_grads = [g for g in grads]
    grads_and_vars = zip(processed_grads, [x, y])
    if i % 5 == 0:
        print(f"step {i}, z = {z.numpy():.2f}, x = {x.numpy():.2f}, y = {y.numpy():.2f},  grads0 = {grads[0].numpy():.2f}, grads1 = {grads[1].numpy():.2f}")
    opt.apply_gradients(grads_and_vars)


# # Conclusion.

# In this article, we looked at the gradient descent method and its implementation with TensorFlow library. Advanced algorithms like "ADAM" generally work better than simple GD. And the result of the optimization depends on the starting point.

# # References.
# * https://en.wikipedia.org/wiki/Gradient_descent
# * https://arxiv.org/abs/1412.6980
# * https://www.tensorflow.org/

# In[ ]:




