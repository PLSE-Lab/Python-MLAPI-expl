#!/usr/bin/env python
# coding: utf-8

# The objective of this experiment is to plot a Quadratic Equation representing an error function and see how to arrive at the minima in the plot.

# In this experiment we will use a simple quadratic equation for error function as below:
# 
# $ w^2 $ + $ 2w $ + $ 2 $

# In[ ]:


# Importing required packages
import matplotlib
matplotlib.use('nbagg')
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


def error(w):
    return (w**2) + (2*w) + 2


# #### Making the data

# In[ ]:


w = list(range(-10,10))
err = []
for i in w:
    err.append(error(i))


# #### Derivative of the error function is $ 2w $ + $ 2 $

# In[ ]:


def gradient(w):
    return 2*w + 2


# Let us write a function for gradient descent that can be reused:

# In[ ]:


def delta(w, eta):
    return eta*gradient(w)

def gradient_descent(eta, w, nb_of_iterations):
    w_err = [np.array([w, error(w)])] # List to store the w, error values
    for i in range(nb_of_iterations):
        dw = delta(w, eta)  # Get the delta w update
        w = w - dw  # Update the current w value
        w_err.append(np.array([w, error(w)]))  # Add w, error to list
    return np.array(w_err)


# ### $ eta $ = 0.2

# In[ ]:


# Set the learning rate
eta = 0.2

#Set the initial parameter
w = 5

# number of gradient descent updates
nb_of_iterations = 20

w_err_02 = gradient_descent(eta, w, nb_of_iterations)


# ### $eta $ = 0.5

# In[ ]:


# Set the learning rate
eta = 0.5

#Set the initial parameter
w = 5

# number of gradient descent updates
nb_of_iterations = 20

w_err_05 = gradient_descent(eta, w, nb_of_iterations)


# ### $eta$ = 0.7

# In[ ]:


# Set the learning rate
eta = 0.7

#Set the initial parameter
w = 5

# number of gradient descent updates
nb_of_iterations = 20

w_err_07 = gradient_descent(eta, w, nb_of_iterations)


# In[ ]:


# Print the final w, and cost
for i in range(0, len(w_err_07)):
    print('w({}): {:.4f} \t cost: {:.4f}'.format(i, w_err_07[i][0], w_err_07[i][1]))


# #### Let us see how to arrive at local minima

# In[ ]:


w = list(range(-10,10))
err = []
for i in w:
    err.append(error(i))


# In[ ]:


plt.figure(figsize=(8, 10))
plt.grid(True)
plt.subplot(311)
plt.plot(w, err)
plt.plot(w_err_02[:,0], w_err_02[:,1],"o")
plt.title(["x vs m","eta = 0.2"])
n = range(1, len(w_err_02[:,0]))
for i, txt in enumerate(n):
    plt.annotate(txt, (w_err_02[:,0][i], w_err_02[:,1][i]))
plt.subplot(312)
plt.plot(w, err)
plt.plot(w_err_05[:,0], w_err_05[:,1],"o")
plt.title(["x vs m","eta = 0.5"])
n = range(1, len(w_err_05[:,0]))
for i, txt in enumerate(n):
    plt.annotate(txt, (w_err_05[:,0][i], w_err_05[:,1][i]))
plt.subplot(313)
plt.plot(w, err)
plt.plot(w_err_07[:,0], w_err_07[:,1],"o")
plt.title(["x vs m","eta = 0.7"])
n = range(1, len(w_err_02[:,0]))
for i, txt in enumerate(n):
    plt.annotate(txt, (w_err_07[:,0][i], w_err_07[:,1][i]))
plt.show()

