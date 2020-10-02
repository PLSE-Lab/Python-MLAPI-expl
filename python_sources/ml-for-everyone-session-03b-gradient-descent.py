#!/usr/bin/env python
# coding: utf-8

# # Lecture 2: Gradient Descent

# This notebook will illustrate gradient descent in a simple linear regression example with a single predictor.  It will also illustrate how to construct functions in python.

# Let's start by importing all of our plotting functionality and numpy:

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


# Here, we construct a simulated dataset:
# $$y = 8x+ \epsilon$$
# where
# $$\epsilon \sim N(0, 2)$$

# In[ ]:


# np.random.rand gives us uniformly distributed random numbers [0, 1], so multiplying by 10 gives us [0, 10]
X = 10*np.random.rand(50) 
# np.random.randn gives us standard normal random numbers, so multiplying by 2 gives us N(0, 2)
eps = 2*np.random.randn(50)
# our final simulated dataset
y = 8*X+eps


# Let's take a look at our dataset:

# In[ ]:


plt.scatter(X, y)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# Let's define a function that takes two arguments-a set of predictions and a set of true y values-and returns the average squared loss:

# In[ ]:


def loss_function(preds, y):
    sq_residuals = (y-preds)**2
    rss = np.sum(sq_residuals)/len(sq_residuals)
    return rss


# Let's define a function that will take coefficients and a set of x values and return the predictions for a linear model:

# In[ ]:


def predict_linear_model(b0, b1, x_to_pred):
    preds = b0+b1*x_to_pred
    return preds


# And finally, a function that will plot our data, a linear model, and return the loss function value:

# In[ ]:


def plot_data_and_preds(b0, b1, x_to_pred):
    preds = predict_linear_model(b0, b1, x_to_pred)

    plt.scatter(X, y)
    plt.plot(X, preds, c="red")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(["Regression Line", "Raw Data"])
    plt.show()

    the_loss = loss_function(preds, y)
    print("loss=%s" % the_loss)
    return the_loss


# In[ ]:


plot_data_and_preds(0, 2, X)


# This function will create a bunch of beta values, plot the linear regression line for each, calculate the loss, and make a plot of the loss as a function of beta.  Notice how to specify a default value for a parameter to a function:

# In[ ]:


def plot_loss_function(y, X, n_points=5):
    # equally spaced array of 5 values between -20 and 20, like the seq function in R
    beta1s = np.linspace(-20, 20, n_points)
    losses = []
    for beta1 in beta1s:
        print("beta1=%s " % beta1)
        loss = plot_data_and_preds(0, beta1, X)
        losses.append(loss)
    plt.scatter(beta1s, losses)
    plt.xlabel("beta1")
    plt.ylabel("J")
    plt.show()


# In[ ]:


#plot_loss_function(y, X)
plot_loss_function(y, X, 20)


# This function will return the value of the gradient of the loss function with respect to $\beta_1$:

# In[ ]:


def gradient_b1(b0, b1, y, X):
    grad = np.sum(-2.0*X*(y-b0-b1*X))/len(X)
    return grad


# In[ ]:


# the gradient at 20 is positive...
gradient_b1(0, 20, y, X)
# and at -10 is negative...
gradient_b1(0, -10, y, X)


# And finally, this function will run our gradient descent algorithm to get to the minumum value of $\beta_1$.

# In[ ]:


def gradient_descent(b0_start, b1_start, y, X, learning_rate=0.01, n_steps=25):
    b1 = b1_start
    print("b1=%s" % b1)
    for i in range(n_steps):
        grad = gradient_b1(b0_start, b1, y, X)
        #print "gradient=%s" % grad
        b1 = b1-learning_rate*grad
        print("b1=%s" % b1)


# In[ ]:


# play with different learning rates: 0.00001, 0.1, 10
gradient_descent(0, 15, y, X, learning_rate=0.01)


# In[ ]:


# things can go terribly wrong and diverge if the learning rate is too high
gradient_descent(0, 20, y, X, learning_rate=10)


# In[ ]:


# a tiny learning rate with converge slowly
gradient_descent(0, 20, y, X, learning_rate=0.00001, n_steps=100)


# In[ ]:




