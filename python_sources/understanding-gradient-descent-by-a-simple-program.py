#!/usr/bin/env python
# coding: utf-8

# **INTRODUCTION:**
# 
# 1. In the below program I have implemented a linear regression model for the given dataset.
# 2. I Found the best fit (slope and intercept) using the batch gradient descent as an optimiser 
# 3. And the cost function as MSE (Mean Squared error)
# 
# **Note:** I did not use any predefined functions to implement gradient descent, so that by going throught the program below will give a clear idea of how gradirnt descent works.

# **GENERAL CONCEPTS:**

# ***What is Gradient Descent?***
# 
# Gradient is a point vector on any steep surface (example: mountain). It has both magnitude (measure of the steepness) and the direction of the steepness. We call it gradient descent because we are trying to descent the gradient by moving to the point where it is less steep (in our case the point where the cost function (MSE) is minimum). In the first image below we are trying to descent the gradient to reach from point A to B.
# 
# ***What is a Concave model?***
# 
# When we plot a linear regression model along with its error in 3D we will always get a concav model (where there will be only one Global minima (B): 
# 
# ![](https://i.imgur.com/IHBZyN0.png)
# 
# To understand better below is not an example of Concave model (where there are many local minima (L1,L2,L3) and one global minima (B) as shown in below picture):
# 
# ![](https://i.imgur.com/2NLAZ5G.png)

# **PROGRAM:**

# 
# 
# **Importing required libraries:**

# In[ ]:


from numpy import *
import matplotlib.pyplot as plt


# **Function to plot the best fit:**

# In[ ]:


def plot_best_fit(intercept, slope):
    axes = plt.gca()
    x_vals = array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-')


# **Function to calculate MSE (Mean Squared Error):**

# In[ ]:


def mean_squared_error(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))


# **Function to calculate gradient descent (b_gradient, m_gradient) and the respective value of slope (new_b) and intercept (new_m):**
# 
# The gradient descent formula is obtained by partial differentiation of cost function (MSE) with respect to slope (b) and intecept (m):
# 
# Mean squred error is the total sum of the difference between the predicted value yi and the expected value y'i 
# 
# ![](https://i.imgur.com/UTV23a0.png) Where, y'i = (MXi + b)
# ![](https://i.imgur.com/rpSB3vh.png)
# 
# 

# In[ ]:


def batch_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


# **Function to implment the logic of the whole program:**
# 
# At one point we will reach a global minima (B) if we are descending our gradient contineously, but if we descend the gradient contineously even after reaching (B) we may end up in point (p) which is not the minima as shown in the below picture: 
# 
# ![](https://i.imgur.com/Jm0sqsn.png)
# 
# At this point (p) MSE will be more than (B). Here is were we will stop the program and the values (slope and intercept) calculated before (p) i.e, at point (B) will be the best slope and intecept with minimum Mean Squared Error
# 
# **Note:** This logic is only applicable for concave models only.

# In[ ]:


def gradient_descent_runner(points, starting_b, starting_m, learning_rate):
    b = starting_b
    m = starting_m
    checker = True
    while(checker):
        b_pre, m_pre = b,m
        error_bef_grad = mean_squared_error(b, m, points)
        b, m = batch_gradient(b, m, array(points), learning_rate)
        error_aft_grad = mean_squared_error(b, m, points)
        if error_aft_grad > error_bef_grad:
            checker = False
    plot_best_fit(b_pre,m_pre)
    return[b_pre,m_pre]


# **Function to Wrap and implement all the above fundctions:**

# In[ ]:


def main_fun():
    points = genfromtxt("../input/distancecycledvscaloriesburned/data.csv", delimiter=",")
    plt.plot(points[:,0], points[:,1], '.')
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, mean_squared_error(initial_b, initial_m, points)))
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate)
    print("After Gradient descent b = {0}, m = {1}, error = {2}".format(b, m, mean_squared_error(b, m, points)))


# In[ ]:


main_fun()

