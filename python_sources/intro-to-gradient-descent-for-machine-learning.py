#!/usr/bin/env python
# coding: utf-8

# **Optimization** is a big part of machine learning. Almost every machine learning algorithm has
# an optimization algorithm at it's core.So this is simple optimization
# algorithm that you can use with any machine learning algorithm.
# 
# **Intuition for Gradient Descent**
# Think of a large bowl like what you would eat serial out of or store fruit in. This bowl is a plot
# of the cost function (f). A random position on the surface of the bowl is the cost of the current
# values of the coeffecients (cost). The bottom of the bowl is the cost of the best set of coefficients,
# the minimum of the function.
# The goal is to continue to try defferent values for the coeffecients, evaluate their cost and
# select new coefficients that have a slightly better (lower) cost. Repeating this process enough
# times will lead to the bottom of the bowl and you will know the values of the coefficients that
# result in the minimum cost.
# 
# **Gradient Descent**
# Gradient descent is an optimization algorithm used to find the values of parameters (coefficients)
# of a function (f) that minimizes a cost function (cost). Gradient descent is best used when the
# parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched
# for by an optimization algorithm.
# 
# > Gradient descent is the most popular optimization strategy in deep learning, in particular an implementation of it called backpropagation. We are using gradient descent as our optimization strategy for linear regression.
# 
# **Gradient Descent Procedure**
# The procedure starts of with initial values for the coeffcients for the function.These could be 0.0 or a small random value.
# coeffcient = 0:0 
# The cost of the coeffcients is evaluated by plugging them into the function and calculating
# the cost.
#  cost = f(coeffcient)
#  
#  cost = evaluate(f(coeffcient))
# The derivative of the cost is calculated. The derivative is a concept from calculus and refers
# to the slope of the function at a given point. We need to know the slope so that we know the
# direction (sign) to move the coeffcient values in order to get a lower cost on the next iteration.
# 
# delta = derivative(cost) 
# Now that we know from the derivative which direction is downhill, we can now update the
# coeffcient values. A learning rate parameter (alpha) must be specifed that controls how much
# the coeffcients can change on each update.
# 
# coeffcient = coeffcient - (alpha * delta)
# his process is repeated until the cost of the coeffcients (cost) is 0.0 or no further improve-
# ments in cost can be achieved.

# > Implementing the alogrithm on small dataset which is linear. this is just to demonstrate gradient descent.
# 

# By applying gradeint descent finding the optimal values of m and  below is the code.

# Here are some helpful links:
# 
# Gradient descent visualization
# https://raw.githubusercontent.com/mattnedrich/GradientDescentExample/master/gradient_descent_example.gif
# 
# Sum of squared distances formula (to calculate our error)
# https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png
# 
# Partial derivative with respect to b and m (to perform gradient descent)
# https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
from numpy import *

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
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

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    points = genfromtxt("../input/linear_demo.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    #print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    print (f' after iteration={num_iterations}  b={b} m={m}', "error=",compute_error_for_line_given_points(b, m, points)) 
    
    x,y = genfromtxt("../input/linear_demo.csv",unpack=True, delimiter=",")
    plt.scatter(x,y)
    new_y=m*x+b
    plt.plot(x, new_y, '-b')
    plt.show()
if __name__ == '__main__':
    run()


# In[ ]:




