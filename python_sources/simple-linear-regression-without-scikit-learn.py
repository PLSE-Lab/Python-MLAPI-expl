#!/usr/bin/env python
# coding: utf-8

# I implementated this algorithm after going throug the Section 1 of Andrew NG's lecture (about Cost Function and Gradient Descent) and Siraj Raval's lecture for implementation of same.

# In[ ]:


# import numpy
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


# This function will plot a best fit line at last
def plot_fit_line(slope, intercept):
    # Plot a line from slope and intercept
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, 'r-')


# In[ ]:


#This is how we calculate and correct our error
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    # itterating for all points
    for i in range (0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        hypothesis = (m * x) + b
        totalError += (hypothesis - y) ** 2
    return totalError/ (2 * float(len(points)))


# In[ ]:


def step_gradient(b_current, m_current, points, learning_rate):
    #gradient descent
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    
    # Here we are just performing the summation
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # b = b + 2((mx+b) - y)
        b_gradient += 2 * ((m_current * x + b_current) - y)
        # m = m + 2((mx+b) - y)x
        m_gradient += 2 * ((m_current * x + b_current) - y) * x

    new_b = b_current - (learning_rate * (1/(2 * N)) * b_gradient)
    new_m = m_current - (learning_rate * (1/(2 * N)) * m_gradient)

    return [new_b, new_m]


# In[ ]:


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m

    # As specified in run(), will iterate 1000 times
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)

    # plot a fit line
    plot_fit_line(m, b)

    # returning calculated final intercept point and slope     
    return [b, m]


# In[ ]:


def run():
    # load dataset    
    points = np.genfromtxt('../input/slr_data.csv', delimiter=',', skip_header=True)
    
    # plot all the actual datapoints 
    plt.plot(points[:,0], points[:,1], '+')
    
    # rate of learning      
    learning_rate = 0.0001

    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000

    print(f"Starting gradient descent at b = {initial_b}, m = {initial_m}, error mean = {compute_error_for_line_given_points(initial_b, initial_m, points)}")
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print(f"After {num_iterations} iterations b = {b}, m = {m}, error mean = {compute_error_for_line_given_points(b, m, points)}")


# In[ ]:


run()


# In[ ]:




