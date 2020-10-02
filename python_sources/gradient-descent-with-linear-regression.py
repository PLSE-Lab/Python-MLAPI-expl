#!/usr/bin/env python
# coding: utf-8

# ## Understanding Gradient Descent with Linear Regression

# Gradient Descent is a optimization algorithm that is used in most Machine Learning models. It starts with a set of initial parameter values and iteratively moving these set of parameter values that minimize a cost function. We achieve the best-fit parameter values by taking the partial derivatives of the parameters. 
# 
# We can use Linear Regression to demonstrate how Gradient Descent can fit a line to a set of data by finding the optimal values (minimized cost function) for the 2 parameters: Intercept and the Slope. The linear regression predictive model is  
# 
# $$ y = mx + b $$
# 
# where m and b are the parameters.

# ### A Linear Regression Example
# Let set up a pair of data sets (X,Y) and each set has 100 data points and it will look something like this:

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(42)
X = 2 * np.random.rand(100,1)
Y = 10 + 4 * X+np.random.randn(100,1) 

plt.style.use('seaborn')
plt.scatter(X, Y, color='black')
plt.title('Training Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


# Let's start with the cost error function, and we will use the Mean Sum Error (MSE). Given a set of values $y_i$, and a set of predicted values $\widehat{y_i}$.
# First we find the error of each data in the 2 sets ($y_i - \widehat{y_i}$), sum up the differences and find the average.
# $$J_{m,b} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \widehat{y_i})^2 $$
# 

# In this case, the predicted values $\widehat{y_i}$ can be found using the linear regression model, and the MSE becomes:
# $$J_{m,b} = \frac{1}{N} \sum_{i=1}^{N} (y_i - (mx_i + b))^2 $$

# The following is MSE implementation when the parameter values are 0 (init_b, init_m). For demonstration, it is implemented iteratively using a loop.

# In[ ]:


n = float(len(X)) # number of data pair
init_b = 0
init_m = 0

def compute_cost_function_point_by_point(m, b, X, Y):
    totalError = 0
    
    for i in range(len(X)):
        x = X[i,0]
        y = Y[i,0]
        totalError += (y - (m * x + b)) ** 2 
        
    return totalError / n

compute_cost_function_point_by_point(init_m,init_b,X,Y)


# And you can see that the result of the cost function is at ~195.1. It means that the error value of the predictive model is at ~195.1 when $m = 0$ and $b = 0$. Remember the goal of this experiment is to minimize this cost function. Now we don't know if it is the best estimation this model can do. So what do we do? We try with different parameter values.
# But how would we do that?
#  
# This is when Gradient Descent and a bit of Calculus concept come in. One way to find any local minimal value is to find the point where the gradient converges. Since we have 2 parameters, we will take 2 partial derivatives.

# $$\frac{\partial J_{m,b}}{\partial m} = -\frac{2}{N} \sum_{i=1}^{N} x_i (y_i - (mx_i + b)) $$
# 
# $$\frac{\partial J_{m,b}}{\partial b} = -\frac{2}{N} \sum_{i=1}^{N} (y_i - (mx_i + b)) $$
# 
# Each iteration, we take a step in the negative gradient and eventually we willa reach a point where it converges. 
# 
# Let look at a implementation below.

# In[ ]:


learning_rate = 0.01


# Learning rate is a hyperparameter, usually ranged between 0 and 1, and it controls how quickly the model is adapted to the problem, or the step size of each iteration. The smaller the error is, the smaller step it will take toward convergence.

# In[ ]:


def compute_gradient_step_by_step(m,b,X,Y,learning_rate):
    b_gradient = 0
    m_gradient = 0
    for i in range(len(X)):
        x = X[i,0]
        y = Y[i,0]
        
        b_gradient += -(2/n) * (y - ((m * x) + b))
        m_gradient += -(2/n) * x * (y - ((m * x) + b))
    new_b = b - (learning_rate * b_gradient)
    new_m = m - (learning_rate * m_gradient)
    return [new_b,new_m]


# In[ ]:


num_iteration = 1000
min_step_size = 0.001


# And it is common practice to run at least 1000 iteration or stop with the gradient step size is less than 0.001

# In[ ]:


def gradient_descent(X,Y,starting_b,starting_m,learning_rate, num_iteration):
    b = starting_b
    m = starting_m
    i = 0
    while i in range(num_iteration) or (b < min_step_size and m < min_step_size):
        b, m = compute_gradient_step_by_step(m, b, X, Y, learning_rate)
        i += 1
    return [b,m]


# In[ ]:


def calculate_predicted_values(X, opt_b, opt_m):
    return X*opt_m + opt_b


# In[ ]:


[g_b, g_m] = gradient_descent(X,Y,init_b,init_m,learning_rate, num_iteration)


# In[ ]:


opt_X = calculate_predicted_values(X, g_b, g_m)


# In[ ]:


plt.style.use('seaborn')
plt.scatter(X, Y, color='black')
plt.plot(X, opt_X)
plt.title('Linear Regression with Gradient Descent ')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()

