#!/usr/bin/env python
# coding: utf-8

# **Logistic Regression: One-Vs-All implemented from scratch**
# 
# This is my first implementation of Logistic Regression One-Vs-All or One-Vs-Rest approach, for multiclass classification. This simple implementation uses only numpy for matrix operations and scipy.optimize to find global minima using advanced optimization algorithms (BFGS, CG etc). No other in-built modules or functions have been used.
# 
# *NOTE: This implementation is for the sole purpose of validating my theoritical understanding of logistic regression and gradient computation. I understand that in reality more advanced algorithms and optimized pre-built libraries are used for better prediction results. *
# 
# Functions implemented:
# * Sigmoid Function - to calculate Sigmoid value of a scalar, vector or matrix
# * Cost Function - to calcuate cost of regression function for given values of coefficients. Objective is to find optimized coefficients which will minimize cost.
# *  ComputeGradient - to compute the gradient or slope of cost function for given values of coefficients. Objective is to find the global minima, i.e. parameters for which the slope/derivative is minimum. 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import optimize # to use minimize function

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))


# **The following function is to calculate value of Sigmoid Function of any variable z.
# z can be a matrix, vector or scalar.
# > sigmoid g(z) = 1/(1 + e^-z)

# In[ ]:


# =============================================================================
# Function to calculate value of Sigmoid Function of any variable z.
# z can be a matrix, vector or scalar
# sigmoid g(z) = 1/(1 + e^-z)
# =============================================================================
def sigmoid(z):
    sig = 1.0/(1.0 + np.exp(-z))
    
    # Due to floating point presision related issues, e^-z might return very 
    # small or very large values, resulting in sigmoid = 1 or 0. Since we will
    # compute log of these values later in cost function, we want to avoid 
    # sig = 1 or 0, and hardcode to following values instead.
    sig[sig == 1.0] = 0.9999
    sig[sig == 0.0] = 0.0001
    
    return sig


# This function computes cost of Logistic Regression with multiple features - Vectorized implementation.
# > Input: data_X = mxn matrix, data_y = m-dim vector, theta = n-dim vector <br>
# > Output: cost = 1-dim vector

# In[ ]:


# =============================================================================
# Compute cost of Logistic Regression with multiple features
# Vectorized implementation
# Input: data_X = mxn matrix, data_y = m-dim vector, theta = n-dim vector
# Output: cost = 1-dim vector
# =============================================================================
def computeCost(theta, data_X, data_y, lambda_reg = 0):
    m = len(data_X) # No of rows
    n = len(data_X[0]) # No of features
    theta = theta.reshape(n,1)
    
    # h(x) = g(z) = g(theta0 + theta1*X1 + theta2*X2 + .. + thetan*Xn)
    # h(x) = g(X * theta) = Sigmoid(X * theta) = m-dim vector
    hx = sigmoid(np.dot(data_X, theta))
    cost = - np.dot(data_y.T, np.log(hx)) - np.dot((1 - data_y).T, np.log(1 - hx))
    
    # This is unregularized cost
    J = cost/m
    
    # Adding regularization. Setting theta0 to 0, because theta0 will not be 
    # regularized
    J_reg = (lambda_reg/(2*m)) * np.dot(theta[1:,:].T, theta[1:,:])
    J = J + J_reg
    
    return J


# This function computes gradient or derivative of cost function over parameter, i.e.
# > d J(Theta)/d Theta
# 
# This is the objective function which returns both gradient and cost.

# In[ ]:


# =============================================================================
# Compute gradient or derivative of cost function over parameter, i.e.
# d J(Theta)/d Theta
# =============================================================================
def computeGradient(theta, data_X, data_y, lambda_reg = 0):
    m = len(data_X) # No of rows
    n = len(data_X[0]) # No of features
    theta = theta.reshape(n,1)
    theta_gradient = np.zeros(theta.shape)
    cost = 0
    #print("==== Inside computeGradient() ====", data_X.shape, data_y.shape)

    cost = computeCost(theta, data_X, data_y, lambda_reg)
    
    hx = sigmoid(np.dot(data_X, theta))
    error = hx - data_y
    theta_gradient = (1/m) * (np.dot(data_X.T, error))
    
    # Apply regularization
    theta_reg = (lambda_reg/m) * theta[1:,:]
    theta_gradient[1:,:] = theta_gradient[1:,:] + theta_reg
    
    #print("==== Inside computeGradient() ====", cost)
    return cost.flatten(), theta_gradient.flatten()


# This is the One vs All method of logistic regression. We can use CG or BFGS for optimization. Also, experiment with the regression parameter lambda and number of iterations.

# In[ ]:


# =============================================================================
# One vs All method of logistic regression
# Used for data with multiple clssification outputs
# =============================================================================
def oneVsAll(data_X, data_y, num_labels, lambda_reg):
    n = data_X.shape[1] # No of features
    all_theta = np.zeros([num_labels, n])
    initial_theta = np.zeros([n, 1])
    
    for label in range(num_labels):
        theta_optimized = optimize.minimize(             computeGradient,             initial_theta,             args=(data_X, data_y == label, lambda_reg),             method = "CG",             jac=True, options={'disp': True, 'maxiter': 150}             )
        #print("OneVsAll: Optimization Result =", theta_optimized)
        theta = theta_optimized.x.reshape(n, 1)
        all_theta[label,:] = theta.T

    return all_theta


# In[ ]:


# Load training data
def loadTrainingData(path):
    train_data = pd.read_csv(path)
    #print(train_data.isnull().sum())
    return train_data


# Testing the implementation.
# 1. Load training data
# 2. Copy target variable and drop from dataset
# 3. Add intercept column with value 1
# 4. Call OneVsAll function to find optimized value of parameter
# 5. Load test data and add intercept
# 6. Predict output on test data using parameter values learnt

# In[ ]:


# Test my implementation of one-vs-all logistic regression algorithm
def test_OneVsAll():
    # This dataset is downloaded from Kaggle
    train_data = loadTrainingData('../input/train.csv')

    # Total number of records
    m = len(train_data)
    
    # Populate y data into a m-dim vector
    # And then drop that column from feature list
    num_labels = len(train_data.label.unique())
    data_y = train_data.label.values.reshape(m, 1)
    train_data = train_data.drop('label', 1)
    
    # Setting first feature to 1, this is the bias/y-intercept or theta0
    train_data.insert(0, 'first_dummy_feature', 1)

    # Populate X (features) data into a mxn matrix
    data_X = train_data.values
    
    # Call one-vs-all calculation
    lambda_reg = 0.5
    all_theta = oneVsAll(data_X, data_y, num_labels, lambda_reg)
    print("OneVsAll: Theta after Advanced Optimization =", all_theta.shape)
    
    # Predict results of test data
    test_data = loadTrainingData('../input/test.csv')
    test_data_m = len(test_data)
    test_data.insert(0, 'first_dummy_feature', 1)
    test_data_X = test_data.values
    
    Z = sigmoid(np.dot(test_data_X, all_theta.T))
    prediction = np.argmax(Z, axis=1)
    print("OneVsAll: Prediction Result =", prediction.shape)
    
    # Prepare submission file
    my_submission = pd.DataFrame({             'ImageId': np.arange(1, test_data_m+1),             'Label': prediction.flatten()})
    my_submission.to_csv('DG_submission.csv', index=False)


# In[ ]:


test_OneVsAll()


# This implementation was for learning purpose only. Please share you thoughts on this can be improved further.
