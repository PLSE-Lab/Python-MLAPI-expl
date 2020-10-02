#!/usr/bin/env python
# coding: utf-8

# # Linear Regression
# [Index](https://www.kaggle.com/veleon/in-depth-look-at-machine-learning-models)
# 
# Linear Regression creates a linear model that makes a prediction by computing a weighted sum of the input features, plus a constant (called _bias term_). These weights combined with the constant create a formula that outputs the predictions.
# 
# The equation:
# 
# $ y = \theta_0+\theta_1x_1+\theta_2x_2+\dotsb+\theta_nx_n$
# 
# * $y$ = predicted value
# * $n$ = number of features
# * $x_i$ = the $i^{th}$ feature value
# * $\theta_j$ = the $j^{th}$ model parameter
# 
# The equation using vectors:
# $ y = \begin{bmatrix}\theta_0\\\theta_1\\\vdots\\\theta_n\end{bmatrix}\cdot\begin{bmatrix}x_0\\x_1\\\vdots\\x_n\end{bmatrix}$
# 
# 
# ## Creating a Dataset
# To try our Linear Regression model we'll need data to use. Using the numpy library we'll generate some random points.

# In[ ]:


import numpy as np
np.random.seed(42)
X = np.random.rand(100,1)
y = 2 + X+np.random.randn(100,1)/7.5


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(X, y, 'ro')
plt.show()


# ## Computing Model Parameters/Weights 
# Now lets compute $\theta$ using the Normal Equation.
# 
# The Normal Equation:
# 
# $ \hat{\theta} = (X^TX)^{-1}X^Ty $
# 
# * $\hat{\theta}$ = the value of $\theta$ that minimizes the cost function
# * $y$ = the vector of target values containing $y^1$ to $y^m$
# * $X$ = the matrix combining all feature values (excluding labels) of all instances in the dataset

# In[ ]:


def computeModelParameters(X,y):
    X_b = np.c_[np.ones((100,1)), X] # concatenate a weight of 1 to each instance
    optimal_theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return optimal_theta

theta = computeModelParameters(X,y)
theta


# So what did we expect? Our formula was $y = 2+x+noise$
# 
# Expectation:
# * $\theta_0 = 2$
# * $\theta_1 = 1$
# 
# Output:
# * $\theta_0 = 2.03$
# * $\theta_1 = 0.94$
# 
# I would say that's pretty close! The noise definitely made it harder to get our exact parameters but with some more data we should be able to get even closer to the original parameters.
# 
# ## Making Predictions
# Now that we have computed $\hat{\theta}$ we can try and predict our generated dataset. 

# In[ ]:


def predictY(x, theta): # predicts a single y value
    return theta[0]+theta[1]*x

def predictAllY(X, theta): # predicts all y values of a matrix
    X_b = np.c_[np.ones((len(X),1)), X] # concatenate 1's for theta_0 * x_0 (because x_0 doesn't exist in our data)
    y_predict = X_b.dot(theta)
    return y_predict
    
y_pred = predictAllY(X, theta)    


# In[ ]:


plt.plot(X, y, 'ro')
plt.plot(X, y_pred, '-')
plt.show()


# ## Comparing to Scikit-Learn
# Now let's compare our model to the Linear Regression model from the Scikit-Learn. 

# In[ ]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
print('Theta 0:', lin_reg.intercept_)
print('Theta 1:', lin_reg.coef_[0])


# Its exactly the same!
# 
# ## Measuring Performance
# To know how well our Linear Regression algorithm performs we'll need a performance measure. The Root Mean Square Error (RMSE) is commonly used for Regression Algorithms. So we'll need to find the values of $\theta$ that minimize the RMSE. But instead of doing that we'll minimize the Mean Square Error (MSE) because this is simpler and still leads to the same result because a smaller mean also equals a smaller root mean.
# 
# The MSE equation:
# 
# $MSE(X, h_\theta)=\frac{1}{m}\displaystyle\sum_{i=1}^{m}(\theta^Tx^{(i)}-y^{(i)})^2$
# 
# * $MSE(X, h_\theta)$ = the cost function measured on the set of examples using the hyptothesis
# * $X$ = matrix combining all feature values (excluding labels) of all instances in the dataset
# * $h_\theta$ = the systems prediction function, also called the hyptothesis
# * $m$ = the number of instances in the dataset
# * $\displaystyle\sum_{i=1}^{m}$ = sum of $(\theta^Tx^{(i)}-y^{(i)})^2$ for $i=1$ to $i=m$
# * $\theta^T$ = transposed vector of model parameters 
# * $x$ = vector of feature values (excluding labels) of the $i^{th}$ instance
# * $y$ = label of the $i^{th}$ instance
# 

# In[ ]:


def calculateMSE(X, y, theta):
    sum = 0
    m = len(X)
    X_b = np.c_[np.ones((m,1)), X] # concatenate 1's for theta_0 * x_0 (because x_0 doesn't exist in our data)
    for i in range(m):
        # Create Prediction Value
        pred = theta.T.dot(X_b[i])
        # Find the Error
        error = pred - y[i]
        # Square the Error
        sqrError = error**2
        # Add the sqrError Up
        sum += sqrError
    return (1/m)*sum[0]

calculateMSE(X,y,theta)


# Next we'll take a look at the Scikit-Learn MSE function.

# In[ ]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y_pred=y_pred, y_true=y)


# Just as before, the function we've created gives (almost) the same result! 

# ## Conclusion
# By creating our own Linear Regression model we have explored the basics of how a linear machine learning model works. We've computed the weights/parameters and succesfully predicted our data. Then we looked at the Scikit-Learn version of the model which computed the same weights/parameters as we did. After this we created our own Mean Square Error calculator function which also gave the same result as the Scikit-Learn version. This means, unless Scikit-Learn is wrong, that we've correctly used both formulas to implement our own Linear Regression model and MSE metric.
# 
# ### Next Kernel:
# [What Is Gradient Descent?](https://www.kaggle.com/veleon/what-is-gradient-descent)
