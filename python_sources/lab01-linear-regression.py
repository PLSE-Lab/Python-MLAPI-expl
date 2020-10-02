#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

print(os.listdir('../input'))


# # Linear Regression
# We use linear regression if we think the data is a linear relationship.
# Simple linear regression model can be described as:
# ## y=mx+b
# 
# *   where m is the slope 
# *   and b is the bias or y intercept
# 
# ![alt text](https://miro.medium.com/max/700/1*6UUSIyncWsJ-0vJKwEcWrw.png)

# # Cost function (Mean squared error, MSE) 
# ![Error function](https://spin.atomicobject.com/wp-content/uploads/linear_regression_error1.png)
# 
# ![alt text](https://miro.medium.com/max/1400/1*A71zTD6_QqUzLhMKj1Rgiw.png)

# In[ ]:


# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0] #get the x value
        y = points[i, 1] #get the y value
        #get sum squared error
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points)) #get the average or MSE


# # Gradient descent
# ![Optimization error](https://media.giphy.com/media/O9rcZVmRcEGqI/giphy.gif)

# ## Partial derivatves of the cost function
# 
# ![Partial derivative](https://spin.atomicobject.com/wp-content/uploads/linear_regression_gradient1.png)

# ## Finding the optimal values for m and b
# 
# ![alt text](https://hackernoon.com/hn-images/0*8yzvd7QZLn5T1XWg.jpg)

# In[ ]:


def step_gradient(b_current, m_current, points, learningRate):
   
    #starting point for gradients
    b_gradient = 0
    m_gradient = 0
    
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        #direction with respect to b and m
        #computing partial derivatives of cost function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    
    #update the b and m values using the partial derivatives
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]


# In[ ]:


def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    #starting b and m
    b = starting_b
    m = starting_m
    
    #gradient descent
    for i in range(num_iterations):
        #update b and m with new more accurate b and m using gradient step
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return [b, m]


# In[ ]:


import os
import numpy as np
import matplotlib.pyplot as plt

file_path = "../input/"

#Step 1 - collect data
points = np.genfromtxt(os.path.join(file_path,'data_test.csv'), delimiter=',')

# Plot the data
x = points[:, 0]
y = points[:, 1]
plt.scatter(x, y, c='blue', alpha=0.8)
plt.title('Scatter plot of data')
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[ ]:


#Step 2 - define hyperparameters
#How fast should the model converge?
learning_rate = 0.0001
#y = mx + b
initial_b = 0 # initial y-intercept guess
initial_m = 0  # initial slope guess
num_iterations = 100000

#Step 3 - train the model
print("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
print("Running...")
[b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
print("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))


# In[ ]:


#Making predictions
y_predict = m*x + b

plt.scatter(x, y, c='blue', alpha=0.8)
plt.title('Linear regression')
plt.xlabel('x')
plt.ylabel('y')
plt.plot([min(x), max(x)], [min(y_predict), max(y_predict)], color='red')  # regression line
plt.show()


# # Check point 1
# Modify your code to minimize the error to less than 112.

# # Check point 2
# Predict new input values (x=[18, 65, 33])
# *   Show the result values
# *   Show the result positions in the regression line 
# 
# 

# In[ ]:


x_new=np.array([18, 65, 33])
y_predict_new = m*x_new + b
print(y_predict_new)

plt.title('Linear regression')
plt.xlabel('x')
plt.ylabel('y')
plt.plot([min(x_new), max(x_new)], [min(y_predict_new), max(y_predict_new)], color='red')  # regression line

plt.scatter(x_new, y_predict_new, c='green')
plt.show()


# # Linear Regression Using Sklearn Library
# Scikit learn provides you two approaches to linear regression:
# 
# *   **LinearRegression** uses least squares method
# *   **SGDRegressor** which is an implementation of stochastic gradient descent
# 
# 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
import math

#read .csv into a DataFrame
dataset = pd.read_csv(os.path.join(file_path,'house_prices.csv'))
size=dataset['sqft_living']
price=dataset['price']

#machine learing handle arrays not dataframes
x = np.array(size).reshape(-1,1)
y = np.array(price).reshape(-1,1)

#we use Linear Regression + fit() is the training
model = LinearRegression()
model.fit(x, y)

#MSE and R value
regression_model_mse = mean_squared_error(x, y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R squared value:", model.score(x,y))

#y=mx+b
#we can get the m and b values after the model fit
#this is the m
print(model.coef_[0])
#this is b in our model
print(model.intercept_[0])

#visualize the dataset with the fitted model
plt.scatter(x, y, color= 'green')
plt.plot(x, model.predict(x), color = 'black')
plt.title ("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

#Predicting the prices
print("Prediction by the model:" , model.predict([[2000]]))


# # Check point 3
# From the example above, Implement linear regression model with mutiple variable using Sklearn (LinearRegression)

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
import math

#read .csv into a DataFrame
dataset = pd.read_csv(os.path.join(file_path,'house_prices.csv'))
size=dataset[['bedrooms','sqft_living']]
price=dataset['price']

#machine learing handle arrays not dataframes
x = np.array(size).reshape(-1,2)
y = np.array(price).reshape(-1,1)

#we use Linear Regression + fit() is the training
model = LinearRegression()
model.fit(x, y)

#R value
print("R squared value:", model.score(x,y))

#y = m1x1 + m2x2 + b
#we can get the m and b values after the model fit
#this is the m
print(model.coef_[0])
#this is b in our model
print(model.intercept_[0])
#Predicting the prices
print("Prediction by the model:" , model.predict([[2, 2000]]))


# # Check point 4
# From the example above, Implement linear regression model with mutiple variable using Sklearn (SGDRegressor)

# In[ ]:


import numpy as np
from sklearn import linear_model


#read .csv into a DataFrame
dataset = pd.read_csv(os.path.join(file_path,'house_prices.csv'))
size=dataset[['bedrooms','sqft_living']]
price=dataset['price']

#machine learing handle arrays not dataframes
x = np.array(size).reshape(-1,2)
y = np.array(price)

model = linear_model.SGDRegressor(max_iter=1000, tol=1e-3, alpha=0.0001)
model.fit(x, y)

#R value
print("R squared value:", model.score(x,y))

#y = m1x1 + m2x2 + b
#we can get the m and b values after the model fit
#this is the m
print(model.coef_[0])
#this is b in our model
print(model.intercept_[0])
#Predicting the prices
print("Prediction by the model:" , model.predict([[2, 2000]]))


# # Check point 5
# Implement non-linear regression model using house_prices.csv as the traning data

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error 
import math

#read .csv into a DataFrame
dataset = pd.read_csv(os.path.join(file_path,'house_prices.csv'))
size=dataset['sqft_living']
price=dataset['price']

#machine learing handle arrays not dataframes
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x = np.array(size).reshape(-1,1)
x_poly = poly_features.fit_transform(np.array(size).reshape(-1,1))
y = np.array(price).reshape(-1,1)

#we use Linear Regression + fit() is the training
model = LinearRegression()
model.fit(x_poly, y)

#R value
print("R squared value:", model.score(x_poly,y))

#y=mx+b
#we can get the m and b values after the model fit
#this is the m
print(model.coef_[0])
#this is b in our model
print(model.intercept_[0])

#visualize the dataset with the fitted model
plt.scatter(x, y, color= 'green')

x_new=np.linspace(0, 15000, 15000).reshape(15000, 1)
x_new_poly = poly_features.transform(x_new)
y_new = model.predict(x_new_poly)
plt.plot(x_new, y_new, "r-", linewidth=2, label="Predictions")
plt.title ("Linear Regression")
plt.xlabel("Size")
plt.ylabel("Price")
plt.show()

