#!/usr/bin/env python
# coding: utf-8

# # Linear Regression Project
# ## Tracy(11/13/2019)_USA Housing Price Prediction
# 
# I want to create a model to put in a few features of a house and returns back an estimate house price. The Linear Regression should be a good path to solve this problem.
# This dataset about a bunch of houses in regions of the United States.
# The data contains the following columns.
# * 'Avg. Area Income': Avg Income of residents of the city house is located in.
# * 'Avg. Area House Age': Avg Age of Houses in same city
# * 'Avg. Area Number of Rooms': Avg Number of Rooms for Houses in same city
# * 'Avg. Area Number of Bedrooms': Avg Number of Bedrooms for Houses in same city
# * 'Area Population': Population of city house is located in
# * 'Price': Price that the house sold at
# * 'Address': Address for the house

# ## Import Libraries and Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


Housing = pd.read_csv('/kaggle/input/usa-housing/USA_Housing.csv')


# ## Data Overview

# In[ ]:


Housing.info()


# In[ ]:


Housing.head()


# In[ ]:


Housing.describe()


# In[ ]:


Housing.columns


# ## Exploratory Data Analysis

# In[ ]:


sns.pairplot(Housing)


# In[ ]:


sns.distplot(Housing['Price'])


# In[ ]:


sns.heatmap(Housing.corr())


# ## Training a Linear Regression Model

# In[ ]:


Housing.columns


# In[ ]:


## First split up data into X array that contains the features to train on, and y array with the target variable.
## I will toss out the Address column because it onlu has text info that th elinear regression model can not use.
X = Housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = Housing['Price']


# In[ ]:


## Split the data into a training set and a testing set.
## I will train model on the training set and then use the test set to evaluate the model
from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# ### Create and Train Model

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm = LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# ### Model Evaluation

# In[ ]:


## Check out coefficients and how we can interpret them.
print(lm.intercept_)


# In[ ]:


coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


# Interpreting the coefficients:
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Income** is associated with an **increase of \$21.63 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area House Age** is associated with an **increase of \$164890.44 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Rooms** is associated with an **increase of \$121297.15 **.
# - Holding all other features fixed, a 1 unit increase in **Avg. Area Number of Bedrooms** is associated with an **increase of \$1814.58 **.
# - Holding all other features fixed, a 1 unit increase in **Area Population** is associated with an **increase of \$15.24 **.

# ### Predictions from Model

# In[ ]:


predictions = lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:


## Residual Histogram
sns.distplot((y_test-predictions),bins=50)


# ## Regression Evaluation Metrics
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# 
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# **Mean Squared Error** (MSE) is the mean of the squared errors:
# 
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# Comparing these metrics:
# 
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
# 
# All of these are **loss functions**, because we want to minimize them.

# In[ ]:


from sklearn import metrics
print('MAE', metrics.mean_absolute_error(y_test, predictions))
print('MSE', metrics.mean_squared_error(y_test, predictions))
print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

