#!/usr/bin/env python
# coding: utf-8

# # Sales Prediction
# ## ( Simple Linear Regression)

# ### Problem Statement
# 
# Build a model which predicts sales based on the money spent on different platforms for marketing.
# 
# ### Data
# Use the advertising dataset given in ISLR and analyse the relationship between 'TV advertising' and 'sales' using a simple linear regression model. 
# 
# In this notebook, we'll build a linear regression model to predict `Sales` using an appropriate predictor variable.

# ## Reading and Understanding the Data

# In[ ]:


# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package

import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns


# In[ ]:


advertising = pd.DataFrame(pd.read_csv("../input/advertising.csv"))
advertising.head()


# ## Data Inspection

# In[ ]:


advertising.shape


# In[ ]:


advertising.info()


# In[ ]:


advertising.describe()


# ## Data Cleaning

# In[ ]:


# Checking Null values
advertising.isnull().sum()*100/advertising.shape[0]
# There are no NULL values in the dataset, hence it is clean.


# In[ ]:


# Outlier Analysis
fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(advertising['TV'], ax = axs[0])
plt2 = sns.boxplot(advertising['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(advertising['Radio'], ax = axs[2])
plt.tight_layout()


# In[ ]:


# There are no considerable outliers present in the data.


# ## Exploratory Data Analysis

# ### Univariate Analysis

# #### Sales (Target Variable)

# In[ ]:


sns.boxplot(advertising['Sales'])
plt.show()


# In[ ]:


# Let's see how Sales are related with other variables using scatter plot.
sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


# In[ ]:


# Let's see the correlation between different variables.
sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)
plt.show()


# As is visible from the pairplot and the heatmap, the variable `TV` seems to be most correlated with `Sales`. So let's go ahead and perform simple linear regression using `TV` as our feature variable.

# ## Model Building

# ### Performing Simple Linear Regression

# Equation of linear regression<br>
# $y = c + m_1x_1 + m_2x_2 + ... + m_nx_n$
# 
# -  $y$ is the response
# -  $c$ is the intercept
# -  $m_1$ is the coefficient for the first feature
# -  $m_n$ is the coefficient for the nth feature<br>
# 
# In our case:
# 
# $y = c + m_1 \times TV$
# 
# The $m$ values are called the model **coefficients** or **model parameters**.
# 
# ---

# ### Generic Steps in model building using `statsmodels`
# 
# We first assign the feature variable, `TV`, in this case, to the variable `X` and the response variable, `Sales`, to the variable `y`.

# In[ ]:


X = advertising['TV']
y = advertising['Sales']


# #### Train-Test Split
# 
# You now need to split our variable into training and testing sets. You'll perform this by importing `train_test_split` from the `sklearn.model_selection` library. It is usually a good practice to keep 70% of the data in your train dataset and the rest 30% in your test dataset

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[ ]:


# Let's now take a look at the train dataset

X_train.head()


# In[ ]:


y_train.head()


# #### Building a Linear Model
# 
# You first need to import the `statsmodel.api` library using which you'll perform the linear regression.

# In[ ]:


import statsmodels.api as sm


# By default, the `statsmodels` library fits a line on the dataset which passes through the origin. But in order to have an intercept, you need to manually use the `add_constant` attribute of `statsmodels`. And once you've added the constant to your `X_train` dataset, you can go ahead and fit a regression line using the `OLS` (Ordinary Least Squares) attribute of `statsmodels` as shown below

# In[ ]:


# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()


# In[ ]:


# Print the parameters, i.e. the intercept and the slope of the regression line fitted
lr.params


# In[ ]:


# Performing a summary operation lists out all the different parameters of the regression line fitted
print(lr.summary())


# ####  Looking at some key statistics from the summary

# The values we are concerned with are - 
# 1. The coefficients and significance (p-values)
# 2. R-squared
# 3. F statistic and its significance

# ##### 1. The coefficient for TV is 0.054, with a very low p value
# The coefficient is statistically significant. So the association is not purely by chance. 

# ##### 2. R - squared is 0.816
# Meaning that 81.6% of the variance in `Sales` is explained by `TV`
# 
# This is a decent R-squared value.

# ###### 3. F statistic has a very low p value (practically low)
# Meaning that the model fit is statistically significant, and the explained variance isn't purely by chance.

# ---
# The fit is significant. Let's visualize how well the model fit the data.
# 
# From the parameters that we get, our linear regression equation becomes:
# 
# $ Sales = 6.948 + 0.054 \times TV $

# In[ ]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# ## Model Evaluation

# ### Residual analysis 
# To validate assumptions of the model, and hence the reliability for inference

# #### Distribution of the error terms
# We need to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of the error terms and see what it looks like.

# In[ ]:


y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[ ]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# The residuals are following the normally distributed with a mean 0. All good!

# #### Looking for patterns in the residuals

# In[ ]:


plt.scatter(X_train,res)
plt.show()


# We are confident that the model fit isn't by chance, and has decent predictive power. The normality of residual terms allows some inference on the coefficients.
# 
# Although, the variance of residuals increasing with X indicates that there is significant variation that this model is unable to explain.

# As you can see, the regression line is a pretty good fit to the data

# ### Predictions on the Test Set

# Now that you have fitted a regression line on your train dataset, it's time to make some predictions on the test data. For this, you first need to add a constant to the `X_test` data like you did for `X_train` and then you can simply go on and predict the y values corresponding to `X_test` using the `predict` attribute of the fitted regression line.

# In[ ]:


# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)


# In[ ]:


y_pred.head()


# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ##### Looking at the RMSE

# In[ ]:


#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))


# ###### Checking the R-squared on the test set

# In[ ]:


r_squared = r2_score(y_test, y_pred)
r_squared


# ##### Visualizing the fit on the test set

# In[ ]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()

