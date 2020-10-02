#!/usr/bin/env python
# coding: utf-8

# <h1> Welcome to this Kernel </h1>

# <p>Data Analytics, we often use <b>Model Development</b> to help us predict future observations from the data we have.</p>
# <p>So, a Model will help us understand the exact relationship between different variables and how these variables are used to predict the result.</p>

# <h4> setup </h4>

# Import libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import matplotlib.pyplot as plt


# load data and store in dataframe df:

# In[ ]:


path="../input/crimes-in-usa-from-1960-to-2014/crime.csv"
df = pd.read_csv(path)
df.head()


# <h2>1. Linear Regression </h2>

# 
# <p>One example of a Data  Model that we will be using is</p>
# <b>**Simple Linear Regression**</b>.
# 
# <br>
# <p>Simple Linear Regression is a method to help us understand the relationship between two variables:</p>
# <ul>
#     <li>The predictor/independent variable (X)</li>
#     <li>The response/dependent variable (that we want to predict)(Y)</li>
# </ul>
# 
# <p>The result of Linear Regression is a <b>linear function</b> that predicts the response (dependent) variable as a function of the predictor (independent) variable.</p>
# 
# 

# $$
#     Y : Target  \ Variable \\
#     X : Predictor \ Variables \\
# $$

#  <b>**Linear function:**</b>
# $$
# Yp = a + b  X
# $$

# <ul>
#     <li> a refers to the <b> intercept </b> of the regression line0 , in others words : the value of Y when X=0 </li>
#     <li> b refers to the <b> slope </b> of the regression

# <h4>Lets now load the modules for linear regression</h4>

# In[ ]:


from sklearn.linear_model import LinearRegression


# ** Create the linear regression object**

# In[ ]:


lr=LinearRegression()


# <h3>How could Burglary help us predict violent crime total in USA?</h3>

# For this example, we want to look at how Burglary can help us predict violent crime total.
# Using simple linear regression, we will create a linear function with "Bulgary" as the predictor variable and the "Violent crime total" as the response variable.

# In[ ]:


X = df[['Burglary']]
Y = df['Violent crime total']


# Fit the linear model using Burglary.

# In[ ]:


lr.fit(X,Y)


# we can output a prediction.

# In[ ]:


Yp=lr.predict(X)
Yp[0:5]


# <h4>**What is the value of the intercept (a)?**</h4>

# In[ ]:


lr.intercept_


# <h4>**What is the value of the slope (b)?**</h4>

# In[ ]:


lr.coef_


# <h3>****What is the final estimated linear model we get?****</h3>

# As we saw above, we should get a final linear model with the structure:

# $$
#    Yp = a + b X
#  $$

# Plugging in the actual values we get:

# <b>**Violent crime total**</b> = 196358.04 - 0.409 x  <b>**Bulgary**</b>

# <h2>2. Multiple Linear Regression</h2>

# <p>What if we want to predict Violent crime total using more than one variable?</p>
# 
# <p>If we want to use more variables in our model to predict Violent crime total, we can use <b>**Multiple Linear Regression**</b>.
# Multiple Linear Regression is very similar to Simple Linear Regression, but this method is used to explain the relationship between one continuous response or target (dependent) variable and <b>two or more</b> predictor (independent) variables.
# Most of the real-world regression models involve multiple predictors. We will illustrate the structure by using four predictor variables, but these results can generalize to any integer:</p>

# $$
# Y: Target \ Variable\\
# X_1 :Predictor\ Variable \ 1\\
# X_2: Predictor\ Variable \ 2\\
# X_3: Predictor\ Variable \ 3\\
# X_4: Predictor\ Variable \ 4\\
# $$

# $$
# a: intercept\\
# b_1 :coefficients \ of\ Variable \ 1\\
# b_2: coefficients \ of\ Variable \ 2\\
# b_3: coefficients \ of\ Variable \ 3\\
# b_4: coefficients \ of\ Variable \ 4\\
# $$

# The equation is given by:

# $$
# Yp = a + b_1 X_1 + b_2 X_2 + b_3 X_3 + b_4 X_4
# $$
