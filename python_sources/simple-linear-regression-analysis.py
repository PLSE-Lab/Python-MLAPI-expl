#!/usr/bin/env python
# coding: utf-8

# **Background:**
# X is the temperature in fahrenheit
# Y is how many times a group of crickets chirp per minute
# 
# Let's calculate the ordinary least squares regression to predict how many chirps per minute are the most common per degree fahrenheit.

# **Outline:**
# * Assign variables for the independent and dependent columns
# * Plot a scattergraph of the data
# * Run the linear regression by:
#     * Creating a new variable (x1) which is composed x and a constant of 1 next to every x.
#     * Create another variable to run the regression. Use the x1 instead of x for this. 
# * Look over the report
# * Calculate the same key metrics using scipy.linregress().
# * Create a def for the linear function
# * Plot the best fit line
# 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# ##Part 1: 
# * Assign variables for the independent and dependent columns
# * Plot a scattergraph of the data
# 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# Import modules and data:
# You'll notice I imported "sklearn.linear_model". We won't be using that, but it's another option to calculate linear regression. 

# In[ ]:


import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('../input/Cricket_chirps.csv')
data


# Assign variables for the independent and dependent variables:

# In[ ]:


X = data['X']
Y = data['Y']
print(data['X'],data['Y'])


# Plot the scattergraph. I like to change the axes so that it starts at zero; it gives more perspective about the data. As the data shows, crickets can only live in temperatures between 60 and 90 degrees F. 

# In[ ]:


plt.scatter(X,Y)
plt.axis([0,95,0,25])
plt.ylabel('Chirps/second')
plt.xlabel('Temperature in F')
plt.show()


# ##Part 2:
# * Run the linear regression by:
#     * Creating a new variable (x1) which is composed x and a constant of 1 next to every x.
#     * Create another variable to run the regression. Use the x1 instead of x for this. 
# * Look over the report
# * Calculate the same key metrics using scipy.linregress().
# 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# 

# We need to set the framework for the constant coefficient to be calculated in the regression equation, so we will create a new variable called X1, which has a row of 1's next to it.

# In[ ]:


X1 = sm.add_constant(X)
X1


# Right after we do that, we will create another variable named results. It will contain the output of the ordinary least squares regression, or OLS. As arguments, we must add the dependent variable y and the newly defined x. At the end, we will need the .fit() method to apply the specific estimation technique to obtain the fit of the model.

# In[ ]:


reg = sm.OLS(Y,X1).fit()
reg.summary()


# Another way to calculate variables with each of the figures is to use the scipy.linregress function.

# In[ ]:


slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)


# In[ ]:


slope


# In[ ]:


intercept


# In[ ]:


r_value**2


# In[ ]:


p_value


# In[ ]:


std_err


# ##Part 3:
# * Create a def for the linear function
# * Plot the best fit line
# 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

# It can be helpful to write a def to input whatever x you want and get a y. 

# In[ ]:


def calc_crick(x):
    return intercept + (x * slope)

calc_crick(85)


# In order to graph the line of best fit, we need to write a def also. The parameter (b) will be filled in with every x value and create the best fit y number for each value. This will be plotted alongside the x on a graph to form the line. 
# 
# 

# In[ ]:


def fitline(b):
    return intercept + slope * b

line = fitline(X)

plt.scatter(X,Y)
plt.plot(X,line)
plt.show()

