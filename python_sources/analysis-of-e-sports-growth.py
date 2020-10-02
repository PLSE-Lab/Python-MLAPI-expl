#!/usr/bin/env python
# coding: utf-8

# Hi Kaggle community!
# 
# This is a beginners analysis on the growth of esports community: it's audience and revenue. I will be applying the methods of machine learning to try to predict the trends, which the growth will follow.
# 
# Let me start by importing the necessary libraries:

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()


# ...and importing the dataset, then splitting it into columns.

# In[ ]:


df = pd.read_csv('/kaggle/input/esports-global-revenue-and-audience-in-20122019/esports_revenue_audience.csv')
year = df[['Year']]
revenue = df[['Estimated e-sports market revenue in million dollars']]
audience = df[['Audience']]
freq_viewers = df[['Frequent viewers in million']]
oc_viewers = df[['Occasional viewers in million']]


# To predict the trends, I will first plot "raw" data:
# 
# - revenue each year,
# - general audience each year,
# - frequent audience each year,
# - and lastly, occasional audience each year
# 
# Notice no data about year 2013, but that won't hinder our process.

# In[ ]:


# Revenue plot
plt.scatter(year,revenue, c="blue", marker="+")
plt.ylabel("E-sports revenue")
plt.xlabel("Year")
plt.title("Revenue each year")
plt.show()


# In[ ]:


# Audience plot
plt.scatter(year,audience, c="blue", marker="+")
plt.ylabel("E-sports audience")
plt.xlabel("Year")
plt.title("Audience each year")
plt.show()


# In[ ]:


# Frequent audience plot
plt.scatter(year,freq_viewers, c="blue", marker="+")
plt.ylabel("E-sports frequent audience")
plt.xlabel("Year")
plt.title("Frequent audience each year")
plt.show()


# In[ ]:


# Occasional audience plot
plt.scatter(year,oc_viewers, c="blue", marker="+")
plt.ylabel("E-sports occasional audience")
plt.xlabel("Year")
plt.title("Occasional audience each year")
plt.show()


# We can see a couple of things in there:
# 
# - revenue plot shows a linear trend, but not "simple linear", more like "polynomialy linear",
# - audience plots are mostly linear, with exception of year 2015, where frequent audience went above expected; that got corrected the following year though
# 
# Since the trends are linear, simple linear regression and polynomial regression should be enough for this problem. Let's start by analyzing the revenue:

# In[ ]:


### Analyzing revenue

# Importing a linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(year, revenue)

plt.scatter(year,revenue, c="blue", marker="+")
plt.plot(year, regressor.predict(year), c='blue')
plt.ylabel("E-sports revenue")
plt.xlabel("Year")
plt.title("Revenue by year")
plt.show()


# We can see, that as mentioned above - simple linear regression won't work - we should try polynomial one.

# In[ ]:


# Polynomial Regression
X = df.iloc[:,0].values # here, I am taking the year column as a simple array, rather than pandas object - else, the plotting won't work later
Y = df.iloc[:,1].values
X = X.reshape(-1, 1)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3) # degree = 2 is default, changing might raise accuracy
X_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, Y)

Y_pred = lin_reg.predict(X_poly)

# Grid rearange for accuracy
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)), 1)

plt.scatter(X, Y, color = 'blue', marker = '+')
plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color = 'red')
plt.ylabel("E-sports revenue")
plt.xlabel("Year")
plt.title("Revenue by year, in million dollars")
plt.show()


# As we can see, this line does more justice to the data. On the site from which the data is (Statistica), was a 2022 prediction - in quantity of 1790 million dollars. Let's check whether we got close:

# In[ ]:


# Prediction of polynomial
lin_reg.predict(poly_reg.fit_transform([[2022]]))


# There is one small caveat to that though. Notice, that from year 2014 the trend follows rather differently, than before 2012. That means, if we ditch the year 2012 the trend becomes a simple linear regression one:

# In[ ]:


# here I am ditching the 2012 value

year = year[1:] 
revenue = revenue[1:]
# Importing the second linear regression model
from sklearn.linear_model import LinearRegression
regressor_two = LinearRegression()
regressor_two.fit(year, revenue)

plt.scatter(year,revenue, c="blue", marker="+")
plt.plot(year, regressor_two.predict(year), c='blue')
plt.ylabel("E-sports revenue")
plt.xlabel("Year")
plt.title("E-sports revenue by year")
plt.show()


# If that's really the case, then the prediction for the following years will be:

# In[ ]:


regressor_two.predict([[2020],[2021],[2022]])


# While 1593 million dollars is not a small sum, it's noticeably smaller that 1731 million dollars. Regardless, I'm excited to see which patch the trend will follow.
# 
# Now, to the audience:

# In[ ]:


year = df[['Year']] # since I ditched the 2012 value from the original, I need to redo this step

# Importing a linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(year, audience)

plt.scatter(year,audience, c="blue", marker="+")
plt.plot(year, regressor.predict(year), c='blue')
plt.ylabel("E-sports audience")
plt.xlabel("Year")
plt.title("Yearly e-sports audience")
plt.show()


# In[ ]:


regressor.predict([[2022]])


# We can see the points fluctuate around the line, but seem to follow it more or less. That's not enough though, so let's try again the polynomial regression.

# In[ ]:


# Polynomial Regression
X = df.iloc[:,0].values
Y = df.iloc[:,-1].values
X = X.reshape(-1, 1)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) # degree = 2 is default, changing might raise accuracy
X_poly = poly_reg.fit_transform(X)
lin_reg_audience = LinearRegression()
lin_reg_audience.fit(X_poly, Y)

Y_pred = lin_reg_audience.predict(X_poly)

# Grid rearange for accuracy
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)), 1)

plt.scatter(X, Y, color = 'blue', marker = '+')
plt.plot(X_grid, lin_reg_audience.predict(poly_reg.fit_transform(X_grid)), color = 'red')
plt.ylabel("E-sports audience")
plt.xlabel("Year")
plt.title("Yearly e-sports audience")
plt.show()


# That's a chart I'd believe more, to be honest. That's why I'm going to use polynomial regression for both groups of viewers:

# In[ ]:


# Polynomial Regression
Y = df.iloc[:,2].values
X = X.reshape(-1, 1)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) # degree = 2 is default, changing might raise accuracy
X_poly = poly_reg.fit_transform(X)
lin_reg_freq = LinearRegression()
lin_reg_freq.fit(X_poly, Y)

Y_pred = lin_reg_freq.predict(X_poly)

# Grid rearange for accuracy
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)), 1)

plt.scatter(X, Y, color = 'blue', marker = '+')
plt.plot(X_grid, lin_reg_freq.predict(poly_reg.fit_transform(X_grid)), color = 'red')
plt.ylabel("E-sports frequent viewership")
plt.xlabel("Year")
plt.title("Yearly e-sports frequent viewership")
plt.show()


# In[ ]:


# Polynomial Regression
Y = df.iloc[:,3].values
X = X.reshape(-1, 1)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2) # degree = 2 is default, changing might raise accuracy
X_poly = poly_reg.fit_transform(X)
lin_reg_occ = LinearRegression()
lin_reg_occ.fit(X_poly, Y)

# Grid rearange for accuracy
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid)), 1)

plt.scatter(X, Y, color = 'blue', marker = '+')
plt.plot(X_grid, lin_reg_occ.predict(poly_reg.fit_transform(X_grid)), color = 'red')
plt.ylabel("E-sports occasional viewership")
plt.xlabel("Year")
plt.title("E-sports yearly occasional viewership")
plt.show()


# Here we notice, that the linear trend (apart from 2014, 2015), continues. As a bonus, predictions for 2022 (in millions of viewers):

# Overall audience

# In[ ]:


lin_reg_audience.predict(poly_reg.fit_transform([[2022]]))


# Frequent audience

# In[ ]:


lin_reg_freq.predict(poly_reg.fit_transform([[2022]]))


# Occasional audience

# In[ ]:


lin_reg_occ.predict(poly_reg.fit_transform([[2022]]))


# That concludes my analysis. If you have any tips on how I might improve, please write in the comments.
