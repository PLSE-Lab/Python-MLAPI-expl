#!/usr/bin/env python
# coding: utf-8

# I am interested in seeing if there are unexpected relationships between the properties of the exoplanets, such that some can be used to predict others. Let's begin!
# 
# First, let us look at how the data is organized.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sci
from scipy import stats
import numpy as np

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

data = pd.read_csv("../input/oec.csv")
data.head(3)

# Any results you write to the current directory are saved as output.


# Next, let's explore the distribution of some of the properties. I am interested in the type flag, distance (in parsecs) from the Sun, and mass.

# In[ ]:


plt.hist(data['TypeFlag'].dropna(), alpha=0.6)
plt.xlabel('Binary Flag')
plt.ylabel('Count')
plt.show()

plt.hist(data['PlanetaryMassJpt'].dropna(), bins=100, alpha=0.6)
plt.xlabel('Planetary Mass (Jpt)')
plt.ylabel('Count')
plt.show()

plt.scatter(data['PlanetaryMassJpt'], data['DistFromSunParsec'])
plt.xlabel('Planetary Mass (Jpt)')
plt.ylabel('Distance from the Sun (parsec)')
plt.show()


# I was wondering  whether planetary mass in some way would correlate with an exoplanet's distance from the Sun. What is interesting in the scatter plot is the distribution is highly skewed along both axes - as do residuals:

# In[ ]:


completeData = data[['PlanetaryMassJpt','DistFromSunParsec']].dropna().astype(float)

residuals = sci.subtract(completeData['PlanetaryMassJpt'], completeData['DistFromSunParsec'])

f, ax = plt.subplots(2)
ax[0].hist(residuals)
plt.xlabel('Residuals')
plt.ylabel('Count')
ax[1].hist(abs(residuals))
plt.show()


# Both absolute and relative errors appear skewed. I fit the absolute error distribution to lognormal PDF to see if log-transforming the data is in order.

# In[ ]:


ar = abs(residuals)
shape, loc, mean = sci.stats.lognorm.fit(ar, floc = 0)
xfit = sci.linspace(ar.min(),ar.max(),500)
pdf = stats.lognorm.pdf(xfit,shape,loc,mean)

plt.hist(ar, 500, normed=1, facecolor = 'grey', alpha=0.6)
plt.plot(xfit, pdf, 'r--', linewidth = 3)
plt.show()


# I decided to log-transform the data to see if the skewness in residuals will be rectified.

# In[ ]:


transformedDst = []
transformedMass = []
for val in list(range(completeData.size)):
    transformedDst.append(math.log(completeData['DistFromSunParsec'][[val]]))
    transformedMass.append(math.log(completeData['PlanetaryMassJpt'][[val]]))
   
residuals = [] 
for val in list(range(completeData.size)):
    residuals.append(transformedMass[val] - transformedDst[val])

x = []
for val in residuals:
    if ~np.isnan(val):
        x.append(val)

mean, std = sci.stats.norm.fit(x)
xfit = sci.linspace(min(x), max(x), 500)
pdf = stats.norm.pdf(xfit, mean, std)

plt.hist(x, 500, normed=1, facecolor='grey', alpha=0.6)
plt.plot(xfit, pdf, 'r--', linewidth=3)
plt.show()


# Since the residuals now appear more Gaussian, a log-transformation of the two exoplanet properties might have a linear relationship. I plotted both along with a linear model trained via gradient descent.

# In[ ]:


transformedDst = np.asarray(transformedDst)
transformedMass = np.asarray(transformedMass)

mask = np.isnan(transformedMass) | np.isnan(transformedDst)
Y=transformedDst[np.logical_not(mask)] # removed NaNs
X=transformedMass[np.logical_not(mask)]

n = Y.size # sample size

it = np.ones(shape=(n,2))
it[:,1] = X

theta = np.zeros(shape=(2,1))

iterations = 2000
alpha = 0.01

## Convergence through gradient descent; adapted from Marcel Caraciolo and Andrew Ng

def cost(X, Y, theta, n):
    predict = X.dot(theta).flatten()
    error = (predict - Y) ** 2
    J = (1/(2*n)) * error.sum()
    return J

def gradient_descent(X, Y, theta, alpha, iterations, n):
    jHistory = np.zeros(shape=(iterations,1))

    for id in range(iterations):
        predict = X.dot(theta).flatten()

        error1 = (predict - Y) * X[:, 0]
        error2 = (predict - Y) * X[:, 1]

        theta[0][0] = theta[0][0] - alpha * error1.sum()/n
        theta[1][0] = theta[1][0] - alpha * error2.sum()/n

        jHistory[id, 0] = cost(X, Y, theta, n)

    return theta, jHistory
## 
print(cost(it, Y, theta, n))

theta, jHistory = gradient_descent(it, Y, theta, alpha, iterations, n)
fit = it.dot(theta).flatten()

print('Regression Coefficients: \n', theta)
print('RMSE: %.2f' %np.sqrt(np.mean((fit - Y) ** 2)))


plt.scatter(transformedMass, transformedDst, s = 50, color = 'darkslateblue', alpha=0.6)
plt.plot(X, fit, color='orange', linewidth=3, linestyle = "--")
plt.xlabel('log(Jupiter Mass)')
plt.ylabel('log(Distance from Sun)')
plt.show()


# There appear to be a linear relationship between an exoplanet's distance from the Sun and its mass. I would probably need to test the linear regression model on more exoplanets after finding a way to identify outliers (which might be compromising the linear fit obtained through gradient descent). I'll also explore whether distance away from the Sun can serve as a predictor for other exoplanet properties. 
# 
# Well, that's it for this notebook!
