#!/usr/bin/env python
# coding: utf-8

# # Principal Component Regression for NIR calibration
# Learned from: https://nirpyresearch.com/principal-component-regression-python/
# We want to build calibration for Brix using NIR spectra from 50 fresh peaches. Each spectrum is composed of 601 data points. To build the calibration we are going to correlate somehow the spectrum of each peach (no pun intended) with its value of Brix measured independently with another method. As you can see already, we are going to reduce 601 data points to a single parameters! That means in practice one never needs this many points. Just a few would be sufficient.
# 
# But which variables are the one that are significant to describe Bri variation? Well, we don't know that in advance, and that is why we are going to use PCA to sort things out. PCA is going to reduce the dimensionality of our data space, getting rid of all the variables (data points) that are highly correlated with one another. In practice we will reduce 601 dimensions to a much lower number, often a few, or even just two, dimensions.
# 
# With PCA we can establish that different pieces of fruit may have different Brix value, but we won't be able to build a model that is able to predict Brix values from future measurements. The reason is that PCA makes no use whatsoever of the additional information we may have, namely the independent data of Brix measured with a different instrument. In order to use the additional information we need to go from a classification problem to a regression problem. Enters PCR.
# 
# # Python implementation of Principal Component Regression
# To put it very simply, PCR is a two-step process:
# 1. Run PCA on our data to decompose the independent variables into 'principal components', corresponding to removing correlated components
# 2. Select a subset of the principal components and run a regression against the calibration values
# 
# In the apparent simplicty of PCR lies also its . main limitataion.Using PCA to extract the principal components in step 1 is done disregarding whichever information we may have on the calibration values. PCA is 'plug and play' technique. It spits out some result regardless of the nature of the inptu data, or its underlying statistics. In other words, PCA is very simple, but sometimes may be wrong.
# 
# These limiataions are inherited by PCR, in the sense that principal components found by PCA may not be the optimal ones for the regression we want to run. This problem is alleviated by Partial Least Square Regression (PLS) which we will discuss in the future. For the moent, let's leverage the simplicty of PCR to understand the basic of the regressio problem, and hopefully enjoy builidng same simple calibration model of our own in Python.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


data = pd.read_csv("../input/peach_spectrabrixvalues.csv")


# In[ ]:


data.head()


# In[ ]:


# Get the data
X = data.values[:, 1:]
y = data['Brix']
wl = np.arange(1100, 2300, 2) # wavelengths
# Plot absorbance spectra
with plt.style.context('ggplot'):
    plt.plot(wl, X.T)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel('Absorbance')
plt.show()


# These are the absorbance spectra of our 50 samples. As mentioned, we divide the PCR problem in two steps. The first is to run PCA on the input data. Before that, we have two processing steps. The first is quite commmon in NIR spectroscopy: take the first derivative of the absorbance data. The second is a staple of PCA: standardize the features to sbustract the mean and reduce to unit variance.

# In[ ]:


# Step 1: PCA on input data
# Define the PCA object
pca = PCA()
# Preprocessing (1): First derivative
d1X = savgol_filter(X, 25, polyorder=5, deriv=1)


# In[ ]:


# number of components
pc = 3
# Preprocessing (2): Standardize features by removing the mean and scaling to unit variance
Xstd = StandardScaler().fit_transform(d1X[:, :])

# Run PCA producing the reduced variable Xreg and select first PC components
Xreg = pca.fit_transform(Xstd)[:, :pc]


# Note that in the last step, after running only `pc` number of components is selected. this will be clear in a bit, when we'll put the whole thing in a standalone function

# In[ ]:


# Create linear regression object
regr = linear_model.LinearRegression()

# Fit
regr.fit(Xreg, y)


# In[ ]:


# Calibration
y_c = regr.predict(Xreg)


# In[ ]:


# Cross-validation
y_cv = cross_val_predict(regr, Xreg, y, cv=10)


# In[ ]:


# Calcuate scores for calibration and cross-validation
score_c = r2_score(y, y_c)


# In[ ]:


score_cv = r2_score(y, y_cv)


# In[ ]:


# calculate the mean square error for calibration and corss validation
mse_c = mean_squared_error(y, y_c)
mse_cv = mean_squared_error(y, y_cv)


# In[ ]:


y_cv


# In[ ]:


print(score_c)


# In[ ]:


print(score_cv)


# In[ ]:


print(mse_c)


# In[ ]:


print(mse_cv)


# We first cfreated the linear regression object and then used it to run the fit. At this point the savvy practitional will distinguish between calibration and cross-validation results. This is an important point, we'll make short digression to cover it.
# 
# # Calibration and cross-validation
# In teh example above, we build a linear regression between the variables Xreg and y. Loosely speaking we build a linear relation between principal components etracted from spectroscopic data and the corresponding Brix values of each sample. So we expect that, if we have done a good job, we should be able to predict th evlue of Brix in other samples not included in our calibration.
# 
# First, how do we know if we've done a good job? The metrics we use are the coefficient of deterination (or R^2) and the mean squared error. The ideas goes like this. With the regr.fit(Xreg, y) command we genrate the coefficients fo the lienar fit (slop m and itnercept 1). With the `y_c = regr.predit(Xreg)` command we can thenuse these coefficiewnts to 'predict' the new values `yc = mXreg + q` of the Brix given the input spectroscopy data. %R^2% tells us how well $y_c$ correlate with y (Ideally we want $r^2$ to be as close as possible to 1). MSE is the meansquared deviation of the $yc$ with respect to $y$ in the units of y, so in our case MSE will be measured in Bx.
# 
# OK, here's the issue. If we take the same data used for the fit and use it again for the prediction, we run the risk of reproducing magnificiently our calibration set, but not being able to preduct any future measurement. This is what data scientists call "overfitting".
# 
# In order to make predictions we want to be able to handle future (unknown) data with good accuracy. For that, we would ideally need an independent set of spectroscopy data, often referred to as valiation data. If we do not have an indepdnent validation set, th enext best thing is to split our input data into calibration and cross-validation sets. Only the calibration data are used to build the regression model. The corss-validation data are then used as (hopefully) independent set to verify the predictive value of our model.
# 
# # Building the calibration model

# In[ ]:


def pcr(X, y, pc):
    '''
        Principal Component Regression in Python
    '''
    # Step 1: PCA on the input data
    # Define the PCA object
    pca = PCA()
    # Preprocessing (1): first derivative
    d1X = savgol_filter(X, 25, polyorder=5, deriv=1)
    
    # Preprocessing (2): standardize features by removing the mean and saling to unit variance
    Xstd = StandardScaler().fit_transform(d1X[:, :])
    
    # Run PCA producing the reduced variable Xreg and select the first pc components
    Xreg = pca.fit_transform(Xstd)[:, :pc]
    
    # Step 2: Regression on selected components
    # Create linear regression object
    regr = linear_model.LinearRegression()
    
    # Fit
    regr.fit(Xreg, y)
    
    # Calibration
    y_c = regr.predict(Xreg)
    
    # Cross-validation
    y_cv = cross_val_predict(regr, Xreg, y, cv=10)
    
    # Calcualte scores for calibration and cross-validation
    score_c = r2_score(y, y_c)
    score_cv = r2_score(y, y_cv)
    
    # Calculate mean square error for calibration and cross validation
    mse_c = mean_squared_error(y, y_c)
    mse_cv = mean_squared_error(y, y_cv)
    
    return (y_cv, score_c, score_cv, mse_c, mse_cv)
    


# In[ ]:


r2s_c = []
r2s_cv = []
mses_c = []
mses_cv = []
for pc in range(1, 20):
    y_cv, score_c, score_cv, mse_c, mse_cv = pcr(X, y, pc)
    r2s_c.append(score_c)
    r2s_cv.append(score_cv)
    mses_c.append(mse_c)
    mses_cv.append(mse_cv)


# In[ ]:


xticks = np.arange(1, 20).astype('uint8')
plt.plot(xticks, r2s_c, '-o', label='Calibration', color='red')
plt.plot(xticks, r2s_cv, '-o', label='Cross Validation', color='blue')
plt.xticks(xticks)
plt.xlabel('Number of PC included')
plt.ylabel('R-squared')
plt.legend()


# In[ ]:


xticks = np.arange(1, 20).astype('uint8')
plt.plot(xticks, mses_c, '-o', label='Calibration', color='red')
plt.plot(xticks, mses_cv, '-o', label='Cross Validation', color='blue')
plt.xticks(xticks)
plt.xlabel('Number of PC included')
plt.ylabel('mse')
plt.legend()


# It seems in both $R^2$ and MSE that number of components as 6 is the best.

# In[ ]:


predicted, r2r, r2cv, mser, mscv = pcr(X, y, pc=6)


# In[ ]:


# Regression plot
z = np.polyfit(y, predicted, 1)
with plt.style.context('ggplot'):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(y, predicted, c='red', edgecolors='k')
    ax.plot(y, z[1] + z[0]*y, c='blue', linewidth=1)
    ax.plot(y, y, color='green', linewidth='1')
    rpd = y.std()/np.sqrt(mscv)
    plt.title('$R^{2}$ (CV): %0.4f, RPD: %0.4f'%(r2cv, rpd))
    
    plt.xlabel('Measured $^{\circ}$Brix')
    plt.ylabel('Predicted $^{\circ}$Brix')
    plt.show()


# The green line represents the ideal, 100%, correlation between measured and predicted values. The blue line is the actual correlation. With $R^2 = 0.43$ the result is obviously not great, and a number of other things are required to improve this figure. Things about multiplicative scatter correlation and feature selection are often used to improve the prediction. At the same time, the main limitation of the PCR approach (the fact the PCA is done without knowledge of the y values) also plays a big role in the final results.
# 
# Note also that we added the information about the [Residual Predictive Deviation information (RPD)](https://rdrr.io/cran/chillR/man/RPD.html). RPD is defined as the standard deviation of the observed values (how varies the y values are) divided by the root means squared error prediction (RMSEP). The RDP takes both prediction error and the variation of the observed (actual) values into account, providign a metric of model validaity that is more objective than the RMSEP and more easily comparable across model validation studies. The greater the RPD, the better the model's predictive capacity.
# 
# Typically better results can be obtained with Partial Least Squares (PLS).
# 
# 

# In[ ]:




