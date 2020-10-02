#!/usr/bin/env python
# coding: utf-8

# # Partial Least Squares Regression in Python
# Learned from: https://nirpyresearch.com/partial-least-squares-regression-python/<br/>
# PLS, acronym of Partial Least Squares, is a widespread regression technique used to analyze near-infrared spectroscopy data.
# 
# PCR is quite simply a regression model built using a number of principal components derived using PCA. PCR is nice and simple but it does not tak einto account anything other than the regression data (e.g., not taking into account about the labels or values need to be predicted - y). That is, our primary reference data are not considered when building a PCR model. That is obviously not optimal, and PLS is a way to fix that.
# 
# In thsi post, we are going to show how to build a simple regression model using PLS in Python.
# 1. Mathematical introduction to the difference between PCR and PLS regression
# 2. Present the basic code for PLS
# 3. Discuss the data we want to analyze and the pre-processing required
# 4. We will build our model using a cross-validation approach
# 
# # Difference between PCR and PLS regression
# Before working on some code, let's briefly discuss the mathematical difference between PCR and PLS.
# 
# Both PLS and PCR perform multiple linear regression, that is they build a linear model, Y = XB + E. Using a common language in statistics, X is the predictor and Y is the response. In NIR analysis, X is the set of spectra, Y is th equantity - or quantities - we want to calibrate for (in our case the brix values). Finally E is an error.
# 
# The matrix X contains highly correlated data and this correlation (unrelated to brix) may obscure the variations we want to measure, that is the variations of the brix content. Both PCR and PLS will get rid of the correlation.
# 
# In PCR, the set of measurements X is treansformed into equivalent $X'=XW$ by a linear transformation $W$, such that all the new 'spectra' (which are the principal components) are linear independent. In statistics $X'$ is called the **factor scores**.
# 
# The linear transformation in PCR is such that it minimises the covariance between the diffrent rows of $X'$. That means this process only uses the spectral data, not the response values.
# 
# This is the key difference between PCR and PLS regression. PLS is based on finding a similar linear transformation, but accomplishes the same task by maximising the covariance between $Y$ and $X'$. In other words, PLS takes into account both spectra and response values and in doing so will improve on some of the limitation on PCR. For these reasons PLS is one of the staples of modern chemometrics.
# 
# # PLS in Python
# `sklearn` already has got a PLS package, so we go ahead and use it without reinventing the wheel. So, first we define teh number of components we want to keep in our PLS regression. Once the PLS object is defined, we fit the regression to the data `x` (the preditor) and `y` (the known response). The third step is to use the model we jsut built to run a cross-validation experiment iusign 10 fold cross-validation.
# 
# When we do not have a large number of spectra, cross-validation is a good way to test the predictive capability of our model.
# 
# 

# In[ ]:


from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


# In[ ]:


data = pd.read_csv("../input/peach-nir-spectra-brix-values/peach_spectrabrixvalues.csv")


# In[ ]:


data.head()


# In[ ]:


y = data['Brix'].values
X = data.values[:, 1:]


# In[ ]:


y.shape


# In[ ]:


X.shape


# In[ ]:


# Plot the data
wl = np.arange(1100, 2300, 2)
print(len(wl))


# In[ ]:


with plt.style.context('ggplot'):
    plt.plot(wl, X.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("Absorbance")


# If required, data can be easily sorted by PCA and corrected with multiplicative scatter correction, however, another simple yet effective way to get rid of baseline and linear variations is to perform second derivative on the data.

# In[ ]:


X2 = savgol_filter(X, 17, polyorder=2, deriv=2)


# In[ ]:


# plot and see
plt.figure(figsize=(8, 4.5))
with plt.style.context('ggplot'):
    plt.plot(wl, X2.T)
    plt.xlabel("Wavelengths (nm)")
    plt.ylabel("D2 Absorbance")
    plt.show()


# The offset is gone and the data look more bunched together.
# 
# Now it's time to get to the optimisation of the PLS regression.

# In[ ]:


def optimise_pls_cv(X, y, n_comp):
    # Define PLS object
    pls = PLSRegression(n_components=n_comp)

    # Cross-validation
    y_cv = cross_val_predict(pls, X, y, cv=10)

    # Calculate scores
    r2 = r2_score(y, y_cv)
    mse = mean_squared_error(y, y_cv)
    rpd = y.std()/np.sqrt(mse)
    
    return (y_cv, r2, mse, rpd)


# In[ ]:


# test with 40 components
r2s = []
mses = []
rpds = []
xticks = np.arange(1, 41)
for n_comp in xticks:
    y_cv, r2, mse, rpd = optimise_pls_cv(X2, y, n_comp)
    r2s.append(r2)
    mses.append(mse)
    rpds.append(rpd)


# In[ ]:


# Plot the mses
def plot_metrics(vals, ylabel, objective):
    with plt.style.context('ggplot'):
        plt.plot(xticks, np.array(vals), '-v', color='blue', mfc='blue')
        if objective=='min':
            idx = np.argmin(vals)
        else:
            idx = np.argmax(vals)
        plt.plot(xticks[idx], np.array(vals)[idx], 'P', ms=10, mfc='red')

        plt.xlabel('Number of PLS components')
        plt.xticks = xticks
        plt.ylabel(ylabel)
        plt.title('PLS')

    plt.show()


# In[ ]:


plot_metrics(mses, 'MSE', 'min')


# In[ ]:


plot_metrics(rpds, 'RPD', 'max')


# In[ ]:


plot_metrics(r2s, 'R2', 'max')


# Notice that all the metrics confirm that 7 components is the best option.
# We now apply it to our solution.

# In[ ]:


y_cv, r2, mse, rpd = optimise_pls_cv(X2, y, 7)


# In[ ]:


print('R2: %0.4f, MSE: %0.4f, RPD: %0.4f' %(r2, mse, rpd))


# In[ ]:


plt.figure(figsize=(6, 6))
with plt.style.context('ggplot'):
    plt.scatter(y, y_cv, color='red')
    plt.plot(y, y, '-g', label='Expected regression line')
    z = np.polyfit(y, y_cv, 1)
    plt.plot(np.polyval(z, y), y, color='blue', label='Predicted regression line')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.plot()


# # Variable selection method for PLS in Python
# RPD of 1.3495 is low and not very stable. Therefore, we could use PLS for variable selection.
