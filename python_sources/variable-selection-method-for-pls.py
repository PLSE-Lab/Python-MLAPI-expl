#!/usr/bin/env python
# coding: utf-8

# # Variable selection method for PLS in Python
# Learned from: https://nirpyresearch.com/variable-selection-method-pls-python/
# 
# RPD of 1.3495 is low and not very stable. Therefore, we could use PLS for variable selection.
# ## Variable selection in chemometrics
# The idea behind variable selection in chemometrics is that when it comes to spectral measurements not all wavelengths are created equals. In visible and NIR spectroscopy especially, it is often hard topredict in advance which wavelength bands will contain most of the signal related to the analysis we want to measure. So, as a first shot, we measure the whole range allowed by our instrument, and then figure out later which bands are more releavant for our calibration.
# 
# To say the same thing in a bit more quantitatie way, we want to check which wavelength bands lead to a better quality model. Actually, in practice, we check which bands give a worse quality model, so we can get rid of them.
# 
# This seems logical enough, but I've deliberately left a fundamental piece of the puzzle out. How dod we even break our spectrum into bands and searching for the worst performing ones?
# 
# For thsi example we'll take a simple approach: We'll filter out wavelength bands based on the strength of the related regression coefficients.
# 
# ## Feature selection by filtering
# Feature selection by filtering is one of the simplest ways of performing wavelegnth selection. For an overview of the methods that are currently used, check out this excellent review paper by T. Mehmood et al.: [A review of variable selection methods in Partial Least Square Regression](https://doi.org/10.1016/j.chemolab.2012.07.010)
# 
# The idea behind this method is simple, and can be summarized as the following:
# 1. Optimize the PLS regression using the full spectrum, for instance using cross-validation or prediction data to quantify its quality
# 2. Extract the regression coefficients form the best model. Each regression coefficient uniquely associates each wavelength with the response. A low absolute value of the regression coefficient means that specific wavelegnth has a low correlation with the quantity of interest.
# 3. Discard the lowest correlation wavelengths accordign to some rule. These wavelengths are the one that typically worsen the quality of the calibration model, therefore by discarding them effectively we expect to improve the metrics associated with our prediction or cross-validation
# 
# One way to discard the lowest correlation wavelengths is to set a threshold and get rid of all wavelengths whose regression coefficients (in absolute value) fall below that threshold. However, this method is sensitive to the choice of threshold, which tends to be subjective choice, or requires a trial-and-error approach.
# 
# Another approach, that is much less subjective, is to discard one wavelength at a time (the one with the lowest absolute value of the associated regression coefficient) and rebuild the calibration model. By choosing the MSE of prediction or cross-validation as metric, the procedure is interated until the MSE decreases. At some point, removing wavelengths will produce a worse calibration, and that is the stopping criterion for the optimisation algorithm.
# 
# Alternatively, one could simply remove a fixed number of wavelengths iteratively, and then check for which number of removed wavelengths the MSE is minimised. Either way we have a method that does not depend on the subjective choice of the thresold and can be applied without changes regardless of the data set being analysed.
# 
# One caveat of this method, at least in its simple implementaiton is that it may get a bit slow for large datasets.

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


# Plot the data
wl = np.arange(1100, 2300, 2)
print(len(wl))


# In[ ]:


def plot_spectrum(X, wl, xLabel, yLabel):
    with plt.style.context('ggplot'):
        plt.plot(wl, X.T)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)


# In[ ]:


plot_spectrum(X, wl, 'Wavelengths (nm)', 'Absorbance')


# If required, data can be easily sorted by PCA and corrected with multiplicative scatter correction, however, another simple yet effective way to get rid of baseline and linear variations is to perform second derivative on the data.

# In[ ]:


# Calculate also the first and the second derrivative
X1 = savgol_filter(X, 11, polyorder=2, deriv=1)
X2 = savgol_filter(X, 13, polyorder=2, deriv=2)


# In[ ]:


plot_spectrum(X1, wl, "Wavelengths (nm)", "SG - first derivative")


# In[ ]:


plot_spectrum(X2, wl, "Wavelengths (nm)", "SG - second derivative")


# In[ ]:


def plot_spectra_vs_pls_coefficients(X, wl,num_comp, yLabel):
    # Define the PLS regression object
    pls = PLSRegression(n_components=num_comp)
    # Fit data
    pls.fit(X, y)
    # Plot spectra
    plt.figure(figsize=(8, 9))
    with plt.style.context('ggplot'):
        ax1 = plt.subplot(211)
        plt.plot(wl, X.T)
        plt.ylabel(yLabel)

        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(wl, np.abs(pls.coef_[:, 0]))
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Absolute value of PLS coefficients")

        plt.show()


# In[ ]:


plot_spectra_vs_pls_coefficients(X, wl, 8, "Absorbance")


# In[ ]:


plot_spectra_vs_pls_coefficients(X1, wl, 8, "SG - first derivative")


# In[ ]:


plot_spectra_vs_pls_coefficients(X2, wl, 8, "SG - second derivative")


# Note also that we might consider to normalize (standardize using z-score, or to make unit variance and zero mean) the data and see if the result is better.

# What we have done is a basic PLS regression with 8 components and used it to fit the absorbance, first derivative, and second derivate data. These are the regression coefficients that quantify the strength of the association between each wavelength and the response.
# We are interested in the absolute value of these coefficients, as large positive and large negative coefficients are equally important for our regression, denoting large positive or large negative correlation with response respectively. In other words we want to identify and discoard the regression coefficients that are close to zero: they are the ones with poor correlation with the response.
# 
# As we can observe, the number of wavelengths associated with the responses are small.
# 
# ## NIR variable selection for PLS regression
# e.g.,<br/>
# `# get the list of indices that sorts the PLS coefficients in ascending order of the absolute value
# sorted_ind = np.argsort(np.abs(pls.coef_[:, 0]))`
# 
# `# sort the spectra accoridng to ascendign absolute value of PLS coefficients.
# Xc = X1[:, sorted_ind]`

# In[ ]:


# define a function to evaluate pls
def pls_evaluate_num_comp(X, y, num_comp):
    pls = PLSRegression(n_components=num_comp)
    y_cv = cross_val_predict(pls, X, y, cv=5)
    mse = mean_squared_error(y_cv, y)
    r2 = r2_score(y_cv, y)
    rpd = y.std()/np.sqrt(mse)
    return (y_cv, mse, r2, rpd)


# In[ ]:


# Try optimize the number of components (without variable selection) => we will use X1
def pls_evaluate_num_comps(X, y, num_comps):
    mses = []
    r2s = []
    rpds = []
    for num_comp in num_comps:
        _, mse, r2, rpd = pls_evaluate_num_comp(X, y, num_comp)
        mses.append(mse)
        r2s.append(r2)
        rpds.append(rpd)
    return (mses, r2s, rpds)


# In[ ]:


def plot_metric(scores, objective, yLabel):
    with plt.style.context('ggplot'):
        plt.plot(num_comps, scores, '-o', color='blue')
        idx = np.argmin(scores) if objective == 'min' else np.argmax(scores)
        plt.plot(num_comps[idx], scores[idx], 'P', color='red', ms=10)
        plt.xlabel("Number of components")
        plt.ylabel(yLabel)
    plt.show()
    return (num_comps[idx], scores[idx])


# In[ ]:


def pls_evaluate_plot_num_comps(X, y, num_comps):
    mses, r2s, rpds = pls_evaluate_num_comps(X, y, num_comps)
    # Plot mses
    num_comp, mse = plot_metric(mses, 'min', 'MSE')
    print(f'The best mse is {mse} with {num_comp} PLS components')
    # Plot r2s
    num_comp, r2  = plot_metric(r2s, 'max', 'R2')
    print(f'The best r2 is {r2} with {num_comp} PLS components')
    # Plot rpds
    num_comp, rpd = plot_metric(rpds, 'max', 'RPD')
    print(f'The best RPD is {rpd} with {num_comp} PLS components')


# In[ ]:


# test with the first 15 components and choose the best for absorbance.
num_comps = np.arange(1, 16)
pls_evaluate_plot_num_comps(X, y, num_comps)


# In[ ]:


# test with the first 15 components and choose the best for Savitzky and Golay first derivative.
num_comps = np.arange(1, 16)
pls_evaluate_plot_num_comps(X1, y, num_comps)


# In[ ]:


# test with the first 15 components and choose the best for Savitzky and Golay second derivative.
num_comps = np.arange(1, 16)
pls_evaluate_plot_num_comps(X2, y, num_comps)


# In[ ]:


def pls_evaluate_and_plot_num_comp(X, y, num_comp):    
    # Evaluate the result with first three components
    y_cv, mse, r2, rpd = pls_evaluate_num_comp(X, y, num_comp)
    # Print the result
    print('MSE: %0.4f' % (mse))
    print('R2: %0.4f' % (r2))
    print('RPD: %0.4f' % (rpd))
    # plot the regression
    p = np.polyfit(y, y_cv, deg=1)
    with plt.style.context('ggplot'):
        plt.figure(figsize=(6, 6))
        plt.scatter(y, y_cv, color='red', edgecolors='black')
        plt.plot(y, y, '-g', label='Expectation')
        plt.plot(y, np.polyval(p, y),'-b', label='Prediction regression')
        plt.legend()
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.plot()
    return (y_cv, mse, r2, rpd)


# In[ ]:


# Test for absorbance at its best number of components (6)
_ = pls_evaluate_and_plot_num_comp(X, y, 6)


# In[ ]:


# Test for the first derivative at its best number of components (3)
_ = pls_evaluate_and_plot_num_comp(X1, y, 3)


# The result for the first derivative is slightly better

# In[ ]:


# Test for the second derivative at its best number of components (4)
_ = pls_evaluate_and_plot_num_comp(X2, y, 4)


# The second deriviate is not as good as the first one.

# ## Comments
# ### Event with the best number of PLS components (3) for the first derivative (best results)
# RPD of 1.3335 is not acceptable.<br/>
# R2: -0.0662 is even worse than the simple base-line prediction as mean of y
# 
# Therefore we will try to select variables

# In[ ]:


def sort_variable(X, y, num_comp):
     # Use PLS using full spectrum (all the wavelengths)
    pls1 = PLSRegression(n_components=num_comp)
    pls1.fit(X, y)
    # Sort the wavelengths by the coefficients
    sorted_ind = np.argsort(np.abs(pls1.coef_[:, 0]))
    # Sort the spectra accordingly
    Xc = X[:, sorted_ind]
    return Xc


# In[ ]:


def pls_evaluate_variable(X, y, num_comp):
    # Array of MSE each time we reduce a wavelength
    mses = np.array([float('inf') for _ in range(X.shape[1])]) # Array of 600 elements (each time we reduce one, but we can't reduce the number of components to smaller than num_comp, thus there will be maximum values for num_comp elements at the end)
    r2s =  np.array([float('-inf') for _ in range(X.shape[1])]) # R2 objective is to maximize (as close to 1 as possible) so the default value is set to -inf
    rpds = np.array([float('-inf') for _ in range(X.shape[1])]) # RPD objective is to maximize (as big as possible) so the default value is set to -inf
    
    # Sort the spectra accordingly
    Xc = sort_variable(X, y, num_comp)
    # Discard wavelength one at a time (but the remained number of wavelengths must be >= num_comp)
    for num_discarded in range(Xc.shape[1] - num_comp):
        Xn = Xc[:, num_discarded:]
        _, mse, r2, rpd = pls_evaluate_num_comp(Xn, y, num_comp)
        mses[num_discarded] = mse
        r2s[num_discarded] = r2
        rpds[num_discarded] = rpd
    return (mses, r2s, rpds)


# In[ ]:


# Helper function
def find_min_2d_indices(x):
    '''
    Find the min index from a 2D array and gives row and column indices for the min element
    Parameters:
        x: the 2D array
    Returns:
        (iIdx, jIdx): iIdx is the row index and jIdx is the column index
    Test:
    >>> find_min_2d_indices(np.array([[1, 2, 3], [1, 0, 1], [2, 2, 4]]))
    (1, 1)
    '''
    idx = np.argmin(x)
    iIdx = idx//x.shape[1]
    jIdx = idx - iIdx * x.shape[1]
    return (iIdx, jIdx)
# Helper function
def find_max_2d_indices(x):
    '''
    Find the max index from a 2D array and gives row and column indices for the min element
    Parameters:
        x: the 2D array
    Returns:
        (iIdx, jIdx): iIdx is the row index and jIdx is the column index
    Test:
    >>> find_max_2d_indices(np.array([[1, 2, 3], [1, 0, 1], [2, 2, 4]]))
    (2, 2)
    '''
    idx = np.argmax(x)
    iIdx = idx//x.shape[1]
    jIdx = idx - iIdx * x.shape[1]
    return (iIdx, jIdx)


# In[ ]:


# Now we evaluate the different PLS with different number of components and different number of variables and collect all the data back
def evaluate_with_different_num_comps(X, y, num_comps):
    comp_mses = [] # array of array of mses (first dimension is for the number of components, second dimension is for all different )
    comp_r2s = [] # similar to mse for r2
    comp_rpds = [] # similar to mse for rpd
    for num_comp in num_comps:
        mses, r2s, rpds = pls_evaluate_variable(X, y, num_comp)
        comp_mses.append(mses)
        comp_r2s.append(r2s)
        comp_rpds.append(rpds)
    comp_mses = np.array(comp_mses)
    comp_r2s = np.array(comp_r2s)
    comp_rpds = np.array(comp_rpds)
    return (comp_mses,comp_r2s, comp_rpds)


# In[ ]:


def print_metric_info(comp_mses, comp_r2s, comp_rpds):
    # considering comp_mses
    min_mse_i, min_mse_j = find_min_2d_indices(comp_mses)
    print(f'Min MSE: component index: {min_mse_i}, variable cut-off index: {min_mse_j}, Components: {num_comps[min_mse_i]}, MSE: {comp_mses[min_mse_i][min_mse_j]}')

    # considering r2s => objective is to maximize r2
    max_r2_i, max_r2_j = find_max_2d_indices(comp_r2s)
    print(f'Max R2: component index: {max_r2_i}, variable cut-off index: {max_r2_j}, Components: {num_comps[max_r2_i]}, R2: {comp_r2s[max_r2_i][max_r2_j]}')
    
    # considering rpds => objective is to maximize rpds
    max_rpd_i, max_rpd_j = find_max_2d_indices(comp_rpds)
    print(f'Max RPD: component index: {max_rpd_i}, variable cut-off index: {max_rpd_j}, Components: {num_comps[max_rpd_i]}, RPD: {comp_rpds[max_rpd_i][max_rpd_j]}')


# In[ ]:


def find_best_components_variables(X, y, num_comps):
    comp_mses, comp_r2s, comp_rpds = evaluate_with_different_num_comps(X, y, num_comps)
    print_metric_info(comp_mses, comp_r2s, comp_rpds)


# In[ ]:


# Try with absorbance
find_best_components_variables(X, y, num_comps)


# In[ ]:


# Now try with Savitzky first derivative
find_best_components_variables(X1, y, num_comps)


# In[ ]:


# Now try with Savitzky second derivative
find_best_components_variables(X2, y, num_comps)


# So if we are considering first derivative, the mse and rpds, the number of components is at the 9th index, and the cut off for variable selection is at 417 variables (cut the first 417)
# Note that the number of components is the index + 1 (since we start with 1 components for the first zero index), but the number of variable is 417 since we start with 0 (discard_index:)
# 
# Similarly for original data or second derivative.
# 
# In this case, the second derivative works best with variable selection.

# In[ ]:


# Now plot results for first derivative
num_comp = 15
num_discarded = 448
# Sort according to the num_comp then select (cut from num_discardeed onward)
Xselected = sort_variable(X, y, num_comp)[:, num_discarded:]
# Now test the performance with the selected num_comp and selected variables
y_cv, mse, r2, rpd = pls_evaluate_and_plot_num_comp(Xselected, y, num_comp)


# In[ ]:


# Now plot results for first derivative
num_comp = 10
num_discarded = 417
# Sort according to the num_comp then select (cut from num_discardeed onward)
Xselected = sort_variable(X1, y, num_comp)[:, num_discarded:]
# Now test the performance with the selected num_comp and selected variables
y_cv, mse, r2, rpd = pls_evaluate_and_plot_num_comp(Xselected, y, num_comp)


# In[ ]:


# Now plot results for first derivative
num_comp = 12
num_discarded = 503
# Sort according to the num_comp then select (cut from num_discardeed onward)
Xselected = sort_variable(X2, y, num_comp)[:, num_discarded:]
# Now test the performance with the selected num_comp and selected variables
y_cv, mse, r2, rpd = pls_evaluate_and_plot_num_comp(Xselected, y, num_comp)


# ## Commment
# It is observable second derivatives and with 12 PLS components perform the best

# In[ ]:




