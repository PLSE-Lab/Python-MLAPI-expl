#!/usr/bin/env python
# coding: utf-8

# # Machine Learning I - Practical I
# 
# Name: {YOUR NAME}
# 
# Course: {NAME OF YOUR PROGRAM}

# This notebook provides you with the assignments and the overall code structure you need to complete the assignment. Each exercise indicates the total number of points allocated. There are also questions that you need to answer in text form. Please use full sentences and reasonably correct spelling/grammar.
# 
# Regarding submission & grading:
# 
# - Solutions can be uploaded to ILIAS until the start of the next lecture. Please upload a copy of this notebook and a PDF version of it after you ran it.
# 
# - Please hand in your own solution. You are encouraged to discuss your code with classmates and help each other, but after that, please sit down for yourself and write your own code. 
# 
# - We will grade not only based on the outputs, but also look at the code. So please use comments make us understand what you intended to do :)
# 
# - For plots you create yourself, all axes must be labeled. 
# 
# - DO NOT IN ANY CASE change the function interfaces.
# 
# - If you are not familiar with python, but used MATLAB before, check out this reference pages listing what you want to use as python equivalent of a certain MATLAB command:
# 
#     https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html
#     
#     http://www.eas.uccs.edu/~mwickert/ece5650/notes/NumPy2MATLAB.pdf
#     
#     http://mathesaurus.sourceforge.net/matlab-python-xref.pdf
#     
#     or, if you prefer to read a longer article, try: 
#     
#     https://realpython.com/matlab-vs-python/#learning-about-pythons-mathematical-libraries
#     
#     

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy import stats
import copy
import pylab
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# ## The  dataset

# The dataset consists of over 20.000 materials and lists their physical features. From these features, we want to learn how to predict the critical temperature, i.e. the temperature we need to cool the material to so it becomes superconductive. First load and familiarize yourself with the data set a bit.

# In[ ]:


data=pd.read_csv('../input/superconduct_train.csv')
print(data.shape)


# In[ ]:


data.head()


# Because the dataset is rather large, we prepare a small subset of the data as training set, and another subset as test set. To make the computations reproducible, we set the random seed.

# In[ ]:


target_clm = 'critical_temp' # the critical temperature is our target variable
n_trainset = 200 # size of the training set
n_testset = 500 #size of the test set


# In[ ]:


# set random seed to make sure every test set is the same
np.random.seed(seed=1)

idx = np.arange(data.shape[0])
idx_shuffled = np.random.permutation(idx) # shuffle indices to split into training and test set

test_idx = idx_shuffled[:n_testset]
train_idx = idx_shuffled[n_testset:n_testset+n_trainset]
train_full_idx = idx_shuffled[n_testset:]

X_test = data.loc[test_idx, data.columns != target_clm].values
y_test = data.loc[test_idx, data.columns == target_clm].values
print('Test set shapes (X and y)', X_test.shape, y_test.shape)

X_train = data.loc[train_idx, data.columns != target_clm].values
y_train = data.loc[train_idx, data.columns == target_clm].values
print('Small training set shapes (X and y):',X_train.shape, y_train.shape)

X_train_full = data.loc[train_full_idx, data.columns != target_clm].values
y_train_full = data.loc[train_full_idx, data.columns == target_clm].values
print('Full training set shapes (X and y):',X_train_full.shape, y_train_full.shape)


# ## Task 1: Plot the dataset [5 pts]
# 
# To explore the dataset, use `X_train_full` and `y_train_full` for two descriptive plots:
# 
# * **Histogram** of the target variable. Use `plt.hist`.
# 
# * **Scatterplots** relating the target variable to one of the feature values. For this you will need 81 scatterplots. Arrange them in one big figure with 9x9 subplots. Use `plt.scatter`. You may need to adjust the marker size and the alpha blending value. 

# In[ ]:


# Histogram of the target variable
plt.hist(y_train_full)
# ADD YOUR CODE HERE


# In[ ]:


#Trials to normalize tha array X_train_full:

normalized_X_train_full=(X_train_full-X_train_full.min())/(X_train_full.max()-X_train_full.min())
normalized_X_train_full.shape
normalized_df=(X_train_full-X_train_full.mean())/X_train_full.std()
import pandas as pd
from sklearn import preprocessing

x = X_train_full #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
normalized_X_train_full.shape


# In[ ]:



    plt.scatter(x_scaled[:,1], y_train_full, c=np.random.rand(3,), alpha=0.5)


# In[ ]:


#Trials to write the loop
# Scatter plots of the target variable vs. features


#fig=plt.figure(figsize=(8, 8))
#columns = 9
#rows = 9
#feat_idxs = range(1,81)

#for idx in feat_idxs:
 #   plt.scatter(X_train_full[:,idx], y_train_full, c=np.random.rand(3,), alpha=0.5)
  #  fig.add_subplot(rows, columns, idx)
#fig, ax = plt.subplots(9, 9, sharex='col', sharey='row', figsize=(10, 10))
#idx = 0
#for lidx in range(9):
 #   for cidx in range(9):
  #      ax[lidx, cidx].scatter( X_train_full[:,idx],y_train_full, 
   #             c=np.random.rand(3,),
    #            alpha=0.5)
#         ax[lidx, cidx].text(0.5, 0.5, str((lidx, cidx, idx)),
#                       fontsize=8, ha='center')
     #   idx += 1

# ADD YOUR CODE HERE
idx = 0
fig, axs = plt.subplots(9, 9, sharex=True, sharey=True, figsize=(10, 10))
# add a big axes, hide frame
fig.add_subplot(111, frameon=False)
# hide tick and tick label of the big axes
plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
plt.grid(False)
plt.xlabel("Different material properties")
plt.ylabel("Critical temperature")

for n in range(81):
    axs[n // 9, n % 9].scatter(x_scaled[:, idx], y_train_full, c = np.random.rand(3,), alpha=0.5)
    idx += 1
                                            


# Which material properties may be useful for predicting superconductivity? What other observations can you make?

# In[ ]:


#from here we can return the name of properties that might be useful and their coordinates of our 9x9 plot
ind = 1
for col in data.columns[1:81]: 
    print(col, ind//9 , ind%9)
    ind += 1
    
    


#  YOUR ANSWER HERE:
#  as we can see from coordinates above, these properties, for instance can be useful:
# std_Density 4 3
# wtd_std_Density 4 4
# mean_ElectronAffinity 4 5
# wtd_mean_ElectronAffinity 4 6
# gmean_ElectronAffinity 4 7
# wtd_mean_Valence 8 0
# gmean_Valence 8 1
# 
# 
# In every subplot we can see an element with max. cr. temperature;
# 
# 
#  

# 

# ## Task 2:  Implement your own OLS estimator [10 pts]
# 
# We want to use linear regression to predict the critical temperature. Implement the ordinary least squares estimator without regularization 'by hand':
# 
# $w = (X^TX)^{-1}X^Ty$
# 
# To make life a bit easier, we provide a function that can be used to plot regression results. In addition it computes the mean squared error and the squared correlation between the true and predicted values. 

# In[ ]:


def plot_regression_results(y_test,y_pred,weights):
    '''Produces three plots to analyze the results of linear regression:
        -True vs predicted
        -Raw residual histogram
        -Weight histogram
        
    Inputs:
        y_test: (n_observations,) numpy array with true values
        y_pred: (n_observations,) numpy array with predicted values
        weights: (n_weights) numpy array with regression weights'''
    
    print('MSE: ', mean_squared_error(y_test,y_pred))
    print('r^2: ', r2_score(y_test,y_pred))
    
    fig,ax = plt.subplots(1,3,figsize=(9,3))
    #predicted vs true
    ax[0].scatter(y_test,y_pred)
    ax[0].set_title('True vs. Predicted')
    ax[0].set_xlabel('True %s' % (target_clm))
    ax[0].set_ylabel('Predicted %s' % (target_clm))

    #residuals
    error = np.squeeze(np.array(y_test)) - np.squeeze(np.array(y_pred))
    ax[1].hist(np.array(error),bins=30)
    ax[1].set_title('Raw residuals')
    ax[1].set_xlabel('(true-predicted)')

    #weight histogram
    ax[2].hist(weights,bins=30)
    ax[2].set_title('weight histogram')

    plt.tight_layout()


# As an example, we here show you how to use this function with random data. 

# In[ ]:


# weights is a vector of length 82: the first value is the intercept (beta0), then 81 coefficients
weights = np.random.randn(82)

# Model predictions on the test set
y_pred_test = np.random.randn(y_test.size)

plot_regression_results(y_test, y_pred_test, weights)


# Implement OLS linear regression yourself. Use `X_train` and `y_train` for estimating the weights and compute the MSE and $r^2$ from `X_test`. When you call our plotting function with the regession result, you should get mean squared error of 599.7.

# In[ ]:


def OLS_regression(X_test, X_train, y_train):
    '''Computes OLS weights for linear regression without regularization on the training set and 
       returns weights and testset predictions.
    
       Inputs:
         X_test: (n_observations, 81), numpy array with predictor values of the test set 
         X_train: (n_observations, 81), numpy array with predictor values of the training set
         y_train: (n_observations,) numpy array with true target values for the training set
         
       Outputs:
         weights: The weight vector for the regerssion model including the offset
         y_pred: The predictions on the TEST set
         
       Note:
         Both the training and the test set need to be appended manually by a columns of 1s to add
         an offset term to the linear regression model.        
    
    '''
    
    # ADD YOUR CODE HERE
    weights = np.dot(np.dot(np.dot(X_train.T, X_train)**(-1), X_train.T), y_train)
    #print('weights: ',weights)
    
    return weights, y_pred


# In[ ]:


np.dot(X_train.T * X_train)


# In[ ]:


np.dot(np.dot(np.dot(X_train.T, X_train)**(-1), X_train.T), y_train)


# In[ ]:


np.dot(X_train.T, X_train)**(-1)) * X_train.transpose() * y_train


# In[ ]:


weights, y_pred = OLS_regression(X_test, X_train, y_train)
plot_regression_results(y_test, y_pred, weights)


# What do you observe? Is the linear regression model good?

# YOUR ANSWER HERE

# ## Task 3: Compare your implementation to sklearn [5 pts]
# 
# Now, familarize yourself with the sklearn library. In the section on linear models:
# 
# https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
# 
# you will find `sklearn.linear_model.LinearRegression`, the `sklearn` implementation of the OLS estimator. Use this sklearn class to implement OLS linear regression. Again obtain estimates of the weights on `X_train` and `y_train` and compute the MSE and $r^2$ on `X_test`.
# 

# In[ ]:


def sklearn_regression(X_test, X_train, y_train):
    '''Computes OLS weights for linear regression without regularization using the sklearn library on the training set and 
       returns weights and testset predictions.
    
       Inputs:
         X_test: (n_observations, 81), numpy array with predictor values of the test set 
         X_train: (n_observations, 81), numpy array with predictor values of the training set
         y_train: (n_observations,) numpy array with true target values for the training set
         
       Outputs:
         weights: The weight vector for the regerssion model including the offset
         y_pred: The predictions on the TEST set
          
         
       Note:
         The sklearn library automatically takes care of adding a column for the offset.     
    
    '''
    
    # ADD YOUR CODE HERE
    
    return weights, y_pred


# In[ ]:


weights, y_pred = sklearn_regression(X_test, X_train, y_train)
plot_regression_results(y_test, y_pred, weights)


# If you implemented everything correctly, the MSE is again 599.74.

# Fit the model using the larger training set, `X_train_full` and `y_train_full`, and again evaluate on `X_test`.

# In[ ]:


weights, y_pred = sklearn_regression(X_test, X_train_full, y_train_full)
plot_regression_results(y_test, y_pred, weights)


#  How does test set performance change? What else changes?

# YOU ANSWER HERE

# ## Task 4: Regularization with ridge regression [15 pts]
# 
# We will now explore how a penalty term on the weights can improve the prediction quality for finite data sets. Implement the analytical solution of ridge regression 
# 
# $w = (XX^T + \alpha I_D)^{-1}X^Ty$
# 
# 
# as a function that can take different values of $\alpha$, the regularization strength, as an input. In the lecture, this parameter was called $\lambda$, but this is a reserved keyword in Python.

# In[ ]:


def ridge_regression(X_test, X_train, y_train, alpha):
    '''Computes OLS weights for regularized linear regression with regularization strength alpha 
       on the training set and returns weights and testset predictions.
    
       Inputs:
         X_test: (n_observations, 81), numpy array with predictor values of the test set 
         X_train: (n_observations, 81), numpy array with predictor values of the training set
         y_train: (n_observations,) numpy array with true target values for the training set
         alpha: scalar, regularization strength
         
       Outputs:
         weights: The weight vector for the regerssion model including the offset
         y_pred: The predictions on the TEST set
          
       Note:
         Both the training and the test set need to be appended manually by a columns of 1s to add
         an offset term to the linear regression model.       
    
    '''

    # ADD YOUR CODE HERE
        
    return weights, y_pred


# Now test a range of log-spaced $\alpha$s (~10-20), which cover several orders of magnitude, e.g. from 10^-7 to 10^7. 
# 
# * For each $\alpha$, you will get one model with one set of weights. 
# * For each model, compute the error on the test set. 
# 
# Store both the errors and weights of all models for later use. You can use the function `mean_squared_error` from sklearn (imported above) to compute the MSE.
# 

# In[ ]:


alphas = np.logspace(-7,7,100)

# ADD YOUR CODE HERE


# Make a single plot that shows for each coefficient how it changes with $\alpha$, i.e. one line per coefficient. Also think about which scale (linear or log) is appropriate for your $\alpha$-axis. You can set this using `plt.xscale(...)`.

# In[ ]:


# ADD YOUR CODE HERE


# Why are the values of the weights largest on the left? Do they all change monotonically? 

# YOUR ANSWER HERE

# Plot how the performance (i.e. the error) changes as a function of $\alpha$. Again, use appropriate scaling of the x-axis. As a sanity check, the MSE value for very small $\alpha$s should be close to the test-set MSE of the unregularized solution, i.e. 599.

# In[ ]:


# ADD YOUR CODE HERE


# Which value of $\alpha$ gives the minimum MSE? Is it better than the unregularized model? Why should the curve reach ~600 on the left?

# YOUR ANSWER HERE

# Now implement the same model using sklearn. Use the `linear_model.Ridge` object to do so.
# 

# In[ ]:


def ridge_regression_sklearn(X_test, X_train, y_train,alpha):
    '''Computes OLS weights for regularized linear regression with regularization strength alpha using the sklearn
       library on the training set and returns weights and testset predictions.
    
       Inputs:
         X_test: (n_observations, 81), numpy array with predictor values of the test set 
         X_train: (n_observations, 81), numpy array with predictor values of the training set
         y_train: (n_observations,) numpy array with true target values for the training set
         alpha: scalar, regularization strength
         
       Outputs:
         weights: The weight vector for the regerssion model including the offset
         y_pred: The predictions on the TEST set
          
       Note:
         The sklearn library automatically takes care of adding a column for the offset.     
   
    
    '''
    
    # ADD YOUR CODE HERE
            
    return weights, y_pred


# This time, only plot how the performance changes as a function of $\alpha$. 

# In[ ]:


# ADD YOUR CODE HERE


# Note: Don't worry if the curve is not exactly identical to the one you got above. The loss function we wrote down in the lecture  has $\alpha$ defined a bit differently compared to sklearn. However, qualitatively it should look the same.

# ## Task 5: Cross-validation [15 pts]
# 
# Until now, we always estimated the error on the test set directly. However, we typically do not want to tune hyperparameters of our inference algorithms like $\alpha$ on the test set, as this may lead to overfitting. Therefore, we tune them on the training set using cross-validation. As discussed in the lecture, the training data is here split in `n_folds`-ways, where each of the folds serves as a held-out dataset in turn and the model is always trained on the remaining data. Implement a function that performs cross-validation for the ridge regression parameter $\alpha$. You can reuse functions written above.

# In[ ]:


def ridgeCV(X, y, n_folds, alphas):
    '''Runs a n_fold-crossvalidation over the ridge regression parameter alpha. 
       The function should train the linear regression model for each fold on all values of alpha.
    
      Inputs: 
        X: (n_obs, n_features) numpy array - predictor
        y: (n_obs,) numpy array - target
        n_folds: integer - number of CV folds
        alphas: (n_parameters,) - regularization strength parameters to CV over
        
      Outputs:
        cv_results_mse: (n_folds, len(alphas)) numpy array, MSE for each cross-validation fold 
        
      Note: 
        Fix the seed for reproducibility.
        
        '''    
    
    cv_results_mse = np.zeros((n_folds, len(alphas)))
    np.random.seed(seed=2)

    
    # ADD YOUR CODE HERE
            
    return cv_results_mse    


# Now we run 10-fold cross-validation using the training data of a range of $\alpha$s.

# In[ ]:


alphas = np.logspace(-7,7,100)
mse_cv = ridgeCV(X_train, y_train, n_folds=10, alphas=alphas)


# We plot the MSE trace for each fold separately:

# In[ ]:


plt.figure(figsize=(6,4))
plt.plot(alphas, mse_cv.T, '.-')
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Mean squared error')
plt.tight_layout()


# We also plot the average across folds:

# In[ ]:


plt.figure(figsize=(6,4))
plt.plot(alphas, np.mean(mse_cv,axis=0), '.-')
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Mean squared error')
plt.tight_layout()


# What is the optimal $\alpha$? Is it similar to the one found on the test set? Do the cross-validation MSE and the test-set MSE match well or differ strongly?

# YOUR ANSWER HERE

# We will now run cross-validation on the full training data. This will take a moment, depending on the speed of your computer. Afterwards, we will again plot the mean CV curves for the full data set (blue) and the small data set (orange).

# In[ ]:


alphas = np.logspace(-7,7,100)
mse_cv_full = ridgeCV(X_train_full, y_train_full, n_folds=10, alphas=alphas)


# In[ ]:


plt.figure(figsize=(6,4))
plt.plot(alphas, np.mean(mse_cv_full,axis=0), '.-')
plt.plot(alphas, np.mean(mse_cv,axis=0), '.-')
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Mean squared error')
plt.tight_layout()


# We zoom in on the blue curve to the very right:

# In[ ]:


plt.figure(figsize=(6,4))
plt.plot(alphas, np.mean(mse_cv_full,axis=0), '.-')
plt.xscale('log')
minValue = np.min(np.mean(mse_cv_full,axis=0))
plt.ylim([minValue-.01, minValue+.02])
plt.xlabel('alpha')
plt.ylabel('Mean squared error')
plt.tight_layout()


# Why does the CV curve on the full data set look so different? What is the optimal value of $\alpha$ and why is it so much smaller than on the small training set?

# YOUR ANSWER HERE

# In[ ]:




