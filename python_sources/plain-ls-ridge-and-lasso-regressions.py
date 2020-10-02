#!/usr/bin/env python
# coding: utf-8

# ### Linear Regression Excersize
# For Freq ML class at the Cooper Union.
# 
# By: Guy Bar Yosef
# 
# In this kernel we will run a linear regression on a dataset provided by the World Health Organization (WHO) to predict the life expectancy of a country given various informaiton about diseases and the like.
# 
# More specifically, we will do plain Least Squares Regression, Ridge Regression, and Lasso Regression(the first two of which will be implemented by hand) and compare our results between the three

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import sys
import matplotlib.pyplot as plt


# ### Data Processing

# In[ ]:


# reading input data
input_data = pd.read_csv("../input/Life Expectancy Data.csv")
input_data.head()


# In[ ]:


# we will ignore countries and years, as they are a big giveaway for multiple years of the same country
x_input = input_data.drop(['Life expectancy ', 'Country', 'Year'], axis=1)  
y_input = input_data['Life expectancy ']


# Before we run any regression, let us clean up our data:
# 1. We will drop the 'Status' feature, as it is a categorical data with only 2 possibilities and when we incorporated it with one-hot encoding we ended up with very large weights, resulting in a large variance in our results.
# 2. We will use an imputer to deal with missing values such as NaN.

# In[ ]:


x_input = x_input.drop(['Status'], axis=1)


# In[ ]:


my_imputer = SimpleImputer(strategy="most_frequent") #strategy accounts for both strings and integer values
x_input = pd.DataFrame( my_imputer.fit_transform(x_input) )
y_input = pd.DataFrame( my_imputer.fit_transform(pd.DataFrame(y_input)) )


# Now let us split our data into a training, validation, and test set.
# The training set will be used for both the linear regression and the ridge regularization version, yet the validation set will only be used in order to tune the lambda hyperparameter of the ridge and lasso regularization. 
# 
# After we have both models, we will evaluate each with the test set and compare.

# In[ ]:


train_x, temp_x, train_y, temp_y = train_test_split(x_input,y_input, test_size=0.2)
val_x, test_x, val_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5)


# ### Plain Linear Regression through Least Squares
# 
# Let us now add a bias column to our input matrix, which will be used in our calculations. Note that this isn't part of our inputs, it is just used to simplify our calculations. Later when we do the lasso regression, we will get rid of it as we are using a package and not implementing it ourselves.

# In[ ]:


def add_bias(data):
    temp = data.copy()
    temp.insert(0,'bias',1)
    return temp


# In[ ]:


b_train_x = add_bias(train_x)
b_val_x = add_bias(val_x)
b_test_x = add_bias(test_x)


# In[ ]:


plain_lin_reg_weights = np.linalg.inv(np.transpose(b_train_x) @ b_train_x) @ np.transpose(b_train_x) @ train_y

plain_lin_reg_predictions = b_test_x @ plain_lin_reg_weights
plain_lin_reg_error = mean_squared_error(plain_lin_reg_predictions, test_y)


# ### Linear Regression with Ridge Regularization

# Before we begin, we will normalize our data (train, validation, and test) and add the bias to it.

# In[ ]:


def normalize (data):
    return (data - data.mean() ) / data.std()


# In[ ]:


norm_train_x = normalize(train_x)
norm_val_x = normalize(val_x)
norm_test_x = normalize(test_x)


# In[ ]:


bnorm_train_x = add_bias(norm_train_x)
bnorm_val_x = add_bias(norm_val_x)
bnorm_test_x = add_bias(norm_test_x)


# Let us now write out a function that will automate the cross-validation needed to find a good lambda, or penalty parameter.

# In[ ]:


# finds the optimal lambda value between the optional parameters for min(mn), max(mx), and jump size
def cross_val(train_x, train_y, val_x, val_y, mn = 0.001, mx = 2, jump = 0.0005):
    best_lambd = mn
    lowest_error = sys.maxsize
    
    for lambd in np.arange(mn, mx, jump):
        cur_weights = np.linalg.inv((np.transpose(train_x) @ train_x) + (lambd*np.eye(len(train_x.axes[1]))) ) @ np.transpose(train_x) @ train_y
        cur_predictions = val_x @ cur_weights
        cur_error = mean_squared_error(cur_predictions, val_y)
        if (lowest_error > cur_error):
            lowest_error = cur_error
            best_lambd = lambd
            
    return best_lambd


# In[ ]:


ridge_lambda = cross_val(bnorm_train_x, train_y, bnorm_val_x, val_y)
ridge_lin_reg_weights = np.linalg.inv((np.transpose(bnorm_train_x) @ bnorm_train_x) + (ridge_lambda * np.eye(len(bnorm_train_x.axes[1]))) ) @ np.transpose(bnorm_train_x) @ train_y

ridge_lin_reg_predictions = bnorm_test_x @ ridge_lin_reg_weights
ridge_lin_reg_error = mean_squared_error(ridge_lin_reg_predictions, test_y)


# ### Least Squares with Lasso Regression
# 
# Note that we are using a pacakge here. However we will find a new penalty coefficient, with an altered cross-validation function (due to the hand-implementation for the ridge regression and the package implementation of the lasso regression)

# In[ ]:


def lasso_cross_val2( train_x, train_y, val_x, val_y, mn = 0.001, mx = 2, jump = 0.0005):
    best_lambd = mn
    lowest_error = sys.maxsize
    for lambd in np.arange(mn, mx, jump):
        temp_model = linear_model.Lasso(alpha=lambd, max_iter=10000)
        temp_model.fit(train_x, train_y)
        cur_error = mean_squared_error(temp_model.predict(val_x), val_y)
        if (lowest_error > cur_error):
            lowest_error = cur_error
            best_lambd = lambd
            
    return best_lambd


# In[ ]:


lasso_lambd = lasso_cross_val2(norm_train_x, train_y, norm_val_x, val_y)
lasso_model = linear_model.Lasso(alpha=lasso_lambd, max_iter=10000)
lasso_model.fit(norm_train_x, train_y)

lasso_lin_reg_predictions = lasso_model.predict(norm_test_x)
lasso_lin_reg_error = mean_squared_error(lasso_lin_reg_predictions, test_y.values)


# #### Lasso Plot
# 
# Lets see which features the lasso regression decided to keep with a lasso plot (very much taken from http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html#sphx-glr-auto-examples-linear-model-plot-lasso-lars-py):
# 

# In[ ]:


alphas, _, coefs = linear_model.lars_path(train_x.values, train_y.values.ravel(), method='lasso')

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.figure(figsize=[20,7])
plt.plot(xx, coefs.T)

plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')

plt.ylim(0,3)
plt.show()


# ### Model Evaluations

# In[ ]:


print('The residual sum of square error is:\nPlain Least Squares Regression: ', plain_lin_reg_error,                                           '\nRidge Regression: ', ridge_lin_reg_error,                                            '\nLasso Regression: ', lasso_lin_reg_error)


# Having run this a few times (each time the training, validation, and test data being split differently) we have seen different regressors achieving the lowest mean squared error. With that said, all three preform very similarily overall. We found, on average, the mean squared error of all three hover around 16.5, meaning that on average we are 4ish years off in our life expectancy predictions.
# 
# Intrestingly, ridge and lasso regularization have not managed to produce consistently superior results to the plain least squares regression. This could be due to a flaw with the data; having multiple years for each country results in much of our input entries looking very similar. This could of course also be a fault of the ridge and lasso implementation, and as such a good follow up will be to use the scikit-learn implementations of plain least squares and ridge, and compare our results.
