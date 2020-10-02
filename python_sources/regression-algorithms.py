#!/usr/bin/env python
# coding: utf-8

# ## Regression algorithms notebook (60 min)

# This is one of the notebooks for the third session of the [Machine Learning workshop series at Harvey Mudd College](http://www.aashitak.com/ML-Workshops/). It involves a gentle introduction overviewing the various regression algorithms. Please follow your own pace. It would be wise to overlook the code at this point and pay attention to the explanation and review the results and charts and develop the conceptual understanding. In the exercise notebook, you will work with another dataset and apply the algorithms learned here. With that practise, the code will become clearer.
# 
# You are most welcome to ask questions and discuss the concepts with the instructor or the TAs.

# We import python modules:

# In[ ]:


import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error

import warnings
warnings.simplefilter('ignore')


# We will learn regression techniques using [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview) dataset. The aim is to predict house prices based on a set of features.

# In[ ]:


path = '../input/'
housing = pd.read_csv(path + 'train.csv')
housing.head()


# In[ ]:


housing.shape


# There are ***81 columns in total***. Before looking deeply into the columns, we might want to first discard the columns that have a lot of missing values. We can revisit later to include and process all the columns, but at first we would want to work with fewer columns.

# In[ ]:


housing.isnull().sum()


# We discard all the columns that have more than 100 missing values.

# In[ ]:


housing = housing.loc[:, housing.isnull().sum() < 100]


# Let us look at the remaining columns.

# In[ ]:


housing.columns


# Let us see how does the columns correlate with the *SalePrice* using the Pearson correlation coefficients that we learned in the first session.

# In[ ]:


correlations = housing.corr()['SalePrice']
correlations


# Let us select only those columns that have a correlation co-efficient of 0.5 or higher in magnitude with the *SalePrice* column.

# In[ ]:


correlations[(correlations > 0.5) | (correlations < -0.5)]


# We pick a few columns for now to train a model. We can anytime revisit and try a different set of columns for prediction.

# In[ ]:


y = housing['SalePrice']

predictor_cols = ['OverallQual', 'YearBuilt', 
                  'YearRemodAdd', 'TotalBsmtSF', 
                  '1stFlrSF', 'GrLivArea', 
                  'FullBath', 'TotRmsAbvGrd', 
                  'GarageCars', 'GarageArea',
                 'Fireplaces', 'LotArea']

X = housing[predictor_cols]
X.head()


# [Features' description](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data):
# * SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
# * OverallQual: Overall material and finish quality
# * YearBuilt: Original construction date
# * YearRemodAdd: Remodel date
# * TotalBsmtSF: Total square feet of basement area
# * 1stFlrSF: First Floor square feet
# * GrLivArea: Above grade (ground) living area square feet
# * FullBath: Full bathrooms above grade
# * TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
# * GarageCars: Size of garage in car capacity
# * GarageArea: Size of garage in square feet
# * Fireplaces: Number of fireplaces
# * LotArea: Lot size in square feet

# Let us have a look at how the sales prices are distributed by plotting the histogram.

# In[ ]:


y.hist();


# It seems like the data is skewed to the right, that is there are a few houses with extra-ordinarily high prices. For linear regression techniques, symmetric data is more conducive to work with, so we take a log transform of the target variable and then plot its distribution.

# In[ ]:


y = np.log1p(y)
y.hist();


# Now, the target variable `y` is more symmetrically distributed (It is closer to [normal (or guassian) distribution also known as the bell curve](https://www.khanacademy.org/math/statistics-probability/modeling-distributions-of-data/more-on-normal-distributions/v/introduction-to-the-normal-distribution) if you are familiar with it).

# The [metric used](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/evaluation) for evaluating the submissions for the competition is given below:
# 
# > Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)
# 
# We have already taken the log transformation for `y` to make sure "that errors in predicting expensive houses and cheap houses will affect the result equally." We have encountered Root-Mean-Squared-Error (RMSE) in a disguised form in the cost function for linear regression and will revisit it later. 

# First we split the data into training and validation set.

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                        random_state = 0)


# Then we train a linear regression model using the training set and calculate the $R^2$ score for both training and validation set. 

# In[ ]:


linreg = LinearRegression().fit(X_train, y_train)

print('R-squared score (training): {:.3f}'
     .format(linreg.score(X_train, y_train)))
print('R-squared score (validation): {:.3f}'
     .format(linreg.score(X_valid, y_valid)))


# As we learned earlier, the coefficient of determination, [R-squared](http://www.fairlynerdy.com/what-is-r-squared/) (denoted by $R^2$), is a statistical measure of how close the data are to the fitted regression line. We note there is quite some difference between the $R^2$ values for the training and the validation set.

# ### Polynomial regression
# An extension of linear regression is polynomial regression, which fits a polynomial curve instead of a line to the data points. 
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/8/8a/Gaussian_kernel_regression.png" width="300" height="350" />
# <p style="text-align: center;"> Polynomial regression curve of degree 3 (cubic) </p> 
# 
#  
# 
# As a refresher, the equation for linear regression with features (variables) $x_1, x_2, \dots, x_n$ is 
# $$ y_{pred} = b + w_1 * x_1 + w_2 * x_2 + \cdots + w_n * x_n$$
# 
# To model polynomial regression, we simply take higher degrees of the features (variables) and build a linear regression model on them.
# For example, the polynomial regression equation for the cubic curve in the above figure would be would be 
# $$ y_{pred} = b + w_1 * x + w_2 * x^2 + w_3 * x^3$$ 
# 
# It can also be thought of as a linear regression model with three variables $x$, $x^2$ and $x^3$. So, we first transform the feature $x$ to create two additional features $x^2$ and $x^3$ by simply taking squares and cubes of the values of feature $x$ and then train a linear regression model on the three features $x$, $x^2$ and $x^3$.

#  We transform the original input features using [`sklearn.preprocessing.PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) to add polynomial features up to degree 2 (quadratic).

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train_poly, X_valid_poly, y_train_poly, y_valid_poly = train_test_split(X_poly, y,
                                                   random_state = 0)


# Then we use the built-in [`LinearRegression()`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) with the polynomial features to build a polynomial regression model.

# In[ ]:


polyreg = LinearRegression().fit(X_train_poly, y_train_poly)

polyreg_train_score = polyreg.score(X_train_poly, y_train_poly)
polyreg_valid_score = polyreg.score(X_valid_poly, y_valid_poly)

print('R-squared score (training): {:.3f}'
     .format(polyreg_train_score))
print('R-squared score (validation): {:.3f}'
     .format(polyreg_valid_score))


# It is great to see that the performance on the validation set has increased noticably by using the polynomial features.

# On account of increased complexity, polynomial regression models can be prone to overfitting. In that case, they can be coupled with Ridge regression, Lasso regression or Elastic Net regression - extensions of linear regression algorithm that use regularization, a method to address overfitting.
# 
# ### Regularization:
# When we have weights that are higher in magnitude, the model is more likely to overfit, so we penalize the weights using a penalty term called regularization parameter (alpha) that keeps the weights small and thereby simplify the model.
# 
# To the linear regression formulation of the cost function, we add the model weights multiplied by the regularization parameter (alpha) so that when the learning process minimizes the cost function while updating the weights, it automatically keeps the weights in check. There are two common ways to add the weights term (using $L1$ and $L2$-norms) and hence the two different algorithms as below.
# 
# Cost function for Linear regression:
# $$ J = \frac{1}{2 n} \sum_{i=1}^n (y^{(i)} - y_{pred}^{(i)})^2 $$
# 
# Cost function for Ridge regression ($L2$-norm):
# $$ J = \frac{1}{2 n} \sum_{i=1}^n (y^{(i)} - y_{pred}^{(i)})^2 + \alpha \sum_{j=1}^m w_j^2$$
# 
# Cost function for Lasso regression ($L1$-norm):
# $$ J = \frac{1}{2 n} \sum_{i=1}^n (y^{(i)} - y_{pred}^{(i)})^2 + \alpha \sum_{j=1}^m |w_j|$$
# 
# 
# For each algorithm, the learning process is the same as linear regression, that is we update the weights using gradient descent to minimize the respective cost function. Since weights are included in the cost function, the learning process takes a balanced approach between minimizing mean-square errors and the weights, thereby reducing overfitting. 
# 
# This technique of regularization by adding the weight term to the cost function using either $L1$ or $L2$-norm is one of the two most commonly used and crucial techniques to address overfitting in deep neural networks as well. The other one being [Dropout regularization](https://medium.com/@amarbudhiraja/https-medium-com-amarbudhiraja-learning-less-to-learn-better-dropout-in-deep-machine-learning-74334da4bfc5) - dropping out units in the network.
# 
# 
# We use [`sklearn.linear_model.Lasso`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) to train a Lasso regression model on the polynomial features.

# In[ ]:


polyreg_lasso = Lasso(alpha=100).fit(X_train_poly, y_train_poly)

print('R-squared score (training): {:.3f}'
     .format(polyreg_lasso.score(X_train_poly, y_train_poly)))
print('R-squared score (validation): {:.3f}'
     .format(polyreg_lasso.score(X_valid_poly, y_valid_poly)))


# Please try different alpha values for the above, such as alpha=5, 500, 5000, etc. and see how it affects the model performance.
# 
# Now we use [`sklearn.linear_model.Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) to train a Ridge regression model. Please try to tune the alpha parameter to find an optimal value.

# In[ ]:


polyreg_ridge = Ridge(alpha=100).fit(X_train_poly, y_train_poly)

print('R-squared score (training): {:.3f}'
     .format(polyreg_ridge.score(X_train_poly, y_train_poly)))
print('R-squared score (validation): {:.3f}'
     .format(polyreg_ridge.score(X_valid_poly, y_valid_poly)))


# To compare the effect of regularization parameter (alpha) as well as to compare different models, we define a few functions to plot the $R^2$ scores and another metric called root mean-squared error (RMSE), explained below. Please feel free to skip the code entirely.

# In[ ]:


def get_scores(reg):
    train_score = reg.score(X_train_poly, y_train_poly)
    valid_score = reg.score(X_valid_poly, y_valid_poly)
    return train_score, valid_score

def get_rmse(reg):
    y_pred_train = reg.predict(X_train_poly)
    train_rmse = np.sqrt(mean_squared_error(y_train_poly, y_pred_train))
    y_pred_valid = reg.predict(X_valid_poly)
    valid_rmse = np.sqrt(mean_squared_error(y_valid_poly, y_pred_valid))
    return train_rmse, valid_rmse

def ridge_validation_curve(alpha):
    reg = Ridge(alpha=alpha).fit(X_train_poly, y_train_poly)
    train_score, valid_score = get_scores(reg)
    train_rmse, valid_rmse = get_rmse(reg)  
    return train_score, valid_score, train_rmse, valid_rmse

def lasso_validation_curve(alpha):
    reg = Lasso(alpha=alpha).fit(X_train_poly, y_train_poly)
    train_score, valid_score = get_scores(reg)
    train_rmse, valid_rmse = get_rmse(reg)  
    return train_score, valid_score, train_rmse, valid_rmse

alphas = [0.1, 1, 5, 25, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 2000]

scores_lasso = [lasso_validation_curve(alpha) for alpha in alphas]
scores_lasso_train = [s[0] for s in scores_lasso]
scores_lasso_valid = [s[1] for s in scores_lasso]
rmse_lasso_train = [s[2] for s in scores_lasso]
rmse_lasso_valid = [s[3] for s in scores_lasso]

scores_ridge = [ridge_validation_curve(alpha) for alpha in alphas]
scores_ridge_train = [s[0] for s in scores_ridge]
scores_ridge_valid = [s[1] for s in scores_ridge]
rmse_ridge_train = [s[2] for s in scores_ridge]
rmse_ridge_valid = [s[3] for s in scores_ridge]

scores_poly_train = [polyreg_train_score]*len(alphas)
scores_poly_valid = [polyreg_valid_score]*len(alphas)
y_pred_train = polyreg.predict(X_train_poly)
rmse_poly_train = [mean_squared_error(y_train_poly, y_pred_train)]*len(alphas)
y_pred_valid = polyreg.predict(X_valid_poly)
rmse_poly_valid = [mean_squared_error(y_valid_poly, y_pred_valid)]*len(alphas)


# Now we plot the $R^2$ scores for the four regression algorithms for different values of alpha for both training and validation set.

# In[ ]:


plt.figure(figsize=(10, 6));
plt.ylim([0.65, 0.9])
plt.xlabel('Regularization parameter (alpha)')
plt.ylabel('R-squared')
plt.title('R-squared scores as function of regularization')

plt.plot(alphas, scores_ridge_train, label='Poynomial with Ridge (training)')
plt.plot(alphas, scores_poly_train, label='Polynomial (training)')
plt.plot(alphas, scores_lasso_train, label='Poynomial with Lasso (training)')

plt.plot(alphas, scores_lasso_valid, label='Poynomial with Lasso (validation)')
plt.plot(alphas, scores_ridge_valid, label='Poynomial with Ridge (validation)')
plt.plot(alphas, scores_poly_valid, label='Polynomial (validation)')
plt.legend(loc=4);


# We revise that higher the $R^2$ score, the better fit is the regression curve to the data points. We also note that a big difference between the $R^2$ scores for the training and validation sets suggests overfitting.  

# Polynomial regression model regularized using Lasso seems to be performing the best since it has the highest $R^2$ score on the validation set and almost no difference between the $R^2$ scores on the training and validation sets.
# Considering the $R^2$ values on the validation set, the order of performance from the best to worse changes is:
# 1. Polynomial regression coupled with Lasso
# 2. Polynomial regression coupled with Ridge
# 3. Polynomial regression

# Let us also look at the root mean-squared error (RMSE). The mean-squared error (MSE) happens to be implicitly included in the cost function for all the regression methods seen so far.
# $$ \frac{1}{n} \sum_{i=1}^n (y^{(i)} - y_{pred}^{(i)})^2 $$
# 
# It is a measure of the error in predicting the target values. This also happens to be the metric used to evaluate the regression model for this dataset.
# 
# We will use [`sklearn.metrics.mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) function followed by `np.sqrt()` (square-root) to calculate the RMSE and plot a graph to compare the RMSE values for the above four models.

# In[ ]:


plt.figure(figsize=(11, 6));
plt.ylim([0.012, 0.3])
plt.xlabel('Regularization parameter (alpha)')
plt.ylabel('Root Mean-squared Error(RMSE)')
plt.title('Root Mean-squared Error(RMSE) as a function of regularization')

plt.plot(alphas, rmse_lasso_valid, label='Poynomial with Lasso (validation)')
plt.plot(alphas, rmse_ridge_valid, label='Poynomial with Ridge (validation)')
plt.plot(alphas, rmse_poly_valid, label='Polynomial (validation)')

plt.plot(alphas, rmse_poly_train, label='Poynomial (training)')
plt.plot(alphas, rmse_lasso_train, label='Poynomial with Lasso (training)')
plt.plot(alphas, rmse_ridge_train, label='Poynomial with Ridge (training)')

plt.legend(loc=1);


# Lower root mean-squared error (RMSE) is preferable. The root mean-squared error (RMSE) on the validation set follows the ascending order:
# 1. Polynomial regression 
# 2. Polynomial regression coupled with Lasso
# 3. Polynomial regression coupled with Ridge
# 
# Polynomial regression coupled with Lasso has little to no difference between the RMSE for training and validation set, which suggests it is likely not overfitting to the training set. However, the simple polynomial regression model has a much lower RMSE value on both training and validation sets compared to all other algorithms. 

# Now we finally predict the examples in the `test.csv` file and [submit our predict to the competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/frequently-asked-questions). 

# In[ ]:


housing_test = pd.read_csv(path + 'test.csv')
Id = housing_test['Id']
X_test = housing_test[predictor_cols]
X_test.head()


# In[ ]:


X_test.isnull().sum()


# We fill in missing values for the columns.

# In[ ]:


X_test = X_test.fillna(method='ffill')


# Now we choose a model to build our final submission, you can pick any. I have simply used the polynomial regression trained on the ***entire data set*** transformed by the polynomial features, that is `X_poly, y`.
# 
# Note that: We don't anymore need to keep the validation set out of training since we have chosen the regression algorithm and the alpha value already, so we are better off using the entire set for training. 

# In[ ]:


reg = LinearRegression().fit(X_poly, y)


# Next we transform the test data features to the quadratic polynomial features using the transformer `poly` that we fit earlier.

# In[ ]:


X_test_poly = poly.transform(X_test)
predictions = reg.predict(X_test_poly)
predictions[:10]


# Since we took the log transform for the target variable at the beginning, for the final answer, we need to use its inverse function, that is exponential to undo the transformation for the predictions.

# In[ ]:


predictions = np.expm1(predictions) 
predictions[:10]


# These are the predictions for the sales prices for the houses. We have built a simple baseline model with a lot of scope for improvement, but it is ready for submission.

# Now we have a look at the sample submission. It is important that our submission file is in correct format to be graded without errors.

# In[ ]:


sample_submission = pd.read_csv(path + 'sample_submission.csv')
sample_submission.head()


# We create a dataframe for submission.

# In[ ]:


submission = pd.DataFrame({'Id': Id,
                          'SalePrice': predictions})

submission.head()


# We save the dataframe as a csv file.

# In[ ]:


submission.to_csv('my_submission.csv', index=False)


# Go to the [competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/evaluation) and make a submission by uploading the `my_submission.csv` file generated thru this notebook that is automatically stored in the same folder as this notebook in your laptop.
# 
# Note: If you are using Kaggle kernels, then
# 1. Commit the notebook
# 2. Leave the edit mode
# 3. Navigate to the output section
# 4. Download the csv file
# 5. Go to the [competition page](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/evaluation) 
# 6. Make a submission by uploading the `my_submission.csv` file

# ### Optional:
# 
# Feature engineering
# 1. Try out different sets of columns, even including those that have missing values by filling them first. You can also explore using polynomial features of higher or lower degrees than 2 and choosing the regularization accordingly.
# 
# 

# ### Acknowledgement:
# 
# The credits for the images used in the above are as follows.
# - Image 1: https://commons.wikimedia.org/wiki/File:Gaussian_kernel_regression.png

# Please proceed on to the [Exercise 3 notebook](https://www.kaggle.com/aashita/exercise-3).

# In[ ]:




