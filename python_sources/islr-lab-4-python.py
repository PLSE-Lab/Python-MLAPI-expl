#!/usr/bin/env python
# coding: utf-8

# # Lab: Cross-Validation and the Bootstrap

# In this lab, we'll explore the four resampling techniques discussed in Chapter 5 of ISLR: the validation set approach, leave-one-out cross-validation (LOOCV), $k$-fold cross-validation, and the bootstrap. Note that some of the commands in this lab may take a while to run. While in the previous labs we used both StatsModels and scikit-learn, in this lab we will focus our attention on just scikit-learn, since StatsModels does not come with built-in classes for performing cross-validation. The two main scikit-learn modules I will be using for this lab are the [model selection module](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection) and the [metrics module](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics). A good starting point for reference is the [user guide section discussing cross validation in scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html) along with the [user guide section discussing model metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html).

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Note that if we wish to use R-style formulas, then we use statsmodels.formula.api
import statsmodels.api as sm
import statsmodels.formula.api as smf


# ## The Validation Set Approach

# First, we'll explore using the validation set approach to estimate the test error rates that result from fitting various linear models with the `Auto` data set.

# In[ ]:


auto_filepath = "../input/ISLR-Auto/Auto.csv"
Auto = pd.read_csv(auto_filepath, na_values = ["?"]).dropna()


# Note that before starting, we use the function `np.random.seed()` in order to set a *seed* for NumPy's random number generator, which is then used in scikit-learn. This is generally a good idea to do when performing analyses that contain an element of randomness, such as cross-validation, in order to have reproducible results.

# In[ ]:


np.random.seed(1)


# To start with, we use the `train_test_split()` function from scikit-learn's `model_selection` module to split the set of observations into four sets: $X$ and $y$ training sets, and $X$ and $y$ test sets. As described in the [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html#sklearn.model_selection.train_test_split), we can adjust the amount of the data set aside for the test sets by specifying a value for the argument `test_size`.

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(Auto["horsepower"], Auto["mpg"], test_size = 0.5)


# Once we have created the vector to denote our training set, we then use `X_train` and `y_train` to fit a linear regression model using only the observations corresponding to those in the training set.

# In[ ]:


reg = LinearRegression()
reg.fit(X_train.values.reshape(-1, 1), y_train)


# After fitting the linear regression of `mpg` onto `horsepower` using the training set, we use the `predict()` function to estimate the response for all 392 observations. Then, we use the `mean_squared_error()` function from scikit-learn's `metrics` module to compute the mean squared error of the 196 observations in the validation set.

# In[ ]:


metrics.mean_squared_error(y_test, reg.predict(X_test.values.reshape(-1, 1)))


# As we can see, the the estimated test mean squared error for the linear regression fit is 24.80. We can then use the `PolynomialFeatures` transformer to estimate the test error for the quadratic and cubic regressions. In order to efficiently apply the polynomial transformation and then fit the least squares regression model, we will combine the steps into a [pipeline](https://scikit-learn.org/stable/modules/compose.html) (for more information, look at the [api page](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline)). Pipelines are useful for convenience and encapsulation, joint parameter selection, and safety. While one can construct `Pipeline` objects directly, we will use the `make_pipeline()` convenience function, which is described [here](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html#sklearn.pipeline.make_pipeline) to simplify things.

# In[ ]:


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


# In[ ]:


quad_pipe = make_pipeline(PolynomialFeatures(degree = 2), LinearRegression())
quad_pipe.fit(X_train.values.reshape(-1, 1), y_train)
pred = quad_pipe.predict(X_test.values.reshape(-1, 1))
metrics.mean_squared_error(y_test, pred)


# In[ ]:


cube_pipe = make_pipeline(PolynomialFeatures(degree = 3), LinearRegression())
cube_pipe.fit(X_train.values.reshape(-1, 1), y_train)
pred = cube_pipe.predict(X_test.values.reshape(-1, 1))
metrics.mean_squared_error(y_test, pred)


# Here we see that the validation set mean squared error rates are 18.85 and 18.81 for the quadratic and cubic fits, respectively. Note that due to the element of randomness in choosing the training set, if we used a different seed (and therefore choose a possibly different training set), we will obtain somewhat different validation set error values.

# In[ ]:


np.random.seed(2)
X_train, X_test, y_train, y_test = train_test_split(Auto["horsepower"], Auto["mpg"], test_size = 0.5)


# In[ ]:


pipe = make_pipeline(PolynomialFeatures(degree = 1), LinearRegression())
pipe.fit(X_train.values.reshape(-1, 1), y_train)
pred = pipe.predict(X_test.values.reshape(-1, 1))
metrics.mean_squared_error(y_test, pred)


# In[ ]:


pipe = make_pipeline(PolynomialFeatures(degree = 2), LinearRegression())
pipe.fit(X_train.values.reshape(-1, 1), y_train)
pred = pipe.predict(X_test.values.reshape(-1, 1))
metrics.mean_squared_error(y_test, pred)


# In[ ]:


pipe = make_pipeline(PolynomialFeatures(degree = 3), LinearRegression())
pipe.fit(X_train.values.reshape(-1, 1), y_train)
pred = pipe.predict(X_test.values.reshape(-1, 1))
metrics.mean_squared_error(y_test, pred)


# Using this training set/validation set split, we get validation set mean squared error values of 23.44, 18.55, and 18.60 for the linear, quadratic, and cubic regression models, respectively. 
# 
# These results are consistent with our findings from the previous chapters that used an approach focused more on the statistics of the coefficient estimates: a quadratic model for predicting `mpg` using `horsepower` performs better than a linear model, and there isn't evidence to suggest that using a cubic model provides a meaningful improvement.

# ## Leave-One-Out Cross-Validation

# Before starting on leave-one-out cross-validation, we'll first go over the simplest way of computing cross-validated metrics: calling the `cross_val_score()` helper function on the estimator and the dataset. There are a few parameters for the `cross_val_score()` helper function to take note of in the context of this lab. These, and more parameters are also discussed on the [documentation page](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score).
# 
# - `estimator`: This is the object used to fit the data. For example it could be a single instance of an estimator class, such as `LinearRegression`, or it can be a pipeline of chained transformers and estimators.
# - `X`: The data to be used for fitting the estimator.
# - `y`: An optional array-like object to try and predict in the case of supervised learning. By default this is `None`.
# - `scoring`: An optional string or scorer callable object/function with the signature `scorer(estimator, X, y)`, which should only return a single value. If no value is provided, the estimator's default scorer (if available) will be used. Note that if we wish to use multiple metrics at once, as well as record fit/score times, then we should use `cross_validate()` instead, as discussed [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate). [Here is a list of the predefined scoring metrics and the names used to call them.](https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules).
# - `cv`: An optional integer, cross-validation generator, or iterable. The possible inputs for `cv` are:
#     - `None`, to use the default of 5-fold cross-validation
#     - an integer, to specify the number of folds for (stratified) $k$-fold cross-validation
#     - a [cross-validation splitter](https://scikit-learn.org/stable/modules/classes.html#splitter-classes)
#     - an iterable yielding (train, test) splits as arrays of indices
#     
# Note that since this lab focuses solely on $k$-fold cross-validation (as leave-one-out cross-validation is equivalent to $n$-fold cross-validation, where $n$ is the number of observations), we could simply provide appropriate integers to the `cv` parameter. For completeness, I will demonstrate how to use `LeaveOneOut` and `KFold` cross-validation splitters. To start out with, we'll import `cross_val_score`, `LeaveOneOut`, and `KFold`.

# In[ ]:


from sklearn.model_selection import cross_val_score, LeaveOneOut, KFold


# Now that we have made the appropriate imports, we'll perform leave-one-out cross-validation to estimate the test mean squared error. This time we will use the [cross-validation splitter class for LOOCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html#sklearn.model_selection.LeaveOneOut).
# 
# Note that scorer objects follow the convention that *higher return values are better than lower return values*. In other words, two given two scores $s_1 < s_2$, then $s_2$ is considered to be the better score. Thus, metrics which measure the distance between the model and the data, such as `metrics.mean_squared_error` are called in `cross_val_score()` by the name `neg_mean_squared_error`, to return the negated value of the metric and follow this convention.

# In[ ]:


# Using LeaveOneOut cross-validation splitter explicitly
X = Auto["horsepower"].values.reshape(-1, 1)
y = Auto["mpg"]
reg = LinearRegression()
loo = LeaveOneOut()
cv_scores = cross_val_score(reg, X, y, scoring = "neg_mean_squared_error", cv = loo)
# Since cv_scores is an array of scores, need to compute the mean afterward
cv_scores.mean()


# Recall that the $k$-fold cross validation estimate for the test mean squared error with $k$ folds is given by
# 
# \begin{equation}
#     \text{CV}_{(k)} = \frac{1}{k} \sum_{i = 1}^k \text{MSE}_i,
# \end{equation}
# 
# where $\text{MSE}_i$ is the mean squared error computed on the $i$th held out fold after fitting the model on the remaining $k - 1$ folds. In the case of leave-one-out cross-validation, $k = n$, the number of obserations in the data, and $\text{MSE}_i$ is the mean squared error $(y_i - \hat{y}_i)^2$ obtained after fitting the statistical learning method on all observations except for $y_i$.
# 
# Before moving on, note that we could use a different scoring method, such as mean absolute error.

# In[ ]:


# Performing leave-one-out cross-validation by passing on the number of observations
# as the argument cv
cv_scores = cross_val_score(reg, X, y, scoring = "neg_mean_absolute_error", cv = X.shape[0])
cv_scores.mean()


# We can make use of a `for` loop to iteratively repeat the procedure of leave-one-out cross-validation for increasingly complex polynomial fits. We will iteratively fit polynomial regressions for polynomials of order $i = 1, \dots, 10$, compute the associated LOOCV error, and store it in the $i$th element of the list `cv_error`. This sort of situation is where using a pipeline to encapsulate both the feature transformer and model class into a single estimator really comes in handy.

# In[ ]:


X = Auto["horsepower"].values.reshape(-1, 1)
y = Auto["mpg"]
loo = LeaveOneOut()
cv_error = []
for i in range(1, 11):
    pipe = make_pipeline(PolynomialFeatures(degree = i), LinearRegression())
    cv_scores = cross_val_score(pipe, X, y, scoring = "neg_mean_squared_error", cv = loo)
    cv_error.append(abs(cv_scores.mean()))
cv_error


# As we can see, there is a sharp drop in the estimated test mean squared error between the linear and quadratic fits, but then no clear improvement from using higher-order polynomials. This agrees with Figure 5.4 in ISLR.

# ## $k$-Fold Cross-Validation

# We can also use the `cross_val_score()` function to perform $k$-fold cross validation by supplying a value `K` to the function call. We'll use $k = 10$, a common choice for $k$, on the `Auto` data set and again use a `for` loop to iteratively compute the $k$-fold cross validation errors corresponding to polynomial fits of order $i = 1, \dots, 10$. Note that while $k$-fold cross validation involves random sampling, it will return consistent results for a given integer value of `cv`, since passing an integer `k` for `cv` tells `cross_val_score()` to use `KFold(n_splits = k)` as the cross-validation splitter. As discussed in [the documentation for the $k$-folds splitter](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold), the constructor for `KFold` can take an optional argument `shuffle`, which is `False` by default, to specify whether or not the data should be shuffled before computing the folds. If the data is not shuffled, then `KFold` will split it into consistent folds every time. 
# 
# *(I need to look more into the source code to check whether or not I am correct in this assessment.)*

# In[ ]:


# Using 10-fold cross-validation by passing the argument cv = 10 to cross_val_score()
X = Auto["horsepower"].values.reshape(-1, 1)
y = Auto["mpg"]
cv_error = []
for i in range(1, 11):
    pipe = make_pipeline(PolynomialFeatures(degree = i), LinearRegression())
    cv_scores = cross_val_score(pipe, X, y, scoring = "neg_mean_squared_error", cv = 10)
    cv_error.append(abs(cv_scores.mean()))
cv_error


# If we do choose to have `KFold` shuffle the data before splitting it into batches, then we should set a random seed to ensure consistent and reproducible results. The two main ways of doing this are using `np.random.seed()` or passing a value to the `random_state` argument in the constructor. 

# In[ ]:


# Using 10-fold cross-validation by passing an instance of KFold with shuffle = True
# In this situation the value of random_state matters
X = Auto["horsepower"].values.reshape(-1, 1)
y = Auto["mpg"]
kfolds = KFold(n_splits = 10, shuffle = True, random_state = 1)
cv_error = []
for i in range(1, 11):
    pipe = make_pipeline(PolynomialFeatures(degree = i), LinearRegression())
    cv_scores = cross_val_score(pipe, X, y, scoring = "neg_mean_squared_error", cv = kfolds)
    cv_error.append(abs(cv_scores.mean()))
cv_error


# Note that the computation time is much shorter than that of LOOCV; the computation was near instantaneous as opposed to taking about 30-45 seconds. While the formula
# 
# \begin{equation}
#     \text{CV}_{(n)} = \frac{1}{n} \sum_{i = 1}^n \left( \frac{y_i - \hat{y}_i}{1 - h_i} \right)^2,
# \end{equation}
# 
# where $\hat{y}_i$ is the $i$th fitted value from the original least squares fit, and $h_i$ is the leverage value of the $i$th observation, could be used, in princlple, to greatly speed up the computation of LOOCV in the case of estimating the mean squared error for least squares or polynomial regression, `cross_val_score()` does not make use of it to provide flexibilty in the estimators and scoring functions that can be used with it.
# 
# As with before, there is still little evidence that using cubic or higher-order polynomial terms leads to lower test error than simply using a quadratic fit.

# ## The Bootstrap

# To illustrate the use of the bootstrap, first we continue with the example shown in Section 5.2 of ISLR using the simulated `Portfolio` data set before moving on to an example involving estimating the accuracy of the linear regression model on the `Auto` data set.

# In[ ]:


portfolio_filepath = "../input/islr-lab-4/Portfolio.csv"
Portfolio = pd.read_csv(portfolio_filepath)


# One of the strengths of the bootstrap approach is that it is very widely applicable and does not require complicated mathematical complications; in Python, we only need to take two main steps to perform a bootstrap analysis. The first is creating a function that computes the statistic of interest. Second, we use the `resample()` function from the `sklearn.utils` module to repeatedly sample observations from the data with replacement and compute the bootstrapped statistics of interest. We start by importing the `resample()` function.

# In[ ]:


from sklearn.utils import resample


# ### Estimating the Accuracy of a Statistic of Interest

# As already noted, we start by working the the `Portfolio` data set from the `ISLR` package, which described in Section 5.2 of ISLR. It consists of 100 simulated pairs of returns for the investments $X$ and $Y$. We are trying to choose the the fraction of our money $\alpha$ to invest in $X$ (investing the remaining $1 - \alpha$ in $Y$) that minimizes the variance of our investment. In other words, we want to minimize $\text{Var}(\alpha X + (1 - \alpha)Y)$. It can be shown that the value of $\alpha$ which minimizes the risk is given by
# 
# \begin{equation}
#     \alpha = \frac{\sigma_Y^2 - \sigma_{XY}}{\sigma_X^2 + \sigma_Y^2 - 2\sigma_{XY}},
# \end{equation}
# 
# where $\sigma_X^2 = \text{Var}(X)$, $\sigma_Y^2 = \text{Var}(Y)$, and $\sigma_{XY} = \text{Cov}(X, Y)$. Since the `Portfolio` data set consists of simulated data, we know that the true values of the parameters were set to $\sigma_X^2 = 1$, $\sigma_Y^2 = 1.25$, and $\sigma_{XY} = 0.5$.
# 
# To illustrate our use of the bootstrap on this data, we first create a function, `alpha()`, which takes the argument `data`, which is the subset of the $(X, Y)$ data used to compute the estimated value of $\alpha$. The function then outputs the estimate for $\alpha$ based on the selected observations.

# In[ ]:


def alpha(data):
    X = data.X
    Y = data.Y
    return ((Y.var() - np.cov(X, Y)[0, 1])/(X.var() + Y.var() - 2*np.cov(X, Y)[0, 1]))


# The function returns an estimate for $\alpha$ by plugging in the estimates for $\sigma_X^2 = \text{Var}(X)$, $\sigma_Y^2 = \text{Var}(Y)$, and $\sigma_{XY} = \text{Cov}(X, Y)$ into the above formula for $\alpha$. For example, we can use the function to estimate $\alpha$ using all 100 observations in the `Portfolio` data set.

# In[ ]:


alpha(Portfolio)


# Next, we can use the `resample()` function to randomly select 100 observations, with replacement. This is equivalent to constructing a new bootstrap data set and recomputing $\hat{\alpha}$ with it.

# In[ ]:


sample = resample(Portfolio, n_samples = 100, random_state = 1)
alpha(sample)


# While we could perform bootstrap analysis by performing this command many times, recording all of the corresponding estimates $\hat{\alpha}$, and computing the resulting standard deviation, it is much more convenient to write a `for` loop to automate the process. Beow we produce 1000 bootstrap estimates for $\alpha$ and then compute the average and standard error.

# In[ ]:


np.random.seed(1)
boot_estimates = np.empty(1000)
for i in range(1000):
    sample = resample(Portfolio)
    boot_estimates[i] = alpha(sample)
print("Bootstrap estimated alpha:", boot_estimates.mean(), 
      "\nBootstrap estimated std. err.:", boot_estimates.std())


# The final output shows that when bootstrapping with the `Portfolio` data, $\hat{\alpha} = 0.5785$ and that the bootstrap estimate for $\text{SE}(\hat{\alpha})$ is $0.0929$. This is fairly close to the true value $\alpha = 0.6$.

# ### Estimating the Accuracy of a Linear Regression Model

# Next, we'll apply the bootstrap to assess the variability of the coefficient estimates and predictions from a statistical learning method. We'll demonstrate that by using the bootstrap to assess the variability of the estimates of $\beta_0$ and $\beta_1$, the intercept and slope terms for the linear regression model that uses `horsepower` to predict `mpg` in the `Auto` data set. We'll then compare the bootstrap estimates with the ones obtained using the formulas for $\text{SE}(\hat{\beta}_i)$ described in Section 3.1.2 of ISLR.
# 
# \begin{equation}
#     \text{SE}(\hat{\beta}_0)^2 = \sigma^2 \left[ \frac{1}{n} + \frac{\bar{x}^2}{\sum_{i = 1}^n (x_i - \bar{x})^2} \right], \,
#     \text{SE}(\hat{\beta}_1)^2 = \frac{\sigma^2}{\sum_{i = 1}^n (x_i - \bar{x})^2}
# \end{equation}
# 
# Recall that $\sigma$ is estimated by using the residual standard error (RSE) $\hat{\sigma}^2$ for the model, which is computed using the formula
# 
# \begin{equation}
#     \text{RSE} = \hat{\sigma} = \sqrt{\frac{1}{n - 2} \sum_{i = 1}^n (y_i - \hat{y}_i)^2}.
# \end{equation}
# 
# To start, we create the function `fit_coefs()` which takes in an array of training $X$ values, an array of training $y$ values, and a scikit-learn estimator class object (e.g. a LinearRegression object for this lab). It then returns a NumPy array consisting of the regression coefficients, with the regression intercept as the last entry.

# In[ ]:


def fit_coefs(X, y, estimator):
    reg = estimator.fit(X, y)
    coefs = reg.coef_
    intercept = reg.intercept_
    return np.append(coefs, intercept)


# First, we'll demonstrate applying this function to the full set of 392 observations in order to compute the estimates of $\beta_0$ and $\beta_1$ on the entire data set using the usual linear regression coefficient estimate formulas from Chapter 3 of ISLR.

# In[ ]:


X = Auto["horsepower"].values.reshape(-1, 1)
y = Auto["mpg"]
reg = LinearRegression()
pd.Series(fit_coefs(X, y, reg), index = ["horsepower", "intercept"])


# Just like we did with `alpha()`, we'll use `fit_coefs()` to create bootstrap estimates for the intercept and slope terms by randomly sampling from among the observations with replacement. Here are two examples before actually peforming the full bootstrap.

# In[ ]:


np.random.seed(1)
sample = resample(Auto)
X = sample["horsepower"].values.reshape(-1, 1)
y = sample["mpg"]
reg = LinearRegression()
pd.Series(fit_coefs(X, y, reg), index = ["horsepower", "intercept"])


# In[ ]:


sample = resample(Auto)
X = sample["horsepower"].values.reshape(-1, 1)
y = sample["mpg"]
reg = LinearRegression()
pd.Series(fit_coefs(X, y, reg), index = ["horsepower", "intercept"])


# Now we use a for loop to compute the standard errors of 1,000 bootstrap estimates for the intercept and slope terms.

# In[ ]:


np.random.seed(17)
reg = LinearRegression()
bootstrap_estimates = pd.DataFrame()
for i in range(1000):
    sample = resample(Auto)
    X = sample["horsepower"].values.reshape(-1, 1)
    y = sample["mpg"]
    coefs = pd.Series(fit_coefs(X, y, reg), index = ["horsepower", "intercept"], name = i)
    bootstrap_estimates = bootstrap_estimates.join(coefs, how = "right")
pd.DataFrame({"original": bootstrap_estimates.mean(axis = 1), "std. error": bootstrap_estimates.std(axis = 1)})


# We got bootstrap estimates of 0.85 for $\text{SE}(\hat{\beta}_0)$ and 0.0073 for $\text{SE}(\hat{\beta}_1)$. Now we'll compare them with the estimates computed using the above formulas, which can be obtained using the `summary()` or `bse()` functions in StatsModels.

# In[ ]:


exog = sm.add_constant(Auto["horsepower"])
endog = Auto["mpg"]
mod = sm.OLS(endog, exog)
res = mod.fit()
print(res.summary())


# In[ ]:


res.bse


# Using the formulas from Section 3.1.2, the standard error estimates are 0.717 for the intercept and 0.0064 for the slope, which are somewhat different from the bootstrap estimates. This is due to the assumptions underlying those formulas used for the estimation. 
# 
# - First, those formulas depend on the unknown parameter $\sigma^2$, the noise variance, which we estimated using the residual sum of squares. While the formulas for the standard errors of the coefficients don't rely on the correctness of the model, our estimate for $\sigma^2$ does. We saw previously that there is a non-linear relationship `mpg` and `horsepower`, which results in inflated residuals from a linear fit. In turn, this will affect our estimate of $\sigma^2$.
# - In addition, the standard formulas make the (somewhat unrealistic) assumption that the $x_i$ are fixed, and that all of the variability comes from the variation in the errors $\epsilon_i$.
# 
# The bootstrap approach doesn't rely on any of these assumptions, so it is likely that the bootstrap estimates are more accurate estimates of the standard errors of $\hat{\beta}_0$ and $\hat{\beta}_1$ than those from the `summary()` or `bse()` functions in StatsModels.
# 
# To conclude, we use the bootstrap to compute standard error estimates for a quadratic model and compare them with the standard linear regression estimates. Since a quadratic model provides a good fit to the data, there is now a better correpondence between the bootstrap estimates and the standard estimates of $\text{SE}(\hat{\beta}_i)$ for $i = 0, 1, 2$.

# In[ ]:


np.random.seed(17)
poly = PolynomialFeatures(degree = 2, include_bias = False)
reg = LinearRegression()
bootstrap_estimates = pd.DataFrame()
for i in range(1000):
    sample = resample(Auto)
    X = poly.fit_transform(sample["horsepower"].values.reshape(-1, 1))
    y = sample["mpg"]
    coefs = pd.Series(fit_coefs(X, y, reg), index = ["horsepower", "horsepower^2", "intercept"], name = i)
    bootstrap_estimates = bootstrap_estimates.join(coefs, how = "right")
pd.DataFrame({"original": bootstrap_estimates.mean(axis = 1), "std. error": bootstrap_estimates.std(axis = 1)})


# In[ ]:


poly = PolynomialFeatures(degree = 2)
exog = poly.fit_transform(Auto["horsepower"].values.reshape(-1, 1))
endog = Auto["mpg"]
mod = sm.OLS(endog, exog)
res = mod.fit()
print(res.summary())


# In[ ]:


res.bse

