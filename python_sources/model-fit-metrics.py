#!/usr/bin/env python
# coding: utf-8

# # Model Fit Metrics
# 
# Once we've built a model, it's important to understand how well it works. To do so, we evaluate the model against one or more metrics. This notebook is an overview of some of the most common metrics used for regression models.
# 
# We'll implement the metrics and test them out on a mocked-up regression target.

# In[ ]:


import numpy as np
from sklearn.linear_model import LinearRegression
clf = LinearRegression()

np.random.seed(42)
X = (np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=0.5))[:, 
                                                                                  np.newaxis]
y = (np.array(sorted(list(range(5))*20)) + np.random.normal(size=100, scale=0.25))[:, 
                                                                                   np.newaxis]

clf.fit(X, y)
y_pred = clf.predict(y)


# ## $R^2$
# 
# ### Discussion
# 
# The first and most immediately useful metric to use in regression is the $R^2$, also known as the coefficient of determination. For a vector of values $y$, a vector of predictions $\hat{y}$, both of length $n$, and a value average $\bar{y}$, $R^2$ is determined by:
# 
# $$R^2(y, \hat{y}) = 1 - \frac{\sum_0^{n-1} (y_i - \hat{y}_i)^2}{\sum_0^{n-1}(y_i - \bar{y})^2}$$
# 
# The coefficient of determination is a measure of how well future samples will be predicted by the model. The best possible score is 1. A constant model which always predicts the average will recieve a score of 0. A model which is arbitrarily worse than an averaging model will recieve a negative score (this shouldn't happen in practice obviously!).
# 
# In practice, it is a "best default" model score: other metrics may be better to use, depending on what you are optimizing for, but the $R^2$ is just generally very good, and should be the first number you look at in most cases.
# 
# $R^2$ is such a popular metric that there are artificial $R^2$ scores, designed to work in a similar way but with completely different underlying mathematics, which are defined for other non-regression operations.
# 
# ### Hand Implementation

# In[ ]:


import numpy as np

def r2_score(y, y_pred):
    rss_adj = np.sum((y - y_pred)**2)
    n = len(y)
    y_bar_adj = (1 / n) * np.sum(y)
    ess_adj = np.sum((y - y_bar_adj)**2)
    return 1 - rss_adj / ess_adj

r2_score(y, y_pred)


# ### Scikit-learn implementation

# In[ ]:


from sklearn.metrics import r2_score
r2_score(y, y_pred)


# ## Residual Sum of Squares (RSS)
# 
# ### Discussion
# 
# The residual sum of squares is the top term in the $R^2$ metric (albeit adjusted by 1 to account for degrees of freedom). It takes the distance between observed and predicted values (the residuals), squares them, and sums them all together. Ordinary least squares regression is designed to minimize exactly *this* value.
# 
# $$\text{RSS} = \sum_0^{n - 1} (y_i - \hat{y}_i)^2$$
# 
# RSS is not very interpretable on its own, because it is the sum of many (potentially very large) residuals. For this reason it is rarely used as a metric, but because it is so important to regression, it's often included in statistical fit assays.
# 
# ### Hand Implementation

# In[ ]:


def rss_score(y, y_pred):
    return np.sum((y - y_pred)**2)


# In[ ]:


rss_score(y, y_pred)


# There is no `scikit-learn` implementation.

# ## Mean Squared Error (MSE)
# 
# ### Discussion
# 
# Mean squared error is the interpretable version of RSS. MSE divides RSS (again adjusted be 1, to account for degrees of freedom) by the number of samples in the dataset to arrive at the average amount of squared error in the model:
# 
# $$\text{MSE} = \frac{1}{n} \cdot \sum_0^{n - 1} (y_i - \hat{y}_i)^2$$
# 
# This is easily interpretable, because it makes a lot of intrinsic sense. Ordinary least squares regression asks that we minimize quadratic error; MSE measures, on average, how much such error is left in the model. However, due to the squaring involved, it is not very robust against outliers.
# 
# ### Hand Implementation

# In[ ]:


def mean_squared_error(y, y_pred):
    return (1 / len(y)) * np.sum((y - y_pred)**2)

mean_squared_error(y, y_pred)


# ### Scikit-learn implementation

# In[ ]:


from sklearn.metrics import mean_squared_error

mean_squared_error(y, y_pred)


# ## Mean Absolute Error
# 
# ### Discussion
# 
# Mean absolute error computes the expected absolute error (or [L1-norm loss](https://www.kaggle.com/residentmario/l1-norms-versus-l2-norms)). Because it involves means, not squared residuals, mean absolute error is more resistant to outliers than MSE is.
# 
# $$\text{MAE}(y, \hat{y}) = \frac{1}{n}\sum_0^{n-1} | y_i - \hat{y}_i |$$

# ### Hand implementation

# In[ ]:


def mean_absolute_error(y, y_pred):
    return (1 / len(y)) * np.sum(np.abs(y - y_pred))
    
mean_absolute_error(y, y_pred)


# ### Scikit-learn implementation

# In[ ]:


from sklearn.metrics import mean_absolute_error
    
mean_absolute_error(y, y_pred)


# ## Median Absolute Error
# 
# ### Discussion
# 
# Mean absolute error computes the median absolute error. Because this value is not only an absolute value, but also a median instead of a mode, this metric is the most resistant metric to outliers that's possible using simple methods.
# 
# $$\text{Mean Absolute Error} = \text{median}(|y_0 - \hat{y}_0, \ldots, |y_n - \hat{y}_n|)$$

# ### Hand implementation

# In[ ]:


def median_absolute_error(y, y_pred):
    return np.median(np.abs(y - y_pred))
    
mean_absolute_error(y, y_pred)


# ### Scikit-learn implementation

# In[ ]:


from sklearn.metrics import median_absolute_error

median_absolute_error(y, y_pred)


# ## Root mean squared error (RMSE)
# 
# ### Discussion
# 
# Root mean squared error is an error metric that's popular in the literature. It is defined as the square root of mean squared error:
# 
# $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n}\sum_0^{n - 1} (y_i - \hat{y}_i)^2}$$
# 
# Since this is just the root of the MSE metric mentioned earlier, we will omit an implementation.
# 
# RMSE is directly comparable to, and serves a similar role as, the MAE, mean absolute error. The difference between the two computationally speaking is that MAE takes the square root of the distance inside the sum, while RMSE takes the square root outside the sum.
# 
# The computational effect is that RMSE is less resistant to outliers, and thus reports a poorer-fitting model when outliers are not properly accounted for. This is considered a good thing when doing cetain things, like performing hyperparameter searches. However, MAE is a more useful reporting statistic because MAE is *interpretable*, while RMSE is not.
# 
# Context for this comparison [here](https://medium.com/human-in-a-machine-world/mae-and-rmse-which-metric-is-better-e60ac3bde13d).
# 
# Note that `scikit-learn` doesn't provide a RMSE evaluator directly...

# ## Explained variance score
# 
# ### Discussion
# 
# The explained variance score is a very clever (IMO) metric which looks at the ratio between the variance of the model/truth differences and the variance of the ground truth alone:
# 
# $$\text{explained variance}(y, \hat{y}) = 1 - \frac{Var({y}) - Var(\hat{y})}{Var(y)}$$
# 
# Hence the moniker "explained variance". The best possible score is 1 (all variance is explained) and the score goes down from there. A further reference on explained variance is [here](https://assessingpsyche.wordpress.com/2014/07/10/two-visualizations-for-explaining-variance-explained/).

# ### Hand implementation

# In[ ]:


def explained_variance_score(y, y_pred):
    return 1 - (np.var(y - y_pred) / np.var(y))

explained_variance_score(y, y_pred)


# ### Scikit-learn implementation

# In[ ]:


from sklearn.metrics import explained_variance_score

explained_variance_score(y, y_pred)


# That concludes this metrics overview section!  Future notebooks will likely explore the comparisons between and decisions about what metric to use in more detail. There are also many other common metrics that we can use that we will consider later as well.
