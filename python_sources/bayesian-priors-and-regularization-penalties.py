#!/usr/bin/env python
# coding: utf-8

# # Empirically Investigating the Relationship Between Bayesian Priors and Regularization Penalty Terms
# 
# Introductions to Bayesian methods often remark that regularizing linear regression with an L2 penalty is equivalent to having a Gaussian prior over the distribution of coefficient terms in a linear model. In this notebook, I would like to explore this empirically, by comparing the cross-validated performance of both a MLE and a Bayesian formulation of linear regression. I will verify that an L2 penalty term corresponds to a Gaussian prior, that an L1 penalty corresponds to a Laplacean prior, and will examine the relationship between the tuning parameter values of the penalties and the dispersions of the prior distributions.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymc3 as pm
import theano.tensor as tt

from scipy.stats import skew

from sklearn.linear_model import Lasso, Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

from priors_penalties_functions import BayesianGLM
from priors_penalties_functions import plot_errors_and_coef_magnitudes, cross_validate_hyperparam_choices

import os
print(os.listdir("../input"))


# ## Import/Preprocess Data

# In[ ]:


train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")


# In[ ]:


train.head()


# ### Preprocessing
# 
# I will follow the methodology in https://www.kaggle.com/apapiu/regularized-linear-models. That is, taking the log of skewed features, creating dummy variables for categorical features, and performing mean imputation.
# 
# After these steps, I will choose just 5 features to work with in order to reduce the feature space to something that is feasible to sample with MCMC. Note that because of this step, predictions made by models will not do very well on the leaderboard (they are relatively simple models to begin with, being variants of simple linear regression). That is okay, because the purpose of this notebook is to explore an equivalence between two formulations of linear regression.

# In[ ]:


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

#log transform the target:
train["SalePrice"] = np.log1p(train["SalePrice"])

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# Create dummy variables
all_data = pd.get_dummies(all_data)

# Mean imputation
all_data = all_data.fillna(all_data.mean())


# In[ ]:


X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = train.SalePrice


# In[ ]:


selector = SelectKBest(f_regression, k=5)
selector.fit(X_train, y)

scaler = StandardScaler()
scaler.fit(X_train, y)

columns = X_train.columns[selector.get_support()]

X_train = pd.DataFrame(selector.transform(scaler.transform(X_train)), columns=columns)
X_test = pd.DataFrame(selector.transform(scaler.transform(X_test)), columns=columns)

X_train.head()


# ## L2 Penalty (Ridge Regression)

# In[ ]:


cv_splitter = KFold(5)


# In[ ]:


alphas = np.logspace(0, 6, num=20)
alphas


# In[ ]:


results_l2 = cross_validate_hyperparam_choices(alphas, X_train, y, cv_splitter, Ridge)
results_l2


# In[ ]:


plot_errors_and_coef_magnitudes(results_l2, "Effect of L2 Penalty on Validation Error & Paramter Magnitude");


# ### Bayesian GLM
# 
# In the probabilistic formulation of linear regression, the response variable, $Y$, is treated as a random variable, equal to the weighted sum of the features, $\beta X$, plus random noise. The noise is typically assumed to be Gaussian, hence the distribution of $Y$ is also Gaussian.
# 
# That is, $Y \sim \mathcal{N}(\beta X, \sigma^2)$, where $\sigma^2$ is the variance of the noise term you would find in the standard formulation of linear regression, $Y = \beta X + \epsilon, \epsilon \sim \mathcal{N}$, $(0 \sigma^2)$.
# 
# Furthermore, we can specify prior distributions over the parameters $\beta$. A common choice is the Gaussian distribution. If we center this distribution around 0, this would indicate that we expect the parameters to be small. Choosing small values for the standard deviation of this prior would correspond to a tighter distribution, indicating a stronger initial belief in small parameters - similar to a large penalty in regularized least-squares regression.
# 
# PyMC3 has a module dedicated to Bayesian GLMs (https://docs.pymc.io/api/glm.html). However, for some reason they will not sample in the Kaggle kernel environment. I suspect it has to do with some Theano backend operation, as my first attempt to make a function from scratch that involved dot products had similar behavior. In any case, I have made my own GLM class due to the problems the library function has in this programming environment. I've modeled it after the scikit-learn API so that I can use it in my cross-validation loop.

# In[ ]:


sigmas = np.sqrt(1 / alphas)
sigmas


# In[ ]:


results_normal = cross_validate_hyperparam_choices(sigmas, X_train, y, cv_splitter, BayesianGLM, 
                                                   is_bayesian=True, bayesian_prior_fn=pm.Normal)
results_normal


# In[ ]:


plot_errors_and_coef_magnitudes(results_normal, 
                                "Effect of Prior Variance on Validation Error & Parameter Magnitude",
                                hyperparam_name="sigma",
                                reverse_x=True);


# ## L1 Penalty (Lasso)

# In[ ]:


results_l1 = cross_validate_hyperparam_choices(alphas, X_train, y, cv_splitter, Lasso)
results_l1


# In[ ]:


plot_errors_and_coef_magnitudes(results_l1, "Effect of L1 Penalty on Validation Error & Parameter Magnitude");


# ## Laplace Prior

# In[ ]:


results_laplace = cross_validate_hyperparam_choices(sigmas, X_train, y, cv_splitter, BayesianGLM, 
                                                   is_bayesian=True, bayesian_prior_fn=pm.Laplace)
results_laplace


# In[ ]:


plot_errors_and_coef_magnitudes(results_laplace, 
                                "Effect of Laplace Prior Variance on Validation Error & Parameter Magnitude",
                                hyperparam_name="sigma",
                                reverse_x=True);

