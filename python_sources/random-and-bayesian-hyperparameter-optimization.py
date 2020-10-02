#!/usr/bin/env python
# coding: utf-8

# ## Grid search
# 
# The simplest form of hyperparameter optimization is picking a few possible likely values and trying them out. The next simplest form of optimization is **grid search**: defining a cross section of points (a grid) that the model is to be evaluated on.
# 
# I give a demonstration of grid search using the `scikit-learn` library in the following Kaggle notebook: ["NYC Buildings, Part 2: Feature Scales, Grid Search"](https://www.kaggle.com/residentmario/nyc-buildings-part-2-feature-scales-grid-search/).
# 
# ## Random search
# 
# Grid search is still a relatively brute-force manner of performing hyperparameter optimization. It wastes a lot of time looking at parameter combinations which, based on combinations already seen, are almost certainly incorrect.
# 
# For parameter spaces with more than three dimensions, it is recommended to switch from grid search to **random search**. In random search instead of iterating over a grid of points we instead sample a list of points to try from a random user-specific probability distribution. This allows the user to bias the search towards the distributions that matter, ultimately resulting in a more effective search:
# 
# ![](https://i.imgur.com/MIQZUEF.png)
# 
# (these notes based on the following blog post: [link](https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/))
# 
# ### Implementation
# Random search is implemented in `scikit-learn` as `RandomizedSearchCV`, covered [here](https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization) in the documentation.
# 
# With neural networks, there are a few different options. For small-scale sequential hyperparameter optimization in Keras, use [talos](https://github.com/autonomio/talos) or [hyperopt](https://github.com/hyperopt/hyperopt). For distribued hyperparameter optimization with any of the mainstream neural network frameworks, use [horovod](https://github.com/horovod/horovod).
# 
# ## Bayesian hyperparameter optimization
# 
# **Hyperparameter optimization** is the process of finding model hyperparameters which are a good fit for the data being modeled.
# 
# **Bayesian hyperparameter optimization** is the answer. The word "Bayesian" comes from Bayes theorem, a formula for constructing an understanding of a probability based on a known prior. Bayesian statistics is a branch of statistics concerned with techniques which utilize such priors. It is often constrasted with frequentist statistics, which is usually more concerned with modeling an underlying distribution instead.
# 
# In Bayesian hyperparameter opt, the known prior is the list of points in the parameter space that have already been tried, and new points are tried based on those known prior points. This has the significant advantage over simple grid search that the global cost minimum can usually be achieved with far fewer iterations. A Bayesian hyperparameter optimization algorithm is any algorithm which uses such knowledge to perform hyperparameter opt.
# 
# ### How it works
# Bayesian search is broadly similar to stochastic gradient descent, the algorithm used in optimizing neural networks (and certain other learning algorithms). In stochastic gradient descent, we feed a few samples of data to the model, evaluate how far from the truth we are using a cost function, construct a gradient based on that information, and then iteratively move into the direction of the gradient until we (hopefully) get to the gloal minima.
# 
# In the case of hyperparameter optimization, each data point corresponds with a new model build, which is expensive. Constructing an accurate gradient is too expensive to be worthwhile. Instead we model the cost surface using a flexible form of approximation&mdash;a Gaussian process. This is the same mechanism used for KDE estimation in e.g. `seaborn.kdeplot`. And instead of following a gradient, we use the model's minima and maxima estimates (which converge to a known value at points on the cost surface that we have already tried, but are scoped-out in between) to select the next candidate point to be tried.
# 
# At the beginning you start with something that looks like this:
# 
# ![](https://i.imgur.com/ZcEq1zg.png)
# 
# And ultimately you end up with something that looks like this:
# 
# ![](https://i.imgur.com/MI6B79W.png)
# 
# ### Difficulties
# Bayesian hyperparameter search is difficult to use in practice because it, ironically, is sensitive to its own hyperparameters. The choice of the covariance function and the kernel function together control the properties of the estimate for the unknown areas in between points, and how well this estimate maps to the functional reality has everything to do with how well Bayesian hyperparemeter search will perform.
# 
# It's also supposedly computationally challenging to implement in a generalized way.
# 
# ### Implementations
# As far as I can tell, there are no mature, well-supported implementations of Bayesian hyperparameter optimization available in Python.
# 
# ### Addendum
# 
# This section of this notebook is based on the following slide deck: [link](https://www.cs.toronto.edu/~rgrosse/courses/csc411_f18/tutorials/tut8_adams_slides.pdf).
