#!/usr/bin/env python
# coding: utf-8

# # Gaussian process regression and classification
# 
# Carl Friedrich Gauss was a great mathematician who lived in the late 18th through the mid 19th century. He is perhaps have been the last person alive to know "all" of mathematics, a field which in the time between then and now has gotten to deep and vast to fully hold in one's head. One of the many things named after him is the Gaussian distribution, also known as the normal distribution. I suspect that if you're reading this you know the normal distribution well!
# 
# A **stochastic process** is a collection of random variables indexed over time or space. Individual points in a stochastic process are related both to the underlying probability distribution governing the process, as well as to other points earlier in the process (as well as later points, obviously). A famous example of a stochastic process is Brownian motion. Basically a stochastric process pairs a probability distribution with a memory of the point's position. If we have create a stochastic process in two dimensions with ten steps in the x direction (stochastic processes are actually continuous, but may be usefully approximated as stepwise), at each step a number is drawn from the underlying distribution and the next point moves that far away from its previous position.
# 
# The path that the point takes is strongly dependent on the probability distribution the steps are drawn from. This probability distribution is known as the **kernel** (not the same usage of this word as the kernels in the kernel trick in e.g. SVMs [here](https://www.kaggle.com/residentmario/kernels-and-support-vector-machine-regularization)). By changing the kernel we fundamentally change the random path that the point walks along. Here, for example, are some selections of walks through time using three different kernel functions:
# 
# ![](https://i.imgur.com/l3wVZyu.png)
# 
# A **Gaussian process** is a stochastic process whose kernel is a Guassian normal distribution. In other words, a stochastic walk is a Gaussian process if every (linear) combination of predictor varibles is multivariate normally distributed. And hence, going the other way, any target variables whose predictor variables basically approximate this foundational distributional property can be modeled using a Gaussian process! This extension of Gaussian processes to regression, and separately to classification, exists in `sklearn` as `GaussianProcessRegressor` and `GaussianProcessClassifier`, respectively.
# 
# Before going into specifics on how this extension works it's worth talking about the strengths and weaknesses of these Gaussian process -based algorithms.
# 
# First of all, Gaussian processes are parametric: they are principled on the assumption that the underlying data is normally distributed and normally jointly distributed. This is a very strong assumption, of course, which will rarely occur fully in practice. The closer to these assumptions a dataset holds, the better the performance GP will have; but it performs surprisingly well even on data where this is relatively far away from true, for the same reason that [Naive Bayes](https://www.kaggle.com/residentmario/primer-on-naive-bayes-algorithms/) works surprisingly well in so many situations: in many cases, simple assumptions can trump distributional fidelity.
# 
# And on the other side of this weakness is a strength. Because these classifiers are parametric they come equipped with reliable probability and variance/co-variance estimates, and may even be used to draw new sample data points. They also extend extremely naturally to the time-series setting, and do not have the censoring issues that some non-parametric time series prediction methods have. And since you have the ability to specify a different kernel functions, they are quite extensible and malleable.
# 
# Because they work by, in part, estimating the covariance matrix of all of the variables in the dataset, GP performance starts to suffer when you have more than a dozen ($12 \times 11 = 132\:\text{est}$) or a couple dozen ($24 \times 23 = 552\:\text{est}$) features. This algorithm is also not sparse: e.g. it will use all of the observations during computation, which limits is generalizability to extremely large datasets.
# 
# ## Gassian process regression
# 
# The `GaussianProcessRegressor` implements Gaussian processes for regression purposes. This algorithm assumes a mean of zero if `normalize_data=False`, and will renormalize the data to make the mean zero if `normaliza_data=True`. Then a covariance matrix for the features in the dataset is estimated. In other words, the algorithm must find the normal distribution which maximizes the log marginal likelihood of each pair of variables&mdash;marginal likelihood because we hold all other variables constant, log for numerical stability, and likelihood in terms of the [likelihood function](https://en.wikipedia.org/wiki/Likelihood_function) of the observed data (slightly more detail in [this CrossValided Q&A pair](https://stats.stackexchange.com/questions/108215/how-to-understand-the-log-marginal-likelihood-of-a-gaussian-process). This is done using an `optimizer` parameter. But log marginal likelihood is non-convex, so it may have multiple local maxima, and thus `n_restarts_optimizer` is provided to allow you to solve for the covariance matrix multiple times and "pick" the values with the highest level of consensus.
# 
# The data the regression is applied to does not necessarily have to be the original data in its original vector space. As with other techniques, like support vector machines, we may detect more complex boundaries by providing a non-linear `kernel`. Also like with support vector machines, the default `kernel` is RBF.
# 
# Finally, regularization may be applied via the `alpha` parameter.

# In[1]:


import numpy as np
rng = np.random.RandomState(0)
import matplotlib.pyplot as plt


def wave(n_points=100, n_points_perturbed=20, pertubation_mult=1):
    X = 5 * rng.rand(n_points, 1)
    y = np.sin(X).ravel()
    perturbation_step_size = n_points // n_points_perturbed
    y[::perturbation_step_size] += 3 * pertubation_mult * (0.5 - rng.rand(X.shape[0] // perturbation_step_size))
    
    # Sort in order of X.
    sort = np.argsort(X[:, 0])
    X = X[sort]
    y = y[sort]
    
    return X, y


X, y = wave(n_points=100, n_points_perturbed=20, pertubation_mult=1)

from sklearn.gaussian_process import GaussianProcessRegressor
clf = GaussianProcessRegressor(random_state=42)
clf.fit(X, y)
y_pred = clf.predict(X)

fig, ax = plt.subplots(1, figsize=(12, 6))
plt.plot(X[:, 0], y, marker='o', color='black', linewidth=0)
plt.plot(X[:, 0], y_pred, marker='x', color='steelblue')
plt.suptitle("$GaussianProcessRegressor(kernel=RBF)$ [default]", fontsize=20)
plt.axis('off')
pass


# Gaussian process regression with the default RBF kernel and only one round of covariance fitting works well with this (trivial) dataset. To see the performance of this technique on more noisy datasets, and with different kernels, try forking this notebook and tweaking the parameters.
# 
# The [`sklearn` documentation](http://scikit-learn.org/stable/modules/gaussian_process.html#gpr-examples) includes lots more text on working with GP regression. Basically it comes down to flexibility: GP regression can implement any kernel that you specify, including ones where you tune white noise or add together several different model effects. The full details are worth investigating for the extremely mathematically curious, but they're kind of a lot to digest. For simple applications and problems, GP regression is very nearly the same as kernel ridge regression, an algorithm I covered in [a previous notebook on non-parametric regression](https://www.kaggle.com/residentmario/non-parametric-regression/).
# 
# ## Gaussian process classification
# 
# An extension of GP to the classification context also exists: `GaussianProcessClassifier`. TODO: investigate.
