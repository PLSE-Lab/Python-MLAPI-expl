#!/usr/bin/env python
# coding: utf-8

# Once you learn the Bayesian statistics, you should come across **the Beta distribution**.
# 
# Beta distribution is something you might have not used at all when you were dealing with a frequentist statistics and a conventional machine learning. However, when it comes to **the Bayesian inference**, which is to estimate the posterior probability distribution given the prior and observations, the Beta distribution is extremely useful.
# 
# What is good with the Beta distribution?
# 
# Well, using the Beta distribution, **you can model the probability itself, just by two positive shape parameters**.
# 
# As the Bayesian inference is all about estimating the probability, this feature of Beta distribution is very helpful to model probabilistic behaviors.
# 
# Furthermore, **the Beta distribution is the conjugate prior probability distribution for the Bernoulli, binomial, negative binomial and geometric distributions**. That means, you can easily compute the posterior distribution by modeling the prior distribution with the Beta distribution.
# 
# Let's have a look at how the Beta distribution looks like as a function of the shape parameters, a and b!

# In[ ]:


# Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("talk")

from scipy.stats import beta


# # Beta distribution (Probability Density Function)

# In[ ]:


# figure and axes
fig, ax = plt.subplots(5, 5, figsize=(20, 20))

# parameters
params = np.array([0.5, 1, 2, 3, 5])
colors = sns.cubehelix_palette(25)

# range
counts = 0
for r, a in enumerate(params):
    for c, b in enumerate(params):
        # probability densitiy function
        x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
        ax[r, c].plot(x, beta.pdf(x, a, b), 'r-', color=colors[counts], lw=10, alpha=0.9)
        ax[r, c].set_title("a = " + str(a) + ", b = " + str(b))
        counts += 1
        if c == 0:
            ax[r, c].set_ylabel("PDF")
        if r == 0:
            ax[r, c].set_xlabel("x")
plt.tight_layout()


# # Beta distribution (cumulative density function)

# In[ ]:


# figure and axes
fig, ax = plt.subplots(5, 5, figsize=(20, 20))

# parameters
params = np.array([0.5, 1, 2, 3, 5])

# range
counts = 0
for r, a in enumerate(params):
    for c, b in enumerate(params):
        # probability densitiy function
        x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)
        ax[r, c].plot(x, beta.cdf(x, a, b), 'r-', color=colors[counts], lw=10, alpha=0.9)
        ax[r, c].set_title("a = " + str(a) + ", b = " + str(b))
        counts += 1
        if c == 0:
            ax[r, c].set_ylabel("CDF")
        if r == 0:
            ax[r, c].set_xlabel("x")
plt.tight_layout()


# Apparently the Beta distribution can take many forms:D Which is your favorite?:)
