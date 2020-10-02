#!/usr/bin/env python
# coding: utf-8

# # Quantile-Quantile or QQ Plot :
# Quantile-Quantile or in short Q-Q plot, in simple terms is method to compare two distributions.
# In a more formal terms, it is a technique to compare whether two sets of sample points are from  or they follow same distributions. 
# 
# However we normally use it to check whether a sample of data points follow a particular distribution. So in this scenario, one distribution is fixed and we know what it is, what it's properties are and we compare the other or unknown sample with it. If the unknown sample of dataset follow given distribution, we will have a scatter plot, where data points will be in a straight line y = x.
# 
# The data points are quantile values of each distribution. The idea is to plot the quantile  values of two distributions/samples and see, if they make a straight line or not. If the quantiles of two sample sets are similar or in a better case, identical then sample set is from the same distribution. In better words, **if the distributions are identical, then the quantiles should be approximately equal.**
# 
# Do note, that the known distribution need not to be only Normal Distribution, it can be any known distribution like, Exponential Distribution, Weibull Distribution, Binomial Distribution etc.
# 
# 
# Let's have some python code, to understand it better.
# 
# 

# In[ ]:


import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# A standard normal distribution, which means mean = 0, and stddev = 1
rvs = stats.norm(loc=0, scale=1)


# In[ ]:


# Generate or take out a sample of points from Normal Disb.
normal_sample = rvs.rvs(size=100000)


# In[ ]:


# We can plot also, to see whether the values are indeed taken from normal distribution
sns.set()
sns.distplot(normal_sample)


# As you can see it totally looks like the cute bell curve everyone talks about. :)

# Let's plot Q-Q plot for these data points against the normal distribution. We should definitely get a straight line.

# In[ ]:


normal_sample = rvs.rvs(size=1000)
stats.probplot(normal_sample, dist="norm", plot=plt)
plt.show()


# Woah, it's almost a straight line. We don't always get a perfect straight line, because remember we have sampled it from a real distribution. As we increase the number of samples more and more, the line created from the points will be straighter.     
# 
# Also, you can notice x-axis represents Theoretical Quantiles which means it's values are spaced to represent the different quantiles of given/known distribution (in our case here, its Normal Disribution) and y-axis is Ordered Values which **represent** the quantiles for the given unknown sample set and red line is the line they should follow. 

# In[ ]:


# This time it's a million
normal_sample = rvs.rvs(size=1000000)
stats.probplot(normal_sample, dist="norm", plot=plt)
plt.show()


# It's a way more better than previous one.
# 
# **NOTE:** If we have less number of sample points then it would be difficult to interpret anything from the plot, of course we have other statistical methods which can work with small samples for this purpose, but for Q-Q plot to make sense we need good number of data points. 
# ##### Rule of thumb is to have at least 30 data points.

# In[ ]:


new_rvs = stats.norm(loc=10, scale=2)
new_normal_sample = new_rvs.rvs(size=10000)
stats.probplot(normal_sample, dist="norm", plot=plt)
plt.show()


# **NOOOOOO !!!!!**
# 
# The reason behind is the math, which I don't wanna talk right now.
# 
# We can also compare two known samples. Let's take Normal Distribution and Gamma Distribution.

# In[ ]:


# We can plot exponential distribution also
sns.set(style="whitegrid")
plt.figure(figsize=(8, 4))
sns.distplot(stats.expon().rvs(size=10000))


# In[ ]:


# Let's compare them
expon_rvs = stats.expon().rvs(size=100000)
normal_rvs = stats.norm().rvs(size=100000)
stats.probplot(x=expon_rvs, dist=stats.norm(), plot=plt)


# As expected, this is not a straight line. Since, these two distributions are different so their Q-Q plot will not be straight line.
# 
# Let's take one more example of comparing Exponential Distribution with Pareto distribution.

# In[ ]:


# I will tell you later what Pareto distribution is.
pareto_rvs = stats.pareto(b=2.62).rvs(size=1000000)
stats.probplot(x=pareto_rvs, dist=stats.expon(), plot=plt)
plt.show()


# Again, as they are not the same distribution, the blue line misses the red line by miles.

# In[ ]:




