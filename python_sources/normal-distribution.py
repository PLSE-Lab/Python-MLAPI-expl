#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# 

# ## What is a Normal (or Gaussian) Distribution?
# A normal distribution often called as Gaussian or Laplace-Gauss Distribution is a continuous probabilty distribution for real-valued random variable. A sample of points is called normally ditributed if they follow certain properties which normal distribution follows. A point on the plot is the probability of sampling a variable on the corresponding x-axis point.
# 
# Now, continuous probability distribution for a real valued ranodom variable is typically described by probability density functions (with the probability of any individual outcome being 0).
# 
# Let's plot PDF of normal distribution.

# In[ ]:


import scipy.stats as stats
rvs = stats.norm(scale=1, loc=0).rvs(1000000)
sns.distplot(rvs)


# ##### Above is the bell curve everyone talks about.

# A normal distribution has two parameters, mu and sigma, where mu is the mean of the distribution and the sigma is standard deviation. Probability of getting selected increases if the value is near the mean. In normal distribution, data points are centered and symmetric around mean.  
# 
# Sigma stands for standard deviation of the distribution. It accounts for the width of the normal distribution. More is the sigma, more will be variability in the values, and more flatter will be the normal distribution which brings down the peak of the ND.  
# 
# Let's see it below

# In[ ]:


def plot_normal_distribution(loc, scale):
    rvs = stats.norm(loc=loc, scale=scale).rvs(1000000)
    sns.distplot(rvs, hist=False, label="stddev=" + str(scale) + ", mean=" + str(loc))
    
plt.figure(figsize=(16, 6))    
plt.subplots
plot_normal_distribution(loc=0, scale=1)
plot_normal_distribution(loc=0, scale=3)
plot_normal_distribution(loc=0, scale=5)
plt.legend()
plt.show()


# As expected, the normal distribution with higher value of stddev, is flatter than the smaller ones. Also, as the smaller stddev is, more higher is the peak of ND.   
# 
# Normal distribution follows 68-95-97 rule, which means the 65% data points will fall within 1 stddev range centered at mean. 95% within 2 standard deviation and 97% within 3 stddev.
# 
# Kurtosis and Skewness values are both 0 for ND. Plot is centered around mean and it is symmetric around mean.
# 
# ND is present in many things in our day to day life.
# 
# By moving the means and keeping the stddev same, the plot will not change it will just move sideways.

# In[ ]:


# We can also move the plot of we change the mean of the distribution
plt.figure(figsize=(14, 5))    
plt.subplot(1, 1, 1)    
plot_normal_distribution(loc=0, scale=2)
plot_normal_distribution(loc=3, scale=2)
plot_normal_distribution(loc=6, scale=2)
plt.legend()
plt.show()


# If we are talking about PDF then we should definitely talk about CDF of probability distribution.

# In[ ]:


def plot_normal_cdf_plot(loc, scale):
    x = np.linspace(-loc-5, loc+5, 10000)
    plt.plot(x, stats.norm.cdf(x, loc=loc, scale=scale), label="stddev=" + str(scale) + ", mean=" + str(loc))
    
plt.figure(figsize=(14, 6))    
plt.subplot(1, 1, 1)     
plot_normal_cdf_plot(loc=0, scale=.5)
plot_normal_cdf_plot(loc=0, scale=1)
plot_normal_cdf_plot(loc=0, scale=1.5)
plot_normal_cdf_plot(loc=0, scale=2.5)
plt.legend()
plt.show()
    


# More is the value of stddev, straighter the cdf tends to become. This happens because the variability of data points increases with stddev. Stddev is measure of variability.

# A normal distribution with mean = 0 and stddev = 1 is called Standard Normal Distribution. 

# In[ ]:




