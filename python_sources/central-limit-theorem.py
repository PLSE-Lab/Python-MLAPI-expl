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


# Central Limit Theorem states that **given a dataset from an unknown distribution, the sample means will approximate the normal distribution**
# 
# Confused ? Let's break it down.
# 
# Suppose we have a dataset of points, from an unknown distribution. Let's sample some points of size 'd' and calculate the mean. We do this step of sampling and calculating the mean multiple times. Then the distrbution of that sample means will be like a bell curve of normal distribution. **Regardless of any distribution from which original dataset was created.** Except for few distributions.
# 
# For CLT to work, the unknown distribution must have a finite variance, which is not the case with Cauchy Distribution. Also, CLT applies to independent, identically distributed variables. 
# 
# Higher the number of points in the samples, CLT works better. For a rule of thumb, samples size should at least 30. **Law of Large Numbers**

# ## What is the use of CLT ?
# 
# The first and foremost use is that we can approximate the mean and the standard deviation of the unknown distribution. Credit goes to CLT.
# 
# Another use is w Hypothesis Testing and constructing confidence intervals.

# In[ ]:


# Let's take some distribution and see if CLT really works or not
import scipy.stats as stats

def plot_sample_means(sample_size, number_of_samples, dist):
    rvs_dist = []
    if dist == "expon":
        # Generate 1000 points from an exponential distribution
        rvs_dist = stats.expon().rvs(10000)
    elif dist == "uniform":
        # Generate 1000 points from an exponential distribution
        rvs_dist = stats.uniform().rvs(10000)        
    # Taking 100 samples of size 30    
    sample_means = [np.random.choice(a=rvs_dist, size=sample_size, replace=True).mean() for _ in range(0, number_of_samples)]
    sns.distplot(sample_means, label="number of different samples: " + str(number_of_samples))


# In[ ]:


plot_sample_means(sample_size=30, number_of_samples=300, dist="expon")


# Is it looking like a normal distribution ? Remember, when we said that CLT requires sufficient sample size. 
# 
# Let's change the **number of sample means** first. That is, number of times sample is drawn and mean is calculated. NOte that, we are not changing the sample size. We will do that later. 

# In[ ]:


plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plot_sample_means(sample_size=30, number_of_samples=50, dist="expon")
plt.subplot(2, 2, 2)
plot_sample_means(sample_size=30, number_of_samples=200, dist="expon")
plt.subplot(2, 2, 3)
plot_sample_means(sample_size=30, number_of_samples=800, dist="expon")
plt.subplot(2, 2, 4)
plot_sample_means(sample_size=30, number_of_samples=1500, dist="expon")


# In[ ]:


# Let' change the size of each sample
plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plot_sample_means(sample_size=30, number_of_samples=200, dist="expon")
plt.subplot(2, 2, 2)
plot_sample_means(sample_size=120, number_of_samples=200, dist="expon")
plt.subplot(2, 2, 3)
plot_sample_means(sample_size=300, number_of_samples=200, dist="expon")
plt.subplot(2, 2, 4)
plot_sample_means(sample_size=1000, number_of_samples=10000, dist="expon")


# Although, everytime above code cell runs, the plots will be all different, but I am really sure that bottom-right plot, which is created with sample size of 1000 and iteration done 10000 times, will be the prettiest normal distribution. That's because of the sizes of samples and iteration is large. **LAW OF LARGE NUMBERS**

# We can try this with original distribution being other distribution, let's do it with uniform distribution

# In[ ]:


plt.figure(figsize=(20, 10))
plt.subplot(2, 2, 1)
plot_sample_means(sample_size=30, number_of_samples=200, dist="uniform")
plt.subplot(2, 2, 2)
plot_sample_means(sample_size=120, number_of_samples=200, dist="uniform")
plt.subplot(2, 2, 3)
plot_sample_means(sample_size=300, number_of_samples=200, dist="uniform")
plt.subplot(2, 2, 4)
plot_sample_means(sample_size=1000, number_of_samples=10000, dist="uniform")


# We can see even if the original distribution is chosen as uniform distribution, results are same.

# In[ ]:




