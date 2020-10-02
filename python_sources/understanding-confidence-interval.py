#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity


# ### Understanding Confidence Interval
# * This notebook simulates 1 million experiments on Udacity engagement dataset
# * On how to accurately interpret confidence interval, refer to my [Medium](https://towardsdatascience.com/understanding-confidence-interval-d7b5aa68e3b) article.

# In[2]:


def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
    """Kernel Density Estimation with Scikit-learn"""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(x[:, np.newaxis])
    # score_samples() returns the log-likelihood of the samples
    log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
    return np.exp(log_pdf)


# __Engagement__ roughly follows exponential distribution.

# In[3]:


engagement = np.loadtxt('../input/engagement.csv')

plt.hist(engagement, bins=100, alpha=0.8, density=True)

x_grid = np.linspace(-1, 1, 100)
pdf_engagement = kde_sklearn(engagement, x_grid, bandwidth=0.007)

plt.plot(x_grid, pdf_engagement, alpha=0.8, lw=2, color='r')

plt.xlabel("engagement")
plt.ylabel("density")

plt.xlim(0, 1)

plt.show()


# In[4]:


mean = np.mean(engagement)
std = np.std(engagement)

print("""
Population mean: %.5f
Population std: %.5f
Population size: %i
"""%(mean, std, len(engagement)))


# ### Examine Sampling distribution

# In[5]:


sample_size = 300
n_trials = 1000000

# draw one million samples, each of size 300
samples = np.array([np.random.choice(engagement, sample_size) 
                    for _ in range(n_trials)])


# In[6]:


# calculate sample mean for each sample
means = samples.mean(axis=1)

# mean of sampling distribution
sample_mean = np.mean(means)

# empirical standard error
sample_std = np.std(means)

analytical_std = std / np.sqrt(sample_size)

print("""
sampling distribution mean: %.5f
sampling distribution std: %.5f
analytical std: %.5f
"""%(sample_mean, sample_std, analytical_std))


# __Sampling distribution__ looks very normal.

# In[7]:


from scipy.stats import t

# sampling distribution
z = 1.96
plt.hist(means, bins=50, alpha=0.9, density=True)
plt.axvline(sample_mean - 1.96 * sample_std, color='r')
plt.axvline(sample_mean + 1.96 * sample_std, color='r')

plt.xlabel('sample_mean')
plt.ylabel('density')

plt.show()


# __Sampping distirbution__ is slightly skewed to the positive side.

# In[8]:


print("lower tail: %.2f%%"%(100 * sum(means < sample_mean - 1.96 * sample_std) / len(means)))
print("upper tail: %.2f%%"%(100 * sum(means > sample_mean + 1.96 * sample_std) / len(means)))


# __QQ plot__ confirms tendency toward normal distribution.

# In[9]:


import pylab 
import scipy.stats as stats


# In[10]:


stats.probplot(means, dist="norm", plot=pylab)
pylab.show()


# ### Confidence Interval
# We get 0.05 false positive rate, under the 95% confidence interval

# In[11]:


# make 95% confidence interval
z = 1.96

se = samples.std(axis=1) / np.sqrt(sample_size)
ups = means + z * se
los = means - z * se
success = np.mean((mean >= los) & (mean <= ups))
fpr = np.mean((mean < los) | (mean > ups))
print("False positive rate: %.3f"%fpr)


# In[12]:


n_points = 8000

# plt.figure(figsize=(14, 6))
plt.scatter(list(range(len(ups[:n_points]))), ups[:n_points], alpha=0.3)
plt.scatter(list(range(len(los[:n_points]))), los[:n_points], alpha=0.3)
plt.axhline(y=0.07727)

plt.xlabel("sample")
plt.ylabel("sample_mean")

