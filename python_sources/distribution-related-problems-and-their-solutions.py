#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 19:00:12 2018
@author: TANMOY DAS
"""


# # Normal Distribution

# ## Math problem 1
# ![Imgur](https://i.imgur.com/6p2Zqtu.png)

# A Gaussian random variable has a mean of 1830 and standard deviation of 460
# Find the probability that the variable will be greater than 2750. 
# Reference: Normal Distribution; Page 294, Chapter 5, FE - IE specific

# In[ ]:


# greater than
import scipy.stats
mean_normal = 8
standard_deviation_normal = 2.5

probability_norm_gt = scipy.stats.norm.sf(15.5, mean_normal,standard_deviation_normal) # greater than
print(probability_norm_gt)


# ## Math Problem 2

# The distribution of weekly incomes follows the normal probability distribution,  with a mean of \$1,000 &
#  a standard deviation of \$100. What is the probability of selecting a shift foreman in the glass industry whose income is: 
# 1. Between \$790 and $1,000?
# 2. Less than \$790?
# 
# Reference: Normal Distribution; Page 235, Chapter 7, Statistical techniques in Business by Lind 

# ![Imgur](https://i.imgur.com/II9XHJK.png)

# ![Imgur](https://i.imgur.com/gBtGIh1.png)

# In[ ]:


import scipy.stats
# in between
mean_normal = 8
standard_deviation_normal = 2.5
probability_norm_lt = scipy.stats.norm.cdf(12, mean_normal,standard_deviation_normal)
probability_norm_gt = scipy.stats.norm.cdf(8, mean_normal,standard_deviation_normal) # greater than
probability_in_between = probability_norm_lt - probability_norm_gt
print(probability_in_between)


# In[ ]:


# less than
mean_normal = 1000
standard_deviation_normal = 100
probability_norm_lt = scipy.stats.norm.cdf(790, mean_normal,standard_deviation_normal)


# ## Problem 3

# In[ ]:


#To find the variate for which the probability is given, let's say the 
#value which needed to provide a 98% probability, you'd use the 
#PPF Percent Point Function
probability_given = scipy.stats.norm.ppf(.98,100,12)


# # Poission Distribution

# ## Math Problem 1

# Assume baggage is rarely lost by Delta Airlines. Most flights do not experience any mishandled bags; some have one bag lost; a few have two bags lost; rarely a flight will have three lost bags; and so on. Suppose a random sample of 1,000 flights shows a total of 300 bags were lost. Determine the probability of losing no bag.
# Source: P 208, Chapter 6, Lind

# In[ ]:


import scipy.stats
mean_poisson = 300/1000
# prob = poisson.cdf(x, mu); x= random variable; mu = mean 
probability_poisson = scipy.stats.poisson.cdf(0, mean_poisson)


# ## Math Problem 2

# Coastal Insurance Company underwrites insurance for beachfront properties along the Virginia, North and South Carolina, and Georgia coasts. It uses the estimate that the probability of a named Category III hurricane (sustained winds of more than 110 miles per hour) or higher striking a particular region of the coast (for example, St. Simons Island, Georgia) in any one year is .05. If a homeowner takes a 30-year mortgage on a recently purchased property in St. Simons, what is the likelihood that the owner will experience at least one hurricane during the mortgage period? Ref: P210, Chapter 6, Lind

# In[ ]:


import scipy.stats
mean_poisson = 30*.05
# n is the number of years, 30 in this case.
# \pi is the probability a hurricane meeting the strength criteria comes ashore.
# \mu is the mean or expected number of storms in a 30-year period.
from IPython.display import display, Math, Latex
display(Math(r'P(x \geq 1) = 1 - P(X=0)'))
probability_poisson = 1 - scipy.stats.poisson.cdf(0, mean_poisson)


# 
