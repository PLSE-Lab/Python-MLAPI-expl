#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from statsmodels.stats.weightstats import zconfint
from statsmodels.stats.weightstats import _zconfint_generic, _tconfint_generic
from math import sqrt
from scipy import stats
# Any results you write to the current directory are saved as output.


# Lets assume that we have two models that has been trained with 5 fold cross validation.
# How to decide which model is better? 
# Due to randomness in augumentation usually it hard to do.
# In this case statistica could help us. 
# 
# ### First thing we could do:
# Calculate 95% confidential intervals for mean score and compare intervals for first and the second model. If they do not cross with each other - choose the model with more scewed interval (for example - for kappa we need to choose model with maximum skew to right due to maximization).
# 
# ### Second:
# if confidential inetrvals is crossing with each other then we cant choose model with maxium skew because there is a chance that true mean kappa (or score), laying in confidential inetrval could be less than we expecting. So in this reason we need to calculate p-value.
# 

# ### Lets see in practice

# In[ ]:


# CV for first model
kappabefore = np.array([0.8765, 0.8711, 0.8476, 0.8471, 0.9164]) #v12
scorebefore = np.array([4.4126, 4.0255, 5.0050, 4.4838, 3.9395]) #v12

# CV for second model
kappaafter = np.array([0.9028, 0.8792, 0.8715, 0.8756, 0.9123]) #v13
scoreafter = np.array([3.8182, 3.9054, 4.0769, 4.3103, 3.7365]) #v13


# In[ ]:


# Some fuctions for premutation test

def permutation_t_stat_1sample(sample, mean):
    t_stat = sum(list(map(lambda x: x - mean, sample)))
    return t_stat

def permutation_zero_distr_1sample(sample, mean, max_permutations = None):
    centered_sample = list(map(lambda x: x - mean, sample))
    if max_permutations:
        signs_array = set([tuple(x) for x in 2 * np.random.randint(2, size = (max_permutations, 
                                                                              len(sample))) - 1 ])
    else:
        signs_array =  itertools.product([-1, 1], repeat = len(sample))
    distr = [sum(centered_sample * np.array(signs)) for signs in signs_array] #####
    return distr

def permutation_test(sample, mean, max_permutations = None, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")
    
    t_stat = permutation_t_stat_1sample(sample, mean)
    
    zero_distr = permutation_zero_distr_1sample(sample, mean, max_permutations)
    
    if alternative == 'two-sided':
        return sum([1. if abs(x) >= abs(t_stat) else 0. for x in zero_distr]) / len(zero_distr)
    
    if alternative == 'less':
        return sum([1. if x <= t_stat else 0. for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        return sum([1. if x >= t_stat else 0. for x in zero_distr]) / len(zero_distr)


# ### Calculating confidential intervals

# In[ ]:


before_mean_std = kappabefore.std(ddof=1)/sqrt(len(kappabefore))
after_mean_std = kappaafter.std(ddof=1)/sqrt(len(kappaafter))
before_mean_std_score = scorebefore.std(ddof=1)/sqrt(len(scorebefore))
after_mean_std_score = scoreafter.std(ddof=1)/sqrt(len(scoreafter))
print('======================== KAPPA ========================')
print('mean kappa before {:.4f}'.format(kappabefore.mean()))
print('mean kappa after {:.4f}'.format(kappaafter.mean()))
print("model before mean kappa 95%% confidence interval", _tconfint_generic(kappabefore.mean(), before_mean_std,
                                                                       len(kappabefore) - 1,
                                                                       0.05, 'two-sided'))
print("model after mean kappa 95%% confidence interval", _tconfint_generic(kappaafter.mean(), after_mean_std,
                                                                       len(kappaafter) - 1,
                                                                       0.05, 'two-sided'))
print('======================== LOSS ========================')
print('mean score before {:.4f}'.format(scorebefore.mean()))
print('mean score after {:.4f}'.format(scoreafter.mean()))
print("model before mean loss 95%% confidence interval", _tconfint_generic(scorebefore.mean(), before_mean_std_score,
                                                                       len(scorebefore) - 1,
                                                                       0.05, 'two-sided'))
print("model after mean loss 95%% confidence interval", _tconfint_generic(scoreafter.mean(), after_mean_std_score,
                                                                       len(scoreafter) - 1,
                                                                       0.05, 'two-sided'))


# Here we see that both intervals for kappa and score is crossing with each other so there is a chance that TRUE mean kappa for second model is LOWER than TRUE kappa for first model.
# In our toy example we have only five obseravtions for kappa and loss (5 fold CV) so we have a chance that we observed skewed results that not close to the true values.
# 
# So now we need to calculate a p-value to decide is second model better for true?
# i will use two test: premutation test and Wiloxon test.
# Permutation test is non parametric test that we could use for variables with unknown distributions.
# Wilcoxon test is for variables with normal distribution.
# We dont know what distribution is exactly we have (we could use special tests to check if distribution is normal) so i show both methods. If it will show different results i would prefer to use permutation test results.

# In[ ]:


print('======================== p-test KAPPA ========================')
_, p = stats.wilcoxon(kappabefore, kappaafter)
print('p-value WilcoxonResult test: %f' % p)
print("p-value permutation test: %f" % permutation_test(kappabefore - kappaafter, 0., max_permutations = 50000))

print('======================== p-test LOSS ========================')
_, p = stats.wilcoxon(scorebefore, scoreafter)
print('p-value WilcoxonResult test: %f' % p)
print("p-value permutation test: %f" % permutation_test(scorebefore - scoreafter, 0., max_permutations = 50000))


# Usually 5% trashold is choosing to decide. In case of 5% we could see >5% p-value for KAPPA and around 5% p-value for SCORE. So what does it mean?
# It means that we could not say that new model make better KAPPA. If we use KAPPA metric to choose model - we have no any evidence that second model is better so we need to go futher in research.
# If we use loss as measure - we have chance that second model is better but not for shure here.

# ### this type of model selecting we could use not only with CV but also with TTA

# In[ ]:




