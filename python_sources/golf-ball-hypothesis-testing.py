#!/usr/bin/env python
# coding: utf-8

# Step 1 : Importing necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp,ttest_ind,levene,shapiro,iqr,mannwhitneyu,wilcoxon,iqr
from statsmodels.stats.power import ttest_power
import scipy.stats as stats


# Step 2 : Reading of Golf data

# In[4]:


golf = pd.read_csv('../input/Golf.csv')


# In[5]:


golf.head()


# In[ ]:


golf.tail()


# In[ ]:


golf.info()


# There are no null values in the dataset 

# In[ ]:


golf.describe()


# In[ ]:


golf.hist()


# In[ ]:


golf.boxplot()


# There are no outliers in both the data for the Current and New golf balls

# Since the same test of driving distance is applied on two populations of Current and the new golf ball, the samples are classified under unpaired samples

# Step 3 : Splitting the data into two samples of Current and New golf balls

# In[ ]:


Current = golf.iloc[:,0]


# In[ ]:


Current.head()


# In[ ]:


iqr(Current, rng = (25,75))


# In[ ]:


meanC = Current.mean()
meanC


# In[ ]:


varC = Current.var()
varC


# In[ ]:


New = golf.iloc[:,1]


# In[ ]:


New.head()


# In[ ]:


iqr(New, rng = (25,75))


# In[ ]:


meanN = New.mean()
meanN


# In[ ]:


varN= New.var()
varN


# Step 4 : Defining Null and Alternate Hypothesis

# The null hypothesis is that the driving distance of current ball is lesser than or equal to New ball and alternate hypothesis
# is that the driving distance of Current ball is greater than the New ball assuming more driving distance is desired.
# 
#  H0 : driving_distance(current) <= driving_distance(new)
#  
#  H1 : driving_distance(current) > driving_distance(new)
#         
# If the null hypothesis is rejected, then it can be concluded that the marketing of New ball with cut resistance and long lasting charecteristics is not desirable for Par inc,

# Step 5 : Testing whether samples are parametric or non parametric by Shapiro Test

# In[ ]:


shapiro(Current)


# In[ ]:


shapiro(New)


# As the P value of shapiro test is greater than 0.05, the null hypothesis of shapiro test that the sample is drawn from the population following normal distribution cannot be rejected.

# Step 6 : t sample testing of the hypothesis

# In[ ]:


t_statistic,p_value = ttest_ind(Current,New)


# In[ ]:


print(t_statistic,p_value)


# The ttest_ind gives 2 tailed probability. As our test is right tailed, the required probability is half of that given by ttest_ind. i.e., 0.1879/2 =0.0989. 
# 
# The Null hypothesis is not rejected as p>0.05

# Step 7 : Levene test for check of variances of population

# In[ ]:


levene(Current,New)


# Since, the pvalue for the levene's test is greater than 0.05, we cannot reject the null hypothesis of levene test,
# which is the population variances of both the samples are same

# As the samples pass levene's test, Pooled standard deviation can be used for calculation of DELTA value required for calculating power of test

# Step 8 : Calculation of POWER OF TEST with pooled standard deviation

# In[ ]:


Pooledstd = np.sqrt(((40-1)*varC+ (40-1)*varN)/(40+40-2))
Pooledstd


# In[ ]:


delta = (meanC - meanN)/Pooledstd
delta


# In[ ]:


print(ttest_power(delta, nobs = 40, alpha = 0.05, alternative = 'larger'))


# In[ ]:


print(ttest_power(delta, nobs = 72, alpha = 0.05, alternative = 'larger'))


# Step 9 : Mannwhitneyu Test of hypothesis

# Assuming, non-parametic data, i.e., the populations from which samples are drawn doesnot follow normal distribution, 2 sample testing of unpaired data can be done by mannwhitneyu model

# In[ ]:


u,p_value = mannwhitneyu(Current,New,alternative = 'greater')


# In[ ]:


print(u,p_value)


# Since, the probability is greater than 0.05, the Null hypothesis cannot be rejected

# Step 10 : Calculation of POWER OF TEST when variances are not equal for the populations

# In[ ]:


critical = stats.t.isf(0.05,76)
critical


# In[ ]:


delta1 = critical - t_statistic
delta1


# In[ ]:


print(ttest_power(delta1, nobs = 40, alpha = 0.05, alternative = 'larger'))


# The power of the test says that there is only 67.29% of the chance that the rejected null hypothesis is actually false.  

# In[ ]:


print(ttest_power(delta1, nobs = 56, alpha = 0.05, alternative = 'larger'))


# Conclusions:
# 
# a) When the population of golf balls is assumed to be parametric,i.e.,  follows normal distribution, the ttest_ind for the unpaired data, gives that the null hypothesis cannot be rejected as p value was 0.0989 which is greater than industrial standard of 0.05 (shapiro test pass case)
# 
# b) When the population of golf balls is assumed to be non-parametic, ie., doesnot follow normal distribution, the mannwhitneyu test gives that the null hypothesis cannot be rejected as p value is 0.1026 which is greater than industrial standards of 0.05.
# (Shapiro test fail case)
# 
# c) The power of the test was calculated to be only 57.9%,  with population of equal variances.  Any sample size above 72, will increase the power of the test and it will approach the required industrial standards of 80% ( levene's test pass case)
# 
# d) The power of the test was calculated to be 67.29%, with populalation of unequal variances.  Any sample size above 56, will increase the power of the test and it will approach the required industrial standards of 80% (levene's test fail case)
# 
# 
# 
# 
#     
# 
#     

# In[ ]:





# In[ ]:




