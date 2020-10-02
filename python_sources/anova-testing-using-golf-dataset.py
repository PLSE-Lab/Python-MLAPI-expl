#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.stats import ttest_1samp,wilcoxon,ttest_ind,shapiro,levene,mannwhitneyu,ttest_rel
import seaborn as sns
from statsmodels.stats.power import ttest_power


# In[ ]:


inp_data=pd.read_csv('../input/dataset-for-anova-testing/SM4-Golf.csv')


# In[ ]:


inp_data.head()


# In[ ]:


inp_data.shape


# In[ ]:


inp_data.info()


# In[ ]:


inp_data.describe()


# # Check Normalization of the columns

# In[ ]:


sns.distplot(inp_data['Current'])


# In[ ]:


sns.distplot(inp_data['New'])


# Thus we see that the two columns follow Normal distribution.

# # ANSWER 1:

# Since there is a concern that the new design may affect the driving distance of the ball. We need to make sure that the new design doesnt affects the driving distance inorder to introduce the new model into the market.
# 
# 
# 
# Hence, We make the assumptions , 
# 
# 
# 
#         The change in the ball dimension has no effect on the driving distance as the NULL HYPOTHESIS   (Mean of the Current and New are almost same)
#         The change in the ball dimesnion has effect on  the driving distance as the ALTERNATIVE HYPOTHESIS    (Mean of the  Current and New are different).

# # ANSWER 2:  

# Using two sample test method to calculate probability:

# In[ ]:


group1=inp_data['Current']
group2=inp_data['New']


# In[ ]:


t_statistic,p_value=ttest_ind(group1,group2)
print(t_statistic,p_value)


# Since Probability is 0.1879 > 0.005
# 
# # p-value is 0.1879
# # Null Hypothesis is accepted

# Since the null hypothesis is accepted , we understand that the new design doesnt affect the driving distance of the ball.
# 
# # We recommend Par Inc., to introduce the new ball 

# # ANSWER 3:

# Using other models to calculate the Probabaility:

# # ttest_1samp:

# In[ ]:


t_statistic,p_value=ttest_1samp(group2-group1,0)


# In[ ]:


print(t_statistic,p_value)


# Since p_value is 0.21>0.005 
# 
# 
# 
# Null hypothesis is accepted

# # mannwhitneyu:

# In[ ]:


t_statistic,p_value=mannwhitneyu(group1,group2)


# In[ ]:


print(t_statistic,p_value)


# Since p_value is 0.102 > 0.5
# 
# 
# 
# 
# Null hypothesis is accepted

# # ttest_rel:

# In[ ]:


t_statistic,p_value = ttest_rel(group2,group1)
print(t_statistic,p_value)


# This is same as the ttest_1samp
# 
# 
# Here also the probability is greater than 0.05 
# 
# 
# So,Null hypothesis is accepted

# # wilcoxon:

# In[ ]:


z_statistic,p_value=wilcoxon(group2-group1)
print(t_statistic,p_value)


# Here also the p_value >0.05 
# 
# 
# Thus Null hypothesis is accepted

# # levene:

# In[ ]:


levene(group1,group2)


# Since pvalue is 0.6 >0.05 
# 
# 
# Null hypothesis is accepted

# # shapiro:

# In[ ]:


shapiro(group2)


# Here probability is 0.3>0.05
# 
# 
# Thus Null hypothesis is accepted

# # Calculation of Pooled Standard deviation and Power of test

# Refer formula in notes

# In[ ]:


group1_var=np.var(group1)
group2_var=np.var(group2)
pooled_SD=np.sqrt(((39*group1_var)+(39*group2_var))/(40+40-2))
print(pooled_SD)


# In[ ]:


power_of_test=ttest_power(pooled_SD, nobs=40, alpha=0.05, alternative="two-sided")
power_of_test


# # ADDITIONAL QUESTIONS IN VIDEO:

# # Confidence interval Calculation

# In[ ]:


n=40
sample_mean_current=np.mean(group1)


# In[ ]:


sigma1=np.std(group1)/np.sqrt(n)


# In[ ]:


conf_interval_Current=stats.t.interval(0.95,   #confidence level is 95%
                              df=n-1,      #degree of freedom
                              loc=sample_mean_current,   #sample mean of group1
                              scale=sigma1)    #SD estimate
print(conf_interval_Current)


# In[ ]:


n=40
sample_mean_new=np.mean(group2)


# In[ ]:


sigma2=np.std(group2)/np.sqrt(n)


# In[ ]:


conf_interval_new=stats.t.interval(0.95,   #confidence level is 95%
                              df=n-1,      #degree of freedom
                              loc=sample_mean_new,   #sample mean of group2
                              scale=sigma2)    #SD estimate
print(conf_interval_new)

