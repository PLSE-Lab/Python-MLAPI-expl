#!/usr/bin/env python
# coding: utf-8

# # Case study
# 1. Card usage has been improved significanlty from last year usage which is 50. (Hint: comparing card usage of post campaign of 1 month with last year hypothesized value 50)
# 2. The last campaign was successful in terms usage of credit card. (Hint comparning means for card usage of pre and post usage of campaign).
# 3. Is there any difference between males and females in terms of credit card usage? (Hint: Comparing means of Card usage for males and females)
# 4. Is there any difference between segments of customers in terms of credit card usage? (Hint: Comparning means of card usage of different segment customers)
# 5. Is there any relation between region and segment? (Hint: Finding the relationship between categorical variables and Segment)
# 6. Is the relationship between card usage in the latest month and pre usage of campaign? (Hint: find the correlation between latest_mon_usage and pre_usage)

# ### Stats modules used in the notebook
# 
#     1. ttest_1samp - One Sample t-test - Argument: Sample 1, hypothetical value
#     2. ttest_rel - Paired sample t-test - Argument: Sample 1, Sample 2
#     3. ttest_ind - Independent sample t-test - Argument: Sample 1, Sample 2
#     4. f_oneway - ANOVA/ F-test - Argument: Smaple 1, Sample 2, sample 3
#     5. chi2_contingency - chisquare test - Argument - cross tab
#     6. stats.pearsonr - Correlation test - Argument - sample 1 and sample 2

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import math
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


cust = pd.read_csv("../input/cust_seg.csv")
cust.head()


# In[ ]:


cust.columns


# In[ ]:


cust.info()


# In[ ]:


cust.describe()


# In[ ]:


cust.Latest_mon_usage.mean()


# In[ ]:


cust.Latest_mon_usage.std()


# 1. Card usage has been improved from last year usage which was 50
# 
# One sample t-test
# 
# H0 (Null hypothesis) Sample_avg = 50
# 
# Ha (ALT hypothesis) Sample_avg > 50
# 
# If this is P < 0.05 and Sample_avg > 50 satisfied we can reject the NULL hypothesis. 
#  else we fail to reject the NULL hypothesis.

# ### One Sample T-Test
# 
# Since we want to check the last month usage the variable we want to check is the last month usage. 

# In[ ]:


stats.ttest_1samp(a = cust.Latest_mon_usage, popmean = 50) # pop mean is the hypothetical value


# In[ ]:


cust.Latest_mon_usage.mean()


# Since the p value is very small we reject the NULL hypothesis. 
# 
# Hence we fail to reject the NULL hypotheiss and accept the ALT hypothesis

# 2. The last campaign was successful in terms of credit card.
# 
# We can compare pre_usage and post_usage of the credit card - Paired sample t-test(dependent sample t-test)
# 
# H0 (NULL hypothesis) Pre_avg = post_avg
# Ha (ALT hypothesis) pre_avg < post_avg
# 
# if p < 0.05 and pre_avg < post_avg, you will reject null 

# ### Two Sample T-Test(Paired)

# In[ ]:


print(cust.pre_usage.mean())
print(cust.Post_usage_1month.mean())
print(cust.post_usage_2ndmonth.mean())


# In[ ]:


stats.ttest_rel(a = cust.pre_usage, b = cust.Post_usage_1month)


# Since the pvalue is not less than 0.05 we fail to reject the Null Hypothesis. Hence the campaign was successful

# 3. Is there any differce in credit card spend usage between males and females?
# 
# **since there are only two categories we can do independent sample t test here.** 
# 
# Comparing two sample averages (both are independent samples)
# 
# H0: males_avg = females_avg
# 
# Ha: males_avg <> females_avg
# 
# if p<0.05, then we reject NULL (there is a relationship between sex and spend)
# 
# else, you fail to reject the NULL (There is no relationship between sex and spend)
# 

# ### Two sample T-Test(Independent)

# In[ ]:


Males_spend = cust.Post_usage_1month[cust.sex == 0]
Females_spend = cust.Post_usage_1month[cust.sex == 1]


# In[ ]:


print(Males_spend.head())
print(Females_spend.head())


# In[ ]:


print(Males_spend.mean())
print(Females_spend.mean())


# In[ ]:


print(Males_spend.std())
print(Females_spend.std())


# In[ ]:


stats.ttest_ind(a = Males_spend, b = Females_spend, equal_var = False)
# equal_var Assume samples have equal variance?


# In[ ]:


# we can use ANOVA as well.
stats.f_oneway(Males_spend, Females_spend)


# Since pvalue is less than 0.05 we reject the NULL hypothesis. Hence, we conclude there is a difference between male spend and female spend

# 4. Is there any difference in credit card spend between segments?
# 
# ANOVA- Ftest
# 
# H0 : S1_avg = S2_avg = S3_avg 
# 
# Ha: One of the segment avg is different from others.
# 
# if p < 0.05, then we reject NULL ,  if p<0.05, there is relationship between segment and spend.
# 
# else, you can not reject NULL.

# ### ANOVA Test

# In[ ]:


cust.segment.value_counts()


# In[ ]:


s1 = cust.Latest_mon_usage[cust.segment == 1]
s2 = cust.Latest_mon_usage[cust.segment == 2]
s3 = cust.Latest_mon_usage[cust.segment == 3]

# perform ANOVA test
stats.f_oneway(s1, s2, s3)


# Pvalue is less than 0.05. Hence, we can reject the NULL hypothesis which means there is a significant difference between the average of the segments.
# 
# If fvalue is high that means the segments are different and when the segments are different that means there is a relationship between the variables

# In[ ]:


print(s1.mean(), s2.mean(), s3.mean() )


# 5. Is there any relationship between region and segment? 
# 
# Chi Square test
# 
# H0 : There is no relationship
# 
# Ha : There is relationship 
# 
# if p < 0.05 , then we reject NULL, 
# 
# else, we fail to reject NULL Hypothesis.
# 

# ### ChiSquare Test

# In[ ]:


t = pd.crosstab(cust.segment, cust.region, margins = True)
t
# actual distribution between segment and region


# In[ ]:


stats.chi2_contingency(observed = t)


# Based on the p value we can say there is a relationship between region and segment. 

# 6. Is there any relationship between card_usage_before and latest_month_usage.
# 
# Correlation
# 
# H0 : There is no relationship
# 
# Ha : There is relationship 
# 
# if p < 0.05 , then we reject NULL, There is relationship between latest month spend and pre usage
# 
# else, we fail to reject NULL Hypothesis.
# 

# ### Correlation

# In[ ]:


print(np.corrcoef(cust.Latest_mon_usage, cust.Post_usage_1month))


# In[ ]:


print(stats.stats.pearsonr(cust.Latest_mon_usage, cust.Post_usage_1month))


# Since pval is less than 0.05 we reject the NULL hypothesis. 
# Hence, there is a relationship. Moreover, Correlation talks about the linear relationship between the variables. That means if one is changing the other is also changing. 

# ### Note:
# 
# if the correlation is equal to zero, that doesn't mean there is no relationship between the variables however that means there is no linear relationship between the variables; they may have a non linear relationship 

# I hope this helps in building the fundamental knowledge in performing the Hypothesis. If you liked this post please do not forget to upvote! :)
