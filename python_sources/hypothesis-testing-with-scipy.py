#!/usr/bin/env python
# coding: utf-8

# **Hypothesis Testing** <br><br>
# I couldn't find a hypothesis testing kernel which was clear enough to me, so I decided to make one. Some stuff might have been omitted if I think it is too confusing for beginners. I do not go into detail on formulas since they are easy to forget and won't be used in 'real' situations. <br><br> I will look at how and when to use a hypothesis test, what the requirments are and how to use standard function to perform hypothesis tests.<br>
# 
# One important thing I want to make clear to beginners is that there are lots of different hypothesis tests and a different test is required depending on *what* you are trying to test.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from statsmodels.stats import weightstats as mstats

df_exams = pd.read_csv('../input/StudentsPerformance.csv')

print("finished imports and dataframe creation")


# In[ ]:


df_exams.head()


# **Tidy up the column names so they're easier to work with**

# In[ ]:


#make the column names easier to work with
df_exams.rename(columns={'race/ethnicity':'ethnicity'},inplace=True)
df_exams.rename(columns={'parental level of education':'parents_education'},inplace=True)
df_exams.rename(columns={'test preparation course':'test_prep_course'},inplace=True)
df_exams.rename(columns={'math score':'math_score'},inplace=True)
df_exams.rename(columns={'reading score':'reading_score'},inplace=True)
df_exams.rename(columns={'writing score':'writing_score'},inplace=True)


# In[ ]:


df_exams.head()


# **Quickly check to see if the data is normally distirubted and if the groups are evenly represented**

# In[ ]:


#group C is over represented in the population
#when sampling we will have to take maybe 50 of each group
#using say 80 of group A seems like a bad idea, since we won't get much variation
#in any random sample we take
df_exams['ethnicity'].value_counts()


# In[ ]:


exams = ['math_score','reading_score','writing_score']
for exam in exams:
    y = df_exams[exam]
    plt.figure(1); plt.title('Normal')
    sns.distplot(y, kde=False, fit=stats.norm)

    plt.figure(2); plt.title('Johnson SU')
    sns.distplot(y, kde=False, fit=stats.johnsonsu)

    fig = plt.figure()
    res = stats.probplot(df_exams[exam], plot=plt)
    plt.show()


# **Graph interpretation**<br><br>
# These graphs show that the data isn't perfectly normally distirubted, the tails on all of them do not fit the normal distribution entirely. However, it is approximately normally distributed, so we should be ok.<br><br>
# Additionally if they were normally distributed, the blue qq plot points would fit the red line perfectly.<br><br>
# Also remember that this is the entire population, not just a sample, and we will be using samples to perform the hypothesis tests which should also be approximately normally distributed. Z tests and T tests are not valid to use if the data is not normally distributed.

# **Hypothesis Tests**<br>
# 
# Hypotheis tests are used to test a theory about data in a scientifically rigourous way, so that we are not reliant on just chance or subjective assumptions.<br><br>
# 
# The tests I will be looking at will be about if a sample mean matches another sample mean, or (using the student scores as an example) if students of say a particular group's score differs from students of another group's score. This will make more sense once we get to the code. Other types of hypothesis tests exist such as tests based on probabilities and binomial distributions etc, but I'm not covering them here just yet.<br>
# 
# When setting up a hypothesis test, there is a null hypothesis (H0) and an alternative hypothesis (H1). A significance level must also be set, usually it is 5%. Significance level is the *highest* p-value which would be acceptable for us reject the null hypothesis.<br><br>
# 
# Hypothesis tests can be one-tailed or two tailed. One tail only tests one side (e.g your H1 might be 'I expect group A scores *higher* than group B'). Two tail (e.g 'I expect group A score *differently* than group B'). as you can see, two tail would account for high or lower 'math_score' results, but one tail only checks for the higher 'math_score'. <br><br>
# 
# Hypothesis tests usually generate a t-score or z-score, which are used to look up a t-distribution table to generate a p-value, which in turn represents the likelihood of us observing a t-score or z-score at least as extreme as the one we observe. p-values are usually automatically generated for us by library functions such as the hypothesis tests in scipy.stats. <br><br>
# 
# p-values as mentioned represent the likelihood of seeing scores at least as extreme as what we have observed. we usually want to see a low p-value (<= 0.05, the chosen significance level) which will confirm our findings.<br><br>
# 
# To start, lets take a sample of 50 students from group A and 50 from group B. Then calculate the Z scores for this data sample.

# In[ ]:


#shuffle the dataframe
df_exams = df_exams.sample(frac=1)

group_a_sample = df_exams[df_exams['ethnicity'] == 'group A']
group_a_sample = group_a_sample[:50]

#if the qq plot points are on the line, the data is normally distributed
fig = plt.figure()
res = stats.probplot(group_a_sample['math_score'], plot=plt)
plt.show()

group_b_sample = df_exams[df_exams['ethnicity'] == 'group B']
group_b_sample = group_b_sample[:50]

fig = plt.figure()
res = stats.probplot(group_b_sample['math_score'], plot=plt)
plt.show()

print(stats.zscore(group_a_sample['math_score']))
print(stats.zscore(group_b_sample['math_score']))


# The Z score tells us how many standard deviations from the mean of the sample the data point lies. -1.78 is 1.78 standard deviations to the left, 0.526 is 0.526 standard deviations to the right
# 
# **Now for a Z test**<br>
# The below code is an example of a z-test. The test is to see if the group a sample's 'mean math_score' is smaller than the population mean. <br><br>
# H0 = The students in group A mean 'math_score' is the same as the population as whole.<br>
# H1 = The students in group A mean 'math_score' is **smaller** than the population as a whole.
# 

# In[ ]:


#in reality you might estimate this value another way for a very large population
population_mean = df_exams['math_score'].mean()
#sample mean 'math_score' (just for illustrative purposes, not strictly required)
sample_group_a_mean = group_a_sample['math_score'].mean()

print(population_mean,sample_group_a_mean)

#this is a 'one sample' test
zstat, pvalue = mstats.ztest(group_a_sample['math_score'],x2=None,value=population_mean,alternative='smaller')

#p-value is very very small < 0.1%; this is enough evidence to reject the null hypothesis
#of course this was expected and obvious, but this is supposed to be a transparent,clear example
print(float(pvalue))


# **T-test**<br>
# in a T test, the sample is assumed to be normally distributed. It is used when the population parameters are not known. There are 3 versions:
# 
# 1. Independent samples t-test which compares mean for two groups<br>
# 2. Paired sample t-test which compares means from the same group at different times<br>
# 3. one sample t-test which tests the mean of a single group against a known mean<br>
# 
# related scipy methods:
# 1. t,p-(two-tailed) = scipy.stats.ttest_ind(group_a,group_b), e.g exam scores of 2 different groups
# 2. t,p-(two tailed) = scipy.stats.ttest_rel(exam_a,exam_b) , e.g same set of students taking 2 exams at different times
# 3. t,p-(two-tailed) = stats.ttest_1samp(sample[],expected_if_null_hypothesis_value)
# 
# A large t-statistic tells us the groups are different. a small t-statistic tell us the groups are similar.
# 
# Below are examples of these 3 test and hypotheses for them. **Depending on the samples that are randomly chosen, they may or may not pass the hypotheses I describe. the notes I've written match what happend when I ran the code.**

# **Independent samples t-test which compares mean for two groups**<br><br>
# Let's try to see if **group_a** scores differently from **group_b** for the 'math_score' feature.<br>
# 
# **Hypothesis statement:**<br>
# H0 = The sample group A mean 'math_score' is **the same** as sample group B .<br>
# H1 = The sample group A mean 'math_score' is **Different** than sample group B.<br>
# testing at 5% significance level. using ttest_ind, this is a two-tailed test.
# 

# In[ ]:


#The variance is not the same for both samples. so we need to specify that in the function
print(group_a_sample['math_score'].var(),group_b_sample['math_score'].var())

tscore,pvalue = stats.ttest_ind(group_a_sample['math_score'],group_b_sample['math_score'],equal_var=False)
print(tscore,pvalue)

print(group_a_sample['math_score'].mean(),group_b_sample['math_score'].mean())
#the scores of group A and B are different, but not statistically significant enough
#according to our p-value so not enough evidence to reject the null hypothesis


# **Paired sample t-test which compares means from the same group at different times**<br><br>
# Let's try to see if **group_a** scores differently on 'math_score' than on 'reading_score'.<br>
# 
# **Hypothesis statement:**<br>
# H0 = The sample group A mean 'math_score' is **the same** as 'reading_score' .<br>
# H1 = The sample group A mean 'math_score' is **Different** than 'reading_score'.<br>
# testing at 5% significance level. using ttest_rel, this is a two-tailed test.

# In[ ]:


tscore,pvalue = stats.ttest_rel(group_a_sample['math_score'],group_a_sample['reading_score'])
print(tscore,pvalue)

print(group_a_sample['math_score'].mean(),group_a_sample['reading_score'].mean())
#the scores for 'math_score' and 'reading_score' are different
#we can reject the null hypothesis as the p-value is within our 5% threshold
#so there is a difference between math_score and reading_score for group a


# **One sample t-test which tests the mean of a single group against a population mean**<br><br>
# Let's try to see if **group_a** sample has a mean 'math_score' the same as the population mean.<br>
# 
# **Hypothesis statement:**<br>
# H0 = The sample group A mean 'math_score' has the same value as the population mean.<br>
# H1 = The sample group A mean 'math_score' is different from the population mean.<br>
# testing at 5% significance level. using ttest_1samp, this is a two-tailed test.

# In[ ]:


tscore,pvalue = stats.ttest_1samp(group_a_sample['math_score'],df_exams['math_score'].mean())
print(tscore,pvalue)

print(group_a_sample['math_score'].mean(),df_exams['math_score'].mean())

#the pvalue is very low (< 5%), so it we reject the null hypothesis
#we can also see that the sample mean is different from the population mean


# **End**<br><br>
# in future additions, I may look at ANOVA hypothesis testing, Chi-square tests and hypothesis tests to use with binomially distributed data and probabilities
