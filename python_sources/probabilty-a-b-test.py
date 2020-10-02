#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# Divyansh Shah - June 2020
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I - Probability](#probability)
# - [Part II - A/B Test](#ab_test)
# - [Part III - Regression](#regression)
# 
# 
# <a id='intro'></a>
# ### Introduction
# 
# A/B tests are very commonly performed by data analysts and data scientists.  
# 
# For this project, I will be working to understand the results of an A/B test run by an e-commerce website.  The goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# <a id='probability'></a>
# #### Part I - Probability
# 
# To get started, let's import our libraries.

# In[ ]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
random.seed(42)


# `1.` Read in the `ab_data.csv` data

# In[ ]:


df = pd.read_csv('../input/ab-data/ab_data.csv')
df.head()


# Number of rows in the dataset.

# In[ ]:


df.shape


# Unique users in the dataset.

# In[ ]:


df['user_id'].nunique()


# The proportion of users converted.

# In[ ]:


df['converted'].mean()


# The number of times the `new_page` and `treatment` don't line up.

# In[ ]:


#Adding the two values
len(df.query('group=="treatment" and landing_page!="new_page"')) + len(df.query('group=="control" and landing_page!="old_page"'))


# Missing values

# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# The rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page. Dropping those rows and storing the new dataframe in **df2**.

# In[ ]:


#Finding the mismatch rows
mismatch_1 = df.query('group=="treatment" and landing_page!="new_page"')
mismatch_2 = df.query('group=="control" and landing_page!="old_page"')

#Dropping those columns by their index
df2 = df.drop(mismatch_1.index)
df2 = df2.drop(mismatch_2.index)


# In[ ]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# Unique **user_id**s in **df2**

# In[ ]:


#No. of unique rows
df2['user_id'].nunique()


# In[ ]:


#Comparing with total size
df2['user_id'].shape


# Looking for duplicated user_ids

# In[ ]:


#User ID of the repeated row
df2[df2['user_id'].duplicated()==True]['user_id']


# In[ ]:


#Info of the repeated column
df2[df2['user_id'].duplicated()==True]


# Removing **one** of the rows with a duplicate **user_id**

# In[ ]:


#Dropping the duplicated row
df2.drop_duplicates(subset='user_id',inplace=True)


# **Probability of an individual converting regardless of the page they receive**

# In[ ]:


df2['converted'].mean()


# **`control` group probability that they converted?**

# In[ ]:


#Probability of control group
df2_control = df2.query('group=="control"')
df2_control['converted'].mean()


# **`treatment` group probability that they converted**

# In[ ]:


#Probability of treatment group
df2_treatment = df2.query('group=="treatment"')
df2_treatment['converted'].mean()


# Probabilty of users received the new page

# In[ ]:


#Split of pages between pages
df2.query('landing_page=="new_page"').shape[0]/df2.shape[0]


# **The Control group had a conversion rate of 12.03% and Treatment group had a conversion rate of 11.88% among all the testing they did on users. Looking at this probability the new page doesn't seem to have much of an impact on the conversion rate as both are nearly equal. But it seems interesting to find further details of why this is happening.**

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, we could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do we run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# For now, consider we need to make the decision just based on all the data provided.  If we want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, the null $H_{0}$ and alternative $H_{1}$ hypotheses be? They are stated below in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **ANSWER** 
# <br>
# $$ H_0: p_{old} \ge p_{new}$$
# <br>
# $$H_1: p_{old} < p_{new}$$

# Assuming under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Performing the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>

# **Convert rate** for $p_{new}$ under the null

# In[ ]:


#Calculating convert rate
p_new = df2.converted.mean() 
p_new


# **Convert rate** for $p_{old}$ under the null<br><br>

# In[ ]:


#Calculating convert rate
p_old = df2.converted.mean()
p_old ## Both are same!!


# $n_{new}$ size of new page group

# In[ ]:


#size of new page group
n_new = len(df2[df2['group'] =='treatment'])
n_new


# $n_{old}$ size of old page group

# In[ ]:


#size of new page group
n_old = len(df2[df2['group'] =='control'])
n_old


# Simulating $n_{new}$ transactions with a convert rate of $p_{new}$ under the null. 

# In[ ]:


#Simulating conversion for new page for 0's and 1's
new_page_converted = np.random.binomial(1, p_new, size=n_new)
new_page_converted


# f. Simulating $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.

# In[ ]:


#Simulating conversion for old page 0's and 1's 
old_page_converted = np.random.binomial(1, p_old, size=n_old)
old_page_converted


# $p_{new}$ - $p_{old}$ for the simulated values

# In[ ]:


#Single simuation difference in mean
diff = new_page_converted.mean() - old_page_converted.mean()
diff


# Simulating for 10,000 iteration $p_{new}$ - $p_{old}$ values. Storing all 10,000 values in **p_diffs**.

# In[ ]:


#Simulating for 10000 times
p_diffs=[]
for _ in range(10000):
    new_converted = np.random.binomial(1, p_new, size=n_new).mean()
    old_converted = np.random.binomial(1, p_old, size=n_old).mean()
    p_diffs.append(new_converted-old_converted)


# In[ ]:


#Converting to array
p_diffs = np.array(p_diffs)


# Histogram of **p_diffs**.

# In[ ]:


#Actual difference from dataset
acc_diff = df2_treatment['converted'].mean() - df2_control['converted'].mean() 
acc_diff


# In[ ]:


plt.hist(p_diffs);
plt.xlabel('Mean of Probability Difference')
plt.ylabel('#Count')
plt.axvline(acc_diff,color='red',label='Actual mean difference')
plt.legend()


# Proportion of the **p_diffs** greater than the actual difference observed in **ab_data.csv** i.e finding the **p-value**

# In[ ]:


#P-Value
(p_diffs > acc_diff).mean()


# **We found the p-value above, it is the probability of the null hypothesis being true. So in order to make a change the p-value should be as low as possible.
# <br>
# Here the p-value is 0.9089 which is very high than the max acceptable error rate of 0.05. Currently we have no statistical evidence to reject the null hypothesis and there is no significant difference between the old and new pages.**<br><br>
# ***Hence, we fail to reject the Null Hypothesis.*** 

# ### Using the Z-Test this time to verify the above findings

# In[ ]:


import statsmodels.api as sm

convert_old = len(df2.query('landing_page == "old_page" and converted ==1 '))
convert_new = len(df2.query('landing_page == "new_page" and converted ==1 '))
n_old = df2.query('landing_page == "old_page"').shape[0]
n_new = df2.query('landing_page == "new_page"').shape[0]


# In[ ]:


#Values of the above
convert_old,convert_new,n_new,n_old


# `stats.proportions_ztest` to compute test statistic and p-value.  [Reference Link](http://knowledgetack.com/python/statsmodels/proportions_ztest/) 

# In[ ]:


#smaller as we need to have p1 < p2 for alternative hypothesis
z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller') 
z_score, p_value


# **From the above, the Z-Score is 1.31 which is less than the critical value of 1.96 for 95% confidence interval so we** ***fail to reject the null hypothesis*** **and the p-value is 0.90 which also determines that** ***we fail to reject the null hypothesis.***  
# <br>
# Source : https://www.statisticshowto.com/probability-and-statistics/find-critical-values/#CVZ

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# In this part, we will try to further verify the result acheived in the previous A/B test by performing regression.<br><br>
# 
# Since each row is either a conversion or no conversion, we will be using **LOGISTIC REGRESSION**

# Creating a column for the intercept, and also create a dummy variable column for which page each user received. **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[ ]:


df2.head()


# In[ ]:


df2['intercept'] = 1


# In[ ]:


df2[['control','ab_page']] = pd.get_dummies(df2['group'])


# In[ ]:


df2.drop('control',axis=1,inplace=True)


# Instantiate the model and fit the model using the two columns to predict whether or not an individual converts.

# In[ ]:


lm = sm.Logit(df2['converted'],df2[['intercept','ab_page']])
result = lm.fit()


# Summary of the model 

# In[ ]:


result.summary()


# **The p-value pf ab_page in the regression model is 0.190 which significantly lower than the p-value in Part II.
# <br>
# This is because the Null and Alternative Hypothesis of Part II and Part III differed.**
# <br>
# $$Part II:$$
# $$ H_0: p_{old} - p_{new} \ge 0$$ 
# $$ H_1: p_{old} - p_{new} < 0$$
# <br>
# $$Part III:$$
# $$ H_0: p_{old} = p_{new}$$
# $$ H_1: p_{old} \ne p_{new}$$
# <br>
# Also in part 2 we implemented a one tailed test whereas in part 3 we implemented two tailed test. 

# **It would yield better results when we use other explanatory variables in the model as they might also have an influence on the decision of whether a user converts or not.
# <br>
# There may also arise an issue of adding more explanatory variables, if there is any collinearity between these variables the model would not give proper results.**

# ### Let's add an effect based on which country a user lives.
# 
# Create dummy variables for these country columns. We will need two columns for the three dummy variables while putting it in the regression model.

# In[ ]:


countries_df = pd.read_csv('../input/ab-data/countries.csv')
df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')


# In[ ]:


df_new[['CA','UK','US']] = pd.get_dummies(df_new['country'])


# In[ ]:


lm = sm.Logit(df_new['converted'],df_new[['CA','US','intercept']])
result = lm.fit()
result.summary()


# **The p-value of both the countries is greater than 0.05, so we do not have enough evidence to show that the countries have an effect on conversion.** <br><br>***We fail to reject the null hypothesis here as well.***

# ### Though we have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  

# In[ ]:


# Adding Interactions
df_new['US_ab_page'] = df_new['US'] * df_new['ab_page']
df_new['CA_ab_page'] = df_new['CA'] * df_new['ab_page']


# In[ ]:


lm = sm.Logit(df_new['converted'],df_new[['CA','US','intercept','ab_page','US_ab_page','CA_ab_page']])
result = lm.fit()
result.summary()


# **The pages interaction with countries also does not have any impact on the conversion rate as the p values are above 0.05.**<br>
# ***Hence we again fail to reject the null hypothesis.***

# <a id='conclusions'></a>
# ## Conclusions
# 
# From the above results where every situation we carried out the p-value was greater than 0.05 and there was no statistical evidence to reject the null hypothesis. We can say that the company should not implement the new page as it does not significant impact on conversion rates and rather continue with the old page. 
