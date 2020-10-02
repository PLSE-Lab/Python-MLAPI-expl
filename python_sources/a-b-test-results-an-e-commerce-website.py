#!/usr/bin/env python
# coding: utf-8

# ## A/B Test Results an e-commerce website
# 
# 
# 
# ## Table of Contents
# - [Introduction](#intro)
# - [Part I Wrangle data & Exploratory Data Analysis](#wrangle)
# - [Part II A/B Test](#ab_test)
# - [Part III Regression](#regression)
# - [Conclusion](#conclusion)
# 
# <a id='intro'></a>
# ### Introduction
# 
#  For this project, I will be working to understand the results of an A/B test run by an e-commerce website. My goal is to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.

# <img src="https://i.imgur.com/UoJrYZQ.png"/>
# 

# <a id='wrangle'></a>
# #### Part I - Wrangle data & Exploratory Data Analysis
# 
# To get started, let's import our libraries.

# In[ ]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy.stats import norm


# In[ ]:


# Read in the dataset and take a look at the top few rows
df = pd.read_csv('../input/ab_data.csv')
df.head()


# In[ ]:


df.info()
#The total of users is 294,478


# In[ ]:


# check the number of unique users in the dataset.
df.user_id.nunique()


# In[ ]:


#Check the proportion of users converted
p= df.query('converted == 1').user_id.nunique()/df.shape[0]

print("The proportion of users converted is {0:.2%}".format(p))


# In[ ]:


# Check the number of times the new_page and treatment don't line up.
l = df.query('(group == "treatment" and landing_page != "new_page" )          or (group != "treatment" and landing_page == "new_page")').count()[0]
print("The number of times the new_page and treatment don't line up is {}".format(l))


# In[ ]:


#Check missing values
df.isnull().sum()


# #### Note
# For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we can not be sure if this row truly received the new or old page. So, I will drop these rows and create a new dataframe.  

# In[ ]:


df2 =df.drop(df.query('(group == "treatment" and landing_page != "new_page" )                       or (group != "treatment" and landing_page == "new_page") or (group == "control" and landing_page != "old_page") or (group != "control" and landing_page == "old_page")').index)


# In[ ]:


# Double Check all of the correct rows were removed 
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# In[ ]:


#Check the number of unique user_ids are in df2
df2.user_id.nunique()


# In[ ]:


df2.head()


# In[ ]:


df['landing_page'].value_counts().plot(kind='bar', figsize=(8,8));


# In[ ]:


df['landing_page'].value_counts().plot(kind='pie', figsize=(8,8));


# In[ ]:





# In[ ]:


lan_rev =df.groupby('landing_page').sum()['converted']

ind = np.arange(len(lan_rev))  # the x locations for the groups
width = 0.35  

plt.subplots(figsize=(18,10))
gen_bars =plt.bar(ind, lan_rev, width, color='g', alpha=.7)
#adv_bars =plt.bar(ind, adv, width, color='b', alpha=.7, label="Adventure")
plt.ylabel('converted',size=14) # title and labels
plt.xlabel('landing_page',size=14)
plt.title('Conversion by landing_page',size=18)
locations = ind + width / 2  # xtick locations
labels = ['old_page', 'new_page']  # xtick labels
plt.xticks(locations, labels)


# In[ ]:


#Check duplicates rows
df2.user_id.duplicated().sum()


# In[ ]:


#Check the repeated user_id
df2[df2.duplicated(['user_id'],keep=False)]['user_id']
print("The user_id repeated is 773192")


# In[ ]:


# Check the row information for the repeat user_id
df2.query('user_id == 773192')


# In[ ]:


#Remove the duplicated rows
df2 = df2.drop(df2.query('user_id == 773192 and timestamp == "2017-01-09 05:37:58.781806"').index)


# In[ ]:


#Check if there is any repeated user_id 
df2.user_id.duplicated().sum()


# In[ ]:


# Calculate the probability of an individual converting regardless of the page they receive
df_prob =df2.query('converted == 1').user_id.nunique()/df2.user_id.nunique()
df_prob

print("The probability of an individual converting regardless of the page they receive is {0:.2%}".format(df_prob))


# In[ ]:


# Calculate the probabilty the individual was in the control group to convert
p_cont = df2.query('converted == 1 and group == "control"').user_id.nunique() /df2.query('group == "control"').user_id.nunique()

print("The probability they converted based on control group is {0:.2%}".format(p_cont))


# In[ ]:


# Calculate the probabilty the individual was in the treatment group to convert
p_treat = df2.query('converted == 1 and group == "treatment"').user_id.nunique() /df2.query('group == "treatment"').user_id.nunique()

print("The probability they converted based on treatment group is {0:.2%}".format(p_treat))


# In[ ]:


# Calculate the probabilty that an individual received the new page
p_n = df2.query('landing_page == "new_page"').user_id.nunique()/df2.user_id.nunique()
#The probability that an individual received the new page is 50.00%
print("The probability that an individual received the new page is {0:.2%}".format(p_n))


# ### Probability results
# >It seems to be that there is insufficient evidence to say that the new treatment page leads to more conversions than the control page. The difference of probability between control (12.04%) and treatment groups (11.88%) is tiny, especially when we compare them with the probability of individual conversion (11.96%)

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Since we do not have  sufficient evidence to say that the new treatment page leads to more conversions than the control page with probability tests, I will run a hypothesis test continuously as each observation was observed with the time stamp associated with each event. 
#  
# `1.`I will consider making the decision only based on all the data provided. Further, I want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%. The null and alternative hypotheses follow below:
# 
# > $$H_0:  P_{new} - P_{old}  \leq  0$$
# 
# 
# > $$H_1: P_{new} - P_{old} > 0$$

# `2.` I will assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page -- that is $p_{new}$ and $p_{old}$ are equal. Furthermore, I will assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# I wil use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# I will perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>

# In[ ]:


# Since P_new and P_old both have "true" success rates equally, their converted rate 
#will have the same result.
p_new = df2.converted.mean()

print("The convert rate for p_new under the null is {0:.4}".format(p_new))


# In[ ]:


# Since P_new and P_old both have "true" success rates equally, their converted rate 
#will have the same result.
p_old = df2.converted.mean()

print("The convert rate for p_old under the null is {0:.4}".format(p_old))


# In[ ]:


# Count the total unique users with new page
n_new = df2.query('landing_page == "new_page" ').count()[0]
n_new


# In[ ]:


# Count the total unique users with old page
n_old = df2.query('landing_page == "old_page" ').count()[0]
n_old


# In[ ]:


#Simulate n_new transactions with a convertion rate of  p_new under the null. 
#Store these n_new 1's and 0's in new_page_converted

new_page_converted = np.random.choice([0,1],n_new, p=(p_new,1-p_new))
new_page_converted


# In[ ]:


#Simulate n_new transactions with a convert rate of  p_old under the null. 
#Store these  n_new 1's and 0's in old_page_converted

old_page_converted = np.random.choice([0,1],n_old, p=(p_old,1-p_old))
old_page_converted


# In[ ]:


# Find the difference between p_new and p_old
#For discovering the difference between p_new and p_old, it is necessary to find out the mean 
#of new_page_converted and old_page_converted.
new_page_converted.mean()


# In[ ]:


old_page_converted.mean()


# In[ ]:


#diff_conv is the difference between p_new and p_old.
diff_conv = new_page_converted.mean() - old_page_converted.mean()
diff_conv


# In[ ]:


# Simulate 10,000 p_new - p_old values with random binomial

new_converted_simulation = np.random.binomial(n_new, p_new,  10000)/n_new
old_converted_simulation = np.random.binomial(n_old, p_old,  10000)/n_old
p_diffs = new_converted_simulation - old_converted_simulation


# In[ ]:


p_diffs = np.array(p_diffs)


# In[ ]:


plt.hist(p_diffs);


# In[ ]:


# Calculate actual difference observed
new_convert = df2.query('converted == 1 and landing_page == "new_page"').count()[0]/n_new
old_convert = df2.query('converted == 1 and landing_page == "old_page"').count()[0]/n_old
obs_diff = new_convert - old_convert
obs_diff


# In[ ]:


#Check the proportion of the p_diffs are greater than the actual difference observed in ab_data.
null_vals = np.random.normal(0, p_diffs.std(), p_diffs.size)


# In[ ]:


plt.hist(null_vals);
plt.axvline(x=obs_diff, color='red')


# In[ ]:


(null_vals > obs_diff).mean()


# #### Sampling distribution analysis
# 
# >1- The proportion of the conversion rate differences were greater than the actual observed difference. The p-value is extremely large (90%) than the type I error rate (5%).That means we fail to reject the null hypothesis.  
# 
# >2- According to Wikipedia, p-value is the probability for a given statistical model that, when the null hypothesis is true, the statistical summary (such as the sample mean difference between two compared groups) would be the same as or of greater magnitude than the actual observed results.
# 
# >3- When the p-value is low (in this project less than 5%), it suggests that the null hypothesis is not true, and we need to consider the alternative hypothesis. Finally, the p-value of 90% indicates that the actual page should be maintained.

# `3.` Now, I will use  stats.proportions_ztest to compute my test statistic and p-value for evaluating if there is a statistically significance difference in conversion rates of the new page and the conversion rates of the old page.
# 
# First, I will calculate the number of conversions for each page, as well as the number of individuals who received each page. The `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[ ]:


convert_old = df2.query('converted == 1 and landing_page == "old_page"').count()[0]
convert_new = df2.query('converted == 1 and landing_page == "new_page"').count()[0]
n_old = df2.query('landing_page == "old_page" ').count()[0]
n_new = df2.query('landing_page == "new_page" ').count()[0]


# In[ ]:


convert_old,convert_new,n_old,n_new


# In[ ]:


z_score, p_value = sm.stats.proportions_ztest(np.array([convert_new,convert_old]),                                              np.array([n_new,n_old]), alternative = 'larger')


# In[ ]:


z_score, p_value


# In[ ]:


norm.cdf(z_score)
#0.09494168724097551 # Tells us how significant our z-score is


# In[ ]:


norm.ppf(1-(0.05/2))
# 1.959963984540054 # Tells us what our critical value at 96% confidence is


# #### Z-test analysis
# >Since the z-score of 1.31 does not exceed the critical value of 1.96, we fail to reject the null hypothesis. The conversion rates of the old page is greater than or equal to the conversion rates of the new. Moreover, there was not a significant difference between the conversion rates of the new page and the conversion rates of the old page (>0.15%).

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# In this final part, I will confirm that the result acheived in the previous A/B test can also be acheived by performing logistic regression. I will use **statsmodels** to fit the regression model to see if there is a significant difference in conversion based on which page a customer receives. <br><br> 

# In[ ]:


df2.head()


# In[ ]:


#Create intercept and dummies columns
df2['intercept'] = 1
df2[['ab_page','old_page']] = pd.get_dummies(df2['landing_page'])
df2 = df2.drop('old_page', axis = 1)


# In[ ]:


df2.head()


# In[ ]:


#Create a model
log = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results = log.fit()
results.summary()


# #### Regression analysis I
# 1- The **p-value** associated with ab_page is 0.19 and the **p-value** in **Part II** was 0.90. So, in both cases, we fail to reject the null hyphothesis because these two p-values are greater than 0.05(Type Error I).
# 
# 2- The difference lies in what each test assumes for their hypothesis. In Part II, the hyphothesis is to analyze if the old page is better unless the new page proves to be definitely greater at a Type I error rate of 5%. In other words,we were concerned with which page had a higher conversion rate, so a one-tailed test. While in the Part III hyphotheses, there is a significant difference in conversion based on which page a customer receives.  The nature of a regression test is not concerned with which had a positive or negative change, specifically. It is concerned with if the condition had any effect at all, so a two-tailed test.

# #### Considering other things that might influence whether or not an individual converts.
# There are many aspects that may influence whether or not an individual converts. For instance, we could consider factors as country, age, gender, city, hour or weekday and try to understand the correlation between them and the effects under the two groups. Another thing to deal with is the Simpson's paradox, in which a trend appears in several different groups of data, but disappears or reverses when these groups are combined. It is sometimes given the descriptive title of reversal paradox or amalgamation paradox.

# #### Regression approach II (add countries)
# Now, I will analyze if the countries have an impact on conversion. So, I will along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. I will read in the **countries.csv** dataset and merge together my datasets on the approporiate rows. 

# In[ ]:


countries_df = pd.read_csv('../input/countries.csv')


# In[ ]:


#Merge the countries data frame with df2
df3 = df2.merge(countries_df, on='user_id', how='inner')
df3.head()


# In[ ]:


df3.country.unique()


# In[ ]:


### Create the necessary dummy variables
df3[['US', 'CA', 'UK']] = pd.get_dummies(df3['country'])
df3 = df3.drop(['country', 'US'], axis = 1)


# In[ ]:


df3.head()


# In[ ]:


#Print Summary
log2 = sm.Logit(df3['converted'], df3[['intercept','ab_page','CA','UK']])
results2 = log2.fit()
results2.summary()


# In[ ]:


# For better visualizing the coef, we exponentiated them with numpy.
1/np.exp(-0.0149),np.exp(0.0408), np.exp(0.0506)


# #### Regression analysis II
# 
# >For each 1 unit decrease in new_page, convert is 1.5 times as likely holding all else constant.
# 
# >For each 1 unit increase in CA, convert is 4.1 times as likely holding all else constant.
# 
# >For each 1 unit increase in UK, convert is 5.2 times as likely holding all else constant.

# Regression approach III (interaction between page and country)
# 
# Now, I will look at an interaction between page and country to see if there significant effects on conversion.  

# In[ ]:


df3.head()


# In[ ]:


#For understanding the interaction between page and country we need two create 
#two columns that multiple ab_page to the country.

df3['CA_new_page']=df3['ab_page']*df3['CA']
df3['UK_new_page']=df3['ab_page']*df3['UK']


# In[ ]:


df3.head()


# In[ ]:


### Print Summary
log3 = sm.Logit(df3['converted'], df3[['intercept', 'ab_page', 'CA', 'UK','CA_new_page','UK_new_page']])


# In[ ]:


results3 = log3.fit()
results3.summary()


# In[ ]:


# For better visualizing the coef, we exponentiated them with numpy.
1/np.exp(-0.0674),np.exp(0.0118), np.exp(0.0175),np.exp(0.0783),np.exp(0.0469)


# #### Regression analysis III
# 
# >For each 1 unit decrease in new_page, convert is 1.5 times as likely holding all else constant.
# 
# >For each 1 unit increase in CA, convert is 1.2 times as likely holding all else constant.
# 
# >For each 1 unit increase in UK, convert is 1.7 times as likely holding all else constant.
# 
# >For each 1 unit increase in CA new_page, convert is 8.1 times as likely holding all else constant.
# 
# >For each 1 unit increase in UK new_page, convert is 4.8 times as likely holding all else constant.

# In[ ]:


X = df3[['CA','UK','CA_new_page','UK_new_page']]
y = df3['converted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)


# In[ ]:


log_mod = LogisticRegression()
log_mod.fit(X_train, y_train)
preds = log_mod.predict(X_test)
confusion_matrix(y_test, preds)


# In[ ]:


accuracy_score(y_test, preds)


# <a id='conclusion'></a>
# ### Conclusion
# 
# The work described in this notebook is based on a database providing details on the conversion rate of two groups(treatment group that holds the new page and control group that holds the old page), on an E-commerce platform from 2017-01-02 to 2017-01-24. The goal was to decide whether the E-commerce website should keep the old page or change to a new. 
# 
# Regarding the quality of the data, we had only 1 row duplicated in a sample with 294,478 rows. So, that problem did not affect the results.  
# 
# To achieve our goal, we performed the following tests: A/B test with z-test and logistic regression models. In A/B test, we found the p-value is higher than type error I, and because this, we fail to reject the null hyphotesis. Then, we saw the z-score was 1.31 which does not exceed the critical value of 1.96, so we fail again to reject the null hyphothesis. 
# 
# After that, we used stasmodels to fit the regression model and we found there is a significant difference in conversion based on which page a customer receives. In the first experiment with regression, we analyzed the individual factors of country and page on conversion. 
# 
# The coefficient of our explanatory variables presented the following results: For each 1 unit decrease in new_page, conversion is 1.5 times as likely holding all else constant; For each 1 unit increase in CA, conversion is 4.1 times as likely holding all else constant; For each 1 unit increase in UK, conversion is 5.2 times as likely holding all else constant. In the last test we interpreted interaction between page and country to see if there significant effects on conversion, we discovered the following results: For each 1 unit increase in CA new_page, conversion is 8.1 times as likely holding all else constant. For each 1 unit increase in UK new_page, conversion is 4.8 times as likely holding all else constant.
# 
# Finally, I strongly recommend gathering more data per period of at least 4 months. I consider the period of 22 days insufficient for making the decision about whether we should keep the old page or change to the new page, even if all tests indicated that we should keep the old one. 
