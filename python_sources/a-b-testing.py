#!/usr/bin/env python
# coding: utf-8

# ## Analyze A/B Test Results
# ### by Luyuan Zhang, June 22nd 2018
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
# For this project, I will be working to understand the results of an A/B test run by an e-commerce website.  My goal is to work through this notebook to help the company understand if they should implement the new page, keep the old page, or perhaps run the experiment longer to make their decision.
# 
# <a id='probability'></a>
# ### Part I - Probability
# 
# import libraries.

# In[ ]:


import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
random.seed(42)


# `1.` Read in the `ab_data.csv` data. Store it in `df`.
# 
# a. Read in the dataset and take a look at the top few rows here:

# In[ ]:


df=pd.read_csv('ab_data.csv')
df.head()


# b. Find the number of rows in the dataset.

# In[ ]:


df.shape


# c. The number of unique users in the dataset.

# In[ ]:


df.user_id.nunique()


# d. The proportion of users converted.

# In[ ]:


(df.converted==1).mean()


# e. The number of times the `new_page` and `treatment` don't line up.

# In[ ]:


((df.group=='treatment') & (df.landing_page=='old_page')).sum()+ ((df.group=='control') & (df.landing_page=='new_page')).sum()


# f. Do any of the rows have missing values?

# In[ ]:


df.info()


# No missing values in any row

# `2.` For the rows where **treatment** is not aligned with **new_page** or **control** is not aligned with **old_page**, we cannot be sure if this row truly received the new or old page.  These rows need to be dropped.  
# 
# a. Create a new dataset **df2** with misaligned rows dropped.

# In[ ]:


#identify misaligned rows
df['misaligned']=((df.group=='treatment') & (df.landing_page=='old_page')) | ((df.group=='control') & (df.landing_page=='new_page'))


# In[ ]:


#extract rows where misgligned==False
df2=df.query('misaligned==False')


# In[ ]:


df2.shape


# In[ ]:


# Double Check all of the correct rows were removed - this should be 0
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]


# `3.` Use **df2** and the cells below to answer questions for **Quiz3** in the classroom.

# a. How many unique **user_id**s are in **df2**?

# In[ ]:


df2['user_id'].nunique()


# b. There is one **user_id** repeated in **df2**.  What is it?

# In[ ]:


df2['user_id'].value_counts().sort_values(ascending=False).head()


# c. What is the row information for the repeat **user_id**? 

# In[ ]:


df2.query('user_id==773192')


# d. Remove **one** of the rows with a duplicate **user_id**, but keep your dataframe as **df2**.

# In[ ]:


df2.drop(1899, axis=0, inplace=True)


# In[ ]:


df2.shape


# `4.` Use **df2** in the below cells to answer the quiz questions related to **Quiz 4** in the classroom.
# 
# a. What is the probability of an individual converting regardless of the page they receive?

# In[ ]:


(df2['converted']==1).mean()


# b. Given that an individual was in the `control` group, what is the probability they converted?

# In[ ]:


actual_pold=(df2.query('group=="control"')['converted']==1).mean()
actual_pold


# c. Given that an individual was in the `treatment` group, what is the probability they converted?

# In[ ]:


actual_pnew=(df2.query('group=="treatment"')['converted']==1).mean()
actual_pnew


# d. What is the probability that an individual received the new page?

# In[ ]:


(df2['landing_page']=='new_page').mean()


# e. Use the results in the previous two portions of this question to suggest if you think there is evidence that one page leads to more conversions?  Write your response below.

# ##### The difference between converted rate of control and treatment groups is very small. I am not convinced that one page leads to more conversions than other. More statistics are needed.

# <a id='ab_test'></a>
# ### Part II - A/B Test
# 
# Notice that because of the time stamp associated with each event, you could technically run a hypothesis test continuously as each observation was observed.  
# 
# However, then the hard question is do you stop as soon as one page is considered significantly better than another or does it need to happen consistently for a certain amount of time?  How long do you run to render a decision that neither page is better than another?  
# 
# These questions are the difficult parts associated with A/B tests in general.  
# 
# 
# `1.` For now, consider you need to make the decision just based on all the data provided.  If you want to assume that the old page is better unless the new page proves to be definitely better at a Type I error rate of 5%, what should your null and alternative hypotheses be?  You can state your hypothesis in terms of words or in terms of **$p_{old}$** and **$p_{new}$**, which are the converted rates for the old and new pages.

# **null:** **$p_{old}$** - **$p_{new}$** >=0**
# 
# **alternative:** **$p_{old}$** - **$p_{new}$** <0**

# `2.` Assume under the null hypothesis, $p_{new}$ and $p_{old}$ both have "true" success rates equal to the **converted** success rate regardless of page - that is $p_{new}$ and $p_{old}$ are equal. Furthermore, assume they are equal to the **converted** rate in **ab_data.csv** regardless of the page. <br><br>
# 
# Use a sample size for each page equal to the ones in **ab_data.csv**.  <br><br>
# 
# Perform the sampling distribution for the difference in **converted** between the two pages over 10,000 iterations of calculating an estimate from the null.  <br><br>
# 
# Use the cells below to provide the necessary parts of this simulation.  If this doesn't make complete sense right now, don't worry - you are going to work through the problems below to complete this problem.  You can use **Quiz 5** in the classroom to make sure you are on the right track.<br><br>

# a. What is the **convert rate** for $p_{new}$ under the null? 

# In[ ]:


pnew_null=(df2['converted']==1).mean()
pnew_null


# ##### under the null, the **$p_{new}$** is 0.1197

# b. What is the **convert rate** for $p_{old}$ under the null? <br><br>

# In[ ]:


pold_null=(df2['converted']==1).mean()
pold_null


# In[ ]:


p_null=pnew_null


# ##### under the null, the **$p_{old}$** is also 0.1197

# c. What is $n_{new}$?

# In[ ]:


n_new=(df2['landing_page']=='new_page').sum()
n_new


# ##### The **$n_{new}$** is 145310

# d. What is $n_{old}$?

# In[ ]:


n_old=(df2['landing_page']=='old_page').sum()
n_old


# ##### The **$n_{old}$** is 145274

# e. Simulate $n_{new}$ transactions with a convert rate of $p_{new}$ under the null.  Store these $n_{new}$ 1's and 0's in **new_page_converted**.

# In[ ]:


new_page_converted=np.random.binomial(n_new, p_null)


# f. Simulate $n_{old}$ transactions with a convert rate of $p_{old}$ under the null.  Store these $n_{old}$ 1's and 0's in **old_page_converted**.

# In[ ]:


old_page_converted=np.random.binomial(n_old, p_null)


# g. Find $p_{new}$ - $p_{old}$ for your simulated values from part (e) and (f).

# In[ ]:


diff=new_page_converted/n_new-old_page_converted/n_old
diff


# h. Simulate 10,000 $p_{new}$ - $p_{old}$ values using this same process similarly to the one you calculated in parts **a. through g.** above.  Store all 10,000 values in **p_diffs**.

# In[ ]:


p_diffs=[]
p_diffs = np.random.binomial(n_new, p_null, 10000)/n_new - np.random.binomial(n_old, p_null, 10000)/n_old   


# i. Plot a histogram of the **p_diffs**.  Does this plot look like what you expected?  Use the matching problem in the classroom to assure you fully understand what was computed here.

# In[ ]:


plt.hist(p_diffs, bins=200)
plt.xlim(-0.005, 0.005)
plt.xlabel('p_diffs')
plt.ylabel('counts')
plt.title('simulated p_diffs distribution')
plt.axvline(0.000, color='red');


# ##### This is what I expected. the center of p_diffs seem be at 0.000. In the simulation, pnew_null and pold_null are equal, so I would expect the center of p_diffs to be at 0. 

# j. What proportion of the **p_diffs** are greater than the actual difference observed in **ab_data.csv**?

# In[ ]:


actual_diff=actual_pnew-actual_pold
actual_diff, actual_pnew, actual_pold


# In[ ]:


actual_diff=actual_pnew-actual_pold
(p_diffs>actual_diff).mean()


# k. In words, explain what you just computed in part **j.**.  What is this value called in scientific studies?  What does this value mean in terms of whether or not there is a difference between the new and old pages?

# ##### What I calculated in part j is the p value. The value 0.90 is much larger than alpha 0.05. Therefore I failed to reject null, and new page is not better in leading to conversion than old page.

# l. We could also use a built-in to achieve similar results.  Though using the built-in might be easier to code, the above portions are a walkthrough of the ideas that are critical to correctly thinking about statistical significance. Fill in the below to calculate the number of conversions for each page, as well as the number of individuals who received each page. Let `n_old` and `n_new` refer the the number of rows associated with the old page and new pages, respectively.

# In[ ]:


import statsmodels.api as sm

convert_old = (df2.query('landing_page=="old_page"')['converted']==1).sum()
convert_new = (df2.query('landing_page=="new_page"')['converted']==1).sum()
n_old = (df2['landing_page']=='old_page').sum()
n_new=(df2['landing_page']=='new_page').sum()

convert_old, convert_new, n_old, n_new         


# m. Now use `stats.proportions_ztest` to compute your test statistic and p-value.  [Here](http://knowledgetack.com/python/statsmodels/proportions_ztest/) is a helpful link on using the built in.

# In[ ]:


z_score, p_value = sm.stats.proportions_ztest([convert_new, convert_old], [n_new, n_old], alternative='larger')
z_score, p_value


# In[ ]:


from scipy.stats import norm
# Tells us how significant our z-score is
print(norm.cdf(z_score))

# for our single-sides test, assumed at 95% confidence level, we calculate: 
print(norm.ppf(1-(0.05)))


# n. What do the z-score and p-value you computed in the previous question mean for the conversion rates of the old and new pages?  Do they agree with the findings in parts **j.** and **k.**?

# ##### z_score is 1.31, less than the critical value 1.64. P value is 0.9, larger than alpha 0.05. Therefore based on z-score and p value I fail to reject null, which is **$p_{old}$** - **$p_{new}$** >=0**. **$p_{new}$ ** is not statistically larger than **$p_{old}$ **.

# <a id='regression'></a>
# ### Part III - A regression approach
# 
# `1.` In this final part, you will see that the result you acheived in the previous A/B test can also be acheived by performing regression.<br><br>
# 
# a. Since each row is either a conversion or no conversion, what type of regression should you be performing in this case?

# ##### Logistic regression.

# b. The goal is to use **statsmodels** to fit the regression model you specified in part **a.** to see if there is a significant difference in conversion based on which page a customer receives.  However, you first need to create a colun for the intercept, and create a dummy variable column for which page each user received.  Add an **intercept** column, as well as an **ab_page** column, which is 1 when an individual receives the **treatment** and 0 if **control**.

# In[ ]:


df2['intercept']=1
df2['ab_page']=pd.get_dummies(df2['group'])['treatment']
df2.head()


# c. Use **statsmodels** to import your regression model.  Instantiate the model, and fit the model using the two columns you created in part **b.** to predict whether or not an individual converts.

# In[ ]:


lm=sm.Logit(df2['converted'], df2[['intercept', 'ab_page']])
results=lm.fit()
results.summary()


# d. Provide the summary of your model below, and use it as necessary to answer the following questions.

# ##### The intercept = -1.9888. exp(-1.9888)=0.13686. This means the conversion rate of baseline case is 0.13686. 
# 
# ##### The coefficient for ab_page (treatment) is -0.015. exp(-0.015)=0.985. 1/0.985=1.015. This means that baseline (control that uses old_page) is 1.015 times likely to result in conversion relative to treatment. 
# 
# ##### p value if 0.19. This is a large number compared to 0.05. Therefore I would not consider the results are statistically significant, and I fail to reject the null, which is that conversion rate does not depend on landing_page.

# e. What is the p-value associated with **ab_page**? Why does it differ from the value you found in the **Part II**?<br><br> 

# ##### The p value is 0.19.
# ##### This is different from results in Part II because the null hypothesis is different.     In Part II, null=new page is not better than old page, alternative=new page is better. So Part II is a one-sided test.      Here, we hope to find out that conversion depends on which landing page users use. The alternative here is that conversion rates are different, and null is there is no difference. So this is a two-sided test 

# f. Now, you are considering other things that might influence whether or not an individual converts.  Discuss why it is a good idea to consider other factors to add into your regression model.  Are there any disadvantages to adding additional terms into your regression model?

# ##### We will be better predicting response if other factors are considered. One disadvantage of including other factors is that explanatory variables might dependent on each other. Some predictors might lose significance, or result in even flipped coefficients

# g. Now along with testing if the conversion rate changes for different pages, also add an effect based on which country a user lives. You will need to read in the **countries.csv** dataset and merge together your datasets on the approporiate rows.  [Here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.join.html) are the docs for joining tables. 
# 
# Does it appear that country had an impact on conversion?  Don't forget to create dummy variables for these country columns - **Hint: You will need two columns for the three dummy varaibles.** Provide the statistical output as well as a written response to answer this question.

# In[ ]:


# read in the country table
country=pd.read_csv('countries.csv')


# In[ ]:


# merge country and df2 tables
df2=pd.merge(df2,country, on=['user_id'])
df2.head()


# In[ ]:


# create dummy columns for country
df2[['CA', 'UK','US']]=pd.get_dummies(df2['country'])
df2.head(1)


# In[ ]:


# run logistic regression on counties
lm=sm.Logit(df2['converted'], df2[['intercept', 'CA', 'UK']])
results=lm.fit()
results.summary()


# ##### Here the baseline is US. 
# ##### The coef for CA is -0.0408. exp(-0.048)=0.953. 1/0.953=1.049. 
#    ##### This means users in US is 1.049 times likely to convert relative to users in CA.
# ##### The coef for UK is 0.0099. exp(0.0099)=1.01.  
#    ##### This means users in UK is 1.01 times likely to convert, relative to users in US. 
# ##### The P values for CA and UK are large, so I won't consider the results are statistically significant.    

# h. Though you have now looked at the individual factors of country and page on conversion, we would now like to look at an interaction between page and country to see if there significant effects on conversion.  Create the necessary additional columns, and fit the new model.  
# 
# Provide the summary results, and your conclusions based on the results.

# In[ ]:


lm=sm.Logit(df2['converted'], df2[['intercept', 'ab_page', 'CA', 'UK']])
results=lm.fit()
results.summary()


# ##### coef for ab_page = -0.0149. exp(-0.0149)=0.985. 1/0.985=1.015.
#    ##### This means users in control group is 1.015 time like to be converted relative to treatment group, holding everything else constant. 
# ##### coef for CA = -0.048. exp(-0.0149)=0.953, 1/0.953=1.049
#    ##### This means users in US are 1.049 times likely to be converted relative to users in CA, holding everything else constant.
# ##### Coef for UK = 0.0099. exp(0.0099)=1.01.  
#    ##### This means users in UK is 1.01 times likely to be converted relative to users in US, holding everything else constant.   
# 
# ##### The coefficients obtained here are not different from those obtained in previous sections. Therefore I conclude that there is no effects whether to include single or multiple variables. The variables ab_page and country seem to be independent on each other.
# ##### The P values for CA and UK are large, so I won't consider the results are statistically significant.   

# ### Part IV - How does day of week affect conversion rate?

# #### I am interested to find out how day of week affects conversion rate.      Are users more likely to convert on a specific weekday? 
# #### To answer this question, I need to get day of week from the timestamp, then get dummies of day of week.

# In[ ]:


df2['datetime']=pd.to_datetime(df2['timestamp'], errors='coerce')


# In[ ]:


df2['dow']=df2['datetime'].dt.weekday_name


# In[ ]:


df2[pd.get_dummies(df2['dow']).columns]=pd.get_dummies(df2['dow'])


# In[ ]:


df2.head(2)


# In[ ]:


# perform logistic regression to see if there is significant difference of conversion rates in different day of week
lm=sm.Logit(df2['converted'], df2[['intercept','ab_page','Friday', 'Monday', 'Saturday', 'Thursday', 'Tuesday',
       'Wednesday']])
results=lm.fit()
results.summary()


# ##### Although the conversion rates in different days of week are more or less different, the difference and p values are not significant enough for me to conclude whether a specific week day had more conversion rate. Therefore I will pick out 2 different days, on which the conversion rates are biggest or smallest, and test if their difference is significant. 

# In[ ]:


# calculate mean conversion rates on each day of week
dow_columns=pd.get_dummies(df2['dow'])
dow_rate=pd.DataFrame([(lambda x:(df2[x] * df2.converted).sum()/df2[x].sum()) (x) for x in dow_columns], index=list(pd.get_dummies(df2['dow']).columns), columns=['conversion_rate'])
dow_rate


# ##### The biggest difference in conversion rates is between Friday and Monday. I will create a sub-dataframe that includes only these two days.

# In[ ]:


# create a sub-dataframe that only included Friday and Monday data
sub_df2=df2.query('dow=="Friday" | dow=="Monday"' )


# In[ ]:


# run a logistic regression to check the significance level of the conversion rate difference between the two days
# Friday is the baseline
lm=sm.Logit(sub_df2['converted'], sub_df2[['intercept', 'ab_page','Monday']])
results=lm.fit()
results.summary()


# ##### On Mondays, users are 1.04 times likely to be converted relative to users in Friday. If type I error is set at 0.05, given the p value here is 0.045, the regression result indicate a significant difference in conversion rate between Monday and Friday. 

# ### Part V - Should we run the test longer time? 
# #### To answer this question, I will:
# #### (1) first calculate the mean conversion rates at each different day as time went on. 
# #### (2) create visual for mean conversion rates vs. days
# #### (3) run a linear regression of the conversion rates on days

# In[ ]:


# check when did the test start and end
df2['datetime'].min(), df2['datetime'].max()


# ##### The test started on 2017-01-02, and ended on 2017-01-24. 
# 
# #### To get mean converion rates of each day, I will group rows by the day. 

# In[ ]:


df2['day']=df2['datetime'].dt.day


# In[ ]:


#create a dataframe that aggregates mean conversion rate of old and new pages of individual day
df2_by_day=pd.DataFrame()
old_df=df2.query('landing_page=="old_page"')
new_df=df2.query('landing_page=="new_page"')
df2_by_day['old_rate']=old_df.groupby('day')['converted'].mean()
df2_by_day['new_rate']=new_df.groupby('day')['converted'].mean()
df2_by_day.reset_index(inplace=True)


# #### view trends of p_old and p_new  vs. day,  to see if there is a visible correlation

# In[ ]:


# create a scatter plot to see if there is increase in conversion rate as time goes on
plt.scatter(df2_by_day['day'],  df2_by_day['old_rate'], color='green',label='p_old')
plt.scatter(df2_by_day['day'], df2_by_day['new_rate'], color='red',label='p_new')
plt.xlabel('day of January')
plt.ylabel('conversion rate')
#plt.ylim(0.11, 0.13)
plt.legend()
plt.title('conversion rate at different days of the month');


# #### view difference p_new-p_old vs. day,  to see if there is visible correlation

# In[ ]:


# create a scatter plot to see if there is increase in conversion rate as time goes on
plt.scatter(df2_by_day['day'],  df2_by_day['new_rate']-df2_by_day['old_rate'], color='blue',label='p_new - p_old')
plt.xlabel('day of January')
plt.ylabel('p_new - p_old')
plt.ylim(-0.05, 0.05)
plt.legend()
plt.title('p_new - p_old');


# #### run linear regression on p_old and day to see if these is statistically significant change in conversion rate

# In[ ]:


df2_by_day['intercept']=1
lm=sm.OLS(df2_by_day['old_rate'], df2_by_day[['intercept', 'day']])
results=lm.fit()
results.summary()


# ##### coef=0.0001, p value=0.427, R-square=0.030. All these indicate that there is no statistically significant correlation between p_old and day. Therefore the conversion rate of old page is not changing with running time

# #### run linear regression on p_new and day to see if these is statistically significant change in conversion rate

# In[ ]:


lm=sm.OLS(df2_by_day['new_rate'], df2_by_day[['intercept', 'day']])
results=lm.fit()
results.summary()


# ##### coef=0.0001, p value=0.439, R-square=0.029. All these indicate that there is no statistically significant correlation between p_new and day. Therefore the conversion rate of new page is not changing with running time.

# #### run a linear regression to check if p_new-p_old changes over time

# In[ ]:


lm=sm.OLS(df2_by_day['new_rate']-df2_by_day['old_rate'], df2_by_day[['intercept', 'day']])
results=lm.fit()
results.summary()


# ##### R-square =0, coef is 0, p value is close to 1. There is definitely not a statistically significant correlation between p_new-p_old and day.

# <a id='conclusions'></a>
# ## Conclusions
# 
# I analyzed user conversion rate vs. their landing_pages. Both A/B testing and logistic regression were employed to answer the same question.
# 
# (1) In A/B testing, I performed a one-sided hypothesis testing: whether new page leads to more conversion rate. The null is new page is not better than old page. The alternative is new page is better. The resulted p value from my test is 0.9, therefore I failed to reject null, and conclude that **new page is not better in leading to conversion**.
# 
# (2) In the logistic regression method, I intend to prove that conversion rate depends on landing page, therefore it is a two-sided hypothesis test. The null is there is no difference between the conversion rate of old and new page. The alternative is there is a difference. The resulted p value from my test is close to 0.2, therefore I failed to reject null, and conclude that there is no statistically significant difference in conversion rate between old and new landing_page. 
# 
# (3) Beside landing_page, I also tested whether country matters in conversion rate. I didn't find any statistically significant evidence that country matters. In other words, **country does not matter**.
# 
# (4) I also tried to find out whether day of week matters. When considering all 7 days of weeks, I didn't find statistically significant evidence to support that a specific day of week has better conversion rate. But when comparing Monday and Friday, I did find that **Monday conversion rate is statistically better than Friday**. 
# 
# (5) I then tried to find out whether running the test longer time improves conversion rate. Neither old_page nor new_page conversion rate improved with time. From the linear regression of p_new-p_old against day, the R-square=0, and p value is close to 1. Therefore is not any indication that running the test longer will improve test results, and I **DO NOT** suggest running longer time. 

# In[ ]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Analyze_ab_test_results_notebook.ipynb'])

