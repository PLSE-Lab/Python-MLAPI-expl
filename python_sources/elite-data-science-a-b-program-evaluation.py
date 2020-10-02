#!/usr/bin/env python
# coding: utf-8

# # **Elite Data Science - A/B Program Evaluation - V2.0 (03/25/2019)**
# 
# ##### John Ostrowski - Digital Marketing Data Scientist
# 
# **Introduction**
# 
# In this next challenge, you'll need to evaluate two different content delivery programs for a foreign language platform.
# 
# A simple way to think about program evaluation problems is to approach them as A/B tests. You can treat one as the control and the other as the test program.
# 
# 
# **Objectives**:
# 
# Should the language platform roll out a drip program or a binge program?
#     1. Which program has higher user engagement?
#     2. If you take different segments of the data, does the relationship still hold?
#     3. Provide actionable insights to the business.
# 
# **This notebook is structured as follows:**
# 
#     1. Data overview
#     2. Question 1
#     3. Question 2
#     4. Conclusions and business suggestions (Question 3)
#     
# **Policy and Ethics for Experiments**
# 
# Here, I'm following the Google A/B Test guidelines [1] for practicing business standards:
# 
# - Risk: The program change test does not expose any participant to risks above the "minimal risk";
# - Benefits: Possible engagement improvement leading to more informed students;
# - Alternatives: Full range of alternatives is kept during the test, opting between both programs or quitting the platform;
# - Data Sensitivity: There is no sensitive data (financial or health). All data is considered anonymized, hard to or impossible for re-identification.

# # 1. Data overview

# In[1]:


# Loading libraries for python
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')


import numpy as np # Linear algebra
import pandas as pd # Data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Data visualization
import seaborn as sns # Advanced data visualization
import re # Regular expressions for advanced string selection
import missingno as msno # Advanced missing values handling
import pandas_profiling # Advanced exploratory data analysis
from scipy import stats # Advanced statistical library
sns.set()
from pylab import rcParams
from scipy import stats # statistical operations
import warnings
warnings.filterwarnings("ignore");

# changing graph style following best practices from Storyteling with Data from Cole Knaflic [3]
sns.set(font_scale=2.5);
sns.set_style("whitegrid");


# In[2]:


# Reading data parsing dates in correct format for users DataFrame
engagement = pd.read_csv(r"../input/engagement.csv")
users = pd.read_csv(r"../input/users.csv", parse_dates=['registration'], infer_datetime_format=True)


# ## Exploratory data analysis
# - The objective of this step is getting to know the working data, what to expect and generate some ideas and hipothesys for solving the purposed questions.
# 
# > *Pro tip: If you haven't worked with pandas_profilling library yet, I strongly suggest you to fork this notebook and try out the commented codes below*
# 
# > *Pro tip: Another library that I really like using it for quick EDA is missingno for checking for missing value sin an intuitive an visual way. If you haven't tried it fork the notebook and play with it!*

# ## EDA preliminar conclusions:
# After running EDA we conclude that:
# - There is no missing values on both data sets, therefore no DLM (Data Loss Management) will be done;
# - Warning for majority of 0 or close to 0 values on hrs_per-week feature;
# - The warning is further understood as 'window shoppers', registered users that didn't spend more than 1 hour in the platform;
# - Data is balanced regarding program categories (binge/drip) with a 50.1 to 49.9% proportion;
# - Most users are in Germany (52%) while least are in Italy (8%);
# - Most used Browser is Chrome (43%) and Opera (2%) the least;
# - IE (22%) and Firefox (22%) are respectively second and third;
# - The sample mean hrs_per_week is 4.6 hours with a sample standard deviation of 2.53 hours;
# - The sample mean and standard deviation segmented by program and by country are very constant, suggesting a very balanced dataset;
# - Most users started the accounts on 2016;
# - Data is overal highly balanced in all tested segments;

# In[3]:


# Initially analyzed each dataset using pandas_profiling
# This is a very insightful library for initial EDA, I left the code as comments due to the large space it takes on screen
# Fork the notebook and run the codes below to try out how powerful pandas_profiling is!

#pandas_profiling.ProfileReport(users)
#pandas_profiling.ProfileReport(engagement[['hrs_per_week', 'browser', 'program']])

# Getting summary statistics for early conclusions excluding user_id
engagement.describe(include="all").drop(['user_id'], axis=1)


# In[7]:


# Let's have a macro view of the hours per week by program
fig1 = sns.boxplot(x='program', y='hrs_per_week', data=engagement);

# All visualizations will follow a similar styling following best practices from Storytelling with Data [3][8][9]
fig1 = plt.gcf()
fig1.set_size_inches(10, 8)
plt.title('Hours per week by program', loc='left', y=1.1);
plt.ylabel("Hours per week [hrs]");
plt.xlabel("Program type");
plt.text(-0.5, 18.5, "Note that median values are similar, but the Binge program has larger variance.\nAlso, Binge has more 'extreme users' denoted by the observed outliers and max value.", fontsize=16, color='grey', style='italic', weight='semibold',);
plt.plot();


# In[8]:


# Checking mean hours per week per program
engagement.groupby(['program']).mean()[['hrs_per_week']]


# In[9]:


# Joining the tables on User_ID for hrs_per_week by country and checking by possible nulls
# Merging on user_is and adding a simple year column for aggregations
df = pd.merge(engagement, users, on='user_id')
df['year'] = df.registration.dt.year

# Checking for null values and exploring further using missingno library
#df.isnull().sum()
#msno.bar(df);


# In[12]:


# Checking engagement distribution for the entire sample (both programs)
fig, ax = plt.subplots()
fig = sns.distplot(df.hrs_per_week, kde=False)
fig.grid(False)
fig = plt.gcf()
fig.set_size_inches(14, 8)
plt.title("Distribution of hours per week for the entire sample (both programs)", loc="left", y=1.1, fontsize=20);
plt.text(-0.8, 5350, "Note that the proportion of users that used the platform between 0 and 1 hour is very significant", fontsize=16, color='grey', style='italic', weight='semibold',);
plt.ylabel("User count");
plt.xlabel("Hours per week [h]");
sns.despine(left=False);
ax.annotate('window shoppers', xy=(0.2, 2700), xytext=(0.6, 3700), fontsize=11,
            arrowprops=dict(facecolor='black', shrink=0.05));
plt.plot();


# In[13]:


# User distributions per country
# Not much of a difference on the aggregate view;
# On the granular view (by program by country), we have once again the conclusion of extreme users in favor to the Binge program;
# However, still on the granular view, all medians are higher for the Drip program while standard deviations are lower, indicating a more regular community.
df.groupby(['country', 'program']).agg({'hrs_per_week':[np.mean, np.std]})


# In[15]:


# Let's check the program distribution for the given period
sns.countplot(x='year', hue='program', data=df);
fig2 = plt.gcf()
fig2.set_size_inches(10, 8)
plt.title('Registrations per year for each program', loc="left", y=1.1);
plt.text(-0.5, 37000, 'Most of the experiment happend during 2016, and data is balanced for the given period', fontsize=16, color='grey', style='italic', weight='semibold',);
plt.legend(loc=0);
plt.ylabel("Count of registrations");
plt.xlabel("Year");
plt.plot();


# In[16]:


# Checking the distribution of registered users throughout the period of analysis
pivot_table = pd.pivot_table(df, index=['country', 'program'], columns='year', aggfunc=np.count_nonzero, values=['registration'], margins=True, )
pivot_table

# Distribution of registered users by country throughout the period of analysis
#sns.catplot('country', col="year", col_wrap=2, hue='program', data=df, kind="count", height=11, aspect=.6);


# # 2. Question 1
# 
# Since we're going to treat this as a controlled experiment where we will initially test the difference between sample means we need to work on a couple theoretical points in orther to make sure our calculations are valid. Most of the statistical knowledge necessary here I learned through online learning [4] [5], and books [6]. 
# 
# ## Hypothesis statement
# - Ho (Null Hypothesis): There is no difference between the average hour per week between programs (ua = ub);
# - Ha (Alternative Hypothesis): The average hour per week is different between programs;
# 
# ## Validity of hypothesis test
# - Normal: Here we can assume that the underlying distribution where the samples came from is normally distributed, but also since our sample size is higher than 30 we fulfill the normal condition for the test;
# - Random: Users were randomly selected to both groups, so we can consider that the observations are random and do not have any source of bias regarding the true population;
# - Indepency: Here we assume the 10% rule for independency, meaning that both samples represent less than 10% of total users that went through the program, therefore we can assume that observations are independent;
# 
# ## Errors and alfa
# - Type I error is the most expensive for this project. A type I error is a false positive, which implies in a mistaken acceptance of the Alternative Hypothesis (Ha). In other words, Type I error for this case would suggest that the company should invest into further development and release the new program even though the program did not generate more engagement than the current program.
# - To minimize Type I error, we will decrease our alfa (alfa = 1 - Confidence level), so it is harder for us to reject the null hypothesis in a first place. Therefore significance level **alfa = 0.01** meaning that we will only reject H0 for events where the probability of occurence is lower than 1 in a hundred.
# 
# ## Early conclusions
# - The fact that confidence intervals at a 95% level overlap in such big proportion (for both the mean hours per week per program and for the difference between means) indicates is an indication that there is no statistically significat difference between programs, the observed difference could be due to randomness as indicated by the intervals;
# - The result of our statistical test indicates a p-value of 0.64, implying that we would see such difference in 64% of the cases where similar sample process is taken. Since p-value > alfa (0.64 > 0.1) we fail to reject the null hypothesis (H0), therefore we don't have enough statistical evidence to suggest Ha;
# - Translating to the business language, there is not enough evidence suggesting to implement the new program due to the fact that the difference in engagement is not statistically significant after the experiment.

# In[18]:


# As a best practice prior to hypothesis testing we evaluate eventual Confidence Interval overlaps at 95%
# So here we're visualizing Z confidence intervals for the sample means by program

fig = sns.pointplot(x="hrs_per_week", y="program", data=engagement, join=False, capsize=0.1)
fig = plt.gcf()
fig.set_size_inches(10, 6)
plt.title("95% Confidence interval for the sample means", loc="left", fontsize=20, y=1.15);
plt.text(4.57307, -0.6, 'The overlap seen between CI suggests that there is no real difference \non the true mean difference between populations ', fontsize=16, color='grey', style='italic', weight='semibold',);
plt.ylabel("");
plt.xlabel("Sample mean");
sns.despine(left=False)
plt.plot();


# In[21]:


# Storing necessary statistics for double checking the statistical test using [7]

# Hours per week mean difference between program
mean_difference = engagement.loc[engagement.program == "binge", "hrs_per_week"].mean() - engagement.loc[engagement.program == "drip", "hrs_per_week"].mean()

# Mean and standard deviation from hour per week per program
mean_hpw_binge = df[df.program == 'binge'].hrs_per_week.mean()
std_hpw_binge = df[df.program == 'binge'].hrs_per_week.std()
n_binge = df[df.program == 'binge'].shape[0]

mean_hpw_drip = df[df.program == 'drip'].hrs_per_week.mean()
std_hpw_drip = df[df.program == 'drip'].hrs_per_week.std()
n_drip = df[df.program == 'drip'].shape[0]


# In[32]:


# Building the confidence interval for the difference of the sample means 
SE = np.sqrt((np.power(std_hpw_binge, 2)/n_binge) + (np.power(std_hpw_drip, 2)/n_drip))

figure(figsize=(7, 6))
plot(mean_difference, "bo", markersize=9)
plot(mean_difference - 1.96*SE, "_", markersize=15, color='b')
plot(mean_difference + 1.96*SE, "_", markersize=15, color='b')
plt.axvline(x=0, ymin=0.045, ymax=0.95);
plt.title("95% Confidence interval for the sample means", loc="left", fontsize=20, y=1.18);
plt.text(-0.055, 0.046, 'The fact that our CI for the mean difference goes \nbeyond zero reinforces the no statistical significant difference \nbetween hour per week for the different programs', fontsize=16, color='grey', style='italic', weight='semibold',);
plt.ylabel("");
plt.plot();


# In[33]:


# T-test using statsmodel for getting our p-value 
# (probability of observing a difference in hours per week as at least or more extreme 
# than the one observed given that the null hypothesis is true)
pval = stats.ttest_ind(df[df.program == 'binge'].hrs_per_week, df[df.program == 'drip'].hrs_per_week, equal_var = False)
print("P-value = {}".format(pval[1].round(4)))


# # 3. Question 2
# 
# Even thoug it is not considered a best practice to keep segmenting the data in order to find a significan p-value to reject H0 [1] I will attempt different test as following:
# 
# - Testing the difference in means by countries;
# - Testing removing window shoppers and extreme outliers;
# 
# Since the data handling is very similar to what was done for Question 1, I will keep it simple and hide the code to focus on the conclusion discussion.
# 
# ## All considerations for normal vaility and error stay constant and similar to Question 1
# 
# ## Early conclusions
# - We noticed that there is a significant amount of users that registered on the platform and did not spend more than 1 hour on the program. We're calling these people 'window shoppers'. This goes along with the warning received at the beggining of ETL from pandas_profiling; 
# - From a marketing perspective we're assuming that they are not the most interesting segment to evaluate the efficienci of the program. For the business, these users can be a new segment for a better remarketing to buy them back since they're easier to become engaged compared to new leads;
# - If we exclude the window shopper segment from our data we start seeing a statistical significat difference on engagement (hours per week spent) for the given period;
# - The statistical significance persists for both Germany and France, failing for Italy due to the higher variability coming from the smaller sample size for this particular country;
# - Outliers were detected following the 3 standard deviations rule, after exclusion and still eliminating the window shoppers, both Germany and France hold a statistical significant mean difference suggesting that the means are different.

# In[34]:


# Histograms for both program types so we can see how many "window shoppers" each one had before eliminating them

bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
fig, ax = plt.subplots(1, 2)
df.loc[(df.program == 'binge') & (df.country == 'Germany'), ['hrs_per_week']].hist(bins=bins, ax=ax[0]);
df.loc[(df.program == 'drip') & (df.country == 'Germany'), ['hrs_per_week']].hist(bins=bins, ax=ax[1]);
ax[0].set_ylim([0, 4200])
ax[0].set_title("Binge distribution")
ax[0].grid(False)
ax[1].set_ylim([0, 4200])
ax[1].set_title("Drip distribution")
ax[1].grid(False)
for i in ax.flat:
    i.set(xlabel='Hours per week [h]', ylabel='Count of users')
fig.suptitle("Distribution of hours per week by program for Germany", fontsize=28, x=0.465, y=1.03);
plt.text(-22, 4550, "Note that the majority of 'window shoppers' were on the Binge program, \nsuch behavior is seens proportionally for France and Italy as well.", fontsize=16, color='grey', style='italic', weight='semibold',);
fig.set_size_inches(16, 10)
plt.plot();


# In[35]:


# Checking summary statistics between programs after removing window shoppers
df_eval0 = df.loc[df.hrs_per_week > 1].groupby(['program']).agg({'hrs_per_week':[np.mean, np.std], 'registration':np.count_nonzero})
df_eval0.columns.set_levels(['sample size (n)','mean','std.dev'],level=1,inplace=True)
df_eval0


# In[36]:


# Checking summary statistics between programs for every country after removing window shoppers
df_eval1 = df.loc[df.hrs_per_week > 1].groupby(['country', 'program']).agg({'hrs_per_week':[np.mean, np.std], 'registration':np.count_nonzero})
df_eval1.columns.set_levels(['sample size (n)','mean','std.dev'],level=1,inplace=True)
df_eval1


# In[37]:


# Testing the difference between means from the entire sample
pval = stats.ttest_ind(df.loc[(df.program == 'binge') & (df.hrs_per_week > 1)].hrs_per_week, df.loc[(df.program == 'drip')  & (df.hrs_per_week > 1)].hrs_per_week, equal_var = False)
print("P-value for the difference between means after removing window shoppers = {}".format(pval[1].round(6)))


# In[38]:


# T-test using statsmodel for getting our p-value - Germany
pval_ger = stats.ttest_ind(df[(df.program == "binge") & (df.country == "Germany") & (df.hrs_per_week >= 1)].hrs_per_week, df[(df.program == "drip") & (df.country == "Germany") & (df.hrs_per_week >= 1)].hrs_per_week, equal_var = False)
pval_fra = stats.ttest_ind(df[(df.program == "binge") & (df.country == "France") & (df.hrs_per_week >= 1)].hrs_per_week, df[(df.program == "drip") & (df.country == "France") & (df.hrs_per_week >= 1)].hrs_per_week, equal_var = False)
pval_it = stats.ttest_ind(df[(df.program == "binge") & (df.country == "Italy") & (df.hrs_per_week >= 1)].hrs_per_week, df[(df.program == "drip") & (df.country == "Italy") & (df.hrs_per_week >= 1)].hrs_per_week, equal_var = False)

print("P-values assesment by country after removing window shoppers:\nP-value for Germany = {} \nP-value for France = {} \nP-value for Italy = {}".format(pval_ger[1].round(6), pval_fra[1].round(6), pval_it[1].round(6)))


# In[39]:


# Removing outliers and evaluating the new sequence of statistical tests
df_nooutliers = df[pd.Series(np.abs(stats.zscore(df.hrs_per_week)) < 3)]
df_eval2 = df_nooutliers.loc[df_nooutliers.hrs_per_week > 1].groupby(['country', 'program']).agg({'hrs_per_week':[np.mean, np.std], 'registration':np.count_nonzero})
df_eval2.columns.set_levels(['sample size (n)','mean','std.dev'],level=1,inplace=True)
df_eval2


# In[40]:


# Testing the difference between means from the entire sample
pval = stats.ttest_ind(df_nooutliers.loc[(df_nooutliers.program == 'binge') & (df_nooutliers.hrs_per_week > 1)].hrs_per_week, df_nooutliers.loc[(df_nooutliers.program == 'drip')  & (df_nooutliers.hrs_per_week > 1)].hrs_per_week, equal_var = False)
print("P-value for the difference between means after removing window shoppers and outliers = {}".format(pval[1].round(6)))

# Checking summary statistics between programs after removing window shoppers and outliers
#df_eval3 = df_nooutliers.loc[df_nooutliers.hrs_per_week > 1].groupby(['program']).agg({'hrs_per_week':[np.mean, np.std], 'registration':np.count_nonzero})
#df_eval3.columns.set_levels(['sample size (n)','mean','std.dev'],level=1,inplace=True)
#df_eval3


# In[41]:


# T-test using statsmodel for getting our p-value - Germany
pval_ger = stats.ttest_ind(df_nooutliers[(df_nooutliers.program == "binge") & (df_nooutliers.country == "Germany") & (df_nooutliers.hrs_per_week >= 1)].hrs_per_week, df_nooutliers[(df_nooutliers.program == "drip") & (df_nooutliers.country == "Germany") & (df_nooutliers.hrs_per_week >= 1)].hrs_per_week, equal_var = False)
pval_fra = stats.ttest_ind(df_nooutliers[(df_nooutliers.program == "binge") & (df_nooutliers.country == "France") & (df_nooutliers.hrs_per_week >= 1)].hrs_per_week, df_nooutliers[(df_nooutliers.program == "drip") & (df_nooutliers.country == "France") & (df_nooutliers.hrs_per_week >= 1)].hrs_per_week, equal_var = False)
pval_it = stats.ttest_ind(df_nooutliers[(df_nooutliers.program == "binge") & (df_nooutliers.country == "Italy") & (df_nooutliers.hrs_per_week >= 1)].hrs_per_week, df_nooutliers[(df_nooutliers.program == "drip") & (df_nooutliers.country == "Italy") & (df_nooutliers.hrs_per_week >= 1)].hrs_per_week, equal_var = False)

print("P-values assesment by country after removing window shoppers and outliers:\nP-value for Germany = {} \nP-value for France = {} \nP-value for Italy = {}".format(pval_ger[1].round(6), pval_fra[1].round(6), pval_it[1].round(6)))


# # Conclusions and business suggestions
# 
# ## Size of difference and statistical significance
# Even after noticing the statistical significance at a 0.01 significance level **I would not suggest this company to further develop the drip program and implement it.**
# 
# **The main reason why lies on the size of the difference, in average the drip program performed 0.1 hour better, this is only 6 minutes which translates in no substancial additional revenue. Therefore significance does not mean importance here.**
# 
# This is a great example of understanding how the huge sample size (n) plays an important role in reducing the variability and improving the power of the analysis which is the probability of not commiting a Type II error (False Negative). Having a large sample size (n) we are able to detect small differences between data sets, however after rejecting the null hypothesis the size of the difference must always be evaluated in order to take business decisions.
# 
# Thank you for reading and analysing this case with me, I hope you could take insights along the way and please provide me feedback so the entire community learns and improves together.
# 
# John Ostrowski.

# ## References
# 
#     [1] https://eu.udacity.com/course/ab-testing--ud257
#     [2] https://hookedondata.org/guidelines-for-ab-testing/ 
#     [3] http://www.storytellingwithdata.com/book
#     [4] https://stattrek.com/tutorials/ap-statistics-tutorial.aspx
#     [5] https://www.khanacademy.org/math/ap-statistics
#     [6] https://www.amazon.com/Naked-Statistics-Stripping-Dread-Data-ebook/dp/B007Q6XLF2
#     [7] http://www.evanmiller.org/ab-testing/t-test.html
#     [8] https://matplotlib.org/api/_as_gen/matplotlib.pyplot.text.html
#     [9] https://medium.com/swlh/storytelling-with-data-part-2-6f4ec8a13585
