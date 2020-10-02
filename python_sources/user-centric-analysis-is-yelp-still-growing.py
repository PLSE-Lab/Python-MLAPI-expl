#!/usr/bin/env python
# coding: utf-8

# # Table of Contents
# 1. [Background](#bg)
# 
#     1.1 [What is this analysis about?](#bg-what)
#     
#     1.2 [Why do we need this analysis?](#bg-why)
#     
#     1.3 [How will we conduct this analysis?](#bg-how)
#     
# 2. [Preparation](#prep)
#     
#     2.1 [Basic Import](#prep-import)
#     
#     2.2 [Function Declaration](#prep-func)
#     
#     2.3 [Data Loading](#prep-load)
#     
# 3. [User Growth Analysis](#growth)
# 
#     3.1 [Basic Information](#growth-info)
#     
#     3.2 [Integrity Check](#growth-integrity)
#     
#     3.3 [Number of New User](#growth-number)
#     
#     3.4 [User Growth](#growth-yoy)
# 
# 4. [User Engagement Analysis](#e8t)
# 
#     4.1 [Basic Information](#e8t-info)
#     
#     4.2 [Number of New Review](#e8t-number)
#     
#     4.3 [Review Growth](#e8t-yoy)
#     
#     4.4 [Monthly Number of Review per User](#e8t-monthly)
#     
# 5. [Conclusion](#conclusion)
# 
# 6. [Future Works](#future)

# <a id="bg"></a>
# # 1. Background
# 
# <a id="bg-what"></a>
# ## 1.1 What is this analysis about?
# We want to see 2 important metrics of a social media: user growth & engagement of yelp. Yelp is social media where user can provide rating, review, and tips for nearby business establishments.
# 
# <a id="bg-why"></a>
# ## 1.2 Why do we need this analysis?
# Yelp has been around since has been around since 2004. It is currently valued at $3.4B, and with such massive size, we can determine better strategy for yelp in the future if we understand how user interact with yelp from time to time. This user behavior analysis can help yelp to determine what to focus on moving forward.
# 
# <a id="bg-how"></a>
# ## 1.3 How will we conduct this analysis?
# We will begin with the simple one, by finding out year-on-year user growth. For the engagement part, we start by defining engagement as review, and then find out the number of review per user. That is, to answer if user become more engaged by calculating average number of review per user.
# 
# **if you find anything suspicious or have any question, feel free to drop a comment :)**

# <img src="https://media.giphy.com/media/3o85xt08p2Y0hanhwQ/giphy.gif" alt="warning: yelp critics" /> <br />
# Here we go! *(source: giphy.com)*

# <a id="prep"></a>
# # 2. Preparation
# 
# <a id="prep-import"></a>
# ## 2.1 Basic Imports

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# options
pd.set_option('display.max_colwidth', -1)

# extra config to have better visualization
sns.set(
    style='whitegrid',
    palette='coolwarm',
    rc={'grid.color' : '.96'}
)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 12


# <a id="prep-func"></a>
# ## 2.2 Function Declaration

# In[163]:


# function to return growth
# later will be used to calculate year-on-year growth
def growth(df, date_col='date', count_col='count', resample_period='AS'):
    df_ri = df.set_index(date_col)
    df_resample = df_ri[count_col].resample(resample_period).sum().to_frame().reset_index()

    df_resample['prev'] = df_resample[date_col] - pd.DateOffset(years=1)

    growth_df = df_resample.merge(
        df_resample,
        how='inner',
        left_on=[date_col],
        right_on=['prev'],
        suffixes=['_l', '_r']
    ).loc[:,['{}_r'.format(date_col), '{}_l'.format(count_col), '{}_r'.format(count_col)]].rename(
        columns={
            '{}_r'.format(date_col) : 'date',
            '{}_l'.format(count_col) : 'count_last_year',
            '{}_r'.format(count_col) : 'count_current'
        }
    )
    
    growth_df['growth'] = growth_df['count_current'] / growth_df['count_last_year']
    return growth_df


# <a id="prep-load"></a>
# ## 2.3 Data Loading

# In[164]:


# load data
usr_df = pd.read_csv("../input/yelp_user.csv")
rvw_df = pd.read_csv("../input/yelp_review.csv")

# data type conversion
usr_df.yelping_since = pd.to_datetime(usr_df.yelping_since)
rvw_df.date = pd.to_datetime(rvw_df.date)

# check what is inside dataframe
print('This is the user dataframe columns:')
print(usr_df.dtypes)
print()
print('This is the review dataframe columns:')
print(rvw_df.dtypes)


# <a id="growth"></a>
# # 3. User Growth Analysis
# 
# <a id="growth-info"></a>
# ## 3.1 Basic Information

# In[165]:


# How many user in the dataset?
'There are {:,} users'.format(len(usr_df))


# <a id="growth-integrity"></a>
# ## 3.2 Integrity Check

# In[166]:


# if there are users with same user_id
count_user_id = usr_df[['user_id','name']].groupby(['user_id']).count().rename(columns={'name' : 'count'})
assert (len(count_user_id[count_user_id['count'] > 1]) == 0), "Multiple user with same user_id"


# <a id="prep-number"></a>
# ## 3.3 Number of New User

# In[182]:


count_yelping_since = usr_df[['yelping_since', 'user_id']].groupby(['yelping_since']).count().rename(columns={'user_id':'count'}).resample('1d').sum().fillna(0)
count_yelping_since['rolling_mean_30d'] = count_yelping_since.rolling(window=30, min_periods=1).mean()
count_yelping_since = count_yelping_since.reset_index()

fig, ax = plt.subplots(figsize=(12,7.5))

_ = count_yelping_since.plot(
    ax=ax,
    x='yelping_since', 
    y='rolling_mean_30d'
)

_ = count_yelping_since.plot(
    ax=ax,
    x='yelping_since', 
    y='count',
    alpha=.3
)

_ = ax.set_title('Daily Number of New User')
_ = ax.legend(['Rolling Mean 30 Days', 'Daily Count'])
_ = ax.set_yticklabels(['{:,.0f}'.format(x) for x in ax.get_yticks()])


# **User growth reach its peak around 2014, plummeted around 2016,  and been constantly dropping until 2017. **
# 
# **Trivia:** We can observe a unique yearly 2 days drop around every year-end. We can zoom in and see that it is around the *Black Friday* and *Christmas Day* of each year. But we will not dig further into that, since it is not really relevant to what we are trying to our analysis goal.

# <a id="growth-yoy"></a>
# ## 3.4 User Growth

# In[193]:


yoy_user_df = growth(count_yelping_since, date_col='yelping_since')
ax1 = yoy_user_df.plot(
    x='date', 
    y='growth', 
    title='Overall Year-on-Year User Growth', 
    figsize=(7.5,5), 
    legend=False,
    linewidth=3
)
_ = ax1.set_yticklabels(['{:,.0f}%'.format(x*100) for x in ax1.get_yticks()])


# Due to extreme growth in the beginning, it is hard to see the effect in more recent years. Elbow is around 2008, let's zoom in after that to analyze further.

# In[194]:


ax2 = yoy_user_df[yoy_user_df['date'] >= '2008-01-01'].plot(
    x='date', 
    y='growth', 
    title='Year-on-Year User Growth After 2008', 
    figsize=(12,7.5),
    legend=False,
    linewidth=3
)
_ = ax2.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax2.get_yticks()])


# As we can see, the growth is **slowing down to about 10% YoY in 2017**. Is 10% Year-on-year growth good or not? We cannot expect to get conclusion only from yelp data, we need to compare with other similar social media platform growth and also yelp's target.
# 
# For many social media, user growth declining is expected due to market saturation
# Usually, after user acquisition is slowing down, other than opening on a new market or hyperlocalization, companies will put more efforts to retain existing customer.
# 
# We will check whether user engagement in yelp is increasing.
# For this analysis, we limit the definition of engagement in yelp as giving review.
# We will explore:
# 1. overall number of review growth
# 2. per user average number of review

# <a id="e8t"></a>
# # 4. User Engagement Analysis
# 
# <a id="e8t-info"></a>
# ## 4.1 Basic Information

# In[170]:


# How many user in the dataset?
'There are {:,} reviews'.format(len(rvw_df))


# <a id="e8t-number"></a>
# ## 4.2 Number of New Review

# In[171]:


#count_review = rvw_df[['date', 'review_id']].groupby(['date']).count().rename(columns={'review_id':'count'})
count_review = rvw_df[['date', 'review_id']].groupby(['date']).count().rename(columns={'review_id':'count'}).resample('1d').sum().fillna(0)
count_review['rolling_mean_30d'] = count_review.rolling(window=30, min_periods=1).mean()
count_review = count_review.reset_index()

fig, ax = plt.subplots(figsize=(12,7.5))

_ = count_review.plot(
    ax=ax,
    x='date', 
    y='rolling_mean_30d',
    style='--'
)

_ = count_review.plot(
    ax=ax,
    x='date', 
    y='count',
    alpha=.3
)

_ = ax.set_title('Daily Number of New Review')
_ = ax.legend(['Rolling Mean 30 Days', 'Daily Count'])
_ = ax.set_yticklabels(['{:,.0f}'.format(x) for x in ax.get_yticks()])


# **Number of review is still growing rapidly**, and we can also observe yearly seasonality from the *Rolling Mean 30 Days*. It is also interesting to dig deeper on the yearly peak, when and why do we have such seasonality.

# <a id="e8t-yoy"></a>
# ## 4.3 Review Growth

# To compare with user growth, let's zoom in after 2008.

# In[183]:


yoy_review = growth(count_review)

fig, ax = plt.subplots(figsize=(12,7.5))

_ = yoy_review[yoy_review['date'] >= '2008-01-01'].plot(
    ax=ax,
    x='date', 
    y='growth',
    linewidth=3
)

_ = yoy_user_df[yoy_user_df['date'] >= '2008-01-01'].plot(
    ax=ax,
    x='date', 
    y='growth',
    style='--',
    linewidth=3
)

_ = ax.set_title('Yearly User and Review Growth Comparison')
_ = ax.legend(['#Review Growth', '#User Growth'])
_ = ax.set_yticklabels(['{:,.0f}%'.format(x*100) for x in ax.get_yticks()])


# As the gap between review growth and user growth widens, we get a sense that **there has been more user giving reviews**. This is a good sign, but we need to analyze further to find out whether more users or just the same user who keep writing reviews.

# <a id="e8t-monthly"></a>
# ## 4.4 Monthly Number of Review per User

# For monthly number of review, we use the *review_count* field from user dataframe. We will create a new field *review_per_month*, which is the number of review divided by number of month the user has been registered.

# In[192]:


# count monthly review per user with review_count and yelping_since
usr_rvw = usr_df[['user_id', 'review_count', 'yelping_since']].copy(deep=True)

# get latest date in dataset,
# safe check, if latest yelping since > latest review date, then take latest yelping since instead
# add buffer 1 month so there is no 0 'month since'
latest_date = max(rvw_df['date'].max(), usr_rvw['yelping_since'].max()) + np.timedelta64(1, 'M') 
usr_rvw['month_since'] = (latest_date - usr_rvw['yelping_since']) / np.timedelta64(1, 'M')
usr_rvw['review_per_month'] = usr_rvw['review_count'] / usr_rvw['month_since']
ax = usr_rvw['review_per_month'].plot.hist(figsize=(7.5,5))
_ = ax.set_yticklabels(['{:,.2f}m'.format(x/1000000) for x in ax.get_yticks()])
_ = ax.set_xlabel('monthly number of review')
_ = ax.set_title('Number of Review per User Distribution')


# There is no insight from the histogram due to outliers,
# we will find out what is the 99th percentile and use the number as threshold to remove the outlier.

# In[174]:


usr_rvw_q = usr_rvw.review_per_month.quantile([.9, .95, .99, 1])
print(usr_rvw_q)


# In[190]:


# We cut at 99% to remove outliers
usr_rvw_rem_outliers = usr_rvw[usr_rvw['review_per_month'] <= usr_rvw_q.loc[0.99]]['review_per_month']
weights = np.ones_like(usr_rvw_rem_outliers) / float(len(usr_rvw_rem_outliers))

ax = usr_rvw_rem_outliers.plot.hist(bins=int(usr_rvw_q.loc[0.99] * 8), weights=weights, figsize=(12,9))
_ = ax.set_xlabel('monthly number of review')
_ = ax.set_yticklabels(['{:.0f}%'.format(x*100) for x in ax.get_yticks()])
_ = ax.set_title('Number of Review per User Distribution')


# In[176]:


'{:.5f}% of users never review'.format(len(usr_rvw[usr_rvw['review_per_month'] == 0]) * 100 / len(usr_rvw[usr_rvw['review_per_month'] < 4]))


# # 4.5 Monthly Number of Review per User

# Now we want to find out if users give more reviews. For this analysis, we need to use the review dataset to find the date of review by each user.

# In[177]:


rvw_eng = rvw_df[['review_id', 'user_id', 'date']].copy(deep=True)
rvw_eng['year'] = rvw_eng['date'].map(lambda x: x.year)

# prepare user dataframe
# to accomodate user who never review
usr_rvw['year_join'] = usr_rvw['yelping_since'].map(lambda x: x.year)


# In[178]:


# find out for each year, what is the distribution of monthly number of review per user

years = rvw_eng['year'].unique()
yearly_rvw_df = pd.DataFrame(columns=['user_id', 'year', 'rvw_p_month'])

for year in years:
    # get all the users that exist this year
    usr_prev_year = usr_rvw[usr_rvw['year_join'] < year][['user_id', 'yelping_since']]
    usr_prev_year['yearly_month_since'] = 12.0 # this means user has joined for the full year
    
    usr_curr_year = usr_rvw[usr_rvw['year_join'] == year][['user_id', 'yelping_since']]
    usr_curr_year['yearly_month_since'] = (pd.Timestamp('{}-01-01'.format(year+1)) - usr_curr_year['yelping_since']) / np.timedelta64(1, 'M')
    
    usr_curr_year_all = usr_curr_year.append(usr_prev_year)
    
    # now get all review done in current year and count by user
    rvw_curr_year = rvw_eng[rvw_eng['year'] == year][['user_id', 'review_id']].groupby(['user_id']).count().rename(columns={'review_id':'count'}).reset_index()
    
    usr_curr_year_all = usr_curr_year_all.merge(rvw_curr_year, on='user_id', how='left')
    usr_curr_year_all['count'].fillna(0.0, inplace=True)
    usr_curr_year_all['rvw_p_month'] = usr_curr_year_all['count'] / usr_curr_year_all['yearly_month_since']
    usr_curr_year_all['year'] = year
    
    yearly_rvw_df = yearly_rvw_df.append(usr_curr_year_all[['user_id', 'year', 'rvw_p_month']])

yearly_rvw_df['year'] = pd.to_numeric(yearly_rvw_df['year'])
         


# Before looking at the user who has reviewed, we want to find out whether among all user whose registered for each year, the proportion of user who has never review decreased or not. If the proportion declining, that means more user are writing review.

# In[221]:


no_review_proportion = (yearly_rvw_df[yearly_rvw_df['rvw_p_month'] == 0].groupby(['year']).count().rvw_p_month / yearly_rvw_df.groupby(['year']).count().rvw_p_month).to_frame().reset_index()

yoy_review['year'] = yoy_review.apply(lambda x: x['date'].year, axis=1)

fig, ax = plt.subplots(figsize=(12,7.5))

_ = no_review_proportion[no_review_proportion['year'] >= 2008].plot(
    ax=ax,
    x='year',
    y='rvw_p_month',
    linewidth=2
)

_ = yoy_review[yoy_review['year'] >= 2008].plot(
    ax=ax,
    x='year', 
    y='growth',
    linewidth=2,
    style='--'
)

_ = ax.set_title('Correlation Between User Who Writes No Review to User Growth')
_ = ax.legend(['Proportion of User Who Writes No Review', '#User Growth'])
_ = ax.set_yticklabels(['{:2.0f}%'.format(x*100) for x in ax.get_yticks()])


# **The proportion of user who write no review is declining each year**. We may assume that the proportion ha been reduced due to user growth plumetting. However, when we compare with the user growth, **we can see that between 2012 to 2014, the user growth is actually improving while the proportion dropping**. This means newer user, starting from around 2013, tend to write at least 1 review every year (more actively engaged). If Yelp did **some major changes between 2013-2014 to boost user review, it has been proven effective**.
# 
# Finally, to find out whether average user writes more, we plot the median monthly number of review per user for each year. We use median because the monthly number of review per user distribution is right-skewed. 

# In[189]:


# We cut at 99% to remove outliers
yearly_rvw_median = yearly_rvw_df[yearly_rvw_df['rvw_p_month'] > 0].groupby('year')['rvw_p_month'].quantile(.5)
ax = yearly_rvw_median.plot.bar(figsize=(12,9), title='Monthly Number of Review per User')
_ = ax.set_ylabel('median (review/month)')


# <a id="conclusion"></a>
# # 5. Conclusion

# ### User Growth
# Based on this dataset sample by yelp, we can see that **the user growth has been declining since around 2015-2016.** During its peak around 2014, yelp can get around **700-800 new user daily whilst only 200-300 after 2017**. Until 2015, the number of user has at least doubled every year, but currently sit at **around 50% growth year-on-year in 2017**.
# 
# Compared to the user growth, **the growth of review is quite contrast**. Until 2017, the daily number of user is still increasing, with peak **around 4,000-4,500 reviews a day**. The year-on-year growth for number of review is still **above 100% even until 2017**. 
# 
# ### User Engagement
# From the user engagement analysis, we found that **only 0.097% of users never review**, which is really low. Most probably due to how the dataset is sampled. From all registered user in this sample, **about 90% write less than 1 review per month.** However, **the proportion of user with 0 review has been dropping**, from almost 80% in 2012 to around 50% in 2017. Looking at only 0.097% of user never review all time, while 50% for all registered user in 2017, we may need to **check churning rate of earlier users**. This may imply that **users tend to write review (or actively engaged) only on the same year they had registered**, but not so afterward.
# 
# As we can observe from **median number of review per user**, it has been steadily **descending to a mere 0.1 review per month**. In another term, 50% of ther users only give about 1 review or less per year.
# 
# ### Conculsion & Suggestion
# We can conclude that: **more user are writing reviews (yeay!), but the number of review per user is actually decreasing (aw..)**
# 
# As review gives value for yelp user, we need to find out way to make user write more review monthly. Although [yelp factsheet](https://www.yelp.com/factsheet) boast a staggering 30 mil mobile app unique users, **if those users are not actively engaged**, let's say write more review, give ratings, or even writing tips, **the churning trend will presumably continue.** Maybe we can start by asking, why do user who has written 1 review did not write another review?
# 
# Suggestion: **(1) win back churn users** and **(2) kickstart new user to write at least 1 review every 3 month** (it is an achievable target, as they have achieved *way back* back in 2005)

# <img src="https://media.giphy.com/media/3o7TKqoaCocZkDzqPC/giphy.gif" alt="people churning"/> <br />
# growin', but ain't stayin'. (*source: giphy.com*)

# <a id="future"></a>
# # 6. Future Works

# Related to user growth and engagement analysis, we can continue the analysis by
# 
# 1. **checking growth and engagement by looking at other features usage**, e.g. tips and check-ins.
# 
# 2. **analysis engagement in respect to how review/rating has been given**, does user write longer review? Or give ratings to more variant business?
# 
# 3. **calculate correlation of growth and engagement to yelp market value**, to measure if growth and engagement is significant to yelp valuation.
