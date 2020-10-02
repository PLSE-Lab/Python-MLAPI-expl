#!/usr/bin/env python
# coding: utf-8

# # User Retention - Meta Kaggle Dataset
# 
# Retention is among the most important metrics to understand. It is crucial in the development of a successful product. In this notebook, we'll use the Meta Kaggle dataset to learn how to visualize, understand, and explore this key performance indicator.
# 
# I'm particularly interested in the usage of Kernels. Therefore this notebook will only look at users which have created/forked at least one notebook.
# 
# In the end, I hope you'll be able to apply what you've learned and look at the retention of your own product.
# 
# <img src="https://image.slidesharecdn.com/metricsandksfsforbuildingahighperformingcustomersuccessteam-150410171710-conversion-gate01/95/customer-success-best-practices-for-saas-retention-metrics-and-ksfs-for-building-a-high-performing-customer-success-team-10-638.jpg?cb=1428686468" alt="thatdbegreat" style="width: 400px"/>
# 
# ## Overview
# Before we dive into the data, let's understand what user retention is and why it's important. 
# 
# ### What is retention?
# At a high level, retention is our ability to make users come back to our product within a certain time period. 
# 
# In our example, we would like users who created a notebook today to use Kernels again when they write their next notebook.
# 
# ### Why is retention important?
# ***Retention tells us if we're building something worth building***. It helps us understand if we're providing value to our users. In ther words, retention tells us whether or not our product has market fit.
# 
# If we provide no value, we don't have a sustainable business.
# 
# Acquiring new users is also expensive. And if we can't retain them, all the money put into user acquisition and growth hacking tactics will go to waste. 
# 
# ### Spoiler Alert! What's a good retention rate?
# Maybe you already know what retention is. You might be simply trying to figure out what is a good number for your retention rate.
# 
# A good retention rate is 15%! No wait... It's 10%. But it could also be 28%, 95% is definitely good I guess - *There's no magic number.* 
# 
# Personally, I was frustrated when I couldn't find that magic number. Only after watching Alex Schultz's  [lecture](https://www.youtube.com/watch?v=n_yHZ_vKjno) about retention that I understood why no particular number existed . I highly encourage you to do the same!
# 
# ## Exploring User Retention of Kernels
# It's time for us to look at the data! 
# 
# We will be looking at what percentage of new users come back in the following week, the week after, and so on. This will tell us how *sticky* Kernels is. Once we have a grasp in the overall user retention, we'll look at ways to make our insights actionable.
# 
# In the following sections we will:
# - Import libraries and load tables
# - Prepare the data for exploration (Data Wrangling).
# - Compute overall week-over-week (WoW) user retention.
# - Explore ways to make our analysis actionable.
# 

# ## Libraries and Tables
# 
# To compute user retention, we need only two tables:
# * A table with user signup dates
# * A table with user events (used to determine the dates an user was active)
# 
# We'll derive both of these tables from the `KernelVersions` dataset.

# In[ ]:


# Imports
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Returns Panda DataFrame for provided CSV filename (without .csv extension).
load = lambda name: pd.read_csv('../input/{}.csv'.format(name))

# Information about public kernel versions. 
# Signup date table and user events table will be derived from this dataframe.
kernel_versions = load('KernelVersions')


# ## Data Wrangling
# Raw data is tipically not ready to be analysed. We first need to clean it up a little. 

# In[ ]:


# Convert CreationDate from string to date. It will make filtering the data easier later on.
kernel_versions['CreationDate'] = (
    pd.to_datetime(kernel_versions['CreationDate'], format='%m/%d/%Y %I:%M:%S %p', cache=True)
    .dt.normalize())


# In[ ]:


# Create 'User Signup' table
# We consider the first time someone created a kernel as their "signup" date.
kernel_users = (
    kernel_versions
    .groupby('AuthorUserId', as_index=False)
    .agg({'CreationDate': 'min'})
    .rename(columns={'AuthorUserId': 'Id', 'CreationDate': 'RegisterDate'})
)

# Assert each row represents a unique user.
assert kernel_users['Id'].nunique() == kernel_users.shape[0], 'kernel_users table is malformed.'

# Display snippet of table.
kernel_users.head()


# In[ ]:


# Create 'User Events' table.
# A user is active on a given day if there's an entry on this table for that user.
#
# If user X has 3 events on date '2018/01/01', we only need one of these events. 
# We remove extra events with the drop_duplicates function.
kernel_user_events = (
    kernel_versions[['AuthorUserId', 'CreationDate']]
    .drop_duplicates()
    .rename(columns={'AuthorUserId': 'Id', 'CreationDate': 'Date'})
)

# Display snippet of table.
kernel_user_events.head()


# In[ ]:


# Merge kernel_users (signup table) with kernel_user_events (events table).
dim_users = pd.merge(
    kernel_user_events,
    kernel_users,
    how='left',
    on='Id')

# Compute the number of weeks between signup and a given event.
dim_users['weeks_from_signup'] = round((dim_users['Date'] - dim_users['RegisterDate']) / np.timedelta64(1, 'W'))
dim_users = dim_users[['Id', 'weeks_from_signup']].drop_duplicates()


# In[ ]:


# Let's only look at the first 8 weeks after signup. 
# This is enough time for the week-over-week retention curve to converge.
dim_users = dim_users[dim_users['weeks_from_signup'] <= 8]

# Convert absolute user count each week as percentage of all users.
cohort_size = dim_users['Id'].nunique()
user_count_by_week = (
    dim_users
    .groupby('weeks_from_signup')
    .agg('count')
    .rename(columns={'Id': 'user_count'})
).reset_index()
user_count_by_week['pct_returned'] = user_count_by_week['user_count'] / cohort_size * 100

# Show retention table
user_count_by_week


# The table above shows that 100% of users were active in the first week (which makes sense since all of them had to signup). 
# 
# In the second week (row 1), 13% of the users returned, and so on.
# 
# Our retention curve looks like:

# In[ ]:


ax = user_count_by_week[['weeks_from_signup', 'pct_returned']].set_index('weeks_from_signup').plot()
ax.set_ylabel('% Active Users')
_ = ax.set_title('Kernels Retention Curve')


# Congratulations! We've just built our first user retention chart!
# 
# The last thing you want to see in a chart like this is the curve touching the x-axis. In other words, 0% of users return to your product at some point! 
# 
# There are users who are still active 8 weeks after they've signed up. This is good news!
# 
# ### A Side Note
# There are many different ways to look at retention. 
# 
# We're looking at the percentage of users who are active N weeks after they've signed up. Some other common approaches are to look at month-over-month retention or look at 7-day (or 30-day) active instead of 1-day active (our case).
# 
# ## So What?
# If we presented these findings in a meeting, the response would most likely be:
# 
# <img src="http://m.memegen.com/uit82m.jpg" alt="Drawing" style="width: 300px;"/>
# 
# However, as Kaggle users, we're not content with just cool. What we really want is:
# 
# <img src="https://i.kym-cdn.com/entries/icons/mobile/000/009/993/tumblr_m0wb2xz9Yh1r08e3p.jpg" alt="Drawing" style="width: 300px;"/>
# 
# To take our insights to the next level, we must make them actionable.
# 
# ## Actionability - Turn Insights Into Action
# We now have our baseline. We know that by the 8th week, 2% of the users are still using Kernels. It's time to explore ways to increase retention.
# 
# Cohort analysis is a great way to develop more actionable insights. The overall user retention is what we've plotted above. However, there are certain groups of users who are more engaged than others. Cohort analysis can help us identify them.
# 
# ### Kernel Categories
# Authors can *tag* their notebooks. To tag a notebook is to associate it with a category.
# 
# <img src="https://image.ibb.co/gZhAu8/tag.png" alt="tag" style="width: 600px" />
# 
# Let's see if user's who add tags to their first notebook behave any differently than our current baseline (all users). For that, we'll join our user events table with the `KernelTags` dataset.
# 

# In[ ]:


# Load KernelTags table.
kernel_tags = load('KernelTags')


# In[ ]:


# Create temporary table to determine if a user's first kernel has a tag.
user_first_kernel = kernel_versions.iloc[kernel_versions.groupby('AuthorUserId')['CreationDate'].idxmin()]
user_first_kernel = pd.merge(
    user_first_kernel,
    kernel_tags,
    how='left',
    on='KernelId',
    suffixes=('', '_kernel_tags'))

# If right side of join is n/a, it's because user's first notebook has no tag/category.
user_first_kernel.loc[pd.notnull(user_first_kernel.TagId), 'TagId'] = 'has_category'
user_first_kernel.loc[pd.isnull(user_first_kernel.TagId), 'TagId'] = 'no_category'
user_first_kernel = user_first_kernel.rename(columns={'TagId': 'has_category'})


# In[ ]:


cohort = 'has_category'

augmented_kernel_users = pd.merge(
    kernel_users,
    user_first_kernel,
    left_on='Id',
    right_on='AuthorUserId',
    suffixes=('', '_b'))[['Id', cohort, 'RegisterDate']]

dim_users = pd.merge(
    kernel_user_events,
    augmented_kernel_users,
    how='left',
    on='Id')
dim_users['weeks_from_signup'] = round((dim_users['Date'] - dim_users['RegisterDate']) / np.timedelta64(1, 'W'))
dim_users = dim_users[['Id', 'weeks_from_signup', cohort]].drop_duplicates()
dim_users = dim_users[dim_users['weeks_from_signup'] <= 6]

assert dim_users['Id'].nunique() == dim_users[dim_users['weeks_from_signup'] == 0].shape[0]

cohort_size = (
    dim_users[dim_users['weeks_from_signup'] == 0]
    .groupby([cohort], as_index=False).agg('count')[[cohort, 'Id']]
    .rename(columns={'Id': 'cohort_size'})
)
cohort_size = cohort_size[cohort_size['cohort_size'] > 1000]


users_by_cohort = (pd.merge(
    dim_users,
    cohort_size,
    on=cohort)
 .groupby(['weeks_from_signup', cohort, 'cohort_size'], as_index=False)
 .agg('count')
 .rename(columns={'Id': 'user_count'})
)

users_by_cohort['pct'] = users_by_cohort['user_count'] / users_by_cohort['cohort_size'] * 100


# In[ ]:


plt.figure(figsize=(8, 6))
for a, b in users_by_cohort.groupby([cohort]):
    plt.plot(b['weeks_from_signup'], b['user_count'] / b['cohort_size'] * 100.0, label=a)
plt.title('Kernels Retention Curve')
plt.ylabel('% Active Users')
plt.xlabel('Weeks From Signup')
plt.legend()
plt.show()


# Wow. It appears that users who added a tag to their first kernel have substatially higher retation rates! Let's look at it in a table.

# In[ ]:


users_by_cohort[['weeks_from_signup', 'has_category', 'pct']].pivot_table(index=['weeks_from_signup'], columns=['has_category'], values=['pct'])


# The difference in retention between the two groups is astonishing. It's certainly worth further exploration and *experimentation*.
# 
# In momentsl like this, it's especially important to remind ourselves that correlation is not causation. Forcing all users to categorise their Kernels won't magically triple user retention.
# 
# ### Actionability
# To make an insight actionable is to first make a "guess" about why the metric behaves a certain way and then run experiments to test your hypothesis.
# 
# We should view our new finding about retention as the starting point of our more guided experimentation. Our *actionable* steps from here are to develop hypotheses and work with engineers to ship A/B experiments.
# 
# Some of our hypotheses could be:
# - Kernels with tags are better indexed by our search, which drives more readers to a kernel, which generates more comments, which ultimately makes an author feel more engaged. 
# - Users who take the time to tag their kernels usually have well written material and get more engagement form the community. Therefore we should find ways to help others write better content.
# 
# Do you have any hypothesis that could explain why new users which tag their notebooks stick around longer? I'd love to hear your thoughts in the comments!
# 
# We've just scratched the surface of user retention analysis. There are many other amazing techniques to help us better understand overall user engagement. We'll explore them in following Kernels! 
# 
# Thank you for reading!
# 
