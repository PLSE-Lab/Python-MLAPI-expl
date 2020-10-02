#!/usr/bin/env python
# coding: utf-8

# Quoting the dataset introduction :
# 
# > Recent studies have found that many forums tend to be dominated by a very small fraction of users. Is this true of Hacker News?
# 
# Let's find out.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import matplotlib
matplotlib.style.use('ggplot')


# Read in CSV to dataframe
df = pd.read_csv('../input/hacker_news_sample.csv', parse_dates=[13], infer_datetime_format=True)

# Examine the fields
df.columns


# In[ ]:


df.shape


# In[ ]:


for column in df.columns:
    print(column, df[column].dtype)


# In[ ]:


df["ranking"].unique()


# In[ ]:


# "ranking" is useless, drop it
del df["ranking"]
df.columns


# In[ ]:


# don't count deleted/invalid/ no user posts
df["deleted"].unique()


# In[ ]:


df["dead"].unique()


# In[ ]:


df = df[pd.isnull(df["dead"])]
df = df[pd.isnull(df["deleted"])]
del df["dead"]
del df["deleted"]
df.columns


# In[ ]:


df = df[pd.notnull(df["by"])]


# In[ ]:


df.shape


# In[ ]:


by_user = df.groupby(["by"]).size()
by_user


# Users "tptacek" and "jacquesm" seem to be the most prolific contributors by a pretty large margin:

# In[ ]:


by_user.nlargest(30)


# Let's see how the data is distributed:

# In[ ]:


by_user.describe()


# In[ ]:


plt.figure(figsize=(10,5))
by_user.plot.box(vert=False, logx=True)


# Most of the users have a low post count but there is a certain number of outliers.

# Almost half of the users have posted only once or less:

# In[ ]:


one_users = by_user[by_user <= 1].count()
total_users = by_user.count()
one_users, total_users, 100 * one_users / total_users


# These users make up for 2.61% of the total number of posts:

# In[ ]:


100 * by_user[by_user <= 1].sum() / by_user.sum()


# If we include users who have made 2 posts we get more than half of users, and twice as much posts:

# In[ ]:


two_users = by_user[by_user <= 2].count()
two_users, total_users, 100 * two_users / total_users


# In[ ]:


100 * by_user[by_user <= 2].sum() / by_user.sum()


# Almost 3 out of 4 have posted less than 5 times, and 96% less than 100 times:

# In[ ]:


100 * by_user[by_user <= 5].count() / total_users


# In[ ]:


100 * by_user[by_user <= 100].count() / total_users


# Now let's make a histogram:

# In[ ]:


plt.figure(figsize=(8,6))
ax = by_user.plot.hist()
ax.set(xlabel="number of posts", ylabel="number of users")
ax


# The discrepancy is so high that we will be better off using a logarithmic scale on both sides.

# In[ ]:


plt.figure(figsize=(8,6))
ax = by_user.plot.hist(logx=True, logy=True, bins=2**np.arange(0, 15))
ax.set(xlabel="number of posts", ylabel="number of users")
ax


# Even with the logarithmic scale we see a clear decrease in number of users when their post count rises.

# User "tptacek" has posted 10657 times, i.e. ~650 times more than the average user:

# In[ ]:


by_user["tptacek"], by_user["tptacek"] / by_user.mean()


# The two most prolific contributors make up for 0.54% of total posts:

# In[ ]:


100 * by_user.nlargest(2).sum() / by_user.sum()


# The 20 top contributors make up for 2.48% of all posts:

# In[ ]:


by_user.nlargest(20)


# In[ ]:


100 * by_user.nlargest(20).sum() / by_user.sum()


# A little more than 1 post out of 4 was made by one of the top 1000 contributors:

# In[ ]:


100 * by_user.nlargest(1000).sum() / by_user.sum()


# Finally, 3 posts out of 4 come from the 20,000 top contributors:

# In[ ]:


100 * by_user.nlargest(20000).sum() / by_user.sum()


# Considering there are 206,482 users, this means roughly **10%** of users wrote more than **75%** of the posts.

# Now, we have considered the data in the whole archive. Users may not have been active during the whole lifespan of the forum. What happens if we analyse user activity during specific time spans?

# In[ ]:


df["year"] = df["timestamp"].dt.year


# In[ ]:


by_year = df.groupby("year").size()
by_year.plot.bar(figsize=(8,6))


# The rate of posts / year is constantly growing except from 2013 to 2014. 2017 is not over at the time of writing, hence the smaller number.

# In[ ]:


# https://stackoverflow.com/a/10374456
user_year = pd.DataFrame({'count' : df.groupby( [ "year", "by"] ).size()}).reset_index()
user_year


# In[ ]:


user_year["year"] = user_year["year"].astype("str")


# In[ ]:


user_year[["year", "count"]].boxplot(by="year", figsize=(10,6))


# Again, there seems to be a similar situation in all years following 2007, with most users concentrated in the low post count but also a number of outliers.

# In[ ]:


# https://stackoverflow.com/a/27844045
group_useryear_sizes = df.groupby( [ "year", "by"] ).size()
group_useryear = group_useryear_sizes.groupby(level="year", group_keys=False)


# Let us see the top poster for each year:

# In[ ]:


group_useryear.nlargest(1)


# And the top 5 for each year:

# In[ ]:


group_useryear.nlargest(5)


# In[ ]:


pareto = group_useryear.apply(lambda x: 100 * x.nlargest(int(x.count() / 5)).sum() / x.sum())
pareto


# In[ ]:


pareto.plot(kind="bar", figsize=(8,6))


# Here we see that for every year except the launch year 2006, **20%** of active users wrote roughly **80%** of the posts. This looks like a striking example of the [Pareto principle][1]. Note that by "active users" we mean users who have posted at least once during that year.
# 
# 
#   [1]: https://en.wikipedia.org/wiki/Pareto_principle

# What if we consider monthly timespans?

# In[ ]:


df["month"] = pd.to_datetime(df["timestamp"].dt.strftime("%Y-%m-01"))


# In[ ]:


group_usermonth_sizes = df.groupby( [ "month", "by"] ).size()
group_usermonth = group_usermonth_sizes.groupby(level="month", group_keys=False)


# In[ ]:


paretomonth = group_usermonth.apply(lambda x: 100 * x.nlargest(int(x.count() / 5)).sum() / x.sum())
paretomonth


# In[ ]:


paretomonth.describe()


# In[ ]:


paretomonth.plot.box(vert=False, figsize=(8,6))


# In[ ]:


paretomonth.plot(figsize=(12,10))


# This does not seem to apply when we consider monthly timespans : what we see is, generally, 20% of active users write roughly 2/3rds of a month's posts.

# Finally, let us see the monthly activity of the top contributors:

# In[ ]:


top_users = group_useryear.nlargest(1).index.levels[1]
list(top_users)


# In[ ]:


user_month = pd.DataFrame({'count' : df.groupby( [ "month", "by"] ).size()}).reset_index()


# In[ ]:


fig, ax = plt.subplots(figsize=(12,10))
for user in top_users:
    user_month_partial = user_month[user_month["by"] == user]
    ax.plot(user_month_partial["month"], user_month_partial["count"], label=user)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.legend()
ax

