#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules

import matplotlib.pyplot as plt
import seaborn as sns

kickstarters = pd.read_csv('../input/kickstarter2018nlp/ks-projects-201801-extra.csv')

# set seed for reproducibility
np.random.seed(0)


# Now that we're set up, let's learn about scaling & normalization. (If you like, you can take this opportunity to take a look at some of the data.)

# # Scaling vs. Normalization: What's the difference?

# 
# One of the reasons that it's easy to get confused between scaling and normalization is because the terms are sometimes used interchangeably and, to make it even more confusing, they are very similar! In both cases, you're transforming the values of numeric variables so that the transformed data points have specific helpful properties. The difference is that, in scaling, you're changing the range of your data while in normalization you're changing the shape of the distribution of your data. Let's talk a little more in-depth about each of these options.
# 

# # Scaling

# This means that you're transforming your data so that it fits within a specific scale, like 0-100 or 0-1. You want to scale data when you're using methods based on measures of how far apart data points, like support vector machines, or SVM or k-nearest neighbors, or KNN. With these algorithms, a change of "1" in any numeric feature is given the same importance.
# 
# For example, you might be looking at the prices of some products in both Yen and US Dollars. One US Dollar is worth about 100 Yen, but if you don't scale your prices methods like SVM or KNN will consider a difference in price of 1 Yen as important as a difference of 1 US Dollar! This clearly doesn't fit with our intuitions of the world. With currency, you can convert between currencies. But what about if you're looking at something like height and weight? It's not entirely clear how many pounds should equal one inch (or how many kilograms should equal one meter).
# 
# By scaling your variables, you can help compare different variables on equal footing. To help solidify what scaling looks like, let's look at a made-up example. (Don't worry, we'll work with real data in just a second, this is just to help illustrate my point.)

# In[ ]:


# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size=1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])


# plot both together to compare
fig, ax= plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original_data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled_data")


# 
# 
# Notice that the shape of the data doesn't change, but that instead of ranging from 0 to 8ish, it now ranges from 0 to 1.

# # Normalization

# Scaling just changes the range of your data. Normalization is a more radical transformation. The point of normalization is to change your observations so that they can be described as a normal distribution.
# 
#     Normal distribution: Also known as the "bell curve", this is a specific statistical distribution where a roughly equal observations fall above and below the mean, the mean and the median are the same, and there are more observations closer to the mean. The normal distribution is also known as the Gaussian distribution.
# 
# In general, you'll only want to normalize your data if you're going to be using a machine learning or statistics technique that assumes your data is normally distributed. Some examples of these include t-tests, ANOVAs, linear regression, linear discriminant analysis (LDA) and Gaussian naive Bayes. (Pro tip: any method with "Gaussian" in the name probably assumes normality.)
# 
# The method were using to normalize here is called the Box-Cox Transformation. Let's take a quick peek at what normalizing some data looks like:

# In[ ]:


# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax = plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original_data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized_data")


# Notice that the shape of our data has changed. Before normalizing it was almost L-shaped. But after normalizing it looks more like the outline of a bell (hence "bell curve").

# For the following example, decide whether scaling or normalization makes more sense.
# 
#     You want to build a linear regression model to predict someone's grades given how much time they spend on various activities during a normal school week. You notice that your measurements for how much time students spend studying aren't normally distributed: some students spend almost no time studying and others study for four or more hours every day. Should you scale or normalize this variable?
#     You're still working on your grades study, but you want to include information on how students perform on several fitness tests as well. You have information on how many jumping jacks and push-ups each student can complete in a minute. However, you notice that students perform far more jumping jacks than push-ups: the average for the former is 40, and for the latter only 10. Should you scale or normalize these variables?
# 

# # Practice scaling

# To practice scaling and normalization, we're going to be using a dataset of Kickstarter campaigns. (Kickstarter is a website where people can ask people to invest in various projects and concept products.)
# 
# Let's start by scaling the goals of each campaign, which is how much money they were asking for.

# In[ ]:


# select the usd_goal_real column
usd_goal = kickstarters.usd_goal_real

# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns=[0])


# plot the original & scaled data together to compare
fig, ax = plt.subplots(1,2)
sns.distplot(usd_goal, ax=ax[0])
ax[0].set_title("Original_data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled_data")


# # Practice normalization

# Ok, now let's try practicing normalization. We're going to normalize the amount of money pledged to each campaign

# In[ ]:


# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = kickstarters.usd_pledged_real > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters.usd_pledged_real.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")


# # Parsing Dates

# we'll be working with two datasets: one containing information on earthquakes that occured between 1965 and 2016, and another that contains information on landslides that occured between 2007 and 2016.

# In[ ]:


import datetime

# Read in our dataset
landslides = pd.read_csv("../input/landslide-events/catalog.csv")

# set seed for reproducibility
np.random.seed(0)


# Now we're ready to look at some dates! (If you like, you can take this opportunity to take a look at some of the data.)

# # Check the data type of our date column

# I'll be working with the date column from the landslides dataframe. The very first thing I'm going to do is take a peek at the first few rows to make sure it actually looks like it contains dates

# In[ ]:


# print the first few rows of the date column
print(landslides['date'].head())


# 
# 
# Yep, those are dates! But just because I, a human, can tell that these are dates doesn't mean that Python knows that they're dates. Notice that the at the bottom of the output of head(), you can see that it says that the data type of this column is "object".
# 
#     Pandas uses the "object" dtype for storing various types of data types, but most often when you see a column with the dtype "object" it will have strings in it.
# 
# you'll notice that there's also a specific datetime64 dtypes. Because the dtype of our column is object rather than datetime64, we can tell that Python doesn't know that this column contains dates.
# 
# We can also look at just the dtype of your column without printing the first few rows if we like:
# 

# In[ ]:


# check the data type of our date column
landslides['date'].dtype


# "O" is the code for "object", so we can see that these two methods give us the same information.
# 

# # Convert our date columns to datetime
# 

# Now that we know that our date column isn't being recognized as a date, it's time to convert it so that it is recognized as a date. This is called "parsing dates" because we're taking in a string and identifying its component parts.
# 
# We can pandas what the format of our dates are with a guide called as "strftime directive". The basic idea is that you need to point out which parts of the date are where and what punctuation is between them. There are lots of possible parts of a date, but the most common are %d for day, %m for month, %y for a two-digit year and %Y for a four digit year.
# 
# Some examples:
# 
#     1/17/07 has the format "%m/%d/%y"
# 
#     17-1-2007 has the format "%d-%m-%Y"
# 
#     Looking back up at the head of the date column in the landslides dataset, we can see that it's in the format "month/day/two-digit year", so we can use the same syntax as the first example to parse in our dates:
# 

# In[ ]:


# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format = "%m/%d/%y")


# 
# 
# Now when I check the first few rows of the new column, I can see that the dtype is datetime64. I can also see that my dates have been slightly rearranged so that they fit the default order datetime objects (year-month-day)

# In[ ]:


# print the first few rows
landslides['date_parsed'].head()


# 
# 
# Now that our dates are parsed correctly, we can interact with them in useful ways.

# 
#     What if I run into an error with multiple date formats? While we're specifying the date format here, sometimes you'll run into an error when there are multiple date formats in a single column. If that happens, you have have pandas try to infer what the right date format should be. You can do that like so:
# 
# landslides['date_parsed'] = pd.to_datetime(landslides['Date'], infer_datetime_format=True)
# 
#     Why don't you always use infer_datetime_format = True? There are two big reasons not to always have pandas guess the time format. The first is that pandas won't always been able to figure out the correct date format, especially if someone has gotten creative with data entry. The second is that it's much slower than specifying the exact format of the dates.
# 

# # Select just the day of the month from our column
# 

# In[ ]:


# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day


# # Plot the day of the month to check the date parsing

# One of the biggest dangers in parsing dates is mixing up the months and days. The to_datetime() function does have very helpful error messages, but it doesn't hurt to double-check that the days of the month we've extracted make sense.
# 
# To do this, let's plot a histogram of the days of the month. We expect it to have values between 1 and 31 and, since there's no reason to suppose the landslides are more common on some days of the month than others, a relatively even distribution. (With a dip on 31 because not all months have 31 days.) Let's see if that's the case:

# In[ ]:


# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()


# In[ ]:


# plot the day of the month
sns.distplot(day_of_month_landslides, kde = False, bins=30)


# Yep, it looks like we did parse our dates correctly & this graph makes good sense to me.

# In[ ]:




