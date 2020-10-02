#!/usr/bin/env python
# coding: utf-8

# * [Scaling vs. Normalization: What's the difference?](#Scaling-vs.-Normalization:-What's-the-difference?)
# * [Practice scaling](#Practice-scaling)
# * [Practice normalization](#Practice-normalization)
# 

# # Get our environment set up
# ________
# 
# The first thing we'll need to do is load in the libraries and datasets we'll be using. 
# 

# In[1]:


# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# read in all our data
kickstarters_2017 = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")

# set seed for reproducibility
np.random.seed(0)


# Now that we're set up, let's learn about scaling & normalization. (If you like, you can take this opportunity to take a look at some of the data.)

# # Scaling vs. Normalization: The difference
# ____
# 
# One of the reasons that it's easy to get confused between scaling and normalization is because the terms are sometimes used interchangeably and, to make it even more confusing, they are very similar! In both cases, we're transforming the values of numeric variables so that the transformed data points have specific helpful properties. The difference is that, in scaling, we're changing the *range* of our data while in normalization we're changing the *shape of the distribution* of our data.
# ___
# 
# ## **Scaling**
# 
# This means that we're transforming our data so that it fits within a specific scale, like 0-100 or 0-1. We want to scale data when we're using methods based on measures of how far apart data points, like [support vector machines, or SVM](https://en.wikipedia.org/wiki/Support_vector_machine) or [k-nearest neighbors, or KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm). With these algorithms, a change of "1" in any numeric feature is given the same importance. 
# 
# For example, we might be looking at the prices of some products in both Yen and US Dollars. One US Dollar is worth about 100 Yen, but if we don't scale our prices methods like SVM or KNN will consider a difference in price of 1 Yen as important as a difference of 1 US Dollar! This clearly doesn't fit with our intuitions of the world. With currency, we can convert between currencies. But what about if we're looking at something like height and weight? It's not entirely clear how many pounds should equal one inch (or how many kilograms should equal one meter).
# 
# By scaling our variables, we can help compare different variables on equal footing. To help solidify what scaling looks like, we look at example.
# 

# In[2]:


# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size = 1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns = [0])

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")


# Notice that the *shape* of the data doesn't change, but that instead of ranging from 0 to 8ish, it now ranges from 0 to 1.
# 
# ___
# ## Normalization
# 
# Scaling just changes the range of our data. Normalization is a more radical transformation. The point of normalization is to change our observations so that they can be described as a normal distribution.
# 
# > **[Normal distribution:](https://en.wikipedia.org/wiki/Normal_distribution)** Also known as the "bell curve", this is a specific statistical distribution where a roughly equal observations fall above and below the mean, the mean and the median are the same, and there are more observations closer to the mean. The normal distribution is also known as the Gaussian distribution.
# 
# In general, we'll only want to normalize our data if we're going to be using a machine learning or statistics technique that assumes your data is normally distributed. Some examples of these include t-tests, ANOVAs, linear regression, linear discriminant analysis (LDA) and Gaussian naive Bayes. (Pro tip: any method with "Gaussian" in the name probably assumes normality.)
# 
# The method we're using to normalize here is called the [Box-Cox Transformation](https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation). Let's take a quick peek at what normalizing some data looks like:

# In[40]:


# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(original_data, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_data[0], ax=ax[1])
ax[1].set_title("Normalized data")


# Notice that the *shape* of our data has changed. Before normalizing it was almost L-shaped. But after normalizing it looks more like the outline of a bell (hence "bell curve"). 
# 
# * We want to build a linear regression model to predict someone's grades given how much time they spend on various activities during a normal school week.  We notice that  our measurements for how much time students spend studying aren't normally distributed: some students spend almost no time studying and others study for four or more hours every day. Should we scale or normalize this variable?
# * We're still working on our grades study, but we want to include information on how students perform on several fitness tests as well. We have information on how many jumping jacks and push-ups each student can complete in a minute. However, we notice that students perform far more jumping jacks than push-ups: the average for the former is 40, and for the latter only 10. Should we scale or normalize these variables?

# # Practice scaling
# ___
# 
# To practice scaling and normalization, we're going to be using a dataset of Kickstarter campaigns. (Kickstarter is a website where people can ask people to invest in various projects and concept products.)
# 
# Let's start by scaling the goals of each campaign, which is how much money they were asking for.

# In[19]:


# select the usd_goal_real column
usd_goal = kickstarters_2017.usd_goal_real

# scale the goals from 0 to 1
scaled_data = minmax_scaling(usd_goal, columns = [0])

# plot the original & scaled data together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(kickstarters_2017.usd_goal_real, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax=ax[1])
ax[1].set_title("Scaled data")


# We can see that scaling changed the scales of the plots dramatically (but not the shape of the data: it looks like most campaigns have small goals but a few have very large ones)

# In[35]:


#selecting the goal column
goal = kickstarters_2017.goal

#Scaling data from 0 to 1
scaled_goal = minmax_scaling(goal, columns = [0])
#plotting the original and scaled
fig, ax = plt.subplots(1,2)
sns.distplot(goal, ax= ax[0])
ax[0].set_title("Original Data")
sns.distplot(scaled_data, ax= ax[1])
ax[1].set_title("Scaled Data")


# # Practice normalization
# ___
# 
# Practicing normalization. We're going to normalize the amount of money pledged to each campaign.

# In[38]:


# get the index of all positive pledges (Box-Cox only takes postive values)
index_of_positive_pledges = kickstarters_2017.usd_pledged_real > 0

# get only positive pledges (using their indexes)
positive_pledges = kickstarters_2017.usd_pledged_real.loc[index_of_positive_pledges]

# normalize the pledges (w/ Box-Cox)
normalized_pledges = stats.boxcox(positive_pledges)[0]

# plot both together to compare
fig, ax=plt.subplots(1,2)
sns.distplot(positive_pledges, ax=ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax=ax[1])
ax[1].set_title("Normalized data")


# It's not perfect (it looks like a lot pledges got very few pledges) but it is much closer to normal!

# In[48]:


#selecting the pledged column
positive_pledged_indexes = kickstarters_2017.pledged > 0

#retrieving pledges with positive values
positive_pledges = kickstarters_2017.pledged.loc[positive_pledged_indexes]

#Normalizing using box-cox
normalized_pledges = stats.boxcox(positive_pledges)[0]

#plotting both together
fig, ax = plt.subplots(1,2)
sns.distplot(positive_pledges, ax = ax[0])
ax[0].set_title("Original Data")
sns.distplot(normalized_pledges, ax = ax[1])
ax[1].set_title("Normalized Data")


# (https://www.kaggle.com/rtatman/the-5-day-regression-challenge). ([These datasets are a good start!](https://www.kaggle.com/rtatman/datasets-for-regression-analysis)) Pick three or four variables and decide if you need to normalize or scale any of them and, if you think you should, practice applying the correct technique.
