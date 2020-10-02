#!/usr/bin/env python
# coding: utf-8

# # Data cleaning challenge day 1 - Handling missing values
# 
# Well, I've been meaning to start a more structured attack on building my Python knowledge. At the moment, I'm lost in the comfort of doing things in R. For a lot of purposes, I'll probably stick to it, however, being able to do everything (or most of the things) I do in R in Python would certainly have its advantages.
# 
# For example, at the moment our company website is built using Python on a Django framework. It would certainly be handy to be able to perform analyses, run reports and build dashboards directly on the backend of the website...
# 
# So, time to hit [Day 1 of Rachael's challenge][1], and cleaning the data is a good place to start.
# 
# [1]: https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values/notebook

# In[21]:


# import numpy and pandas

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# read in the San Francisco building permits data
sfPermits = pd.read_csv("../input/Building_Permits.csv")

# set seed for reproducibility
np.random.seed(0)


# ### Look at a couple of rows from the sf_permits dataset. Do you notice any missing data?

# In[22]:


sfPermits.sample(5)


# Quite a few missing values visible already, and we've only looked at the first five rows of the dataset, cleaning will be required...
# 
# ### Find out what percent of the sf_permit dataset is missing

# In[23]:


# Calculate total number of cells in dataframe
totalCells = np.product(sfPermits.shape)

# Count number of missing values per column
missingCount = sfPermits.isnull().sum()

# Calculate total number of missing values
totalMissing = missingCount.sum()

# Calculate percentage of missing values
print("The SF Permits dataset contains", round(((totalMissing/totalCells) * 100), 2), "%", "missing values.")


# ### Look at the columns Street Number Suffix and Zipcode from the sf_permits datasets. Both of these contain missing values. Which, if either, of these are missing because they don't exist? Which, if either, are missing because they weren't recorded?

# In[24]:


missingCount[['Street Number Suffix', 'Zipcode']]


# Looks like a lot more missing values for street number suffix than zipcode. Let's check out the percentages:

# In[31]:


print("Percent missing data in Street Number Suffix column =", (round(((missingCount['Street Number Suffix'] / sfPermits.shape[0]) * 100), 2)))
print("Percent missing data in Zipcode column =", (round(((missingCount['Zipcode'] / sfPermits.shape[0]) * 100), 2)))


# As every address has a Zipcode, it looks like the missing values for this column are due to the values not being recorded. For the Street Number Suffix column, it is likely very few properties will have a suffix to the number, I see a lot of 3s, 18s, 46s, but not nearly as many 36A or 18B, so it is likely that these are missing as they don't exist.
# 
# ### Try removing all the rows from the sf_permits dataset that contain missing values. How many are left?

# In[32]:


sfPermits.dropna()


# If we drop all rows that contain a missing value, we greatly simplify our dataset. So simple, we can go for an early lunch. Every row contains at least one missing value (well, we know from our Street Number Suffix answer above that simply eliminating those gets rid of nearly 99% of our data), so we end up with a dataframe of column headers.
# 
# ### Now try removing all the columns with empty values. Now how much of your data is left?

# In[35]:


sfPermitsCleanCols = sfPermits.dropna(axis=1)
sfPermitsCleanCols.head()


# In[36]:


print("Columns in original dataset: %d \n" % sfPermits.shape[1])
print("Columns with na's dropped: %d" % sfPermitsCleanCols.shape[1])


# Well, that gives us a clean set of values, but we've sacrificed a lot of variables in the process...
# 
# ### Your turn! Try replacing all the NaN's in the sf_permit data with the one that comes directly after it and then [replace all the reamining na's with 0]

# In[39]:


imputeSfPermits = sfPermits.fillna(method='ffill', axis=0).fillna("0")

imputeSfPermits.head()


# That's certainly a nicer way to do things, but still quite a simplistic method. For EDA and preliminary analysis it's a good way to get started, but choosing an imputation method based on the type of data in each column would be a logical next step.
# 
# Either way, coming from R, it's been a good exercise to start getting my Python more up to scratch; thanks Rachael!
