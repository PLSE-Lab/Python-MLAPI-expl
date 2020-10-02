#!/usr/bin/env python
# coding: utf-8

# This is my first proper Kernel. Hello kagglers! And thanks for all your help so far! As it looks like I'll be spending more time with Python in my future, I'm not done with your help yet...
# 
# For now though, onto the challenges, and a couple of nice little challenges to my woeful Python skills they R (it was UK pun day when I did this)
# 
# Time to get to work...

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")


# Challenge the first: **Which countries don't use ppm as their unit of measurement?**

# In[ ]:


# define first query
queryOne = """SELECT DISTINCT country, unit
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE unit != 'ppm'
        """


# In[ ]:


# use the very useful query_to_pandas_safe to avoid scanning too much data
ppmNot = open_aq.query_to_pandas_safe(queryOne)

# display returned data
ppmNot


# Quite a few; this SI thing must really be catching on...

# Challenge the second: **Which pollutants have a value of exactly 0?**
# As has been pointed out in another kernel I came across, this question is perhaps not the most specific, and could be interpreted in a number of different ways.
# 
# Also, can any pollutant have a value of *exactly* 0? In what volume of test sample? Does a 0 actually mean 'below the level of detection'?
# 
# However, as this is day 1 of the scavenger hunt, I'm going to assume that it's an introductory question and choose to interpret it in the way that gives the most simple SQL query. 
# 
# Which I think is this one:

# In[ ]:


queryTwo = """SELECT DISTINCT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """


# In[ ]:


noPollute = open_aq.query_to_pandas_safe(queryTwo)
        
noPollute


# We might want a bit more information than that I suppose. Some of those pollutants may have a value of 0 in just a single location, some may be at zero in multiple locations. Let's change the query a little bit to see which pollutants look as though they're below the levels of detection in the most places:

# In[ ]:


queryThree = """SELECT pollutant
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value = 0
        """
        
noPolluteCounts = open_aq.query_to_pandas_safe(queryThree)
        
# count the number of times each pollutant hit that magic zero figure
noPolluteCounts.pollutant.value_counts()


# Go go gadget sulphur dioxide reduction measures! Okay, so, leaping to conclusions there without having looked at most of the dataset, but hey, it's day one and I did a thing using Python.
# 
# Here's to many more kernels in the future.
