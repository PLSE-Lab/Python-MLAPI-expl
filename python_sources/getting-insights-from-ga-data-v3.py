#!/usr/bin/env python
# coding: utf-8

# # Getting insights from GA data #
# 
# ## What you will build ##
# 
# In this tutorial, you'll learn how to get GA data from Bigquery
# 
# We are going to use a public bigquery dataset, specifically  [google_analytics_sample](https://bigquery.cloud.google.com/dataset/bigquery-public-data:google_analytics_sample). 
# 
# ## What you will learn ##
# 
# * How to use python on ipython notebook to fetch GA data
# 
# ## Setup ##
# 
# * There is no prerequites for running this notebook, as a small sample of dates will be extracted for demonstrating purposes
# * If you want to do serious analysis, would recommend datalab/colab 
# 

# ## Fetch data from Bigquery ##
# 
# You can also run these queries in [BigQuery](https://bigquery.cloud.google.com/table/bigquery-public-data:google_analytics_sample.ga_sessions_20170801).
# Here we examine the data structure for Bigquery to understand each columns.
# 

# In[ ]:


print('What tables do I have?')
import pandas as pd
from bq_helper import BigQueryHelper
bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")
bq_assistant.list_tables()[:5]


# In[ ]:


QUERY = """
    SELECT 
        *  -- Warning, be careful when doing SELECT ALL
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801` 
    LIMIT 10
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head()


# ### what does the schema look like? ###

# In[ ]:


schema = bq_assistant.table_schema("ga_sessions_20160801")
schema


# ### Can I look for specific columns? ###

# In[ ]:


schema[schema['name'].str.contains("page")]


#  ## Get some useful data  ##

# ### Working with Sessions (AKA visits) ###

# In[ ]:


schema[schema['name'].str.contains("visit")]


# **Visits on a particular day**

# In[ ]:


QUERY = """
    SELECT
        SUM(totals.visits) as visits
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801` 
    LIMIT 10
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head()


# **Visits for a number of days**

# In[ ]:


QUERY = """
    SELECT
        SUM(totals.visits) as visits
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 
    LIMIT 10
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head()


# **Visits by day for a number of days**

# In[ ]:


QUERY = """
    SELECT
        date as date,
        SUM(totals.visits) as visits
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 
    GROUP BY date
    ORDER BY date ASC
    LIMIT 31
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head(31)


# ### Working with users ###

# In[ ]:


QUERY = """
    SELECT
        fullVisitorId
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 
    LIMIT 10
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head()


# ### How do I count them? ###

# In[ ]:


# This way doesnt work!
QUERY = """
    SELECT
        fullVisitorId,
        COUNT(*) as the_count
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 
    GROUP BY fullVisitorId
    LIMIT 10
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head(5)


# ### The wrong way to count visitors ###

# In[ ]:


# This way doesnt work!
QUERY = """
    SELECT
        COUNT(fullVisitorId) as the_count
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 
    LIMIT 10
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head(5)


# ### A right way to count visitors ###

# In[ ]:


# This way works!
QUERY = """
    SELECT
        COUNT(DISTINCT fullVisitorId) as the_count
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 
    LIMIT 10
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head(5)


# ### Working with Pageviews ###

# In[ ]:


schema[schema['name'].str.contains("page")]


# In[ ]:


QUERY = """
    SELECT
        date as date,
        SUM(totals.pageviews) as pageviews
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 
    GROUP BY date
    ORDER BY date ASC
    LIMIT 31
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head()


# ### Working with Revenue ###

# In[ ]:


schema[schema['name'].str.contains("Revenue")]


# In[ ]:


QUERY = """
    SELECT
        date as date,
        SUM(totals.transactionRevenue) as transactionRevenue
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 
    GROUP BY date
    ORDER BY date ASC
    LIMIT 31
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head()


# ### What is going on with the high revenue numbers!!! ###
# 
# The Google Analytics Schema: (http://goo.gl/3gPcJH)
#     
# 

# In[ ]:


QUERY = """
    SELECT
        date as date,
        SUM(totals.transactionRevenue)/1e6 as transactionRevenue       ## 1e6 - means 10 to the power of 6 (or 1,000,000) AKA 10^6
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 
    GROUP BY date
    ORDER BY date ASC
    LIMIT 31
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head(31)


# ### Working with custom dimensions ###

# In[ ]:


schema[schema['name'].str.contains("custom")]


# In[ ]:


QUERY = """
    SELECT
        fullVisitorId,
        customDimensions
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 
    LIMIT 10
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head(10)


# **The custom dimensions are all nested up in those square brackets. We will need to unnest them to be able to use them**

# In[ ]:


QUERY = """
    SELECT
        fullVisitorid,
        customDimensions,
        cds.Value,
        cds.Index
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`, 
        UNNEST(customDimensions) AS cds                                        #### this bit breaks out the bracket stuff
    WHERE _TABLE_SUFFIX BETWEEN '20170101'AND '20170131' 
    AND index = 4
    LIMIT 10
"""

df = bq_assistant.query_to_pandas(QUERY)
df.head()


# ****----------------------------------------------------------------------------------------------------------------------------------------------------------------****

# # YOUR TURN! #

# ## Question 1: How much revenue was there in June 2017 ##

# In[ ]:


## put your code for question 1 here


# ## Question 2: In June 2017, which traffic source has brought the maximum revenue?  ##

# In[ ]:


## put your code for question 2 here


# **----------------------------------------------------------------------------------------------------------------------------------------------------------------**

# # ADVANCED SECTION  #
# ##  Get user journey for one user ##
# 
# * To get user journey, for each user, we need user_id to identify the user, and then a series of activities in chronicle order.

# In[ ]:


QUERY = """
  SELECT 
      fullVisitorId, 
      TIMESTAMP_SECONDS(visitStartTime) AS visitStartTime,
      TIMESTAMP_ADD(TIMESTAMP_SECONDS(visitStartTime), INTERVAL hits.time MILLISECOND) AS hitTime,
      hits.page.pagePath,
      hits.type
  FROM 
      `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga, 
      unnest(hits) as hits
  WHERE fullVisitorId = '0509972280802528263' AND
      _TABLE_SUFFIX BETWEEN '20170801' AND '20170801'
  ORDER BY hitTime 
  LIMIT 50
  """
df = bq_assistant.query_to_pandas(QUERY)
df


# ## Get user journey for many users
# 
# * To get user journey, for each user, we need user_id to identify the user, and then a series of activities in chronicle order.

# In[ ]:


QUERY = """  
  SELECT 
      fullVisitorId, 
      TIMESTAMP_SECONDS(visitStartTime) AS visitStartTime,
      TIMESTAMP_ADD(TIMESTAMP_SECONDS(visitStartTime), INTERVAL hits.time MILLISECOND) AS hitTime,
      hits.page.pagePath,
      hits.type
  FROM 
      `bigquery-public-data.google_analytics_sample.ga_sessions_*` AS ga, 
      unnest(hits) as hits
  WHERE _TABLE_SUFFIX BETWEEN '20170801' AND '20170801'
  ORDER BY fullVisitorId, hitTime 
  LIMIT 500
  """
df = bq_assistant.query_to_pandas(QUERY)
df

