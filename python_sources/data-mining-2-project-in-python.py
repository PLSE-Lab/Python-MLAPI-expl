#!/usr/bin/env python
# coding: utf-8

# # Data Mining 2 Project
# 
# To do:
# 1. Figure out how to get one table into a data frame: DONE
# 2. Save the data frame to CSV file: DONE
# 3. Examine the list of lists: DONE
# 4. Identify columns to remove
# 5. Build new query excluding unwanted columns
# 6. Start downloading data to CSV
# 
# ==
# Examine assumptions required for statistical modeling plan:
# - Pros of planned methods
# - Cons of planned methods
# 

# ## Get Access To the Data and set up libaries

# In[ ]:


import bq_helper
import pandas as pd
import os
#from bq_helper import BigQueryHelper

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
# https://github.com/SohierDane/BigQuery_Helper/blob/master/bq_helper.py

# Establish Helper Object for data scanning
google_analytics = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="google_analytics_sample")

# Another example of how to get the data
# bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

# Create list of tables to later assist with queries
tablelist = google_analytics.list_tables()
print(len(tablelist))
print("First table:", tablelist[0],"  Last table:", tablelist[-1])


# The above table list confirms that the data start at August 1, 2016 (20160801) and end at August 1, 2017 (20170801)
# 
# Now examine the overall structure of a table

# In[ ]:


print("Table ID:", tablelist[0])
print(google_analytics.head(tablelist[0]).columns)
google_analytics.head(tablelist[0], num_rows=3)


# In[ ]:


print("Table ID:", tablelist[0])
google_analytics.table_schema(tablelist[0])


# ## Queries

# ### Total Size of All Data
# First, I'll find the size of the entire table in Gigabytes by using a funciton native to the bq_helper package

# In[ ]:


# Determine Total Datasize
query_everything = """
#standardSQL
SELECT *
FROM 
    # enclose table names with WILDCARDS in backticks `` , not quotes ''
    `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
    _TABLE_SUFFIX < '20170802'
"""
google_analytics.estimate_query_size(query_everything)


# # Test abilbity to use table list
# from google.cloud import bigquery
# client = bigquery.Client()
# 
# query_test = '''
# #standardSQL
# SELECT *
# FROM 
#     # enclose table names with WILDCARDS in backticks `` , not quotes ''
#     `bigquery-public-data.google_analytics_sample.ga_sessions_*`
# WHERE
#     _TABLE_SUFFIX < @myval
# '''
# query_params = [
#     bigquery.ScalarQueryParameter('myval', 'INT64', 20170802)
# ]
# job_config = bigquery.QueryJobConfig()
# job_config.query_parameters = query_params
# query_job = client.query(
#     query_test,
#     # Location must match that of the dataset(s) referenced in the query.
#     location='US',
#     job_config=job_config)
# 
# #print(query_test)
# google_analytics.estimate_query_size(query_job)
# 

# ### Select 1 table
# 
# 1. Start by esimtating the query size
# 2. If query size is acceptable, save as data frame

# query_oneTable = """
# #standardSQL
# SELECT *
# FROM 
#     # enclose table names with WILDCARDS in backticks `` , not quotes ''
#     `bigquery-public-data.google_analytics_sample.ga_sessions_20160801`
# """
# google_analytics.estimate_query_size(query_oneTable)

# oneTable = google_analytics.query_to_pandas_safe(query_oneTable, max_gb_scanned=.1)
# print(oneTable.head(3))

# print(oneTable.shape, oneTable.columns)

# ### Examine Column Names

# for col in oneTable.columns:
#     print(col, ": ", type(oneTable[col][0]))
#     print(oneTable[col][0])
#     #print(oneTable[col])
#     

# ### Test writing to CSV

# # myfilepath = "C:/Users/Jusitn/Documents/Python Scripts/DataMining2"
# oneTable.to_csv("20160801.csv", encoding='utf-8', index=False)

# # Combine Query and CSV

# In[ ]:


#20160801 - 20160807
query = """
#standardSQL
SELECT *
FROM 
    # enclose table names with WILDCARDS in backticks `` , not quotes ''
    `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
    _TABLE_SUFFIX BETWEEN '20160801' AND '20160807'
"""
print("Est query size (GB):", google_analytics.estimate_query_size(query))
oneTable = google_analytics.query_to_pandas_safe(query, max_gb_scanned=.2)
#print(oneTable.head(3))
print(oneTable.shape, oneTable.columns)


# In[ ]:


oneTable.to_csv("20160801_0807.csv",  encoding='utf-8', index=False) #default encoding in Python 3 is  encoding='utf-8' but special characters force ASCII
print("Query: ", query, "to CSV completed")


# # 20160803
# query3 = """
# #standardSQL
# SELECT *
# FROM 
#     # enclose table names with WILDCARDS in backticks `` , not quotes ''
#     `bigquery-public-data.google_analytics_sample.ga_sessions_20160803`
# """
# print(google_analytics.estimate_query_size(query3))
# oneTable3 = google_analytics.query_to_pandas_safe(query3, max_gb_scanned=.1)
# #print(oneTable.head(3))
# print(oneTable3.shape)
# oneTable3.to_csv("20160803.csv", encoding='utf-8', index=False)

# # 20180604
# query4 = """
# #standardSQL
# SELECT *
# FROM 
#     # enclose table names with WILDCARDS in backticks `` , not quotes ''
#     `bigquery-public-data.google_analytics_sample.ga_sessions_20180604`
# """
# print(google_analytics.estimate_query_size(query4))
# oneTable4 = google_analytics.query_to_pandas_safe(query4, max_gb_scanned=.1)
# #print(oneTable.head(3))
# print(oneTable4.shape)
# oneTable4.to_csv("20180604.csv", encoding='utf-8', index=False)

# In[ ]:





# ## Test Reading from CSV
# **Important Note:** after perfomring the 'Test writing to CSV', I had to click on the 'Data' tab in the Kernel, look at the left hand column, and add my own kernel as a data source so that i could see the CSV files in my directory.
# THEN, be sure to select the table and click "download"

# import pandas as pd
# googledata = pd.read_csv('../input/20160802.csv')
# #DOES NOT WORK:  pd.read_csv('../input/data-mining-2-project-in-python/20160801.csv')
# 
# googledata.head(1)

# for col in googledata.columns:
#     print(col, ": ", type(googledata[col][0]))
#     print(googledata[col][0])
#     #print(oneTable[col])
# 

# The above data 

# googledata['trafficSource'][0]

# # Example Queries
# 
# What is the total number of transactions generated per device browser in July 2017?

# query1 = """SELECT
# device.browser,
# SUM ( totals.transactions ) AS total_transactions
# FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
# WHERE
# _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'
# GROUP BY
# device.browser
# ORDER BY
# total_transactions DESC;
#         """
# google_analytics.estimate_query_size(query1)
# #response1 = google_analytics.query_to_pandas_safe(query1)
# #response1.head(10)
# 

# The real bounce rate is defined as the percentage of visits with a single pageview. What was the real bounce rate per traffic source?

# query2 = """SELECT
# source,
# total_visits,
# total_no_of_bounces,
# ( ( total_no_of_bounces / total_visits ) * 100 ) AS bounce_rate
# FROM (
# SELECT
# trafficSource.source AS source,
# COUNT ( trafficSource.source ) AS total_visits,
# SUM ( totals.bounces ) AS total_no_of_bounces
# FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
# WHERE
# _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
# GROUP BY
# source )
# ORDER BY
# total_visits DESC;
#         """
# response2 = google_analytics.query_to_pandas_safe(query2)
# response2.head(10)

# What is the average amount of money spent per session in July 2017?

# query6 = """SELECT
# ( SUM(total_transactionrevenue_per_user) / SUM(total_visits_per_user) ) AS
# avg_revenue_by_user_per_visit
# FROM (
# SELECT
# fullVisitorId,
# SUM( totals.visits ) AS total_visits_per_user,
# SUM( totals.transactionRevenue ) AS total_transactionrevenue_per_user
# FROM
# `bigquery-public-data.google_analytics_sample.ga_sessions_*`
# WHERE
# _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
# AND
# totals.visits > 0
# AND totals.transactions >= 1
# AND totals.transactionRevenue IS NOT NULL
# GROUP BY
# fullVisitorId );
#         """
# response6 = google_analytics.query_to_pandas_safe(query6, max_gb_scanned=10)
# response6.head(10)

# What is the sequence of pages viewed?

# query7 = """SELECT
# fullVisitorId,
# visitId,
# visitNumber,
# hits.hitNumber AS hitNumber,
# hits.page.pagePath AS pagePath
# FROM
# `bigquery-public-data.google_analytics_sample.ga_sessions_*`,
# UNNEST(hits) as hits
# WHERE
# _TABLE_SUFFIX BETWEEN '20170701' AND '20170731'
# AND
# hits.type="PAGE"
# ORDER BY
# fullVisitorId,
# visitId,
# visitNumber,
# hitNumber;
#         """
# response7 = google_analytics.query_to_pandas_safe(query7, max_gb_scanned=10)
# response7.head(10)
