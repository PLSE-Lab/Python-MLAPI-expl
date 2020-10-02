#!/usr/bin/env python
# coding: utf-8

# ## INTRODUCTION
# 
# This is a basic exploration of bit coin transactions to hone Google bigquery skills for SQL scavanger hunt. Lets import the necessary libraries.

# In[ ]:


import bq_helper
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ## Day 1 
# Dataset : Air Quality
# 1.  Which countries use a unit other than ppm to measure any type of pollution? (Hint: to get rows where the value isn't something, use "!=")
# 2.  Which pollutants have a value of exactly 0?

# In[ ]:


openaq = bq_helper.BigQueryHelper("bigquery-public-data", "openaq")
openaq.head("global_air_quality")


# In[ ]:


query = """SELECT distinct 
                country 
            from `bigquery-public-data.openaq.global_air_quality` where 
            unit != 'ppm';
        """
openaq.estimate_query_size(query) #20.6555436
open_aq= openaq.query_to_pandas_safe(query, max_gb_scanned=0.1)
open_aq.head(10)


# In[ ]:


#2.  Which pollutants have a value of exactly 0?
query = """SELECT distinct 
                pollutant 
            from `bigquery-public-data.openaq.global_air_quality` where 
            value = 0;
        """
openaq.estimate_query_size(query) #20.6555436
open_aq= openaq.query_to_pandas_safe(query, max_gb_scanned=0.1)
open_aq.head(10)


# ## Day2
# Dataset : pet_records
# 1. How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
# 2. How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
# 3. Optional extra credit: read about aggregate functions other than COUNT() and modify one of the queries you wrote above to use a different aggregate function.

# In[ ]:


hacker_news = bq_helper.BigQueryHelper("bigquery-public-data", "hacker_news")
hacker_news.list_tables()


# In[ ]:


hacker_news.head("full")


# In[ ]:


# query to pass to 
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
hacker_news.estimate_query_size(query) #20.6555436
Q1= openaq.query_to_pandas_safe(query, max_gb_scanned=0.26)
Q1.head(10)


# In[ ]:


#Q3  Optional extra credit: read about aggregate functions other than COUNT() and 
#modify one of the queries you wrote above to use a different aggregate function.
#hacker_news.head("comments")
query = """SELECT SUM(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY title
        """

hacker_news.estimate_query_size(query) #20.6555436
Q3= openaq.query_to_pandas_safe(query, max_gb_scanned=0.2)
Q3.head(10)


# In[ ]:


# 2. How many comments have been deleted?
hacker_news.head("comments")
query = """SELECT COUNT(id) as num_deleted_comments
            FROM `bigquery-public-data.hacker_news.comments`
            where deleted=TRUE
        """

hacker_news.estimate_query_size(query) #20.6555436
Q2= openaq.query_to_pandas_safe(query, max_gb_scanned=0.26)
Q2.head(10)


# ## Day 3
# Dataset : nhtsa_traffic_fatalities
# 1. Which hours of the day do the most accidents occur during?
# Return a table that has information on how many accidents occurred in each hour of the day in 2015, sorted by the the number of accidents which occurred each hour. Use either the accident_2015 or accident_2016 table for this, and the timestamp_of_crash column. (Yes, there is an hour_of_crash column, but if you use that one you won't get a chance to practice with dates. :P)
# Hint: You will probably want to use the EXTRACT() function for this.
# 2. Which state has the most hit and runs?
# Return a table with the number of vehicles registered in each state that were involved in hit-and-run accidents, sorted by the number of hit and runs. Use either the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name and hit_and_run columns.

# In[ ]:


import bq_helper
accidents = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="nhtsa_traffic_fatalities")
accidents.list_tables()


# In[ ]:


accidents.head('accident_2016')


# In[ ]:


# Which state has the most hit and runs?
#Return a table with the number of vehicles registered in each state 
#that were involved in hit-and-run accidents, sorted by the number of hit and runs. 
#Use either the vehicle_2015 or vehicle_2016 table for this, especially the registration_state_name 
#and hit_and_run columns.
accidents.head("vehicle_2016")
query = """SELECT EXTRACT(HOUR FROM timestamp_of_crash) ,
                    COUNT(consecutive_number)                  
            FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
            GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
            ORDER BY COUNT(consecutive_number) DESC
        """
accidents.estimate_query_size(query)
Q1= accidents.query_to_pandas_safe(query,max_gb_scanned=0.1)
Q1.head(10)
import matplotlib.pyplot as plt
plt.plot(Q1.f1_)
plt.title("Number of accdents by hour number 2016")


# In[ ]:


#1. Which hours of the day do the most accidents occur during?
#Return a table that has information on how many accidents occurred in each hour of the day in 2015, 
#sorted by the the 
#number of accidents which occurred each hour. Use either the accident_2015 or 
#accident_2016 table for this, and the timestamp_of_crash column. (Yes, there is an hour_of_crash column, but if you use that one you won't get a chance to practice with dates. :P)
query = """SELECT registration_state_name, 
                    COUNT(vehicle_number) as Number_of_vehicles
          FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2016`
          WHERE hit_and_run = "Yes"
          GROUP BY registration_state_name
          ORDER BY Number_of_vehicles DESC
        """
accidents.estimate_query_size(query)
Q2= accidents.query_to_pandas_safe(query,max_gb_scanned=0.1)
Q2.head(10)


# ## Day 4
# Dataset : Bitcoin blockchain

# In[ ]:


# checking out the structure and tables
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()


# In[ ]:


bitcoin_blockchain.table_schema("blocks")


# In[ ]:


bitcoin_blockchain.table_schema("transactions")


# In[ ]:


# How many bitcoin transactions were made eachday in 2017
query = """WITH time AS 
            (
                SELECT 
                EXTRACT(DATE FROM TIMESTAMP_MILLIS(timestamp)) AS date,
                transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            SELECT 
                COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM date) AS day,
                EXTRACT(MONTH FROM date) AS month,
                EXTRACT(YEAR FROM date) AS Year
            FROM time
            GROUP BY day, month, year
            ORDER BY day, month, year
        """
bitcoin_blockchain.estimate_query_size(query) #20.6555436
BTC_Per_Day= bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)


# In[ ]:


# Lets Plot 
#BTC_Per_Day.head()
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(BTC_Per_Day.transactions)
plt.title("Daily Bitcoin Transcations For 2017")


# In[ ]:


# How many transactions are associated with each merkle root?
query2 = """SELECT 
                merkle_root,
                count(transaction_id) as number_of_transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY merkle_root
                ORDER BY number_of_transactions DESC
        """
bitcoin_blockchain.estimate_query_size(query2)# 36.84488421678543


# In[ ]:


# The estimated size of the query is 37 gb
trans_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned = 37)
a= trans_per_merkle.head(10)


# In[ ]:


# Top 10 Merkle Roots
a= trans_per_merkle.head(10)
print(a)


# ## Day 5
# * How many commits (recorded in the "sample_commits" table) have been made in repos written in the Python programming language? (I'm looking for the number of commits per repo for all the repos written in Python.
# * You'll want to JOIN the sample_files and sample_commits questions to answer this.
#     Hint: You can figure out which files are written in Python by filtering results from the "sample_files" table using WHERE path LIKE '%.py'. This will return results where the "path" column ends in the text ".py", which is one way to identify which files have Python code.

# In[ ]:


github  = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="github_repos")

github .list_tables()

github.table_schema("sample_files")


# In[ ]:


query = """WITH python_repos AS (
    SELECT repo_name
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.py')
SELECT commits.repo_name, COUNT(commit) AS commit_count
FROM `bigquery-public-data.github_repos.sample_commits` AS commits
INNER JOIN python_repos
    ON  python_repos.repo_name = commits.repo_name
GROUP BY commits.repo_name
ORDER BY commit_count DESC

"""
github.estimate_query_size(query)     #query size is approx 5GB
num_commits = github.query_to_pandas_safe(query, max_gb_scanned=10)
display(num_commits)

