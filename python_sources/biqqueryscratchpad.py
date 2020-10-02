#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Day 1: Intro to SQL and BigQuery

# In[ ]:


from google.cloud import bigquery


# In[ ]:


# Create a "Client" object
client = bigquery.Client()


# In[ ]:


# Construct a reference to the "hacker_news" dataset
dataset_ref = client.dataset("hacker_news", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)


# In[ ]:


# List all the tables in the "hacker_news" dataset
tables =list(client.list_tables(dataset))

# Print names of all tables in the dataset (there are four!)
for table in tables:  
    print(table.table_id)


# In[ ]:


# Construct a reference to the "full" table
table_ref = dataset_ref.table("full")

# API request - fetch the table
table = client.get_table(table_ref)


# In[ ]:


# Print information on all the columns in the "full" table in the "hacker_news" dataset
table.schema


# In[ ]:


# Preview the first five lines of the "full" table
client.list_rows(table, max_results=5).to_dataframe()


# In[ ]:


# Preview the first five entries in the "by" column of the "full" table
client.list_rows(table, selected_fields=table.schema[:1], max_results=5).to_dataframe()


# In[ ]:


# Count tables in the dataset
num_tables=len(tables)


# In[ ]:


#How many columns in the `comments` table have `TIMESTAMP` data?
table=client.get_table(dataset_ref.table("comments"))
table.schema
# here num_columns=1


# In[ ]:


# Write the code here to explore the data 
client.list_rows(table, max_results=5).to_dataframe()


# # Day 2: SELECT, FROM, WHERE

# In[ ]:


dataset_ref = client.dataset("openaq", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# List all the tables in the "openaq" dataset
tables = list(client.list_tables(dataset))

# Print names of all tables in the dataset (there's only one!)
for table in tables:  
    print(table.table_id)


# In[ ]:


# Preview the first five lines of the "global_air_quality" table
client.list_rows(table, max_results=5).to_dataframe()


# In[ ]:


# Query to select all the items from the "city" column where the "country" column is 'US
query = """
        SELECT city
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """


# In[ ]:


# Set up the query
query_job = client.query(query)
# API request - run the query, and return a pandas DataFrame
us_cities = query_job.to_dataframe()


# In[ ]:


us_cities.head(10),us_cities.shape


# In[ ]:


# What five cities have the most measurements?
us_cities.city.value_counts().head()


# In[ ]:


query = """
        SELECT city, source_name
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """


# In[ ]:


# Set up the query
query_job = client.query(query)
# API request - run the query, and return a pandas DataFrame
us_cities = query_job.to_dataframe()
us_cities.head()


# In[ ]:


#You can select all columns with a * like this
query = """
        SELECT *
        FROM `bigquery-public-data.openaq.global_air_quality`
        WHERE country = 'US'
        """


# In[ ]:


# Query to get the score column from every row where the type column has value "job"
query = """
        SELECT score, title
        FROM `bigquery-public-data.hacker_news.full`
        WHERE type = "job" 
        """

# Create a QueryJobConfig object to estimate size of query without running it
dry_run_config = bigquery.QueryJobConfig(dry_run=True)

# API request - dry run query to estimate costs
dry_run_query_job = client.query(query, job_config=dry_run_config)

print("This query will process {} bytes.".format(dry_run_query_job.total_bytes_processed))


# In[ ]:


# Only run the query if it's less than 100 MB
ONE_HUNDRED_MB = 100*1000*1000
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_HUNDRED_MB)

# Set up the query (will only run if it's less than 100 MB)
safe_query_job = client.query(query, job_config=safe_config)

# API request - try to run the query, and return a pandas DataFrame
safe_query_job.to_dataframe()


# In[ ]:


# Only run the query if it's less than 1 GB
ONE_GB = 1000*1000*1000
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=ONE_GB)

# Set up the query (will only run if it's less than 1 GB)
safe_query_job = client.query(query, job_config=safe_config)

# API request - try to run the query, and return a pandas DataFrame
job_post_scores = safe_query_job.to_dataframe()

# Print average score for job posts
job_post_scores.score.mean()


# ## Exercise
# ### 1) Units of measurement
# 
# Which countries have reported pollution levels in units of "ppm"?  In the code cell below, set `first_query` to an SQL query that pulls the appropriate entries from the `country` column.

# In[ ]:


# Query to select countries with units of "ppm"
first_query = """SELECT DISTINCT country
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE unit='ppm'
                """ # Your code goes here

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
first_query_job = client.query(first_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
first_results = first_query_job.to_dataframe()

# View top few rows of results
print(first_results.head())


# ### 2) High air quality
# 
# Which pollution levels were reported to be exactly 0?  
# - Set `zero_pollution_query` to select **all columns** of the rows where the `value` column is 0.
# - Set `zero_pollution_results` to a pandas DataFrame containing the query results.

# In[ ]:


# Query to select all columns where pollution levels are exactly 0
zero_pollution_query ="""SELECT *
                FROM `bigquery-public-data.openaq.global_air_quality`
                WHERE value=0
                """  # Your code goes here

# Set up the query
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(zero_pollution_query, job_config=safe_config)

# API request - run the query and return a pandas DataFrame
zero_pollution_results = query_job.to_dataframe() # Your code goes here

print(zero_pollution_results.head())


# # Day 3: GROUP BY, HAVING, COUNT
# 
# * Which Hacker News comments generated the most discussion?

# In[ ]:


# Query to select comments that received more than 10 replies
query_popular = """
                SELECT parent, COUNT(id) AS comment_number
                FROM `bigquery-public-data.hacker_news.comments`
                GROUP BY parent
                HAVING COUNT(id) > 10
                """
# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query_popular, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
popular_comments = query_job.to_dataframe()

# Print the first five rows of the DataFrame
popular_comments.head()


# In[ ]:


# Improved version of earlier query, now with aliasing & improved readability
query_improved = """
                 SELECT parent, COUNT(1) AS NumPosts
                 FROM `bigquery-public-data.hacker_news.comments`
                 GROUP BY parent
                 HAVING COUNT(1) > 10
                 """
# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query_improved, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
popular_comments = query_job.to_dataframe()

# Print the first five rows of the DataFrame
popular_comments.head()


# # Note:
# * Count works differently depending on what you ask for. Count(*) counts the number of registries, Count(column) returns the count of non-empty rows, and Count Distinct(column) returns unique values.
# * It doesn't matter if you use 0,1,* or, 56, it's just a placeholder for the empty function that needs kwargs to run but performance wise they're almost the same
# * A Normalized database should have a unique key
# * Pandas package works on one node. Which slows it down. SQL is naturally parallelized on a cluster.
# * count() can't be used with WHERE clause

# ## Exercise:
# 

# In[ ]:





# # Day 4: ORDER BY clause, Dates, Extract
# ## ORDER BY:
# * > It sorts the results returned by query
# * > By default sorting is in ascending manner
# * > To make it descending use "DESC" arguments
# ## Dates:
# * > There are two ways that dates can be stored in BigQuery: as a DATE or as a DATETIME.
# * > DATE format: YYYY-[M]M-[D]D
# * >The DATETIME format is like the date format, but with time added at the end.
# * > important in time series problem
# * > [https://cloud.google.com/bigquery/docs/reference/legacy-sql#datetimefunctions](http://https://cloud.google.com/bigquery/docs/reference/legacy-sql#datetimefunctions)
# ## EXTRACT:
# * > to look at part of a date, like the year or the day
# * > used to get information from Dates
# * > can be only used with Dates
# * > can be used along with WHERE clause.

# In[ ]:


# Construct a reference to the "nhtsa_traffic_fatalities" dataset
dataset_ref = client.dataset("nhtsa_traffic_fatalities", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "accident_2015" table
table_ref = dataset_ref.table("accident_2015")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "accident_2015" table
client.list_rows(table, max_results=5).to_dataframe()


# In[ ]:


# Query to find out the number of accidents for each day of the week
query = """
        SELECT COUNT(consecutive_number) AS num_accidents, 
               EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week,
               EXTRACT(QUARTER FROM timestamp_of_crash) AS quarter
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        GROUP BY day_of_week,quarter
        ORDER BY quarter,num_accidents DESC
        """
# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**9)
query_job = client.query(query, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
accidents_by_day = query_job.to_dataframe()

# Print the DataFrame
accidents_by_day


# ## Excercise

# In[ ]:


from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "world_bank_intl_education" dataset
dataset_ref = client.dataset("world_bank_intl_education", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "international_education" table
table_ref = dataset_ref.table("international_education")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "international_education" table
client.list_rows(table, max_results=5).to_dataframe()


# > * Which countries spend the largest fraction of GDP on education?  
# 

# In[ ]:


# 1) Government expenditure on education
country_spend_pct_query = """
                          SELECT DISTINCT country_name,AVG(value) AS avg_ed_spending_pct
                          FROM `bigquery-public-data.world_bank_intl_education.international_education`
                          WHERE indicator_code='SE.XPD.TOTL.GD.ZS' and year>2009 and year<2018
                          GROUP BY country_name
                          ORDER BY avg_ed_spending_pct DESC
                          """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
country_spend_pct_query_job = client.query(country_spend_pct_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
country_spending_results = country_spend_pct_query_job.to_dataframe()

# View top few rows of results
print(country_spending_results.head())


# > *  Write a query below that selects the indicator code and indicator name for all codes with at least 175 rows in the year 2016.

# In[ ]:


#restrict to codes that are reported by many countries. 
code_count_query = """SELECT  indicator_code,indicator_name,COUNT(1) AS num_rows
                      FROM `bigquery-public-data.world_bank_intl_education.international_education`
                      WHERE year=2016
                      GROUP BY  indicator_code, indicator_name
                      HAVING num_rows>=175
                      ORDER BY num_rows DESC
                        """

# Set up the query
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
code_count_query_job = client.query(code_count_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
code_count_results = code_count_query_job.to_dataframe()

# View top few rows of results
print(code_count_results.head())


# # DAY 5: AS and WITH.....AS
# > * use **AS** to rename the columns generated by queries, which is also known as aliasing.
# * insert it right after the column you select.
# * A common table expression (or CTE) is a temporary table that you return within your query. CTEs are helpful for splitting your queries into readable chunks, and you can write queries against them.
# * WITH .....AS is used  to get a CTE.
#         * Syntax: """ WITH  CTE_name AS  (query) """
# *  It's important to note that CTEs only exist inside the query where you create them, and you can't reference them in later queries. So, any query that uses a CTE is always broken into two parts: 
#         * (1) first, we create the CTE, and then
#         * (2) we write a query that uses the CTE.

# In[ ]:


# Construct a reference to the "crypto_bitcoin" dataset
dataset_ref = client.dataset("crypto_bitcoin", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "transactions" table
table_ref = dataset_ref.table("transactions")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "transactions" table
client.list_rows(table, max_results=5).to_dataframe()


# * find out how many Bitcoin transactions were made each day for the entire timespan of a bitcoin transaction dataset.

# * count the number of transactions for each date and sorts the table so that earlier dates appear first.

# In[ ]:


# Query to select the number of transactions per date, sorted by date
query_with_CTE = """ 
                 WITH time AS 
                 (
                     SELECT DATE(block_timestamp) AS trans_date
                     FROM `bigquery-public-data.crypto_bitcoin.transactions`
                 )
                 SELECT COUNT(1) AS transactions,
                        trans_date
                 FROM time
                 GROUP BY trans_date
                 ORDER BY trans_date
                 """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query_with_CTE, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
transactions_by_date = query_job.to_dataframe()

# Print the first five rows
transactions_by_date.head()


# *  plot the raw results to show us the number of Bitcoin transactions per day over the whole timespan of this dataset.

# In[ ]:


transactions_by_date.set_index('trans_date').plot()


# ## EXERCISE

# # DAY 6 JOINS

# > * Using JOIN, we can write a query to create a table with just two columns: the name of the pet and the name of the owner.
# > * INNER JOIN => OR Operation
# > * OUTER JOIN=> AND Operation
# > * what happens when the key is not unique in one of the table out of two while doing inner join ?
# Ans:  Records multiply. It gets messy.
# . 

# In[ ]:


# Construct a reference to the "github_repos" dataset
dataset_ref = client.dataset("github_repos", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "licenses" table
licenses_ref = dataset_ref.table("licenses")

# API request - fetch the table
licenses_table = client.get_table(licenses_ref)

# Preview the first five lines of the "licenses" table
client.list_rows(licenses_table, max_results=5).to_dataframe()


# In[ ]:


# Construct a reference to the "sample_files" table
files_ref = dataset_ref.table("sample_files")

# API request - fetch the table
files_table = client.get_table(files_ref)

# Preview the first five lines of the "sample_files" table
client.list_rows(files_table, max_results=5).to_dataframe()


# In[ ]:


# Query to determine the number of files per license, sorted by number of files
query = """
        SELECT L.license, COUNT(1) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` AS sf
        INNER JOIN `bigquery-public-data.github_repos.licenses` AS L 
            ON sf.repo_name = L.repo_name
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 10 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
query_job = client.query(query, job_config=safe_config)

# API request - run the query, and convert the results to a pandas DataFrame
file_count_by_license = query_job.to_dataframe()
# Print the DataFrame
file_count_by_license


# In[ ]:




