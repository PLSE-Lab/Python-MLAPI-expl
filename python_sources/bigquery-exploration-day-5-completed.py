#!/usr/bin/env python
# coding: utf-8

# ## **Understanding the Essentials of SQL and BigQuery** 

# In[ ]:


import bq_helper


# In[ ]:


hacker_news = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                      dataset_name = "hacker_news")


# In[ ]:


hacker_news.list_tables()


# In[ ]:


hacker_news.table_schema("full")


# In[ ]:


hacker_news.head("full")


# In[ ]:


hacker_news.head("full", selected_columns = "by", num_rows = 10)


# In[ ]:


query = """SELECT score 
FROM `bigquery-public-data.hacker_news.full` 
WHERE type = "job" """
hacker_news.estimate_query_size(query)


# In[ ]:


hacker_news.query_to_pandas_safe(query, max_gb_scanned = 0.1)


# In[ ]:


job_post_scores = hacker_news.query_to_pandas_safe(query)


# In[ ]:


job_post_scores.score.mean()


# In[ ]:


job_post_scores.to_csv("job_post_scores.csv")


# ## Day 1
# ### ** Global Air Quality Dataset**

# In[ ]:


import bq_helper
open_aq = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data',
                                  dataset_name = 'openaq')
open_aq.list_tables()


# In[ ]:


open_aq.head("global_air_quality")


# In[ ]:


query = """SELECT city
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE country = 'US'
"""


# In[ ]:


us_cities = open_aq.query_to_pandas_safe(query)


# In[ ]:


us_cities.city.value_counts().head()


# In[ ]:


# Question 1 - Countries using anything other than ppm as unit of measurement
query1 = """SELECT distinct(country)
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE unit != 'ppm'
"""


# In[ ]:


country_list = open_aq.query_to_pandas_safe(query1)
country_list.head()


# In[ ]:


country_list.shape


# In[ ]:


# Question 2 - I understand that this means which one pollutant has 0 value globally, meaning that 
# essentially that doesn't exist on the planet/ at the measurement sites. But this is not true
# for any pollutant. Switching to finding pollutants with value 0 anywhere.
query2 = """SELECT DISTINCT pollutant, value as `Pollutant_Level`
FROM `bigquery-public-data.openaq.global_air_quality`
WHERE value = 0
"""

pollutant_levels = open_aq.query_to_pandas_safe(query2)


# In[ ]:


pollutant_levels


# ## Day 2
# ### **Having, Group BY and Count()**

# In[ ]:


import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                      dataset_name = "hacker_news")
hacker_news.head('comments')


# In[ ]:


query4 = """SELECT parent, COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
GROUP BY parent
HAVING COUNT(id) > 10
ORDER BY COUNT(id) DESC
"""
popular_stories = hacker_news.query_to_pandas_safe(query4)
popular_stories.head()


# In[ ]:


# Question 1 - How many stories are there of each type in full table?
query5 = """SELECT type, COUNT(id)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
ORDER BY COUNT(id) DESC
"""
hacker_news.estimate_query_size(query5)


# In[ ]:


story_type = hacker_news.query_to_pandas_safe(query5)
story_type.shape
story_type.head()


# In[ ]:


# Question 2 - How many comments were deleted?
query6 = """SELECT COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
WHERE deleted = True
"""
hacker_news.estimate_query_size(query6)


# In[ ]:


deleted_comments = hacker_news.query_to_pandas_safe(query6)
deleted_comments


# In[ ]:


# Question 3 Alternative
query7 = """SELECT COUNTIF(deleted = True)
FROM `bigquery-public-data.hacker_news.comments`
"""
hacker_news.estimate_query_size(query7)


# In[ ]:


del_comments_alt = hacker_news.query_to_pandas_safe(query7)
del_comments_alt


# ## Day 3
# ### **Dates and ORDER BY Clause**

# In[ ]:


import bq_helper
accidents = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                    dataset_name = "nhtsa_traffic_fatalities")
accidents.head('accident_2015')


# In[ ]:


query8 = """SELECT COUNT(consecutive_number),
EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY EXTRACT(DAYOFWEEK FROM timestamp_of_crash)
ORDER BY COUNT(consecutive_number) DESC
"""
accidents.estimate_query_size(query8)


# In[ ]:


accident_by_weekday = accidents.query_to_pandas_safe(query8)
accident_by_weekday


# In[ ]:


import matplotlib.pyplot as plt
plt.scatter(x = accident_by_weekday.f1_,y = accident_by_weekday.f0_)


# In[ ]:


# Question 1 - Which hour of the day do most crashes happen at?
query9 = """SELECT COUNT(consecutive_number),
EXTRACT(HOUR FROM timestamp_of_crash)
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY EXTRACT(HOUR FROM timestamp_of_crash)
ORDER BY COUNT(consecutive_number) DESC
"""
accidents.estimate_query_size(query9)


# In[ ]:


accident_by_time = accidents.query_to_pandas_safe(query9)
accident_by_time
plt.scatter(x = accident_by_time.f1_, y = accident_by_time.f0_)


# In[ ]:


# Question 2 - Which state has the most hit and run cases?
query10 = """SELECT registration_state_name, COUNTIF(hit_and_run = "Yes")
FROM `bigquery-public-data.nhtsa_traffic_fatalities.vehicle_2015`
GROUP BY registration_state_name
ORDER BY COUNTIF(hit_and_run = "Yes") DESC
"""

accidents.estimate_query_size(query10)


# In[ ]:


hit_and_run = accidents.query_to_pandas_safe(query10)
hit_and_run.head()


# ## Day 4
# ### ** WITH and AS**

# In[ ]:


import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                             dataset_name = "bitcoin_blockchain")


# In[ ]:


query11 = """WITH time AS
(
    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
    transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)
SELECT EXTRACT(YEAR FROM trans_time) AS Year,
EXTRACT(MONTH FROM trans_time) AS Month,
COUNT(transaction_id) AS transactions
FROM time
GROUP BY Year, Month
ORDER BY Year, Month 
"""

bitcoin_blockchain.estimate_query_size(query11)


# In[ ]:


trans_per_month = bitcoin_blockchain.query_to_pandas_safe(query11, max_gb_scanned = 21)
trans_per_month.head()


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(trans_per_month.transactions)


# In[ ]:


# Question 1 - How many bitcoin transactions were made each day in 2017?

query12 = """WITH trans_all AS
(
    SELECT TIMESTAMP_MILLIS(timestamp) as Trans_Time,
    transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)
SELECT EXTRACT(DAYOFYEAR FROM Trans_Time) AS Day_Of_Year,
COUNT(transaction_id) as Number_Of_Transactions
FROM trans_all
WHERE EXTRACT(YEAR FROM Trans_Time) = 2017
GROUP BY Day_Of_Year
ORDER BY Day_Of_Year
"""
bitcoin_blockchain.estimate_query_size(query12)


# In[ ]:


trans_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query12, max_gb_scanned = 21)
trans_per_day_2017.head()


# In[ ]:


plt.plot(trans_per_day_2017.Number_Of_Transactions)
plt.title("Number of Bitcoin Transactions Per Day in 2017")
plt.xlabel("Day Number")


# In[ ]:


# Question 2 - How many transactions are associated with each merkle root?
query13 = """SELECT merkle_root,
COUNT(transaction_id)
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
GROUP BY merkle_root
ORDER BY COUNT(transaction_id) DESC
"""
bitcoin_blockchain.estimate_query_size(query13)


# In[ ]:


trans_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query13, max_gb_scanned = 37)
trans_per_merkle_root.head()


# ## Day 5 
# ### **JOIN**

# In[ ]:


import bq_helper
github = bq_helper.BigQueryHelper(active_project = 'bigquery-public-data',
                                 dataset_name = 'github_repos')


# In[ ]:


github.table_schema('licenses')


# In[ ]:


github.table_schema('sample_files')


# In[ ]:


query14 = """SELECT L.license, COUNT(sf.path) AS Number_Of_Files
FROM `bigquery-public-data.github_repos.sample_files` as sf
INNER JOIN `bigquery-public-data.github_repos.licenses` as L
ON sf.repo_name = L.repo_name
GROUP BY license
ORDER BY Number_Of_Files DESC
"""
github.estimate_query_size(query14)


# In[ ]:


file_count_by_license = github.query_to_pandas_safe(query14, max_gb_scanned = 6)
file_count_by_license.head()


# In[ ]:


# Question 1 - How many commits made in repos written in Python?
github.table_schema('sample_files')
github.table_schema('sample_commits')


# In[ ]:


query15 = """WITH sf_python AS
(
    SELECT DISTINCT repo_name
    FROM `bigquery-public-data.github_repos.sample_files`
    WHERE path LIKE '%.py'
)
SELECT sf.repo_name AS Repo, COUNT(sc.commit) AS Number_Of_Commits
FROM sf_python AS sf
INNER JOIN `bigquery-public-data.github_repos.sample_commits` AS sc
USING(repo_name)
GROUP BY Repo
ORDER BY Number_Of_Commits DESC
"""
github.estimate_query_size(query15)


# In[ ]:


commits_per_repo = github.query_to_pandas_safe(query15, max_gb_scanned = 6)
commits_per_repo.head()

