#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# The last exercise in [Intro to SQL](https://www.kaggle.com/learn/sql) built the queries for an **Expert-Finder** website. This website could identify the StackOverflow users that answered questions on any technical topic.
# 
# But the StackOverflow data is very large. So, if your website serves 1000s of requests a day, you should optimize the queries to lower operating cost and keep your website snappy. In this exercise, you'll practice optimizing these types of queries to see how efficient you can make them.
# 
# ## Quick data overview
# 
# As a reminder, here are the tables in the publicly available Stack Overflow dataset.

# In[ ]:


from google.cloud import bigquery

# Create client object to access database
client = bigquery.Client()

# Specify dataset for high level overview of data
dataset_ref = client.dataset("stackoverflow", "bigquery-public-data")
dataset = client.get_dataset(dataset_ref)

# List all the tables
tables = client.list_tables(dataset)
for table in tables:  
    print(table.table_id)


# ## Review structure of answers data
# Your primary focus is finding users who answered questions. For this, you will need to use the `posts_answers` table. Here is a preview of this data.

# In[ ]:


table_ref = dataset_ref.table("posts_answers")
table = client.get_table(table_ref)
# See the first five rows of data
client.list_rows(table, max_results=5).to_dataframe()


# You may notice that the `tags` field is empty. Here's a quick overview of the `posts_questions` table, which can be joined to the `posts_answers` table, and which has useful `tags` data for most questions.

# In[ ]:


table_ref = dataset_ref.table("posts_questions")
table = client.get_table(table_ref)
client.list_rows(table, max_results=5).to_dataframe()


# In[ ]:


bytes_per_gb = 2**30
job_config = bigquery.QueryJobConfig(dry_run=True)
result = client.query(
    (
"""
WITH bq_questions as (
SELECT title, accepted_answer_id 
FROM `bigquery-public-data.stackoverflow.posts_questions` 
WHERE tags like '%bigquery%' and accepted_answer_id is not NULL
)
SELECT ans.* 
FROM bq_questions inner join `bigquery-public-data.stackoverflow.posts_answers` ans
ON ans.Id = bq_questions.accepted_answer_id
"""
    ),
    job_config=job_config
)
print("This query will process {} GB.".format(result.total_bytes_processed // bytes_per_gb))


# In[ ]:


bytes_per_gb = 2**30
job_config = bigquery.QueryJobConfig(maximum_bytes_billed=1)
result = client.query(
    (
"""
WITH bq_questions as (
SELECT title, accepted_answer_id 
FROM `bigquery-public-data.stackoverflow.posts_questions` 
WHERE tags like '%bigquery%' and accepted_answer_id is not NULL
)
SELECT ans.* 
FROM bq_questions inner join `bigquery-public-data.stackoverflow.posts_answers` ans
ON ans.Id = bq_questions.accepted_answer_id
"""
    ),
    job_config=job_config
)
result.to_dataframe()


# In[ ]:


query = """
        WITH bq_questions AS 
        (
            SELECT title, accepted_answer_id 
            FROM `bigquery-public-data.stackoverflow.posts_questions` 
            WHERE tags like '%bigquery%' and accepted_answer_id is not NULL
        )
        SELECT ans.* 
        FROM bq_questions 
        INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` ans
            ON ans.Id = bq_questions.accepted_answer_id
        """

result = client.query(query).result().to_dataframe()


# In[ ]:


result.head()

