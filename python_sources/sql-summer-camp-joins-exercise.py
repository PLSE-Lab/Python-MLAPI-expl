#!/usr/bin/env python
# coding: utf-8

# **[SQL Home Page](https://www.kaggle.com/learn/intro-to-sql)**
# 
# ---
# 

# # Introduction
# 
# [Stack Overflow](https://stackoverflow.com/) is a widely beloved question and answer site for technical questions. You'll probably use it yourself as you keep using SQL (or any programming language). 
# 
# Their data is publicly available. What cool things do you think it would be useful for?
# 
# Here's one idea:
# You could set up a service that identifies the Stack Overflow users who have demonstrated expertise with a specific technology by answering related questions about it, so someone could hire those experts for in-depth help.
# 
# In this exercise, you'll write the SQL queries that might serve as the foundation for this type of service.
# 
# As usual, run the following cell to set up our feedback system before moving on.

# In[ ]:


# Set up feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.sql.ex6 import *
print("Setup Complete")


# Run the next cell to fetch the `stackoverflow` dataset.

# In[ ]:


from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "stackoverflow" dataset
dataset_ref = client.dataset("stackoverflow", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)


# # Exercises
# 
# ### 1) Explore the data
# 
# Before writing queries or **JOIN** clauses, you'll want to see what tables are available. 
# 
# *Hint*: Tab completion is helpful whenever you can't remember a command. Type `client.` and then hit the tab key. Don't forget the period before hitting tab.

# In[ ]:


# Get a list of available tables 
list_of_tables = []
temp = list(client.list_tables(dataset)) # Your code here

for table in temp:
    list_of_tables.append(table.table_id)

# Print your answer
print(list_of_tables)

# Check your answer
q_1.check()


# For the solution, uncomment the line below.

# In[ ]:


#q_1.solution()


# ### 2) Review relevant tables
# 
# If you are interested in people who answer questions on a given topic, the `posts_answers` table is a natural place to look. Run the following cell, and look at the output.

# In[ ]:


# Construct a reference to the "posts_answers" table
answers_table_ref = dataset_ref.table("posts_answers")

# API request - fetch the table
answers_table = client.get_table(answers_table_ref)

# Preview the first five lines of the "posts_answers" table
client.list_rows(answers_table, max_results=5).to_dataframe()


# It isn't clear yet how to find users who answered questions on any given topic. But `posts_answers` has a `parent_id` column. If you are familiar with the Stack Overflow site, you might figure out that the `parent_id` is the question each post is answering.
# 
# Look at `posts_questions` using the cell below.

# In[ ]:


# Construct a reference to the "posts_questions" table
questions_table_ref = dataset_ref.table("posts_questions")

# API request - fetch the table
questions_table = client.get_table(questions_table_ref)

# Preview the first five lines of the "posts_questions" table
client.list_rows(questions_table, max_results=5).to_dataframe()


# Are there any fields that identify what topic or technology each question is about? If so, how could you find the IDs of users who answered questions about a specific topic?
# 
# Think about it, and then check the solution by running the code in the next cell.

# In[ ]:


q_2.solution()


# ### 3) Selecting the right questions
# 
# A lot of this data is text. 
# 
# We'll explore one last technique in this course which you can apply to this text.
# 
# A **WHERE** clause can limit your results to rows with certain text using the **LIKE** feature. For example, to select just the third row of the `pets` table from the tutorial, we could use the query in the picture below.
# 
# ![](https://i.imgur.com/RccsXBr.png) 
# 
# You can also use `%` as a "wildcard" for any number of characters. So you can also get the third row with:
# 
# ```
# query = """
#         SELECT * 
#         FROM `bigquery-public-data.pet_records.pets` 
#         WHERE Name LIKE '%ipl%'
#         """
# ```
# 
# Try this yourself. Write a query that selects the `id`, `title` and `owner_user_id` columns from the `posts_questions` table. 
# - Restrict the results to rows that contain the word "bigquery" in the `tags` column. 
# - Include rows where there is other text in addition to the word "bigquery" (e.g., if a row has a tag "bigquery-sql", your results should include that too).

# In[ ]:


# Your code here
questions_query = """
                  SELECT id, title, owner_user_id
                  FROM `bigquery-public-data.stackoverflow.posts_questions`
                  WHERE tags LIKE '%bigquery%'
                  """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
questions_query_job = client.query(questions_query, job_config = safe_config) # Your code goes here

# API request - run the query, and return a pandas DataFrame
questions_results = questions_query_job.to_dataframe() # Your code goes here

# Preview results
print(questions_results.head())

# Check your answer
q_3.check()


# For a hint or the solution, uncomment the appropriate line below.

# In[ ]:


#q_3.hint()
#q_3.solution()


# ### 4) Your first join
# Now that you have a query to select questions on any given topic (in this case, you chose "bigquery"), you can find the answers to those questions with a **JOIN**.  
# 
# Write a query that returns the `id`, `body` and `owner_user_id` columns from the `posts_answers` table for answers to "bigquery"-related questions. 
# - You should have one row in your results for each answer to a question that has "bigquery" in the tags.  
# - Remember you can get the tags for a question from the `tags` column in the `posts_questions` table.
# 
# Here's a reminder of what a **JOIN** looked like in the tutorial:
# ```
# query = """
#         SELECT p.Name AS Pet_Name, o.Name AS Owner_Name
#         FROM `bigquery-public-data.pet_records.pets` as p
#         INNER JOIN `bigquery-public-data.pet_records.owners` as o 
#             ON p.ID = o.Pet_ID
#         """
# ```
# 
# It may be useful to scroll up and review the first several rows of the `posts_answers` and `posts_questions` tables.  

# In[ ]:


display(client.list_rows(questions_table, max_results=5).to_dataframe())
display(client.list_rows(answers_table, max_results=5).to_dataframe())


# In[ ]:


# Your code here
answers_query = """
                SELECT a.id, a.body, a.owner_user_id
                FROM `bigquery-public-data.stackoverflow.posts_questions` AS q 
                INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                    ON q.id = a.parent_id
                WHERE q.tags LIKE '%bigquery%'
                """

# Set up the query
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
answers_query_job = client.query(answers_query, job_config = safe_config) # Your code goes here

# API request - run the query, and return a pandas DataFrame
answers_results = answers_query_job.to_dataframe() # Your code goes here

# Preview results
print(answers_results.head())

# Check your answer
q_4.check()


# For a hint or the solution, uncomment the appropriate line below.

# In[ ]:


#q_4.hint()
#q_4.solution()


# ### 5) Answer the question
# You have the merge you need. But you want a list of users who have answered many questions... which requires more work beyond your previous result.
# 
# Write a new query that has a single row for each user who answered at least one question with a tag that includes the string "bigquery". Your results should have two columns:
# - `user_id` - contains the `owner_user_id` column from the `posts_answers` table
# - `number_of_answers` - contains the number of answers the user has written to "bigquery"-related questions

# In[ ]:


# Your code here

bigquery_experts_query = """
                        WITH merger AS
                        (
                            SELECT a.id, a.body, a.owner_user_id
                            FROM `bigquery-public-data.stackoverflow.posts_questions` AS q 
                            INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                                ON q.id = a.parent_id
                            WHERE q.tags LIKE '%bigquery%'
                        )
                        SELECT owner_user_id AS user_id, count(1) AS number_of_answers
                        FROM merger
                        GROUP BY owner_user_id
                        """

# Set up the query
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
bigquery_experts_query_job = client.query(bigquery_experts_query, job_config = safe_config) # Your code goes here

# API request - run the query, and return a pandas DataFrame
bigquery_experts_results = bigquery_experts_query_job.to_dataframe() # Your code goes here

# Preview results
print(bigquery_experts_results.head())

# Check your answer
q_5.check()


# For a hint or the solution, uncomment the appropriate line below.

# In[ ]:


#q_5.hint()
#q_5.solution()


# ### 6) Building a more generally useful service
# 
# How could you convert what you've done to a general function a website could call on the backend to get experts on any topic?  
# 
# Think about it and then check the solution below.

# In[ ]:


q_6.solution()


# # Congratulations!
# 
# You know all the key components to use BigQuery and SQL effectively. Your SQL skills are sufficient to unlock many of the world's largest datasets.
# 
# Want to go play with your new powers?  Kaggle has BigQuery datasets available [here](https://www.kaggle.com/datasets?sortBy=hottest&group=public&page=1&pageSize=20&size=sizeAll&filetype=fileTypeBigQuery).
# 
# # Feedback
# 
# Bring any questions or feedback to the [Learn Discussion Forum](https://www.kaggle.com/learn-forum).

# ---
# **[SQL Home Page](https://www.kaggle.com/learn/intro-to-sql)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
