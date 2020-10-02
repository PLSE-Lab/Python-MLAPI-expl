#!/usr/bin/env python
# coding: utf-8

# **[Advanced SQL Home Page](https://www.kaggle.com/learn/advanced-sql)**
# 
# ---
# 

# # Introduction
# 
# Now that you know how to query nested and repeated data, you're ready to draw interesting insights from the [GitHub Repos](https://www.kaggle.com/github/github-repos) dataset.  
# 
# Before you get started, run the following cell to set everything up.

# In[ ]:


# Set up feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.sql_advanced.ex3 import *
print("Setup Complete")


# # Exercises
# 
# ### 1) Who had the most commits in 2016?
# 
# GitHub is the most popular place to collaborate on software projects. A GitHub **repository** (or repo) is a collection of files associated with a specific project, and a GitHub **commit** is a change that a user has made to a repository.  We refer to the user as a **committer**.
# 
# The `sample_commits` table contains a small sample of GitHub commits, where each row corresponds to different commit.  The code cell below fetches the table and shows the first five rows of this table.

# In[ ]:


from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "github_repos" dataset
dataset_ref = client.dataset("github_repos", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "sample_commits" table
table_ref = dataset_ref.table("sample_commits")

# API request - fetch the table
sample_commits_table = client.get_table(table_ref)

# Preview the first five lines of the table
client.list_rows(sample_commits_table, max_results=5).to_dataframe()


# Run the next code cell to print the table schema. 

# In[ ]:


# Print information on all the columns in the table
sample_commits_table.schema


# Write a query to find the individuals with the most commits in this table in 2016.  Your query should return a table with two columns:
# - `committer_name` - contains the name of each individual with a commit (from 2016) in the table
# - `num_commits` - shows the number of commits the individual has in the table (from 2016)
# 
# Sort the table, so that people with more commits appear first.
# 
# **NOTE**: You can find the name of each committer and the date of the commit under the "committer" column, in the "name" and "date" child fields, respectively.

# In[ ]:


# Write a query to find the answer
max_commits_query = """
                    SELECT
                        committer.name AS committer_name,
                        COUNT(*) AS num_commits
                    FROM `bigquery-public-data.github_repos.sample_commits`
                    WHERE EXTRACT(YEAR FROM committer.date) >= 2016
                    GROUP BY committer_name
                    ORDER BY num_commits DESC
                    """

# Check your answer
q_1.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_1.hint()
#q_1.solution()


# ### 2) Look at languages!
# 
# Now you will work with the `languages` table.  Run the code cell below to print the first few rows.

# In[ ]:


# Construct a reference to the "languages" table
table_ref = dataset_ref.table("languages")

# API request - fetch the table
languages_table = client.get_table(table_ref)

# Preview the first five lines of the table
client.list_rows(languages_table, max_results=5).to_dataframe()


# Each row of the `languages` table corresponds to a different repository.  
# - The "repo_name" column contains the name of the repository,
# - the "name" field in the "language" column contains the programming languages that can be found in the repo, and 
# - the "bytes" field in the "language" column has the size of the files (in bytes, for the corresponding language).
# 
# Run the following code cell to print the table schema.

# In[ ]:


# Print information on all the columns in the table
languages_table.schema


# Assume for the moment that you have access to a table called `sample_languages` that contains only a very small subset of the rows from the `languages` table: in fact, it contains only three rows!  This table is depicted in the image below.
# 
# ![](https://i.imgur.com/qAb5lZ2.png)
# 
# How many rows are in the table returned by the query below?
# 
# ![](https://i.imgur.com/Q5qYAtz.png)
# 
# Fill in your answer in the next code cell.

# In[ ]:


# Fill in the blank
num_rows = 6

# Check your answer
q_2.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_2.hint()
#q_2.solution()


# ### 3) What's the most popular programming language?
# 
# Write a query to leverage the information in the `languages` table to determine which programming languages appear in the most repositories.  The table returned by your query should have two columns:
# - `language_name` - the name of the programming language
# - `num_repos` - the number of repositories in the `languages` table that use the programming language
# 
# Sort the table so that languages that appear in more repos are shown first.

# In[ ]:


# Write a query to find the answer
pop_lang_query = """
                 SELECT
                     language.name AS language_name,
                     COUNT(*) AS num_repos
                 FROM `bigquery-public-data.github_repos.languages`,
                 UNNEST(language) AS language
                 GROUP BY language_name
                 ORDER BY num_repos DESC
                 """
# Check your answer
q_3.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_3.hint()
#q_3.solution()


# ### 4) Which languages are used in the repository with the most languages?
# 
# For this question, you'll restrict your attention to the repository with name `'polyrabbit/polyglot'`.
# 
# Write a query that returns a table with one row for each language in this repository.  The table should have two columns:
# - `name` - the name of the programming language
# - `bytes` - the total number of bytes of that programming language
# 
# Sort the table by the `bytes` column so that programming languages that take up more space in the repo appear first.

# In[ ]:


# Your code here
all_langs_query = """
                  SELECT
                      language.name AS name,
                      language.bytes AS bytes
                  FROM `bigquery-public-data.github_repos.languages`,
                      UNNEST(language) AS language
                  WHERE repo_name = 'polyrabbit/polyglot'
                  ORDER BY bytes DESC
                  """

# Check your answer
q_4.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_4.hint()
#q_4.solution()


# # Keep going
# 
# Learn how to make your queries **[more efficient](https://www.kaggle.com/alexisbcook/writing-efficient-queries)**.

# ---
# **[Advanced SQL Home Page](https://www.kaggle.com/learn/advanced-sql)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
