#!/usr/bin/env python
# coding: utf-8

# **[SQL Micro-Course Home Page](https://www.kaggle.com/learn/SQL)**
# 
# ---
# 

# # Introduction
# 
# You have the tools to obtain data from a single table in whatever format you want it. But what if the data you want is spread across multiple tables?
# 
# That's where **JOIN** comes in! **JOIN** is incredibly important in practical SQL workflows. So let's get started.
# 
# # Example
# 
# We'll use our imaginary `pets` table, which has three columns: 
# - `ID` - ID number for the pet
# - `Name` - name of the pet 
# - `Animal` - type of animal 
# 
# We'll also add another table, called `owners`.  This table also has three columns: 
# - `ID` - ID number for the owner (different from the ID number for the pet)
# - `Name` - name of the owner
# - `Pet_ID` - ID number for the pet that belongs to the owner (which matches the ID number for the pet in the `pets` table)
# 
# ![](https://i.imgur.com/Rx6L4m1.png) 
# 
# To get information that applies to a certain pet, we match the `ID` column in the `pets` table to the `Pet_ID` column in the `owners` table.  
# 
# ![](https://i.imgur.com/eXvIORm.png)
# 
# For example, 
# - the `pets` table shows that Dr. Harris Bonkers is the pet with ID 1. 
# - The `owners` table shows that Aubrey Little is the owner of the pet with ID 1. 
# 
# Putting these two facts together, Dr. Harris Bonkers is owned by Aubrey Little.
# 
# Fortunately, we don't have to do this by hand to figure out which owner goes with which pet. In the next section, you'll learn how to use **JOIN** to create a new table combining information from the `pets` and `owners` tables.
# 
# # JOIN
# 
# Using **JOIN**, we can write a query to create a table with just two columns: the name of the pet and the name of the owner. 
# 
# ![](https://i.imgur.com/fLlng42.png)
# 
# We combine information from both tables by matching rows where the `ID` column in the `pets` table matches the `Pet_ID` column in the `owners` table.
# 
# In the query, **ON** determines which column in each table to use to combine the tables.  Notice that since the `ID` column exists in both tables, we have to clarify which one to use.  We use `p.ID` to refer to the `ID` column from the `pets` table, and `o.Pet_ID` refers to the `Pet_ID` column from the `owners` table.
# 
# > In general, when you're joining tables, it's a good habit to specify which table each of your columns comes from. That way, you don't have to pull up the schema every time you go back to read the query.
# 
# The type of **JOIN** we're using today is called an **INNER JOIN**. That means that a row will only be put in the final output table if the value in the columns you're using to combine them shows up in both the tables you're joining. For example, if Tom's ID number of 4 didn't exist in the `pets` table, we would only get 3 rows back from this query. There are other types of **JOIN**, but an **INNER JOIN** is very widely used, so it's a good one to start with.

# # Example: How many files are covered by each type of software license?
# 
# GitHub is the most popular place to collaborate on software projects. A GitHub **repository** (or **repo**) is a collection of files associated with a specific project.  
# 
# Most repos on GitHub are shared under a specific legal license, which determines the legal restrictions on how they are used.  For our example, we're going to look at how many different files have been released under each license. 
# 
# We'll work with two tables in the database.  The first table is the `licenses` table, which provides the name of each GitHub repo (in the `repo_name` column) and its corresponding license.  Here's a view of the first five rows.

# In[ ]:


from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

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


# The second table is the `sample_files` table, which provides, among other information, the GitHub repo that each file belongs to (in the `repo_name` column).  The first several rows of this table are printed below.

# In[ ]:


# Construct a reference to the "sample_files" table
files_ref = dataset_ref.table("sample_files")

# API request - fetch the table
files_table = client.get_table(files_ref)

# Preview the first five lines of the "sample_files" table
client.list_rows(files_table, max_results=5).to_dataframe()


# Next, we write a query that uses information in both tables to determine how many files are released in each license.

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

# Set up the query
query_job = client.query(query)

# API request - run the query, and convert the results to a pandas DataFrame
file_count_by_license = query_job.to_dataframe()


# It's a big query, and so we'll investigate each piece separately.
# 
# ![](https://i.imgur.com/QeufD01.png)
#     
# We'll begin with the **JOIN** (highlighted in blue above).  This specifies the sources of data and how to join them. We use **ON** to specify that we combine the tables by matching the values in the `repo_name` columns in the tables.
# 
# Next, we'll talk about **SELECT** and **GROUP BY** (highlighted in yellow).  The **GROUP BY** breaks the data into a different group for each license, before we **COUNT** the number of rows in the `sample_files` table that corresponds to each license.  (Remember that you can count the number of rows with `COUNT(1)`.) 
# 
# Finally, the **ORDER BY** (highlighted in purple) sorts the results so that licenses with more files appear first.
# 
# It was a big query, but it gave us a nice table summarizing how many files have been committed under each license:  

# In[ ]:


# Print the DataFrame
file_count_by_license


# You'll use **JOIN** clauses a lot and get very efficient with them as you get some practice.
# 
# # Your Turn
# 
# You are on the last step.  Finish it by solving **[these exercises](https://www.kaggle.com/kernels/fork/682118)**.

# ---
# **[SQL Micro-Course Home Page](https://www.kaggle.com/learn/SQL)**
# 
# 
