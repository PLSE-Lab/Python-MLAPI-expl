#!/usr/bin/env python
# coding: utf-8

# <table>
#     <tr>
#         <td>
#         <center>
#         <font size="+5">SQL Scavenger Hunt: Day 5</font>
#         </center>
#         </td>
#     </tr>
# </table>

# ## Example: How many files are covered by each license?
# ____
# 
# Today we're going to be using the GitHub Repos dataset. GitHub is an place for people to store & collaborate on different versions of their computer code. A "repo" is a collection of code associated with a specific project. 
# 
# Most public code on Github is shared under a specific license, which determines how it can be used and by who. For our example, we're going to look at how many different files have been released under each licenses. 
# 
# First, of course, we need to get our environment ready to go:

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")


# Now we're ready to get started on our query. This one is going to be a bit of a beast, so stick with me! The only new syntax we'll see is around the JOIN clause, everything is something we've already learned. :)
# 
# First, I'm going to specify which columns I'd like to be returned in the final table that's returned to me. Here, I'm selecting the COUNT of the "path" column from the sample_files table and then calling it "number_of_files". I'm *also* specifying that I was to include the "license" column, even though there's no "license" column in the "sample_files" table.
# 
#         SELECT L.license, COUNT(sf.path) AS number_of_files
#         FROM `bigquery-public-data.github_repos.sample_files` as sf
# Speaking of the JOIN clause, we still haven't actually told SQL we want to join anything! To do this, we need to specify what type of join we want (in this case an inner join) and how which columns we want to JOIN ON. Here, I'm using ON to specify that I want to use the "repo_name" column from the each table.
# 
#     INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
#             ON sf.repo_name = L.repo_name
# And, finally, we have a GROUP BY and ORDER BY clause that apply to the final table that's been returned to us. We've seen these a couple of times at this point. :)
# 
#         GROUP BY license
#         ORDER BY number_of_files DESC
#  Alright, that was a lot, but you should have an idea what each part of this query is doing. :) Without any further ado, let' put it into action.

# In[ ]:


# You can use two dashes (--) to add comments in SQL
query = ("""
        -- Select all the columns we want in our joined table
        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            ON sf.repo_name = L.repo_name -- what columns should we join on?
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)

file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=6)


# Whew, that was a big query! But it gave us a nice tidy little table that nicely summarizes how many files have been committed under each license:  

# In[ ]:


# print out all the returned results
print(file_count_by_license)


# And that's how to get started using JOIN in BigQuery! There are many other kinds of joins (you can [read about some here](https://cloud.google.com/bigquery/docs/reference/standard-sql/query-syntax#join-types)), so once you're very comfortable with INNER JOIN you can start exploring some of them. :)

# # Scavenger hunt Question ?
# ___
# 
# Now it's your turn! Here is the question I would like you to get the data to answer. Just one today, since you've been working hard this week. :)
# 
# *  How many commits (recorded in the "sample_commits" table) have been made in repos written in the Python programming language? (I'm looking for the number of commits per repo for all the repos written in Python.
#     * You'll want to JOIN the sample_files and sample_commits questions to answer this.
#     * **Hint:** You can figure out which files are written in Python by filtering results from the "sample_files" table using `WHERE path LIKE '%.py'`. This will return results where the "path" column ends in the text ".py", which is one way to identify which files have Python code.
# 

# In[ ]:


repo_query = ("""
        -- Select all the columns we want in our joined table
        SELECT sf.repo_name ,COUNT(C.commit) as number_of_python_commits
        FROM `bigquery-public-data.github_repos.sample_files` as sf
        -- Table to merge into sample_files
        INNER JOIN `bigquery-public-data.github_repos.sample_commits` as C 
            ON sf.repo_name = C.repo_name -- what columns should we join on?
        WHERE sf.path LIKE '%.py'
        GROUP BY sf.repo_name
        ORDER BY number_of_python_commits DESC
        """)


# In[ ]:


github.estimate_query_size(repo_query)


# In[ ]:


repo_commit_count_for_python = github.query_to_pandas_safe(repo_query, max_gb_scanned=6)


# In[ ]:


print(repo_commit_count_for_python)

