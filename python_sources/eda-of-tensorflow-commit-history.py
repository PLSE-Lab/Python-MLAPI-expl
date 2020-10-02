#!/usr/bin/env python
# coding: utf-8

# # Overview
# 
# In this kernel I'll  do some basic exploration of the commit messages and programming languages for the tensorflow library. I'll be using bigquery to query the 3 Terabyte Github Repo dataset  and use the bq_helper tool to get the queried data into pandas dataframe. 
# 
# Questions that were explored here are : 
# 
# * Question 1 : What are the most popular languages used in Tensorflow and tensorflow based projects?
# * Question 2 : What are the Top commit subjects Lines in tensorflow Repository? 
# * Question 3 : Who are the most prominent committers? 
# * Question 4 : Finding the commit date range for the current snapshot of data.
# * Question 5 : What are the Most Frequent Words in Tensorflow Commit Messages ? 
# 
# New concepts I had to learn is to how to use arrays and structs in bigquery. 
# 
# Reference : 
# * https://cloud.google.com/bigquery/docs/reference/standard-sql/arrays
# 

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
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use("ggplot")

# Any results you write to the current directory are saved as output.


# First I import bq_helper package and create the BigQueryHelper object.

# In[ ]:


import bq_helper


# In[ ]:


github_repos = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                        dataset_name="github_repos")


# In[ ]:


# List all the tables in the database

github_repos.list_tables()


# In[ ]:


# lets check out the schema for the languages table

github_repos.table_schema("languages")


# In[ ]:


# head is used to get a snapshot for the first few rows of this table
github_repos.head("languages")


# # Question 1 : What are the most popular languages used in Tensorflow and tensorflow based projects?

# In[ ]:


query1 = """SELECT repo_name, language FROM 
             `bigquery-public-data.github_repos.languages`
             WHERE repo_name LIKE "%tensorflow%";
             """


# In[ ]:


github_repos.estimate_query_size(query1)


# In[ ]:


lang_results = github_repos.query_to_pandas_safe(query1)


# In[ ]:


lang_results.head(10)


# It looks like after quering for the repo_names with the three keywords : tensorflow, pytorch and keras, I ended up getting other people's forks for the main repository, but I also ended up getting all the derivative projects(e.g keras and keras-rl for reinforcement learning) and tutorials . 

# In[ ]:


unique_programming_languages = []

for lang in lang_results['language']:
    for unique_lang in lang:
        unique_programming_languages.append(unique_lang["name"])


# In[ ]:


unique_programming_languages = pd.Series(unique_programming_languages)


# In[ ]:


unique_programming_languages.value_counts(normalize=True)[0:20].plot(kind="bar",figsize=(10,10),
                                                       title="Programming Language Count For Deep Learning Repositories"
                                                        , rot = 80)


# Python, C++, C are expected along with Jupyter Notebooks, but I was surprised to see Go, Ruby etc in the top 20 language list for tensorflow repositories. It's probably because tensorflow supports many languages. We can drill down later to see which repositories use Go or Ruby, but as for now I'll move to the commit messages.

# In[ ]:


github_repos.table_schema("sample_commits")


# In[ ]:


github_repos.head("sample_commits")


# # Question 2 : What are the Top commit subjects Lines in tensorflow Repository? 
# 
# Note From here I'm limiting myself to tensorflow official repository only.

# In[ ]:


query2 = """SELECT subject, COUNT(*) AS commit_count
            FROM `bigquery-public-data.github_repos.sample_commits`
            WHERE repo_name = 'tensorflow/tensorflow'
            GROUP BY subject;"""
            


# In[ ]:


github_repos.estimate_query_size(query2)


# In[ ]:


q2 = github_repos.query_to_pandas_safe(query2)


# In[ ]:


q2.set_index("subject").sort_values("commit_count", ascending=True)[-10:].plot(kind="barh",
                                                                              title="Top 10 Subject Lines In Tensorflow Commit Messages")


# # Question 3 : Who are the most prominent committers? 

# Initially I wanted to stick to the sample commits, but that dataset does not have many samples. After figuring out how to work with bigquery structs and arrays of strings, I ended up querying the commits table.  Note here the repo name is an array of string, so to query it I'm using OFFSET to index the array with a 0-based indexing.  The query size is 93 GB, but I don't mind going over the limits for a while since this months Kaggle limit is in terabytes. 

# In[ ]:


github_repos.table_schema('commits')


# In[ ]:


query3 =  """SELECT author.name, COUNT(*) AS count FROM
            `bigquery-public-data.github_repos.commits`
            WHERE repo_name[OFFSET(0)] = 'tensorflow/tensorflow'
            GROUP BY author.name
            ; """
            


# In[ ]:


github_repos.estimate_query_size(query3)


# In[ ]:


q3 = github_repos.query_to_pandas(query3)


# In[ ]:


q3.set_index("name").sort_values('count', ascending= False)[0:5].plot(kind='bar')


# I expected there to be more commits, I'll check out the number of commits to the major 3 deep learning libraries over the years for comparison to understand the date range of this dataset better.

# In[ ]:


q4 = """SELECT author.date, COUNT(*) AS count FROM
            `bigquery-public-data.github_repos.commits`
            WHERE repo_name[OFFSET(0)] = 'tensorflow/tensorflow'
            GROUP BY author.date
            ; """


# In[ ]:


github_repos.estimate_query_size(q4)


# In[ ]:


q4 = github_repos.query_to_pandas(q4)


# In[ ]:


q4.head()


# In[ ]:


q4.date = pd.to_datetime(q4.date)


# In[ ]:


q4.set_index('date').plot()


# Looks like since the dataset is being continuously updated, I've only data from September 25th to October 10th here. Even if I do text analysis here on the commit messages it'd not much interesting without all the data. It might be interesting to at least check most frequent words though. 

# # Question 5 : What are the Most Frequent Words in Tensorflow Commit Messages ? 

# This time instead of checking out only the subjects I'll consider the messages.

# In[ ]:


q5 = """SELECT message FROM
            `bigquery-public-data.github_repos.commits`
            WHERE repo_name[OFFSET(0)] = 'tensorflow/tensorflow'
            ; """


# In[ ]:


github_repos.estimate_query_size(q5)


# In[ ]:


q5 = github_repos.query_to_pandas(q5)


# In[ ]:


q5.head()


# In[ ]:


from wordcloud import WordCloud,STOPWORDS


# In[ ]:


text = q5.message.str.cat().lower()


# In[ ]:


stopwords = STOPWORDS


# In[ ]:


cloud = WordCloud(background_color="white",stopwords = stopwords,max_words=500,colormap='cubehelix').generate(text)


# In[ ]:


plt.figure(figsize = (8,7))
plt.imshow(cloud, interpolation='bilinear', aspect='auto')
plt.axis("off")
plt.title("Most Frequent Words In Tensorflow Commit Messages")


# In[ ]:




