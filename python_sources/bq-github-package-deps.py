#!/usr/bin/env python
# coding: utf-8

# # Extracting data of popular testing framework lised in package.json in github projects via BigQuery

# This notebook contains steps of how to extract data of popular testing framework lised in package.json in github projects via BigQuery
# 
# Below is a list of interested testing framework that we will use in the query:
# * mocha
# * jest
# * jasmine
# * qunit
# * funcunit
# * cypress
# * puppeteer
# * chai
# * sinon

# ## Where the data is collected?

# We will extract data from the Kaggle's BigQuery dataset **GitHub Repos** which contains a full snapshot of the content of more than 2.8 million open source GitHub repositories  since 2008.
# 
# See more details about this dataset here: https://www.kaggle.com/github/github-repos
# 
# Note that this dataset was **LAST UPDATED** on **2019-03-20** at the time this note book this run (2019-06-27)

# <hr />

# ## Data Wrangling

# ### Data Requirements
# 
# 1. Extract data only github project which
#    * contains package.json
#    * package.json contains at lease one of the interested testing frameworks in the 'dependencies' or 'devDependencies' keys
#    
# 2. Each row will contains the following columns:
#    * repository name
#    * Name of all listed dependencies
#    * Columns where their name is one of the interested testing framework and its value is a Boolean value which indicate whether that particular testing framework is present in the dependencies listed in package.json

# ### Steps
# 
# Below are steps that are required to collect the required data
# 1. Create a subtable called **tf** which created from:
#    * Filter only rows which contains **package.json** in its file path in the **files** table. 
#    * SELECT the **id** and **repo_name** columns
# 2. Create a new table called **t_dep** by 
#    * JOIN the subtable **tf** with the **contents** table on the **id** column 
#    * SELECT the **id** and **repo_name** columns of the subtable tf, and create a new column **package_dep** which contains a list of all packages listed in the ***dependencies*** or ***devDependencies*** in package.json using text in the content column
# 3. Create a new table called **t_dep_check** by 
#    * SELECT the **repo_name** and **package_dep** columns of the table **t_dep**
#    * Create a new column **is_interested** by checking whether a value in the **package_dep** columh contains any interested testing framework. Set this value if any interested framework to True is found, and False otherwise.
# 4. Finally, only return data where the **is_interested** value is True****

# ## Let's get started!

# After listing out all requirements and steps, let start the data wrangling process!

# Let's first start by loading all required packages

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


# In[ ]:


from google.cloud import bigquery
client = bigquery.Client()


# ### Construct a query

# The following query was inspired by
# 
# https://www.kaggle.com/ibadia/using-javascript-with-bigquery-simple-tutorial

# First, create a [User-Defined function](https://cloud.google.com/bigquery/docs/reference/standard-sql/user-defined-functions) that we will use to get a list of dependencies in package.json

# In[ ]:


javascript_query='''
CREATE TEMPORARY FUNCTION 
getDependenciesInPackageJson(content STRING)
RETURNS STRING
LANGUAGE js AS
"""

var res = '';
try {
    var x = JSON.parse(content);
    
    var list_dep = [];
    if (x.dependencies) {
      list_dep = Object.keys(x.dependencies);
    }
    
    var list_devdep = [];
    if (x.devDependencies) {
      list_devdep = Object.keys(x.devDependencies);
    }
    
    var list_alldep = list_dep.concat(list_devdep)
    res = list_alldep.join(',')
    

} catch (e) {}

return res;
""";
'''
print (javascript_query)


# Then, create a query for the steps mentioned earlier

# In[ ]:


# --- Specify the table names 

# # Use sample tables when testing out the query
# ds_files = 'bigquery-public-data.github_repos.sample_files'
# ds_contents = 'bigquery-public-data.github_repos.sample_contents'

ds_files = 'bigquery-public-data.github_repos.files'
ds_contents = 'bigquery-public-data.github_repos.contents'


# --- Specify a list of interested testing framework
list_fw = [
    'mocha', 
    'jest',
    'jasmine',
    'qunit',
    'funcunit',
    'cypress',
    'puppeteer',
    'chai',
    'sinon'
]


# In[ ]:


my_sql_query=('''
WITH t_dep AS (
    SELECT 
        tf.id AS id, 
        tf.repo_name AS repo_name, 
        getDependenciesInPackageJson(tc.content) AS package_dep
    FROM (
        SELECT id, repo_name, path
        FROM `{}`
            WHERE path LIKE "package.json" 
    ) AS tf
    LEFT JOIN
      `{}` tc
    ON
      tf.id = tc.id
),

t_dep_check AS (
    SELECT repo_name, package_dep,
        REGEXP_CONTAINS(package_dep, r"{}") AS is_interested
    FROM t_dep
)

SELECT repo_name, package_dep
FROM t_dep_check
    WHERE is_interested
''').format(ds_files, ds_contents, '|'.join(list_fw))


# Finally, concatenate the UDF and query to construct a final query

# In[ ]:


final_query=javascript_query+my_sql_query
print (final_query)


# Before extracting data from BigQuery, Let's see how much data will be processed in this query to ensure that we won't exceed the free tier quota.

# In[ ]:


my_job_config = bigquery.job.QueryJobConfig()
my_job_config.dry_run = True

my_job = client.query(final_query, job_config=my_job_config)
BYTES_PER_GB = 2**30
my_job.total_bytes_processed / BYTES_PER_GB


# With the current query, it will process around 2.5 TB and we should find a better optimized query for this.
# 
# However, since we still have enough quota left and we need this data, we will proceed with this query.

# In[ ]:


query_contents = client.query(final_query)

# Create a dataframe from the queried results
df_contents = query_contents.to_dataframe()


# ## Data Transformation

# After getting the queried data, we will transform it first before exporting it to a .csv file.
# 
# Below is what we will do in this section:
# 1. Sort data by the **repo_name** column
# 2. Then, create new columns for all interested testing framework and assign a Boolean value to indicate whether a particular repository depends on those testing framework or not

# In[ ]:


# Make a copy of this dataframe before cleaning & transforming it
df_interested = df_contents.copy()

# Sort by the 'repo_name' column
df_interested = df_interested.sort_values(by='repo_name')


# In[ ]:


# Inspect the data 
df_contents.head()


# Inspect the size of the output data

# In[ ]:


df_interested.shape


# Loop through each interested testing framework and update its column by checking whether the **package_dep** contains the current testing framework in the loop

# In[ ]:


for cur_fw in list_fw:
    df_interested[cur_fw] = df_interested.package_dep.str.contains(cur_fw)


# Inspect how many repositories depend on those testing frameworks

# In[ ]:


df_interested[list_fw].sum(axis=0).sort_values(ascending=False)


# In[ ]:


df_interested.to_csv("github_package_deps_June_2019.csv",index=False)


# In[ ]:




