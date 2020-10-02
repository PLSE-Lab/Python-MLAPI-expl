#!/usr/bin/env python
# coding: utf-8

# # INTRO
# ___
# 
# We'll take a look at some of the basic statements of SQL. We'll use BigQuery, a database system that lets you apply SQL to huge datasets. The current example uses a dataset called `openaq` which contain one table called `global_air_quality`

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
open_aq = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="openaq")

# print all the tables in this dataset
open_aq.list_tables()


# Using the `head()` function, let's take a look at the first  5 rows of the dataset

# In[ ]:


# print the first 5 rows of the "global_air_quality" dataset
open_aq.head("global_air_quality")


# # SELECT (columns) FROM (table)
# 
# The first SQL statements is `SELECT`, the `SELECT` statement is used to select data from a database. We can choose whatever column in our table in the `SELECT` statement. In this particular case, we'll select the `'city'` and `'country'` collumns from the `global_air_quality table`.
# 
# ** > We need to put our SQL statements inside the quotation mark ("""   """) **

# In[ ]:


# query to select all the items from the "city" and "country" column
query1 = """SELECT city, country
            FROM `bigquery-public-data.openaq.global_air_quality`
        """


# Now we can use our query above and turns it into Pandas Dataframe using the `open_aq.query_to_pandas_safe` function.

# In[ ]:


# only run this query if it's less than 100 MB
query1_result = open_aq.query_to_pandas_safe(query1, max_gb_scanned=0.1)


# Now I've got a dataframe called query1_result. We can take a look how our query works by simply typing the dataframe name

# In[ ]:


query1_result


# # WHERE
# 
# The next SQL statement that we're gonna look is the` WHERE` statement. The`WHERE` syntax is used to extract only those records that fulfill a specified condition. 
# 
# For example, let's take a look of the name of the city in the US where the pollutan value is over 100. We can simply just make another SQL query.

# In[ ]:


# query to select all items in "city", "pollutant" and "value" column which has 
# pollutant  value more than 100 and  have US as the country
query2 = """SELECT city, pollutant, value
            FROM `bigquery-public-data.openaq.global_air_quality`
            WHERE value > 100 AND country='US'
          """


# In[ ]:


query2_result = open_aq.query_to_pandas_safe(query2)


# In[ ]:


query2_result


# That's it for the first part of the SQL basics, see you again in the next part where we'll talk about `aggregation` and groupby statement
