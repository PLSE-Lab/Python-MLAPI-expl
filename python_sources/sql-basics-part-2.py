#!/usr/bin/env python
# coding: utf-8

# # INTRO
# ___
# 
# After we learn the most basic SQL statement which is `SELECT`, we'll take a look at some more  of the basic statements of SQL. We're still using `openaq` dataset  which contain one table called `global_air_quality`. Just like the first part, we'll need to set things up first before we can start applying the SQL query

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


# # AGGREGATE FUNCTION
# 
# An aggregate function  allows you to perform a calculation on a set of values to return a single  value. Usually it will be followed by GROUPBY statement. The most commonly used aggregate function in SQL are `SUM(), MAX(), MIN(), AVG(), COUNT()`
# 
# Say we want to find out how many different countries are in the dataset. The easiest way is using the `COUNT()` function and combining it with the `DISTINCT()` statement. The `DISTINCT()` statement is used to prevent the query of counting the same country twice because it will only return different values.
# 
# 
# ** > We need to put our SQL statements inside the quotation mark ("""   """) **

# In[ ]:


# query to count number of different countries
query1 = """SELECT COUNT(DISTINCT(country))
            FROM `bigquery-public-data.openaq.global_air_quality`
        """


# In[ ]:


# only run this query if it's less than 100 MB
query1_result = open_aq.query_to_pandas_safe(query1, max_gb_scanned=0.1)


# In[ ]:


#check the query result which is now a dataframe
query1_result


# The result seems fine, except that the colulmn name is `f0_` which is a little confusing, we can rename the column using aliases in our query

# In[ ]:


# query to count number of different countries
query2 = """SELECT COUNT(DISTINCT(country)) AS number_of_countries
            FROM `bigquery-public-data.openaq.global_air_quality`
        """


# In[ ]:


# only run this query if it's less than 100 MB
query2_result = open_aq.query_to_pandas_safe(query2, max_gb_scanned=0.1)


# In[ ]:


#check the query result which is now a dataframe
query2_result


# That looks better, SQL aliases basically are used to give a table, or a column in a table, a temporary name. This is very useful so that we don't get confused because it makes the table more readable.

# # GROUP BY and ORDER 

# The `GROUP BY` statement allows you to separate data into groups, which can be aggregated independently of one another. It is often used with aggregate functions` (COUNT, MAX, MIN, SUM, AVG)` to group the result-set by one or more columns.
# 
# For example, if you want to see list of countries and the average value of pollutant for each of the countries, you can do that using `GROUP BY` statement

# In[ ]:


# query to select the country and its average value of pollutant
# columns that are not included within an aggregate function and must be included in the GROUP BY 
query3 = """SELECT country, AVG(value)
            FROM `bigquery-public-data.openaq.global_air_quality`
            GROUP BY country
            ORDER BY 2 DESC
          """


# In[ ]:


# only run this query if it's less than 100 MB
query3_result = open_aq.query_to_pandas_safe(query3, max_gb_scanned=0.1)


# In[ ]:


#check the query result which is now a dataframe
query3_result


# Now we have a dataframe that shows list of every countries with the average pollutant

# That's it for the second part of the SQL basics, hope to see you again in the next part where we'll talk about subquery.
