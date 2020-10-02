#!/usr/bin/env python
# coding: utf-8

# **[SQL Micro-Course Home Page](https://www.kaggle.com/learn/SQL)**
# 
# ---
# 

# # Introduction
# 
# So far, you've learned how to use several SQL clauses.  For instance, you know how to use **SELECT** to pull specific columns from a table, along with **WHERE** to pull rows that meet specified criteria.  You also know how to use aggregate functions like **COUNT()**, along with **GROUP BY** to treat multiple rows as a single group.  
# 
# Now you'll learn how to change the order of your results using the **ORDER BY** clause, and you'll explore a popular use case by applying ordering to dates.  To illustrate what you'll learn in this tutorial, we'll work with a slightly modified version of our familiar `pets` table.
# 
# ![](https://i.imgur.com/b99zTLv.png)
# 
# # ORDER BY
# 
# **ORDER BY** is usually the last clause in your query, and it sorts the results returned by the rest of your query.
# 
# Notice that the rows are not ordered by the `ID` column.  We can quickly remedy this with the query below.
# 
# ![](https://i.imgur.com/6o9LuTA.png)
# 
# The **ORDER BY** clause also works for columns containing text, where the results show up in alphabetical order.
# 
# ![](https://i.imgur.com/ooxuzw3.png)
# 
# You can reverse the order using the **DESC** argument (short for 'descending').  The next query sorts the table by the `Animal` column, where the values that are last in alphabetic order are returned first.
# 
# ![](https://i.imgur.com/IElLJrR.png)
#  
# # Dates
# 
# Next, we'll talk about dates, because they come up very frequently in real-world databases. There are two ways that dates can be stored in BigQuery: as a **DATE** or as a **DATETIME**. 
# 
# The **DATE** format has the year first, then the month, and then the day. It looks like this:
# 
# ```
# YYYY-[M]M-[D]D
# ```
#     
# * `YYYY`: Four-digit year
# * `[M]M`: One or two digit month
# * `[D]D`: One or two digit day
# 
# So `2019-01-10` is interpreted as January 10, 2019.
# 
# The **DATETIME** format is like the date format ... but with time added at the end.
# 
# # EXTRACT
# 
# Often you'll want to look at part of a date, like the year or the day. You can do this with **EXTRACT**.  We'll illustrate this with a slightly different table, called `pets_with_date`.
# 
# ![](https://i.imgur.com/vhvHIh0.png)
# 
# The query below returns two columns, where column `Day` contains the day corresponding to each entry the `Date` column from the `pets_with_date` table: 
#             
# ![](https://i.imgur.com/PhoWBO0.png)
# 
# SQL is very smart about dates, and we can ask for information beyond just extracting part of the cell. For example, this query returns one column with just the week in the year (between 1 and 53) for each date in the `Date` column: 
# 
# ![](https://i.imgur.com/A5hqGxY.png)
# 
# You can find all the functions you can use with dates in BigQuery in [this documentation](https://cloud.google.com/bigquery/docs/reference/legacy-sql#datetimefunctions) under "Date and time functions".  

# # Example: Which day of the week has the most fatal motor accidents?
# 
# Let's use the US Traffic Fatality Records database, which contains information on traffic accidents in the US where at least one person died.
# 
# We'll investigate the `accident_2015` table. Here is a view of the first few rows.  (_We have hidden the corresponding code. To take a peek, click on the "Code" button below._)

# In[ ]:


from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "nhtsa_traffic_fatalities" dataset
dataset_ref = client.dataset("nhtsa_traffic_fatalities", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "accident_2015" table
table_ref = dataset_ref.table("accident_2015")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "accident_2015" table
client.list_rows(table, max_results=5).to_dataframe()


# Let's use the table to determine how the number of accidents varies with the day of the week.  Since:
# - the `consecutive_number` column contains a unique ID for each accident, and
# - the `timestamp_of_crash` column contains the date of the accident in DATETIME format,
# 
# we can:
# - **EXTRACT** the day of the week (as `day_of_week` in the query below) from the `timestamp_of_crash` column, and
# - **GROUP BY** the day of the week, before we **COUNT** the `consecutive_number` column to determine the number of accidents for each day of the week.
# 
# Then we sort the table with an **ORDER BY** clause, so the days with the most accidents are returned first.

# In[ ]:


# Query to find out the number of accidents for each day of the week
query = """
        SELECT COUNT(consecutive_number) AS num_accidents, 
               EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week
        FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
        GROUP BY day_of_week
        ORDER BY num_accidents DESC
        """


# As usual, we run it as follows:

# In[ ]:


# Set up the query
query_job = client.query(query)

# API request - run the query, and convert the results to a pandas DataFrame
accidents_by_day = query_job.to_dataframe()

# Print the DataFrame
accidents_by_day


# Notice that the data is sorted by the `num_accidents` column, where the days with more traffic accidents appear first.
# 
# To map the numbers returned for the `day_of_week` column to the actual day, you might consult [the BigQuery documentation](https://cloud.google.com/bigquery/docs/reference/legacy-sql#dayofweek) on the DAYOFWEEK function. It says that it returns "an integer between 1 (Sunday) and 7 (Saturday), inclusively". So, in 2015, most fatal motor accidents in the US occured on Sunday and Saturday, while the fewest happened on Tuesday.

# # Your Turn
# **ORDER BY** can make your results easier to interpret. **[Try it yourself](https://www.kaggle.com/kernels/fork/682087)**.
# 

# ---
# **[SQL Micro-Course Home Page](https://www.kaggle.com/learn/SQL)**
# 
# 
