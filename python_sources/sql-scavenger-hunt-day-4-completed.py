#!/usr/bin/env python
# coding: utf-8

# <table>
#     <tr>
#         <td>
#         <center>
#         <font size="+5">SQL Scavenger Hunt: Day 4.</font>
#         </center>
#         </td>
#     </tr>
# </table>

# ## Example: How many Bitcoin transactions are made per month?
# ____
# 
# Now let's work through an example with a real dataset. Today, we're going to be working with a Bitcoin dataset (Bitcoin is a popular but volatile cryptocurrency). We're going to use a common table expression (CTE) to find out how many Bitcoin transactions were made per month for the entire timespan of this dataset.
# 
# First, just like the last three days, we need to get our environment set up:

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")


# Now we're going to write a query to get the number of transactions per month. One problem here is that this dataset uses timestamps rather than dates, and they're stored in this dataset as integers. We'll have to convert these into a format that BigQuery recognizes using TIMESTAMP_MILLIS(). We can do that using a CTE and then write a second part of the query against the new, temporary table we created. This has the advantage of breaking up our query into two, logical parts. 
# 
# * Convert the integer to a timestamp
# * Get information on the date of transactions from the timestamp
# 
# You can see the query I used to do this below.

# In[ ]:


query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month 
            ORDER BY year, month
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)


# Since they're returned sorted, we can just plot the raw results to show us the number of Bitcoin transactions per month over the whole timespan of this dataset.

# In[ ]:


# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")


# Pretty cool, huh? :)
# 
# As you can see, common table expressions let you shift a lot of your data cleaning into SQL. That's an especially good thing in the case of BigQuery because it lets you take advantage of BigQuery's parallelization, which means you'll get your results more quickly.

# # Scavenger hunt Questions ?
# 
# * How many Bitcoin transactions were made each day in 2017?
#     * You can use the "timestamp" column from the "transactions" table to answer this question. You can check the [notebook from Day 3](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-3/) for more information on timestamps.
# * How many transactions are associated with each merkle root?
#     * You can use the "merkle_root" and "transaction_id" columns in the "transactions" table to answer this question. 
#     * Note that the earlier version of this question asked "How many *blocks* are associated with each merkle root?", which would be one block for each root. Apologies for the confusion!
# 

# In[ ]:


query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) AS year,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                WHERE year = 2017
            )
            SELECT year,
                EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                COUNT(transaction_id) AS transactions
            FROM time
            GROUP BY year, day
            ORDER BY day            
        """


# In[ ]:


# note that max_gb_scanned is set to 21, rather than 1
transactions_on_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)


# In[ ]:


transactions_on_2017

