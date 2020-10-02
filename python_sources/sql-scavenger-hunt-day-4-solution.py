#!/usr/bin/env python
# coding: utf-8

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

# # Scavenger hunt
# ___
# 
# > **Important note**: Today's dataset is bigger than the ones we've used previously, so your queries will be more than 1 Gigabyte. You can still run them by setting the "max_gb_scanned" argument in the `query_to_pandas_safe()` function to be large enough to run your query, or by using the `query_to_pandas()` function instead.
# 
# Now it's your turn! Here are the questions I would like you to get the data to answer. Practice using at least one alias in each query. 
# 
# * How many Bitcoin transactions were made each day in 2017?
#     * You can use the "timestamp" column from the "transactions" table to answer this question. You can check the [notebook from Day 3](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-3/) for more information on timestamps.
# * How many transactions are associated with each merkle root?
#     * You can use the "merkle_root" and "transaction_id" columns in the "transactions" table to answer this question. 
#     * Note that the earlier version of this question asked "How many *blocks* are associated with each merkle root?", which would be one block for each root. Apologies for the confusion!
# 
# In order to answer these questions, you can fork this notebook by hitting the blue "Fork Notebook" at the very top of this page (you may have to scroll up). "Forking" something is making a copy of it that you can edit on your own without changing the original.

# **Solution starts here!** First we'll see how many Bitcoin transactions took place on each day in 2017.

# In[ ]:


# Bitcoin transactions each day in 2017

dayTransactionsQuery = """WITH time AS
                        (
                            SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                        )
                    SELECT EXTRACT(DAYOFYEAR FROM trans_time) AS day, 
                        EXTRACT(YEAR FROM trans_time) AS year,
                        COUNT(transaction_id) AS transactions
                    FROM time
                    WHERE EXTRACT(YEAR FROM trans_time) = 2017
                    GROUP BY year, day
                    ORDER BY year, day
                    """ 

print(bitcoin_blockchain.estimate_query_size(dayTransactionsQuery))

dayTransactions = bitcoin_blockchain.query_to_pandas_safe(dayTransactionsQuery, max_gb_scanned=21)


# In[ ]:


print(dayTransactions.shape)

dayTransactions.head()


# Now we know that we selected the correct number of days (by extracting DAYOFYEAR, not DAY). And a plot of 2017 transactions by day:

# In[ ]:


dayTransactions.plot("day","transactions", title="Bitcoin Transactions Per Day in 2017")


# Next! Number of transactions per merkle root [(link for the uninitiated)](https://en.wikipedia.org/wiki/Merkle_tree):

# In[ ]:


transactionsPerMerkleQuery = """SELECT merkle_root, COUNT(transaction_id) AS transactions
                                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                                GROUP BY merkle_root
                                ORDER BY transactions DESC
                            """

transactionsPerMerkle = bitcoin_blockchain.query_to_pandas_safe(transactionsPerMerkleQuery, max_gb_scanned = 37)


# In[ ]:


transactionsPerMerkle.head()


# Well, that looks like what we want. Hooray again!
