#!/usr/bin/env python
# coding: utf-8

# Thursday. About a quarter to nine. These are the voyages of the MacBook Pro Early 2011. It's continuing mission to explore strange new datasets, to seek out new insights and new other stuff, to continue to go (albeit increasingly slowly), where just a few thousand have gone before...
# 
# It's [day 4 of the SQL Scavenger Hunt][1]! Let's get it on.
# [1]: https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-4

# Today, it's all about **AS** and **WITH**. Let's hook up with our aliases...
# 
# Before we do that though, let's round up the usual suspects:

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")


# First scavenger hunt challenge: **How many Bitcoin transactions were made each day in 2017?**

# In[ ]:


query = """ WITH time AS 
            (
                SELECT DATE(TIMESTAMP_MILLIS(timestamp)) AS properTime,
                    transaction_id AS transactions
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            
            SELECT properTime AS day, COUNT(transactions) AS dailyTransactions
            FROM time
            WHERE EXTRACT(YEAR FROM properTime) = 2017
            GROUP BY day
            ORDER BY day
        """

# note that max_gb_scanned is set to 21, rather than 1
transPerDay = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)


# Time for the great reveal, well, the `head()` at least...

# In[ ]:


transPerDay.head()


# In[ ]:


# time to try to plot my first ever graph in Python...
# import plotting library
import matplotlib.pyplot as plt

# plot daily bitcoin transactions
plt.plot(transPerDay.dailyTransactions)
plt.title("Daily Bitcoin Transcations in 2017")


# Second scavenger hunt challenge: **How many transactions are associated with each merkle root?**

# In[ ]:


query2 = """SELECT merkle_root AS merkleRoot, COUNT(transaction_id) AS transPerMerk
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkleRoot
            ORDER BY transPerMerk DESC
        """
# fire up the query
merkles = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=40)

# cross your fingers...
merkles


# In[ ]:


# basic plot
plt.boxplot(merkles.transPerMerk)
plt.title("Distribution of Transactions Per Merkle Root")
plt.ylabel("Transactions")


# And there we have it, or, at least, I hope we do. 80% of the way through and my SQL is already 800% better.
# 
# Thanks Rachael and fellow kagglers
