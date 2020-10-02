#!/usr/bin/env python
# coding: utf-8

# ## Environment Setup and Overview

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head("transactions")


# ## How many Bitcoin transactions are made per month?

# Since the timestamps of transactions are stored as integers, we need first to convert them into valid timestamps before we extract the month from each. Then, we can query the number of transactions being made each month.
# 
# This can be accomplished in one query statement split into two parts using the *WITH*...*AS* clauses as shown below.

# In[ ]:


query = """
        WITH time AS
        (
            SELECT TIMESTAMP_MILLIS(timestamp) as trans_time, transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`        
        )
        
        SELECT EXTRACT(YEAR FROM trans_time) as Year,
            EXTRACT (MONTH FROM trans_time) as Month,
            COUNT(transaction_id) as Transactions
        FROM time
        GROUP BY Year, Month
        ORDER BY Year, Month
        """
trans_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned = 21)
trans_per_month.head(10)


# In[ ]:


# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(trans_per_month.Transactions)
plt.title("Monthly Bitcoin Transcations")


# ## SQL Scavenger Hunt: Day 4 Task

# **1. How many Bitcoin transactions were made each day in 2017?**

# In[ ]:


query = """
        WITH time AS
            (
            SELECT TIMESTAMP_MILLIS(timestamp) as trans_time, transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
        SELECT 
            EXTRACT(YEAR FROM trans_time) as Year,
            EXTRACT(MONTH FROM trans_time) as Month,
            EXTRACT(DAY FROM trans_time) as Day,
            count(transaction_id) as Transactions
        FROM time
        GROUP BY Year, Month, Day
        HAVING Year = 2017
        ORDER BY Year, Month, Day        
        """
trans_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned = 40)
trans_per_day


# In[ ]:


# plot daily bitcoin transactions
plt.plot(trans_per_day.Transactions)
plt.title("Daily Bitcoin Transcations")


# **2. How many transactions are associated with each merkle root?**

# In[ ]:


query = """
        SELECT merkle_root as MerkleRoots, count(transaction_id) as Transactions
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY MerkleRoots
        ORDER BY Transactions DESC
        """
trans_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned = 40)
trans_per_merkle.head(20)


# 
