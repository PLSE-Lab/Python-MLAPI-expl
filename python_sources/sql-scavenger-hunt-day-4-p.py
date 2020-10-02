#!/usr/bin/env python
# coding: utf-8

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

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

bitcoin_blockchain.head('transactions')


# **How many Bitcoin transactions were made each day in 2017?**

# In[ ]:


query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DAY FROM trans_time) AS day,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY year, month, day 
            ORDER BY year, month, day
        """
# note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_per_day.head()


# In[ ]:


# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_day.transactions)
plt.title("Daily Bitcoin Transcations in 2017")


# **How many transactions are associated with each merkle root?**

# In[ ]:


query = """ 
           SELECT COUNT(transaction_id) AS num_trans, merkle_root
           FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           GROUP BY merkle_root 
           ORDER BY num_trans DESC
        """
# note that max_gb_scanned is set to 21, rather than 1
merkle_root_trans = bitcoin_blockchain.query_to_pandas(query)
merkle_root_trans.head()


# Please feel free to ask any questions you have in this notebook or in the [Q&A forums](https://www.kaggle.com/questions-and-answers)! 
# 
# Also, if you want to share or get comments on your kernel, remember you need to make it public first! You can change the visibility of your kernel under the "Settings" tab, on the right half of your screen.
