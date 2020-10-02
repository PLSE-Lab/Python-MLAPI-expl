#!/usr/bin/env python
# coding: utf-8

# # Importing Packages and Datasets

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")


# ## Example: How many Bitcoin transactions are made per month?

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

transactions_per_month.head()


# ### Data Plotting

# In[ ]:


# import plotting library
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,8)
# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")


# # Scavenger hunt
# 
# ## Question 1 : How many Bitcoin transactions were made each day in 2017?
# >     * You can use the "timestamp" column from the "transactions" table to answer this question. You can check the [notebook from Day 3](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-3/) for more information on timestamps.

# In[ ]:


# Question 1 : How many Bitcoin transactions were made each day in 2017?

# Answer 1 : Find the query below

query_1 = """ WITH time AS 
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
            GROUP BY day, year, month
            HAVING year = 2017
            ORDER BY year, month, day
        """


transactions_in_2017 = bitcoin_blockchain.query_to_pandas(query_1)

transactions_in_2017.head()


# ### Data Plotting

# In[ ]:


# plot the data
# import plotting library
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,8)

# plot monthly bitcoin transactions
plt.plot(transactions_in_2017.transactions)
plt.title('Daily Number of Transcations in 2017')


# # Scavenger hunt
# 
# ## Question 2 : How many blocks are associated with each merkle root?
#     * You can use the "merkle_root" and "block_id" columns in the "blocks" table to answer this question. 

# In[ ]:


# Question 2 : How many blocks are associated with each merkle root?

# Answer 2 : Find the query below

query_2 = """
            SELECT 
                COUNT(DISTINCT merkle_root) AS Merkle_Root, 
                COUNT(DISTINCT block_id) as Blocks
            FROM `bigquery-public-data.bitcoin_blockchain.blocks`
            ORDER BY merkle_root 
         """

blocks_vs_merkle_root = bitcoin_blockchain.query_to_pandas(query_2)

blocks_vs_merkle_root.head()


# Here we see that there is the same number of blocks as merkel roots. Now we have to check how many blocks are assignated to each merkel root.

# In[ ]:


query_2a = """ WITH merkels AS 
            (
                SELECT merkle_root,
                    COUNT(DISTINCT block_id) AS blocks
                FROM `bigquery-public-data.bitcoin_blockchain.blocks`
                GROUP BY merkle_root
            )
            SELECT COUNT(blocks) AS blocks,
                MIN(blocks) AS max_merkle_blocks,
                MAX(blocks) AS min_merkle_blocks
            FROM merkels
        """
# note that max_gb_scanned is set to 21, rather than 1
#block_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
block_per_merkle = bitcoin_blockchain.query_to_pandas(query_2a)


# In[ ]:


block_per_merkle.head()


# Hence, we conclude that there is only one block associated with each merkel_block.
