#!/usr/bin/env python
# coding: utf-8

# <table>
#     <tr>
#         <td>
#         <center>
#         <font size="+1">If you haven't used BigQuery datasets on Kaggle previously, check out the <a href = "https://www.kaggle.com/rtatman/sql-scavenger-hunt-handbook/">Scavenger Hunt Handbook</a> kernel to get started.</font>
#         </center>
#         </td>
#     </tr>
# </table>
# 
# ___ 
# 
# ## Previous days:
# 
# * [**Day 1:** SELECT, FROM & WHERE](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-1/)
# * [**Day 2:** GROUP BY, HAVING & COUNT()](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-2/)
# * [**Day 3:** ORDER BY & Dates](https://www.kaggle.com/rtatman/sql-scavenger-hunt-day-3/)
# 
# ____
# 
# **Here are my answers for Day 4 of the SQL Scavenger Hunt!**
# 

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
#     * You can use the "merkle_root" and "block_id" columns in the "transactions" table to answer this question. 
#     * Note that the earlier version of this question asked "How many *blocks* are associated with each merkle root?", which would be one block for each root. Apologies for the confusion!
# 
# In order to answer these questions, you can fork this notebook by hitting the blue "Fork Notebook" at the very top of this page (you may have to scroll up). "Forking" something is making a copy of it that you can edit on your own without changing the original.

# In[ ]:


# import package with helper functions 
import bq_helper

#Import plotting library
import matplotlib.pyplot as plt

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")


# In[ ]:


transact_query = """ WITH transact AS 
            (
                SELECT DATE(TIMESTAMP_MILLIS(timestamp)) AS transaction_date,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT transaction_date AS date, COUNT(transaction_id) AS transactions
            FROM transact
            WHERE EXTRACT(YEAR FROM transaction_date) = 2017
            GROUP BY date 
            ORDER BY date
        """


# In[ ]:


transactions_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(transact_query, max_gb_scanned=21)


# In[ ]:


plt.figure(figsize=(12,6))
plt.plot(transactions_per_day_2017.date, transactions_per_day_2017.transactions)
plt.title('Bitcoin Transactions in 2017 Per Day')
plt.xlabel('Day in 2017')
plt.ylabel('Number of Transactions')


# In[ ]:


print(transactions_per_day_2017.head(10))
print()
print('Number of total entries : '+ str(len(transactions_per_day_2017)))


# In[ ]:


merkle_query = """ WITH m_root AS 
            (
                SELECT merkle_root AS Merkle,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS Transactions, merkle
            FROM m_root
            GROUP BY merkle 
            ORDER BY Transactions DESC
        """


# In[ ]:


blocks_per_merkle = bitcoin_blockchain.query_to_pandas_safe(merkle_query, max_gb_scanned=40)


# In[ ]:


top20_merkle = blocks_per_merkle.head(20)
print('Total Number of Different Merkle Roots : ' + str(len(blocks_per_merkle)))
print()
print('Top 20 Merkle Root Counts')
print(top20_merkle)


# Thank you for taking the time to look through my solutions, any feedback is greatly appreciated!  :) 
