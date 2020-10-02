#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")


# Then write the code to answer the questions below.
# 
# #### Note
# This dataset is bigger than the ones we've used previously, so your queries will be more than 1 Gigabyte. You can still run them by setting the "max_gb_scanned" argument in the `query_to_pandas_safe()` function to be large enough to run your query, or by using the `query_to_pandas()` function instead.
# 
# ## Questions
# #### 1) How many Bitcoin transactions were made each day in 2017?
# * You can use the "timestamp" column from the "transactions" table to answer this question. You can go back to the [order-by tutorial](https://www.kaggle.com/dansbecker/order-by) for more information on timestamps.

# In[ ]:


bitcoin_blockchain.list_tables()


# In[ ]:


bitcoin_blockchain.head('transactions')


# In[ ]:


query1 = '''WITH time AS
            (
             SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, 
                    transaction_id
             FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             )
             SELECT COUNT(transaction_id) AS transactions,
                    EXTRACT(DAYOFYEAR FROM trans_time) AS day,
                    EXTRACT(YEAR FROM trans_time) AS year
             FROM time
             GROUP BY year, day
             ORDER BY year, day
         '''


# In[ ]:


daily_transactions = bitcoin_blockchain.query_to_pandas(query1)


# In[ ]:


daily_transactions.head(10)


# In[ ]:


daily_transactions_2017 = daily_transactions[daily_transactions.year == 2017]
daily_transactions_2017.head()


# In[ ]:


import matplotlib.pyplot as plt
x = daily_transactions_2017.day
y = daily_transactions_2017.transactions
plt.figure(figsize = (12,6))
plt.plot(x,y)
plt.title('Bitcoin transactions made each day in 2017')
plt.xlabel('day')
plt.ylabel('transactions')
plt.show()


# 
# #### 2) How many transactions are associated with each merkle root?
# * You can use the "merkle_root" and "transaction_id" columns in the "transactions" table to answer this question. 

# In[ ]:


query2 = '''SELECT COUNT(transaction_id) AS Transactions, 
                   merkle_root AS Merkle_Root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY Transactions
         '''


# In[ ]:


merkle = bitcoin_blockchain.query_to_pandas(query2)


# In[ ]:


merkle


# ---
# # Keep Going
# [Click here](https://www.kaggle.com/dansbecker/joining-data) to learn how to combine multiple data sources with the JOIN command.
# 
# # Feedback
# Bring any questions or feedback to the [Learn Discussion Forum](kaggle.com/learn-forum).
# 
# ----
# 
# *This tutorial is part of the [SQL Series](https://www.kaggle.com/learn/sql) on Kaggle Learn.*
