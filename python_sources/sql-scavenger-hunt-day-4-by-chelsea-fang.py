#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import bq_helper as bq
bitcoin=bq.BigQueryHelper(active_project="bigquery-public-data",
                         dataset_name="bitcoin_blockchain")


# In[ ]:


query1="""
        WITH time AS
        (
            SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
        FROM time
        GROUP BY year, month
        ORDER BY year, month
"""
result1=bitcoin.query_to_pandas_safe(query1,max_gb_scanned=21)
print(result1)


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(result1.transactions)
plt.title("Monthly Bitcoin Transactions")


# In[ ]:


bitcoin.head('transactions')


# * How many Bitcoin transactions were made each day in 2017?
# *         You can use the "timestamp" column from the "transactions" table to answer this question. You can check the notebook from Day 3 for more information on timestamps.
# * How many transactions are associated with each merkle root?
# *         You can use the "merkle_root" and "transaction_id" columns in the "transactions" table to answer this question.
# *         Note that the earlier version of this question asked "How many blocks are associated with each merkle root?", which would be one block for each root. Apologies for the confusion!
# 

# In[ ]:


query2="""
        WITH year2017 AS
        (
            SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
         SELECT COUNT(transaction_id) as transaction, EXTRACT(DAYOFYEAR FROM trans_time) as day
         FROM year2017
         WHERE EXTRACT(YEAR FROM trans_time)=2017
         GROUP BY day    
         ORDER BY day
"""
result2=bitcoin.query_to_pandas_safe(query2,max_gb_scanned=21)
print(result2)


# In[ ]:


plt.plot(result2.transaction)


# * How many transactions are associated with each merkle root?
# *         You can use the "merkle_root" and "transaction_id" columns in the "transactions" table to answer this question.
# *         Note that the earlier version of this question asked "How many blocks are associated with each merkle root?", which would be one block for each root. Apologies for the confusion!

# In[ ]:


query3="""
        SELECT COUNT(transaction_id) as transaction, merkle_root
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY merkle_root
"""
result3=bitcoin.query_to_pandas_safe(query3,max_gb_scanned=38)
print(result3)

