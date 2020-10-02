#!/usr/bin/env python
# coding: utf-8

# # Question 1 - How many Bitcoin transactions were made each day in 2017?
# You can use the "timestamp" column from the "transactions" table to answer this question. 
# 

# In[ ]:


import pandas as pd
import bq_helper


# In[ ]:


btc = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                dataset_name="bitcoin_blockchain")


# In[ ]:


btc.list_tables()


# only two tables in the one, blocks and transactions

# In[ ]:


trans_dat = btc.head('transactions')


# In[ ]:


trans_dat


# In[ ]:


trans_dat.columns


# In[ ]:


#How many Bitcoin transactions were made each day in 2017?
q1 = """
WITH time AS(
    SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
        transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    )            
    SELECT COUNT(transaction_id) AS transactions,
        EXTRACT(DAY FROM trans_time) AS day,
        EXTRACT(MONTH FROM trans_time) AS month
    FROM time
    WHERE EXTRACT(YEAR FROM trans_time) = 2017
    GROUP BY month, day
    ORDER BY month, day
"""


# In[ ]:


btc.estimate_query_size(q1)


# In[ ]:


q1_ans = btc.query_to_pandas_safe(q1 ,max_gb_scanned=21)


# In[ ]:


q1_ans


# # Question 2 - How many transactions are associated with each merkle root?
# You can use the "merkle_root" and "transaction_id" columns in the "transactions" table to answer this question.

# In[ ]:


trans_dat


# In[ ]:


q2 = """
    SELECT merkle_root AS merkle, COUNT(transaction_id) AS transactions
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    GROUP BY merkle
    ORDER BY transactions
""" 


# In[ ]:


btc.estimate_query_size(q2)


# In[ ]:


q2_ans = btc.query_to_pandas_safe(q2 ,max_gb_scanned=37)


# In[ ]:


q2_ans

