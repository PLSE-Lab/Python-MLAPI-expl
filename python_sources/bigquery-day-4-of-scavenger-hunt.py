#!/usr/bin/env python
# coding: utf-8

# ## 1) How many Bitcoin transactions were made each day in 2017?

# In[ ]:


import bq_helper as bq


# In[ ]:


bitcoin_blockchain = bq.BigQueryHelper(active_project = "bigquery-public-data",
                                      dataset_name = "bitcoin_blockchain")


# In[ ]:


query = """WITH time AS
           (
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
           ORDER BY month, day ASC    
"""


# In[ ]:


bitcoin_blockchain.estimate_query_size(query)


# In[ ]:


transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query,max_gb_scanned=22)


# In[ ]:


transactions_per_day


# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (18,9)
plt.plot(transactions_per_day.transactions)
plt.title("Daily Transactions for 2017")


# ## 2) How many transactions are associated with each merkle root?

# In[ ]:


query2 = """SELECT COUNT(transaction_id) AS transactions,
                   merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions DESC
"""


# In[ ]:


bitcoin_blockchain.estimate_query_size(query2)


# In[ ]:


transactions_per_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned=38)


# In[ ]:


transactions_per_merkle_root.head(10)


# In[ ]:


len(transactions_per_merkle_root)


# In[ ]:




