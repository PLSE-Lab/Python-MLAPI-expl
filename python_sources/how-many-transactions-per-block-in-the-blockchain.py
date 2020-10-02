#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")


# In[ ]:


query = """
#standardSQL
SELECT 
   T.block_id
  --,ROUND((SUM(O.output_satoshis) / 100000000), 3) AS tot_BTC
  ,COUNT(DISTINCT T.transaction_id) AS num_transactions

        FROM `bigquery-public-data.bitcoin_blockchain.transactions` T
        CROSS JOIN UNNEST(outputs) O
        GROUP BY  T.block_id
        ORDER BY block_id DESC 

"""
bitcoin_blockchain.estimate_query_size(query)


# In[ ]:


results = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=100)


# In[ ]:


results.to_csv("output.csv")


# In[ ]:


results.head()


# In[ ]:




