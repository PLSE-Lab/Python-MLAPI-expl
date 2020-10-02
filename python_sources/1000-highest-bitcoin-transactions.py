#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import our bq_helper package
import bq_helper
# create a helper object for our bigquery dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", dataset_name = "bitcoin_blockchain")


# In[ ]:


query = """#standardSQL
        SELECT 
               T.block_id
              ,T.transaction_id
              ,ROUND((SUM(O.output_satoshis) / 100000000), 3) AS tot_BTC
              --,COUNT(DISTINCT I.input_pubkey_base58) AS num_inputs
              ,COUNT(O.output_satoshis) AS num_outputs
                    FROM `bigquery-public-data.bitcoin_blockchain.transactions` T
                    CROSS JOIN UNNEST(outputs) O
                    --CROSS JOIN UNNEST(inputs) I
                    GROUP BY  T.block_id, T.transaction_id
                    ORDER BY tot_BTC DESC 
                    LIMIT 100
             """


# In[ ]:


bitcoin_blockchain.estimate_query_size(query)


# In[ ]:


results = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=100)


# In[ ]:


results.to_csv("satoshis.csv")


# In[ ]:


results.head()


# In[ ]:


import matplotlib.pyplot as plt
amounts = results["tot_BTC"]
plt.hist(amounts, bins = 100)


# In[ ]:




