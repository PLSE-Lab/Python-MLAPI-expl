#!/usr/bin/env python
# coding: utf-8

# In this starter kernel, I'll show you how to use the BigQuery Python clienty library in Kernels to query data from the Bitcoin Blockchain. We'll examine how many addresses receive bitcoin daily.
# 
# For general resources about working with BigQuery datasets on Kaggle, check out [this forum post](https://www.kaggle.com/product-feedback/48573).

# In[20]:


from google.cloud import bigquery
import pandas as pd

client = bigquery.Client()

# Query by Allen Day, GooglCloud Developer Advocate (https://medium.com/@allenday)
query = """
#StandardSQL
SELECT o.day, COUNT(DISTINCT(o.output_key)) AS recipients FROM
(
SELECT TIMESTAMP_MILLIS((timestamp - MOD(timestamp,86400000))) AS day, outputs.pubkey_base58 AS output_key, satoshis 
FROM 
  `bitcoin-bigquery.bitcoin.blocks` 
    JOIN
  UNNEST(transactions) AS transactions
    JOIN 
  UNNEST(transactions.outputs) AS outputs
) AS o
GROUP BY day
ORDER BY day
"""

query_job = client.query(query)

iterator = query_job.result(timeout=30)
rows = list(iterator)

# Transform the rows into a nice pandas dataframe
transactions = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Look at the first 10 headlines
transactions.head(10)


# In the first ten days of the Bitcoin Blockchain, only one transaction was recorded on January 16th, 2009. This seems to jibe with the history of Bitcoin as noted [here](https://en.wikipedia.org/wiki/History_of_bitcoin).
# 
# Let's plot the data over time to visualize the growth of Bitcoin (and the Blockchain).

# In[21]:


import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

transactions.plot()


# Unsurprisingly, the number of daily recipients has grown dramatically. I encourage readers to fork this kernel and improve my absolutely basic plot and/or explore additional questions in this fascinating dataset.
