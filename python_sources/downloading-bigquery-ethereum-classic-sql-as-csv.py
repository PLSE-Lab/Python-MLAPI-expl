#!/usr/bin/env python
# coding: utf-8

# # Using Etherum Classic Dataset to Download BigQuery SQL Output As CSV
# 
# In this notebook, we will be exploring how to download Etherum Classic queries as CSV from Kaggle.
# 

# In[ ]:


import numpy as np
import pandas as pd
from google.cloud import bigquery


# We will start out by composing an SQL query for the dataset.
# 
# We will then be storing the output of the query inside a Pandas Dataframe. After that, we will save our dataframe as a CSV.
# 
# The query we will be writing will get us the daily list of miners and how many blocks they mined.
# 
# We would write it out like this:
# ```
# SELECT miner, 
#     DATE(timestamp) as date,
#     COUNT(miner) as total_block_reward
# FROM `bigquery-public-data.crypto_ethereum_classic.blocks` 
# GROUP BY miner, date
# HAVING COUNT(miner) > 0
# ```

# In[ ]:


client = bigquery.Client()
sql = """
SELECT miner, 
    DATE(timestamp) as date,
    COUNT(miner) as total_block_reward
FROM `bigquery-public-data.crypto_ethereum_classic.blocks` 
GROUP BY miner, date
HAVING COUNT(miner) > 0
ORDER BY date, miner, total_block_reward DESC
"""

# Run a Standard SQL query using the environment's default project
df = client.query(sql).to_dataframe()
df


# In[ ]:


df.to_csv("etc.csv")


# In[ ]:




