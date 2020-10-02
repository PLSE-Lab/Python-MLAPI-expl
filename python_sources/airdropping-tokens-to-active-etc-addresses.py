#!/usr/bin/env python
# coding: utf-8

# # Finding active ETC addresses
# 
# Let's find recently active ETC addresses and airdrop them some tokens!

# In[ ]:


import numpy as np
import pandas as pd
import os
from google.cloud import bigquery


# In[ ]:


client = bigquery.Client()
ethereum_classic_dataset_ref = client.dataset('crypto_ethereum_classic', project='bigquery-public-data')


# In[ ]:


query = """
SELECT from_address AS address
FROM `bigquery-public-data.crypto_ethereum_classic.transactions`
WHERE block_number > 7500000
GROUP BY from_address
"""

query_job = client.query(query)
iterator = query_job.result()


# In[ ]:


rows = list(iterator)
# Transform the rows into a nice pandas dataframe
active_users = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))


# In[ ]:


# Post-process: convert to airdrop-tool format
active_users['tokens'] = ['1000000000000000000' for x in rows] # to airdrop 1 token (with 18 decimals) to each address
active_users.shape


# ## Exporting
# 
# The produced json file can be used in the [airdrop-tool](https://forum.saturn.network/t/how-to-setup-and-launch-your-own-airdrop/2366). Simply download it from [here](https://www.kaggle.com/neuralhax0r/airdropping-tokens-to-active-etc-addresses).

# In[ ]:


out = active_users.to_json(orient='records')
with open('airdrop.json', 'w') as f:
    f.write(out)

