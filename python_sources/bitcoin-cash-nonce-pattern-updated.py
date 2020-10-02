#!/usr/bin/env python
# coding: utf-8

# Please [read this](https://www.kaggle.com/smasuda/bitcoin-nonce-pattern-updated) for background.
# 

# In[ ]:


from google.cloud import bigquery
import pandas as pd
client = bigquery.Client()
query = "select date(timestamp) as date, nonce,number from `bigquery-public-data.crypto_bitcoin_cash.blocks`"


# In[ ]:


query_job = client.query(query)
df = query_job.to_dataframe()


# In[ ]:


#check the first 5 result, to ensure we get what we want.
#additionally do data sanity check by looking at block exploerers such as 
#https://blockexplorer.com/block/{block_id}

pd.set_option('display.max_colwidth', -1)
df.head(5)


# In[ ]:


import struct
def noncehex_to_int(nonce):
    if len(nonce) < 8:
        nonce = ("0"  * (8- len(nonce))) + nonce
        
    try:
        (ret, ) = struct.unpack('>I', bytes.fromhex(nonce))
        return ret
    except:
        print(nonce)
        
df['nonce_int'] = df['nonce'].map(noncehex_to_int)


# In[ ]:


from matplotlib import pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 10
plt.plot_date( df['date'], df['nonce_int'], ms=0.1);
plt.title("bch");
plt.xlabel("block timestamp");
plt.ylabel("nonce");

