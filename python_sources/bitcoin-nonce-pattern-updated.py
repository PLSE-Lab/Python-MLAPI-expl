#!/usr/bin/env python
# coding: utf-8

# An interesting pattern was found by [planB](https://twitter.com/100trillionUSD) in bitcoin's nonce, which is considered to be uniform distribution while it was not. TL;DR - See the plot at the bottom of this kernel.
# 
# [Nonce](https://en.bitcoinwiki.org/wiki/Nonce) is a parameter in a bitcoin's block where a miner adjusts it to find a block hash whose number of leading zero is more than the one agreed by bitcoin protocol (called "difficulty". The more leading zeros means more difficult to find such a hash). Strictly speaking it doesn't have to be uniformly distributed since a miner can choose it at will; however, any bias (such as incremental search from zero) might conflict with competing miners' search, ending up with a less success rate to win the mining race. In other words, an interest here is why we see a pattern implying many miners have the same bias in common at the risk of lowering the success rate - or a simple bug in commonly used ASICs? What do you think?
# 
# The pattern was originally found by [this tweet](https://twitter.com/100trillionUSD/status/1081217034485149697).
# Patterns are explored in other blockchains as well:
# 
# * [BCH](https://twitter.com/goldenmiffy/status/1081522233040723968)
# * [XMR](https://twitter.com/khannib/status/1082280569449447424)
# * [ETH](https://twitter.com/StopAndDecrypt/status/1082291450619088897), notice the slope lines at the bottom around 3,000,000.
# * [MONA](https://twitter.com/visvirial/status/1082825684572024832)
# 
# The discussions on why spread in different tweets or [reddit](https://www.reddit.com/r/Bitcoin/comments/adddja/the_weird_nonce_pattern/). The focus is more around  the pattern of four blank bands on the right hand side; it looks as if miners skipped certain nonce ranges.
# 
# In this reproduce case, because the dataset available doesn't seem to have block height which the original tweet used, I used block timestamp here instead. The result looks the same.
# 
# **Updated as the data schema has changed**

# In[ ]:


from google.cloud import bigquery
import pandas as pd
client = bigquery.Client()
query = "select date(timestamp) as date, nonce,number from `bigquery-public-data.crypto_bitcoin.blocks`"


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
plt.title("btc");
plt.xlabel("block timestamp");
plt.ylabel("nonce");

