#!/usr/bin/env python
# coding: utf-8

# [The previous kernel](https://www.kaggle.com/smasuda/bitcoin-nonce-pattern) looked into bitcoin's nonce distribution over time. How about blockhash then?
# 
# It has been already [analyzed](https://en.bitcoin.it/wiki/Distribution_of_nonces_and_hashes), and this kernel aims to be a working reproduce case of the existing work.  Again, I used block timestamp as a proxy of block height.
# 
# As you can see in the histogram below, blockhash **is not** uniformly distributed among blocks; it is upper-bounded by [difficulty](https://en.bitcoin.it/wiki/Difficulty), which becomes yet more obvious if you plot it over time.
# 
# The interesting nonce pattern can also be observed in the last plot - although the point of the chart here would be to show how blockhash is uniformly distributed over nonce under a certain difficulty range.

# In[ ]:


from google.cloud import bigquery
import pandas as pd
client = bigquery.Client()
query = "select timestamp as timestamp, nonce,block_id from `bigquery-public-data.bitcoin_blockchain.blocks`"


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


#convert hexdecimal blockhash into int. Apply log scale for plot.
import math
df['blockhash_int'] = df['block_id'].apply(lambda x : int(x,16))
df['blockhash_log10'] = df['blockhash_int'].apply(lambda x : math.log10(x))

# remove the outliers
df = df[ (df['timestamp'] > 1230854400000) & (df['timestamp'] < 1600000000000) ]
df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms')


# In[ ]:


from matplotlib import pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 10

plt.hist(df['blockhash_log10'], bins=100, histtype='stepfilled')
plt.title('histogram of log10(blockhash)')
plt.xlabel('log10(blockhash)')
plt.ylabel('frequency')
plt.show();


# In[ ]:


plt.plot_date( df['timestamp'], df['blockhash_log10'], ms=0.1);
plt.title("log10(blockhash) vs block time");
plt.xlabel("block time");
plt.ylabel("blockhash in log10");


# In[ ]:


fig = plt.figure(figsize=(15,10))
sc=plt.scatter( df['nonce'], df['blockhash_log10'], s=0.01, c=df['timestamp']);
fig.colorbar(sc)
plt.title("log10(blockhash) vs nonce, color=block timestamp");
plt.xlabel("nonce");
plt.ylabel("blockhash in log10");
plt.show();

