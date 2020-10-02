#!/usr/bin/env python
# coding: utf-8

# I  found an interesting relationship between IP count and download.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
nrows=None
dtypes = {
        'ip'            : 'uint32',
        'is_attributed' : 'uint8',
        }
train_df = pd.read_csv('../input/train.csv',dtype=dtypes,nrows=nrows,usecols=['ip','is_attributed'])
ip_grp = train_df[['ip','is_attributed']].groupby(['ip']).agg(['count', 'sum']).is_attributed.sort_values(by='count').reset_index()
ip_grp.columns = ['ip', 'occurrences', 'download_count']
df = ip_grp[['download_count','occurrences']].groupby('occurrences').agg(['count', 'sum'])['download_count']
df.columns = ['num_IPs', 'num_downloads']
df['cvr_x_occurrences'] = df.num_downloads /df.num_IPs
df = df.reset_index()
df.head(10)


# Regardless of the number of occurrences, download count from the IP is one.
# 
# This trend applies to IPs with an occurrence count of 400 or less.

# In[2]:


thre = 400
_df = df[df.occurrences<thre].copy()
plt.plot(_df.occurrences, _df.cvr_x_occurrences)


# For example, even if he clicks 1 times or 3 times, he usually downloads once and only once.

# In[3]:


train_df[train_df.ip.isin(ip_grp[ip_grp.occurrences==1].head(10).ip)].sort_values(['ip','is_attributed'])


# In[4]:


train_df[train_df.ip.isin(ip_grp[ip_grp.occurrences==3].head(10).ip)].sort_values(['ip','is_attributed'])


# This tendency disappears when the number of appearances becomes 400 or more.

# In[5]:


thre = 3000
_df = df[df.occurrences<thre].copy()
_df['roll'] = _df.cvr_x_occurrences.rolling(window=int(10)).mean()
plt.plot(_df.occurrences, _df.roll)


# There is not much proportion of IP with occurrence count of 400 or less, 
# but the number of downloads is large.
# 
# As you can see below,  the percentage of clicks from the IP addresses with 50 clicks or less 
# is only 1%, but the percentage of download accounts for 40%. This is significant for AUC.

# In[6]:


sum_download = sum(df.num_downloads)
df['cum_download'] = df.num_downloads.cumsum()
df['cum_download_ratio'] = df['cum_download']/sum_download
df['pv'] = df.num_IPs*df.occurrences
sum_pv = sum(df.pv)
df['cum_pv'] = df.pv.cumsum()
df['cum_pv_ratio'] = df.cum_pv / sum_pv
df['cvr'] = df.num_downloads / df.pv
df[['occurrences','cum_download_ratio','cum_pv_ratio']].head(50)


# Probably this is the effect of fraud access.
# 
# Or I may possibly have some misunderstanding.

# In[ ]:




