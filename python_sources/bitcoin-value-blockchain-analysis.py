#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from sklearn import preprocessing

pd.set_option('display.float_format', lambda x: '%.4f' % x)


# In[ ]:


dfBlockchainData = pd.read_csv('../input/bitcoin_dataset.csv')
dfBlockchainData.set_index('Date', inplace=True)


# In[ ]:


nRow, nCol = dfBlockchainData.shape
minFilled = dfBlockchainData.describe()[(dfBlockchainData.describe().index == 'count')].T.min()['count']
print(f'Dataset contains {nRow} lines (at least {((minFilled/nRow) * 100).round(4)}% completed) and {nCol} columns, described below')
print(dfBlockchainData.info())


# ## Column Definition 
# 
# btc_market_price
# 
# btc_total_bitcoins
# 
# btc_market_cap
# 
# btc_trade_volume
# 
# btc_blocks_size
# 
# btc_avg_block_size
# 
# btc_n_orphaned_blocks
# 
# btc_n_transactions_per_block
# 
# btc_median_confirmation_time
# 
# btc_hash_rate
# 
# btc_difficulty
# 
# btc_miners_revenue
# 
# btc_transaction_fees
# 
# btc_cost_per_transaction_percent
# 
# btc_cost_per_transaction
# 
# btc_n_unique_addresses
# 
# btc_n_transactions
# 
# btc_n_transactions_total
# 
# btc_n_transactions_excluding_popular
# 
# btc_n_transactions_excluding_chains_longer_than_100
# 
# btc_output_volume
# 
# btc_estimated_transaction_volume
# 
# btc_estimated_transaction_volume_usd
