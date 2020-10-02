#!/usr/bin/env python
# coding: utf-8

# This kernel is part 2 of 3, based on [code originally written](https://gist.github.com/allenday/1207a2c0b962d65b101d1e753ebc1c52) by [Allen Day](https://twitter.com/allenday) and modified by myself and Meg Risdal. It will be used to visualize a directed graph representing Bitcoin transactions that follow the first known exchange of Bitcoin for goods on May 17, 2010 made by [Laszlo Hanyecz](https://en.bitcoin.it/wiki/Laszlo_Hanyecz).
# 
# This stage handles the post-processing of data pulled from BigQuery [in step 1](https://www.kaggle.com/sohier/tracing-the-10-000-btc-pizza/). At the end, we'll have a graph of transactions to export to [a third kernel for plotting in R](https://www.kaggle.com/mrisdal/visualizing-the-10k-btc-pizza-transaction-network).

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('../input/transactions.csv')
df['date_time'] = pd.to_datetime(df.timestamp * 1000000)


# In[ ]:


df.head(3)


# In[ ]:


# the original pizza wallet
BASE_SEEDS = ['1XPTgDRhN8RFnzniWCddobD9iKZatrvH4']
SATOSHI_PER_BTC = 10**7


# In[ ]:


df.head(3)


# In[ ]:


def dig_row(row, seeds, transactions, min_satoshis, trace_from_key):
    if row['satoshis'] < min_satoshis:
        return None
    trace_columns = {True: 'input_key', False: 'output_key'}
    if row[trace_columns[trace_from_key]] not in seeds:
        return None
    seeds.add(row['output_key'])
    transactions.append(row)


def single_pass_dig(initial_seeds, input_df, initial_datetime=None, min_satoshis=0, trace_from_key=True):
    df = input_df.copy()
    active_seeds = {i for i in initial_seeds}
    if trace_from_key and initial_datetime is not None:
        df = df[df['date_time'] >= initial_datetime]
    elif not(trace_from_key) and initial_datetime is not None:
        df = df[df['date_time'] <= initial_datetime]
    df.sort_values(by=['timestamp'], ascending=trace_from_key, inplace=True)
    transactions = []
    df.apply(lambda row: dig_row(row, active_seeds, transactions, min_satoshis, trace_from_key), axis=1)
    return pd.DataFrame(transactions)


# In[ ]:


# setting seed date to +/- 1 day from the actual pizza transaction to avoid
# worrying about when in the day the pizza was purchased
future_transactions = single_pass_dig(BASE_SEEDS, df, initial_datetime=pd.to_datetime("May 16, 2010"))
past_transactions = single_pass_dig(BASE_SEEDS, df, initial_datetime=pd.to_datetime("May 18, 2010"), trace_from_key=False)


# In[ ]:


total_flows = future_transactions[['input_key', 'output_key', 'satoshis']].groupby(
    by=['input_key', 'output_key']).sum().reset_index()
total_flows.head(3)


# In[ ]:


total_flows.to_csv('total_flows.csv', index=False)


# In[ ]:


total_flows.info()

