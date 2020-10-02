#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import bq_helper
import matplotlib.pyplot as plt






# In[ ]:


bitcoin = bq_helper.BigQueryHelper(active_project='bigquery-public-data',dataset_name='bitcoin_blockchain')


# In[ ]:


bitcoin.table_schema('blocks')


# # Question 1 : How many Bitcoin transactions were made each day in 2017?

# In[ ]:


bitcoin.table_schema('transactions')


# In[ ]:


query = """WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,EXTRACT(DATE FROM trans_time) AS DATE FROM time
            GROUP BY DATE 
            ORDER BY DATE """


# In[ ]:


bitcoin.estimate_query_size(query)


# In[ ]:


date_trend_transactions = bitcoin.query_to_pandas_safe(query,max_gb_scanned=21)


# In[ ]:


date_trend_transactions.head()


# In[ ]:


plt.plot(date_trend_transactions.transactions)
plt.title("Transaction trend -Day wise")


# # Question 2 - How many transactions are associated with each merkle root?

# In[ ]:


query=""" WITH merkle as (SELECT merkle_root, transaction_id as transaction from `bigquery-public-data.bitcoin_blockchain.transactions`) SELECT count(transaction) as transaction_count , merkle_root as merkle from merkle group by merkle """


# In[ ]:


bitcoin.estimate_query_size(query)


# In[ ]:


bitcoin.query_to_pandas_safe(query,max_gb_scanned=50)


# End of Day 4 ...Awaiting for more...
