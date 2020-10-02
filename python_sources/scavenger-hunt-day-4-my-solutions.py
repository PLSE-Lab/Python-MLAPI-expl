#!/usr/bin/env python
# coding: utf-8

# # Scavenger Hunt Day 4: WITH and AS
# 
# Today we are learning about aliasing, or providing shortcuts that can help with length and readability of SQL queries.  Let's roll.
# 
# ## Import the Libraries:

# In[ ]:


# import general libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import our trusty bigquery helper library
import bq_helper


# ## Create our bigquery helper instance:

# In[ ]:


# create helper
bit_helper = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                     dataset_name = 'bitcoin_blockchain')


# ## Examine our dataspace:

# In[ ]:


# Check out the tables in the space
bit_helper.list_tables()


# In[ ]:


# Check out the head of one of the tables
bit_helper.head('transactions')


# Enough piddling around, let's get on with it.
# 
# # Scavenger Hunt!
# 
# ## First Item:
# 
# How many Bitcoin transactions were made each day in 2017?
# * You can use the "timestamp" column from the "transactions" table to answer this question. 
# * You can check the notebook from Day 3 for more information on timestamps.
# 
# ## Generate and Run Query:

# In[ ]:


# build a query
query1 = """
        WITH dates AS
        (
        SELECT TIMESTAMP_MILLIS(timestamp) as datetime, transaction_id
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        )
        
        SELECT 
        EXTRACT(DATE from datetime) as date,
        COUNT(transaction_id) as transactions
        FROM dates
        WHERE EXTRACT(YEAR from datetime)=2017
        GROUP BY date
        ORDER BY date
"""

# check out data usage for our query
bit_helper.estimate_query_size(query1)


# That's a big query.  Let's go ahead and run it.  We'll still use ...to_pandas_safe, but we'll set the limit to 21GB to hopefully cover the query.

# In[ ]:


# run that suckah!
bit_transactions = bit_helper.query_to_pandas_safe(query1, max_gb_scanned=21)


# ## Check out the results

# In[ ]:


bit_transactions.head(10)


# ## Graph it!

# In[ ]:


# Let's add a 30-day moving average
# first use the 'date' column as the index
bit_transactions.set_index('date', inplace=True)
bit_transactions['30dMA'] = bit_transactions['transactions'].rolling(30).mean()

plt.style.use('ggplot')
bit_transactions.plot(figsize=(12,8))
plt.title('Bitcoin Transactions Per Day from {} to {}'.format(bit_transactions.index[0], bit_transactions.index[-1]))


# Just for fun, let's look at the maximum and minimum number of transactions during that period.

# In[ ]:


min = bit_transactions.transactions.min()
max = bit_transactions.transactions.max()
mindate = bit_transactions.transactions.argmin()
maxdate = bit_transactions.transactions.argmax()

print("The lowest number of transactions was {} which occurred on {}.".format(min, mindate))
print("The highest number of transactions was {} which occurred on {}.".format(max, maxdate))


# ## Second Item:
# How many transactions are associated with each merkle root?
# * You can use the "merkle_root" and "transaction_id" columns in the "transactions" table to answer this question.
# 
# ## Build and Run Query:

# In[ ]:


# build a query
query2 = """
        SELECT COUNT(transaction_id) as transactions, merkle_root
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY merkle_root
        ORDER BY transactions DESC
"""

# check the data usage of our query
bit_helper.estimate_query_size(query2)


# In[ ]:


# run that suckah!
trans_per_root = bit_helper.query_to_pandas_safe(query2, max_gb_scanned=37)


# In[ ]:


trans_per_root.head(10)


# In[ ]:


trans_per_root.tail(10)


# In[ ]:




