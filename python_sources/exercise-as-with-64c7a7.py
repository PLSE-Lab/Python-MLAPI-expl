#!/usr/bin/env python
# coding: utf-8

# # Get Started
# 
# After forking this notebook, run the code in the following cell:

# In[ ]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head("transactions",5)


# Then write the code to answer the questions below.
# 
# #### Note
# This dataset is bigger than the ones we've used previously, so your queries will be more than 1 Gigabyte. You can still run them by setting the "max_gb_scanned" argument in the `query_to_pandas_safe()` function to be large enough to run your query, or by using the `query_to_pandas()` function instead.
# 
# ## Questions
# #### 1) How many Bitcoin transactions were made each day in 2017?
# * You can use the "timestamp" column from the "transactions" table to answer this question. You can go back to the [order-by tutorial](https://www.kaggle.com/dansbecker/order-by) for more information on timestamps.

# In[ ]:


query = '''
            with time as
            (
                select TIMESTAMP_MILLIS(timestamp)as trans_time,transaction_id from `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            select count(transaction_id) as count_transid,
                    extract(day from trans_time) as days,
                    extract(month from trans_time) as months,
                    extract(year from trans_time) as year
                    from time
                    group by year,months,days
                    having year = 2017
                    order by year,months,days
        '''
bitcoin = bitcoin_blockchain.query_to_pandas_safe(query,max_gb_scanned=23)
bitcoin.head()


# #### 2) How many transactions are associated with each merkle root?
# * You can use the "merkle_root" and "transaction_id" columns in the "transactions" table to answer this question. 

# In[ ]:


query2 = '''select merkle_root,count(transaction_id) as transaction from `bigquery-public-data.bitcoin_blockchain.transactions` group by merkle_root order by transaction desc
            '''
trans_per_root = bitcoin_blockchain.query_to_pandas_safe(query2, max_gb_scanned = 45)
trans_per_root.head()


# ---
# # Keep Going
# [Click here](https://www.kaggle.com/dansbecker/joining-data) to learn how to combine multiple data sources with the JOIN command.
# 
# # Feedback
# Bring any questions or feedback to the [Learn Discussion Forum](kaggle.com/learn-forum).
# 
# ----
# 
# *This tutorial is part of the [SQL Series](https://www.kaggle.com/learn/sql) on Kaggle Learn.*
