#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.cloud import bigquery
import pandas as pd
from bq_helper import BigQueryHelper

client = bigquery.Client()

# Query by Allen Day, GooglCloud Developer Advocate (https://medium.com/@allenday)
query1 = """
#standardSQL
SELECT
  o.day,
  COUNT(DISTINCT(o.output_key)) AS recipients
FROM (
  SELECT
    TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) AS day,
    output.output_pubkey_base58 AS output_key
  FROM
    `bigquery-public-data.bitcoin_blockchain.transactions`,
    UNNEST(outputs) AS output ) AS o
GROUP BY
  day
ORDER BY
  day
"""

query_job = client.query(query1)

iterator = query_job.result(timeout=30)
rows = list(iterator)

# Transform the rows into a nice pandas dataframe
transactions = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))

# Look at the first 10 headlines
transactions.head(10)


# In[ ]:


import matplotlib
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

transactions.plot('day','recipients')


# In[ ]:


# https://github.com/SohierDane/BigQuery_Helper
from bq_helper import BigQueryHelper

# This establishes an authenticated session and prepares a reference to the dataset that lives in BigQuery.
bq_assistant = BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")


# In[ ]:


bq_assistant.list_tables()


# In[ ]:


bq_assistant.table_schema("blocks")


# In[ ]:


bq_assistant.table_schema("transactions")


# In[ ]:


bq_assistant.head("blocks",10)


# In[ ]:


bq_assistant.head("transactions",10)


# In[ ]:


QUERY_TEMPLATE = """
SELECT
    TIMESTAMP_MILLIS((timestamp - MOD(timestamp,86400000))) AS day,
    transaction_id,
    block_id,
    previous_block,
    inputs.input_pubkey_base58 AS input_key,
    outputs.output_pubkey_base58 AS output_key,
    outputs.output_satoshis as satoshis,
    version
FROM `bigquery-public-data.bitcoin_blockchain.transactions`
    JOIN UNNEST (inputs) AS inputs
    JOIN UNNEST (outputs) AS outputs
WHERE inputs.input_pubkey_base58 IN UNNEST({0})
    AND outputs.output_satoshis  >= {1}
    AND inputs.input_pubkey_base58 IS NOT NULL
    AND outputs.output_pubkey_base58 IS NOT NULL
GROUP BY day, transaction_id, block_id, previous_block, input_key, output_key, satoshis, version
"""


# In[ ]:


def trace_transactions(target_depth, seeds, min_satoshi_per_transaction, bq_assistant):
    """
    Trace transactions associated with a given bitcoin key.

    To limit the number of BigQuery calls, this function ignores time. 
    If you care about the order of transactions, you'll need to do post-processing.

    May return a deeper graph than the `target_depth` if there are repeated transactions
    from wallet a to b or or self transactions (a -> a).
    """
    MAX_SEEDS_PER_QUERY = 500
    query = QUERY_TEMPLATE.format(seeds, min_satoshi_per_transaction)
    print(f'Estimated total query size: {int(bq_assistant.estimate_query_size(query)) * MAX_DEPTH}')
    results = []
    seeds_scanned = set()
    for i in range(target_depth):
        seeds = seeds[:MAX_SEEDS_PER_QUERY]
        print(f"Now scanning {len(seeds)} seeds")
        query = QUERY_TEMPLATE.format(seeds, min_satoshi_per_transaction)
        transactions = bq_assistant.query_to_pandas(query)
        results.append(transactions)
        # limit query kb by dropping any duplicated seeds
        seeds_scanned.update(seeds)
        seeds = list(set(transactions.output_key.unique()).difference(seeds_scanned))
    return pd.concat(results).drop_duplicates()


# In[ ]:


MAX_DEPTH = 4
BASE_SEEDS = ['1XPTgDRhN8RFnzniWCddobD9iKZatrvH4']
#SATOSHI_PER_BTC = 10**7
#MIN_BITCOIN_PER_TRANSACTION = 1/1000
#min_satoshi = MIN_BITCOIN_PER_TRANSACTION * SATOSHI_PER_BTC


# In[ ]:


df_t = trace_transactions(MAX_DEPTH, BASE_SEEDS, 0, bq_assistant)


# In[ ]:


df_t.to_csv("transactions.csv", index=False)


# In[ ]:


df_t.head(10)


# In[ ]:


df_t.info()


# In[ ]:


df_t.sort_values(by='day', ascending=False)


# In[ ]:


df_t.isnull().any()


# In[ ]:


df_exact = df_t[df_t['block_id'] == df_t['previous_block']]
df_exact


# In[ ]:


df_t['block_id'].value_counts()


# In[ ]:





# In[ ]:





# In[ ]:


QUERY_TEMPLATE_ = """
SELECT
    TIMESTAMP_MILLIS((timestamp - MOD(timestamp,86400000))) AS day,
    T.transaction_id AS transactions_id,
    block_id,
    previous_block,
    O.output_satoshis AS satoshis,
    version
FROM `bigquery-public-data.bitcoin_blockchain.blocks`
    JOIN(SELECT* 
        FROM 'transactions'.T)AS T
    JOIN(SELECT*
        FROM 'outputs'.O)AS O

GROUP BY day, transaction_id, block_id, previous_block, satoshis, version
"""


# In[ ]:


def trace_transactions(target_depth, seeds, min_satoshi_per_transaction, bq_assistant):
    """
    Trace transactions associated with a given bitcoin key.

    To limit the number of BigQuery calls, this function ignores time. 
    If you care about the order of transactions, you'll need to do post-processing.

    May return a deeper graph than the `target_depth` if there are repeated transactions
    from wallet a to b or or self transactions (a -> a).
    """
    MAX_SEEDS_PER_QUERY = 500
    query = QUERY_TEMPLATE_.format(seeds, min_satoshi_per_transaction)
    print(f'Estimated total query size: {int(bq_assistant.estimate_query_size(query)) * MAX_DEPTH}')
    results = []
    seeds_scanned = set()
    for i in range(target_depth):
        seeds = seeds[:MAX_SEEDS_PER_QUERY]
        print(f"Now scanning {len(seeds)} seeds")
        query = QUERY_TEMPLATE_.format(seeds, min_satoshi_per_transaction)
        transactions = bq_assistant.query_to_pandas(query)
        results.append(transactions)
        # limit query kb by dropping any duplicated seeds
        seeds_scanned.update(seeds)
        seeds = list(set(transactions.output_key.unique()).difference(seeds_scanned))
    return pd.concat(results).drop_duplicates()


# In[ ]:


MAX_DEPTH = 4
BASE_SEEDS = ['1XPTgDRhN8RFnzniWCddobD9iKZatrvH4']
#SATOSHI_PER_BTC = 10**7
#MIN_BITCOIN_PER_TRANSACTION = 1/1000
#min_satoshi = MIN_BITCOIN_PER_TRANSACTION * SATOSHI_PER_BTC


# In[ ]:


df_b = trace_transactions(MAX_DEPTH, BASE_SEEDS, 0, bq_assistant)


# In[ ]:


df_t.to_csv("transactions.csv", index=False)


# In[ ]:





# In[ ]:


def trace_transactions(target_depth, seeds, min_satoshi_per_transaction, bq_assistant):
    """
    Trace transactions associated with a given bitcoin key.

    To limit the number of BigQuery calls, this function ignores time. 
    If you care about the order of transactions, you'll need to do post-processing.

    May return a deeper graph than the `target_depth` if there are repeated transactions
    from wallet a to b or or self transactions (a -> a).
    """
    MAX_SEEDS_PER_QUERY = 500
    query = QUERY_TEMPLATE.format(seeds, min_satoshi_per_transaction)
    print(f'Estimated total query size: {int(bq_assistant.estimate_query_size(query)) * MAX_DEPTH}')
    results = []
    seeds_scanned = set()
    for i in range(target_depth):
        seeds = seeds[:MAX_SEEDS_PER_QUERY]
        print(f"Now scanning {len(seeds)} seeds")
        query = QUERY_TEMPLATE.format(seeds, min_satoshi_per_transaction)
        transactions = bq_assistant.query_to_pandas(query)
        results.append(transactions)
        # limit query kb by dropping any duplicated seeds
        seeds_scanned.update(seeds)
        seeds = list(set(transactions.output_key.unique()).difference(seeds_scanned))
    return pd.concat(results).drop_duplicates()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


query2 = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month 
            ORDER BY year, month
        """

# note that max_gb_scanned is set to 21, rather than 1
transactions_per_month = bq_assistant.query_to_pandas(query2)#query_to_pandas_safe


# In[ ]:


transactions_per_month.tail(10)


# In[ ]:


transactions_per_month.plot('year','transactions')
plt.title("Monthly Bitcoin Transcations")


# In[ ]:


df = bq_assistant.query_to_pandas(query1)


# In[ ]:


print('Size of dataframe: {} Bytes'.format(int(df.memory_usage(index=True, deep=True).sum())))


# In[ ]:





# In[ ]:




