import pandas as pd

import bq_helper

# Define the token to analyze
token = 'QSP'
token_address = '0xb8c77482e45f1f44de1745f52c74426c631bdd52'

# Helper object for BigQuery Ethereum dataset
eth = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="ethereum_blockchain")

# # Compute total tokens received for each address
# queryReceived = """
# SELECT tx.to_address as addr, 
#       SUM(CAST(tx.value AS float64)/POWER(10,18)) as amount_received
# FROM `bigquery-public-data.ethereum_blockchain.token_transfers` as tx,
#      `bigquery-public-data.ethereum_blockchain.tokens` as tokens
# WHERE tx.token_address = tokens.address 
#   AND tokens.address = '{t}'
# GROUP BY 1
# """.format(t=token_address)

# # Estimate how big this query will be
# eth.estimate_query_size(queryReceived)

# df_received = eth.query_to_pandas_safe(queryReceived, max_gb_scanned=15)

# # Compute total tokens sent for each address
# querySent = """
# SELECT tx.from_address as addr, 
#       SUM(CAST(value AS float64)/POWER(10,18)) as amount_sent
# FROM `bigquery-public-data.ethereum_blockchain.token_transfers` as tx,
#      `bigquery-public-data.ethereum_blockchain.tokens` as tokens
# WHERE tx.token_address = tokens.address 
#   AND tokens.address = '{t}'
#   AND tx.from_address <> '0x0000000000000000000000000000000000000000'
# GROUP BY 1
# """.format(t=token_address)

query = """
WITH receivedTx AS (
    SELECT tx.to_address as addr, 
           tokens.symbol as symbol, 
           SUM(CAST(tx.value AS float64)/POWER(10,18)) as amount_received
    FROM `bigquery-public-data.ethereum_blockchain.token_transfers` as tx,
        `bigquery-public-data.ethereum_blockchain.tokens` as tokens
    WHERE tx.token_address = tokens.address 
        AND tokens.address = '{t}'
    GROUP BY 1, 2
),

sentTx AS (
    SELECT tx.from_address as addr, 
           tokens.symbol as symbol, 
           SUM(CAST(tx.value AS float64)/POWER(10,18)) as amount_sent
    FROM `bigquery-public-data.ethereum_blockchain.token_transfers` as tx,
         `bigquery-public-data.ethereum_blockchain.tokens` as tokens
    WHERE tx.token_address = tokens.address 
        AND tokens.address = '{t}'
        AND tx.from_address <> '0x0000000000000000000000000000000000000000'
    GROUP BY 1, 2
),

walletBalances AS (
    SELECT sum(r.amount_received) - sum(s.amount_sent) as balance
    FROM receivedTx as r, sentTx as s
    WHERE r.addr = s.addr
    GROUP BY r.addr
)

SELECT sum(w.balance) as circulating_supply
FROM walletBalances as w
    
""".format(t=token_address)

# Estimate how big this query will be
eth.estimate_query_size(query)

# df_sent = eth.query_to_pandas_safe(querySent, max_gb_scanned=15)

# # Combine send and received totals for each address
# df = df_sent.merge(df_received, left_on="addr", right_on="addr")

# Determine the current balance of each address
# balances = df["amount_received"] - df["amount_sent"]

# total_supply = 976442388.321185
# actual_supply = balances.sum()
# # Compute the remaining, unallocated tokens or if 
# # current token circulation exceeds the total expected supply
# print("Total Supply in Contract: {0}".format(total_supply))
# print("Actual Total Supply: {0}".format(actual_supply))

# if total_supply < actual_supply:
#     print("Token Overflow: True")
# else:
#     print("Token Overflow: False")
            
# print("Unallocated tokens: {0}".format(supply.sum()[0] - balances.sum()))

df = eth.query_to_pandas_safe(query, max_gb_scanned=18)

print(df.head())