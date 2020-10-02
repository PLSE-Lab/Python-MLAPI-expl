#!/usr/bin/env python
# coding: utf-8

# We show how to extract meaningful statistics of accounts of the Ethereum main network, from the data set hosted by Google BigQuery. 
# 
# **Active accounts**
# 
# First, we find which accounts are more active, by searching for the probability that an account is modified in a generic block (we don't distinguish between transactions from/to its address). This is an index of the accounts' **activity**.
# 
# To find the most popular accounts, I wrote the following query (1).  The number of addresses in Ethereum main network is becoming quite large, hence we filter out those that are rarely updated (they have been part of less than 5 transactions). Moreover, we limit the study to a range of blocks after the "explosion" of the **crypto bubble** (blocks between min_block_number and max_block_number).
# 
# As suggested [here](https://www.kaggle.com/mrisdal/visualizing-average-ether-costs-over-time?utm_medium=partner&utm_source=cloud&utm_campaign=big+data+blog+ethereum), we set a safety limit to control the quota of free queries provided by Kaggle. For this kernel, you will read a maximim of 120 GB from BigQuery. As the database size is constantly increasing, in the future you will incur in a error, if the amount of scanned data is greater than max_gb_scanned=40.

# In[ ]:


from google.cloud import bigquery
from bq_helper import BigQueryHelper
import pandas as pd

bq_assistant = BigQueryHelper("bigquery-public-data", "ethereum_blockchain")
client = bigquery.Client()

min_block_number = 5100000
max_block_number = 6400000

# find average values and sort
query = """
SELECT
  address, SUM(n_updates) AS updates
FROM
(
  SELECT
      address, COUNT(*) AS n_updates
  FROM
  (
  SELECT DISTINCT
    from_address AS address, block_number AS block_number
  FROM
    `bigquery-public-data.ethereum_blockchain.transactions`
  WHERE
    block_number > %d
    AND
    block_number < %d
  )
  GROUP BY 
    address

  UNION ALL

  SELECT 
      address AS address, COUNT(*) AS n_updates
  FROM
  (
  SELECT DISTINCT
    to_address AS address, block_number AS block_number
  FROM
    `bigquery-public-data.ethereum_blockchain.transactions`
  WHERE
    block_number > %d
    AND
    block_number < %d
  )
  GROUP BY 
    address
)
WHERE
  n_updates >= 5
  AND
  address IS NOT NULL
GROUP BY 
  address
ORDER BY 
  updates DESC
"""

most_populars = bq_assistant.query_to_pandas_safe(query % (min_block_number, max_block_number, min_block_number, max_block_number), max_gb_scanned=40)
print("Retrieved " + str(len(most_populars)) + " accounts.")
blocks_int = max_block_number - min_block_number
most_populars = most_populars.sort_values(by='updates', ascending=False)
most_populars["probability"] = most_populars["updates"] / (blocks_int*1.0)
print(most_populars.head(10))


# Ok, we obtained the list of most popular addresses and printed the **top ten**. A quick check of the addresses on a blockchain explorer (https://etherscan.io/) tells us that they are all associated with currency exchanges or popular decentralized applications (**dApps**). We might expect that the accounts follow a power-law, where the most "active" accounts are owned by centralized exchanges. Is it true?

# In[ ]:


from scipy.optimize import curve_fit
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

def func_powerlaw(x, m, c, c0):
    return c0 + x**m * c

blocks_int = max_block_number - min_block_number

# Compute probabilities
most_populars["probability"] = most_populars["updates"] / (blocks_int*1.0)
most_populars["idxs"] = range(1, len(most_populars) + 1)

# Fit curve
sol = curve_fit(func_powerlaw, most_populars["idxs"], most_populars["probability"], p0 = np.asarray([float(-1),float(10**5),0]))
fitted_func = func_powerlaw(most_populars["idxs"], sol[0][0], sol[0][1], sol[0][2])
print("Fit with values {} {} {}".format(sol[0][0], sol[0][1], sol[0][2]))

# Plot fit vs samples (only for the first 2000)
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,5))
plt.loglog(most_populars["probability"].tolist()[1:10000],'o')
plt.loglog(fitted_func.tolist()[1:10000])
plt.xlabel("Account index (by descending popularity)")
plt.ylabel("Relative frequency [1/block]")
plt.show()


# We plot the relative frequency of accounts that are modified at least once per hour (there are approximately 10k of them in the main network).
# Their relative frequency doesn't follow a "simple" power-law, because the 20 most popular accounts have "equal power", i.e. they show similar activity. A *broken* power-law would be more appropriate, as we have observed in a [recent paper](https://arxiv.org/pdf/1807.07422.pdf). In that document, you can find more insights about how to use the statistics of accounts to design better protocols.

# **Statistics of specific accounts**
# 
# We can also observe the activity of specific accounts. For example, the following query returns the block numbers (and their timestamp) at which an account has been modified. We apply it to two addresses: one associated with [CryptoKitties](https://www.cryptokitties.co/) dApp, and one with [Bittrex](https://bittrex.com/) exchange, that are characterized by similar relative frequency of updates.

# In[ ]:


query = """
SELECT 
      timestamp, number
    FROM
      `bigquery-public-data.ethereum_blockchain.blocks`
INNER JOIN 
(
    SELECT DISTINCT
              from_address AS address, block_number AS block_number
            FROM
              `bigquery-public-data.ethereum_blockchain.transactions`
            WHERE
              from_address = '%s'
              AND
              block_number > %d
              AND
              block_number < %d

    UNION DISTINCT

    SELECT DISTINCT
              to_address AS address, block_number AS block_number
            FROM
              `bigquery-public-data.ethereum_blockchain.transactions`
            WHERE
              to_address = '%s'
              AND
              block_number > %d
              AND
              block_number < %d
) as InnerTable
ON 
    `bigquery-public-data.ethereum_blockchain.blocks`.number = InnerTable.block_number;
"""

# CryptoKitties address
adx_1 = most_populars.iloc[4].address 
transax_1 = bq_assistant.query_to_pandas_safe(query % (adx_1, min_block_number, max_block_number, adx_1, min_block_number, max_block_number), max_gb_scanned=40)
print("Retrieved " + str(len(transax_1)) + " blocks for account %s." % (adx_1) )
transax_1.sort_values(by="number", ascending=True, inplace=True)
    
# Bittrex address    
adx_2 = most_populars.iloc[3].address 
transax_2 = bq_assistant.query_to_pandas_safe(query % (adx_2, min_block_number, max_block_number, adx_2, min_block_number, max_block_number), max_gb_scanned=40)
print("Retrieved " + str(len(transax_2)) + " blocks for account %s." % (adx_2) )
transax_2.sort_values(by="number", ascending=True, inplace=True)
    
transax = list()
transax.append(transax_1)
transax.append(transax_2)


# From the results, it is possible to extract several interesting statistics. For example, the cumulative density function (CDF) of number of blocks between two consecutive updates to the account. The knowledge of the CDF is important to design and evaluate the performances of blockchain lightweight protocols. For example, we can decide to sync our blockchain client periodically, and tune the period based on the statistics of the account that we are interested in.

# In[ ]:


# plot the Empirical CDF
plt.figure(figsize=(15,5))
for t in transax:
    t.sort_values(by="number", inplace=True)
    tx_d = t.diff()
    tx_d = tx_d.iloc[1:]
    count = np.sort(tx_d["number"].values)
    cdf = np.arange(len(count)+1)/float(len(count))
    plt.plot(count, cdf[:-1])

plt.axis([0, 20, 0, 1])
plt.xlabel("Number of blocks without updates [n]")
plt.ylabel("Empirical CDF ( Pr[x <= n] )")
plt.show()


# We can also look for specific activity patterns. For example, in the figure below we show that, even if the two accounts show the same relative frequency *on average*, their activity is not stationary.
# 
# Other question: on which **day of the week** are the two accounts more active? The bar figures show that CryptoKitties, that is a game, is particularly used during the weekends. On the other hand, Bittrex is active on Mondays and Tuesdays and on Friday, that is quite expected for a trading platform.

# In[ ]:


# activity during time
txp_1 = transax_1["timestamp"].groupby(transax_1["timestamp"].dt.floor('d')).size().reset_index(name='CryptoKitties')
txp_2 = transax_2["timestamp"].groupby(transax_2["timestamp"].dt.floor('d')).size().reset_index(name='Bittrex')
txp_1 = txp_1[10:-10]
txp_2 = txp_2[10:-10]


fig, ax = plt.subplots(1, 1, figsize=(15, 8))
ax = txp_1.plot(x="timestamp", y="CryptoKitties", ax=ax)
ax = txp_2.plot(x="timestamp", y="Bittrex", ax=ax)
plt.ylabel("Active blocks/day")

# patterns
f = plt.figure(figsize=(15,5))
ax = f.add_subplot(121)
ax2 = f.add_subplot(122)

plt.subplot(1, 2, 1)
txp_1 = transax_1["timestamp"].groupby(transax_1["timestamp"].dt.day_name()).count().sort_values()
txp_1 /= sum(txp_1)
txp_1.plot(kind="bar", ax=ax)
plt.xlabel("Day of the week")
plt.ylabel("Normalized count")
plt.title("CryptoKitties")

plt.subplot(1, 2, 2)
txp_2 = transax_2["timestamp"].groupby(transax_2["timestamp"].dt.day_name()).count().sort_values()
txp_2 /= sum(txp_2)
txp_2.plot(kind="bar", ax=ax2)
plt.xlabel("Day of the week")
plt.ylabel("Normalized count")
plt.title("Bittrex")


# **Conclusion**
# 
# * We have introduced some basic tools that can be used for the statistical characterization of blockchain updates. 
# * Besides the study of economic/human interactions that happen on chain, the characterization allows to model the data flow generated by blockchain software. This is a valuable information for the management of the IT infrastructures.
# * We have extracted one feature ("an account is modified (or not) in a block"), because it was the simplest approach, but we can extend this kernel by distinguishing between transactions from/to the account, and their cardinality.
