#!/usr/bin/env python
# coding: utf-8

# # Measuring Decentralization of Ethereum Classic using Nakamoto Library on Python with Google BigQuery
# 
# [Nakamoto](https://github.com/YazzyYaz/nakamoto-coefficient) is a python library built to measure the Nakamoto Coefficient as presented by the Medium article "[Quantifying Decentralization](https://news.earn.com/quantifying-decentralization-e39db233c28e)".
# 
# Nakamoto is a measurement of the minimum entity needed to compromise a subsystem of the blockchain, or 51% attack it. It is not just limited to the hashrate of the blockchain but can used to measure wealth distribution, exchange volume, code contribution. 
# 
# Nakamoto module here will go over each subsystem example and how to generate the Nakamoto Coefficient, the Gini Coefficient (to measure inequality distribution), and the Lorenz Curve (to measure curve to perfect equality).
# 
# This notebook will also be using Google BigQuery with the Ethereum Classic Dataset to measure two subsytems of the Nakamoto Coefficient.
# 
# Let's get started!
# 
# The module we will be running is called `nakamoto` and can be installed by running the following chunk.

# In[ ]:


import numpy as np
import pandas as pd
import os
from google.cloud import bigquery
# For this Notebook, we will be using the Nakamoto python 
# module found here: https://github.com/YazzyYaz/nakamoto-coefficient
get_ipython().system('pip install nakamoto')


# In[ ]:


client = bigquery.Client()
ethereum_dataset_ref = client.dataset('crypto_ethereum_classic', project='bigquery-public-data')


# ## Nakamoto Coefficient of Wealth Distribution on Blockchain Addresses
# 
# Here, we will first measure the wealth distribution and income inequality among all Ethereum Classic addresses, where we limit our measurement to the top 10k accounts.
# 
# We provide the query here for getting the top 10k balances for the day which was taken from [the article here](https://medium.com/google-cloud/how-to-query-balances-for-all-ethereum-addresses-in-bigquery-fb594e4034a7). 

# In[ ]:


# SQL query needed to get top 10K Ethereum Classic balances for the day
query = """
#standardSQL
-- MIT License
-- Copyright (c) 2018 Evgeny Medvedev, evge.medvedev@gmail.com
with double_entry_book as (
    -- debits
    select to_address as address, value as value
    from `bigquery-public-data.crypto_ethereum_classic.traces`
    where to_address is not null
    and status = 1
    and (call_type not in ('delegatecall', 'callcode', 'staticcall') or call_type is null)
    union all
    -- credits
    select from_address as address, -value as value
    from `bigquery-public-data.crypto_ethereum_classic.traces`
    where from_address is not null
    and status = 1
    and (call_type not in ('delegatecall', 'callcode', 'staticcall') or call_type is null)
    union all
    -- transaction fees debits
    select miner as address, sum(cast(receipt_gas_used as numeric) * cast(gas_price as numeric)) as value
    from `bigquery-public-data.crypto_ethereum_classic.transactions` as transactions
    join `bigquery-public-data.crypto_ethereum_classic.blocks` as blocks on blocks.number = transactions.block_number
    group by blocks.miner
    union all
    -- transaction fees credits
    select from_address as address, -(cast(receipt_gas_used as numeric) * cast(gas_price as numeric)) as value
    from `bigquery-public-data.crypto_ethereum_classic.transactions`
)
select address, 
sum(value) / 1000000000 as balance
from double_entry_book
group by address
order by balance desc
limit 10000
"""

# We pass the query to the client
query_job = client.query(query)
iterator = query_job.result()


# In[ ]:


rows = list(iterator)
# Transform the rows into a nice pandas dataframe
balances = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))


# In[ ]:


# Now that we have a dataframe of the top 100k balances for the day on ETC,
# it's time to import it into a CustomSector in nakamoto to begin analysis

from nakamoto.sector import CustomSector

# data passed into nakamoto must be as a numpy array
balance_data = np.array(balances['balance'])
type(balance_data)


# In[ ]:


# We build a config dictionary for Plotly like this:
nakamoto_config = {
    'plot_notebook': True,
    'plot_image_path': None
}

# We also need a currency name and sector type, which is used for plotting information
currency = 'ETC'
sector_type = 'daily balance'

# Since our balance data is sorted descending, we need to flip the data
# for a proper gini and lorenz, otherwise the coefficient comes back negative
balance_data = balance_data[::-1]

# Now, we instantiate the balance object
balance_sector = CustomSector(balance_data,
                             currency,
                             sector_type,
                             **nakamoto_config)


# ## Gini
# The Gini coefficient is a value between 0 and 1, with 0 being perfect equality and 1 being perfect inequality. It helps understand measurement of inequality. In this example, we measure inequality distribution using balances per address. The assumption here is that each person has only 1 address; Top addresses also belong to exchanges, which use their own accounting software on their app; Only top 10k balances analyzed, so not a complete picture.

# In[ ]:


# Let's get back the gini coefficient
balance_sector.get_gini_coefficient()


# ## Nakamoto Coefficient
# 
# Nakamoto coefficient is a measure of the minimum amount of entities required to achieve > 51% control or influence to compromise a subsystem. In this example, we are measuring the minimum number of entities required to control more than 51% of the entire ETC circulating supply.

# In[ ]:


balance_sector.get_nakamoto_coefficient()


# ## Lorenz Curve
# 
# The lorenz curve can plot distribution of inequality. It has a straight line used as a measure of perfect equality. The curve, called the Lorenz curve highlights how far away the entities are from perfect equality.
# 
# We pass a line through the 51% mark to highlight the influence of the minimum number of entities who are capable of compromising the system. The minimum Nakamoto coefficient is the number of entities in red.

# In[ ]:


balance_sector.get_plot()


# ## Nakamoto Coefficient of Miner Reward Count By Miner Address on Ethereum Classic
# 
# Here, we will use Google BigQuery Ethereum Classic Dataset to run an SQL query to get back the total number of times a miner received a block reward in the last month. We will measure the income inequality by block reward count. This of course assumes the rewards are divided by entities, but of course those addresses can belong to pools.

# In[ ]:


query = """
#standardSQL
-- MIT License
-- Copyright (c) 2019 Yaz Khoury, yaz.khoury@gmail.com
WITH mined_block AS (
  SELECT miner, DATE(timestamp)
  FROM `bigquery-public-data.crypto_ethereum_classic.blocks` 
  WHERE DATE(timestamp) > DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH)
  ORDER BY miner ASC)
SELECT miner, COUNT(miner) AS total_block_reward 
FROM mined_block 
GROUP BY miner 
ORDER BY total_block_reward ASC
"""

# We pass the query to the client
query_job = client.query(query)
iterator = query_job.result()


# In[ ]:


rows = list(iterator)
# Transform the rows into a nice pandas dataframe
mining_rewards = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))
mining_rewards_data = np.array(mining_rewards['total_block_reward'])


# In[ ]:


sector_type = 'mining_rewards'
mining_rewards_sector = CustomSector(mining_rewards_data,
                             currency,
                             sector_type,
                             **nakamoto_config)


# Now, we will once again get the Gini, Nakamoto and Lorenz Plot of the Mining Reward Counts Per Miner in the Past 30 Days.

# In[ ]:


# Mining Gini
mining_rewards_sector.get_gini_coefficient()


# In[ ]:


# Nakamoto Coefficient
mining_rewards_sector.get_nakamoto_coefficient()


# In[ ]:


mining_rewards_sector.get_plot()


# ## Market Nakamoto Coefficient Of Ethereum Classic Volume By Exchange
# 
# We will now measure the total 24 hour volume by the exchanges that trade Ethereum Classic. We will look into the distribution of ETC inequality among exchanges and the minimum amount of exchanges needed to compromise 51% of total circulating supply of ETC in exchanges.
# 
# For this, we use the special sector class `Market`. We pass it a CoinMarketCap `#market` type of url like the following we will use for ETC: `https://coinmarketcap.com/currencies/ethereum-classic/#markets`

# In[ ]:


from nakamoto.sector import Market

market_url = "https://coinmarketcap.com/currencies/ethereum-classic/#markets"
market_sector = Market(currency, market_url, **nakamoto_config)


# In[ ]:


# Market Gini
market_sector.get_gini_coefficient()


# In[ ]:


market_sector.get_nakamoto_coefficient()


# In[ ]:


market_sector.get_plot()


# ## Client and Geography Nakamoto Coefficient
# 
# Here, we have built in sector classes for `Geography` (mining nodes by country) and `Client` (mining nodes by client software) for ETC and ETH because the built-in sector takes data from [Ethernodes.org](http://ethernodes.org) which doesn't distinguish between `chainId`. We plan on being able to separate two datasets, but for now, it's assumed as the Client and Geography Nakamoto coefficients of EVM datasets.

# In[ ]:


from nakamoto.sector import Client, Geography

client_sector = Client(currency, **nakamoto_config)
geography_sector = Geography(currency, **nakamoto_config)


# In[ ]:


# Client Gini
client_sector.get_gini_coefficient()


# In[ ]:


# Geography Gini
geography_sector.get_gini_coefficient()


# In[ ]:


# Client Nakamoto
client_sector.get_nakamoto_coefficient()


# In[ ]:


# Geography Nakamoto
geography_sector.get_nakamoto_coefficient()


# In[ ]:


client_sector.get_plot()


# In[ ]:


geography_sector.get_plot()


# ## Mininmum Nakamoto and Maximum Gini Coefficient of All Ethereum Classic Sectors
# 
# To get an overall idea of all the subsystems and the minimum nakamoto of all minimum nakamoto coefficients in the sectors as well as maximum gini which is most centralized/inequal sector, we use `Nakamoto` class for analysis.

# In[ ]:


from nakamoto.coefficient import Nakamoto


sector_list = [geography_sector, 
               market_sector, 
               client_sector,  
               balance_sector,
               mining_rewards_sector]

nakamoto = Nakamoto(sector_list)


# In[ ]:


# Minimum Nakamoto Coefficient
nakamoto.get_minimum_nakamoto()


# In[ ]:


# Maximum Gini Coefficient
nakamoto.get_maximum_gini()

