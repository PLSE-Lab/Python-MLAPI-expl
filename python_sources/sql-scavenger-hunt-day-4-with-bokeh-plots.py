#!/usr/bin/env python
# coding: utf-8

# # Scavenger Hunt Day 4 - exercises
# ___
# In this notebook we're going to answer the below 2 questions which are exercises from Rachel Tatman's SQL Scavenger Hunt Day 4. 
# 
# * How many Bitcoin transactions were made each day in 2017?
# 
# * How many transactions are associated with each merkle root?
# 
# ___
# #### Before trying to answer the above questions, we need to import the "bq_helper" package and initialize the BigQueryHelper object. We will also print the Bitcoin Blockchain dataset's schema.

# In[ ]:


# Import package with helper functions 
import bq_helper

# Create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")


# In[ ]:


# Print the Bitcoin Blockchain dataset's schema
bitcoin_blockchain.list_tables()


# In[ ]:


# Display the first 5 rows of 'transactions' table 
bitcoin_blockchain.head('transactions')


# ___
# #### Now we're going to write a query to get the number of transactions per day in 2017. Since our dataset uses timestamps instead of dates and they are stored as integers, we will have to convert them using BigQuery's TIMESTAMP_MILLIS() function.

# In[ ]:


query_1 = """ WITH timetable AS
             (
                 SELECT TIMESTAMP_MILLIS(timestamp) AS transaction_time, 
                     transaction_id
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             )
            SELECT COUNT(transaction_id) AS NumOfTransactions,
                EXTRACT(DAY FROM transaction_time) AS Day,
                EXTRACT(MONTH FROM transaction_time) AS Month,
                EXTRACT(YEAR FROM transaction_time) AS Year
            FROM timetable
            WHERE EXTRACT(YEAR FROM transaction_time) = 2017
            GROUP BY year,month,day
            ORDER BY NumOfTransactions DESC
          """


# In[ ]:


# Estimate query size
bitcoin_blockchain.estimate_query_size(query_1)


# In[ ]:


# Note that max_gb_scanned is set to 21, rather than 1
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query_1, max_gb_scanned=21)


# In[ ]:


# Print first 5 rows
transactions_per_day.head(5)


# Let's plot the results for the first 50 days with the biggest number of transactions

# In[ ]:


# Import pandas library and create new 'Date' column which we'll use for plotting purposes
import pandas as pd
transactions_per_day['Date'] = pd.to_datetime(transactions_per_day[['Year','Month','Day']]) 

# Display first 5 rows
transactions_per_day.head()


# In[ ]:


# Import Bokeh plotting library
from bokeh.io import show, output_notebook
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import NumeralTickFormatter
output_notebook()


# In[ ]:


# Initialize ColumnDataSource object
source_1 = ColumnDataSource(data=dict(
    dats = transactions_per_day['Date'][:50],
    nums = transactions_per_day['NumOfTransactions'][:50]
))

# Set plot properties
daily_plot = figure(x_axis_label = "Date", y_axis_label = "Number of Transactions", 
                    x_axis_type='datetime', tools = "pan,box_zoom,reset", plot_width = 800,
                    plot_height = 500, title="Transactions Per Day")

daily_plot.circle('dats', 'nums', size = 5, color = '#FFD447', source = source_1)
daily_plot.yaxis[0].formatter = NumeralTickFormatter(format="0,0")
show(daily_plot)


# As we can observe, the biggest number of transactions occured in December 2017, reaching its maximum on 14th of December 2017 with 490644 transactions. In fact, 2017 was a very remarkable year for the whole cryptocurrency market. The fact that Bitcoin price has risen from 800 dollars in January 2017 up to it's "ATH - All Time High"  = 20000 dollars on 17th of December 2017 on some Korean Exchanges, caused lots of media companies to publish news about crypto tokens. This in turn led to the massive hype which resulted in increased demand for cryptocurrencies and forced some exchanges to block the registration possibilities due to the lack of processing resources. 

# ___
# #### Now, let's answer the second question - how many transactions are associated with each merkle root?

# In[ ]:


query_2 = """  WITH timetable AS
             (
                 SELECT TIMESTAMP_MILLIS(timestamp) AS transaction_time, 
                     transaction_id, 
                     merkle_root AS MerkleTree
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
             )
            SELECT COUNT(transaction_id) AS NumOfTransactions, MerkleTree,
                EXTRACT(DAY FROM transaction_time) AS Day,
                EXTRACT(MONTH FROM transaction_time) AS Month,
                EXTRACT(YEAR FROM transaction_time) AS Year
            FROM timetable
            GROUP BY MerkleTree, Year, Month, Day
            ORDER BY NumOfTransactions DESC
          """


# In[ ]:


# Estimate query size
bitcoin_blockchain.estimate_query_size(query_2)


# In[ ]:


# Note that max_gb_scanned is set to 21, rather than 1
transactions_per_merkle = bitcoin_blockchain.query_to_pandas_safe(query_2, max_gb_scanned=40)


# In[ ]:


transactions_per_merkle.head(10)


# As we can see, there is a downward trend in the number of transactions associated with each MerkleTree. To confirm this, let's plot the results.

# In[ ]:


# Initialize ColumnDataSource object
source_2 = ColumnDataSource(data=dict(
    index = list(transactions_per_merkle.index.values)[:100],
    nums = transactions_per_merkle['NumOfTransactions'][:100],
))

# Set plot properties
merkle_plot = figure(x_axis_label = "Merkle index", y_axis_label = "Number of Transactions",
                   tools = "pan,box_zoom,reset", plot_width = 800,
                   plot_height = 500, title = "Transactions Per Merkle")

merkle_plot.circle(x = 'index' ,y = 'nums', size = 5, color = '#30FF49', source = source_2)
merkle_plot.yaxis[0].formatter = NumeralTickFormatter(format="0,0")
show(merkle_plot)


# Plotting the results confirmed our hypothesis that there is downward trend in the number or transactions per Merkle Tree.
