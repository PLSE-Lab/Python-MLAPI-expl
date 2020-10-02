#!/usr/bin/env python
# coding: utf-8

# # Some Basic Analysis on Bitcoins

# In[ ]:


import time
start_time=time.time()
import numpy as np
import operator
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)
plt.rcParams['figure.figsize']=(12,5)
from google.cloud import bigquery
from bq_helper import BigQueryHelper
client = bigquery.Client()
bq_assistant = BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")


# # We want to find out the total SUM of satoshis transacted each day,
# ### Example on day1                                                    
# ### If A gave 5 bitcoin to B
# ### B gave 3 bitcoin to C.  
# ### C gave 2 bitcoin to A
# ### Then total bitcoin transacted are:
# ### 5+3+2=10 Bitcoins on day 1

# ### I used the query on this kernel: https://www.kaggle.com/mrisdal/visualizing-daily-bitcoin-recipients  and modified it to get the sum of all satoshis transacted each day

# In[ ]:


# Modified the query on the below kernel to get sum of all satoshis spent each day 
# https://www.kaggle.com/mrisdal/visualizing-daily-bitcoin-recipients
q = """
SELECT
  o.day,
  SUM(o.output_price) AS sum_output_price
FROM (
  SELECT
    TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) AS day,
    output.output_satoshis AS output_price
  FROM
    `bigquery-public-data.bitcoin_blockchain.transactions`,
    UNNEST(outputs) AS output ) AS o
GROUP BY
  o.day
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))

Grouped_df=bq_assistant.query_to_pandas_safe(q, max_gb_scanned=10)
Grouped_df=Grouped_df.sort_values(by=["sum_output_price"], ascending=False)


# ## Satoshi is the smallest unit of a bitcoin all transactions are recorded in terms of satoshis. 
# ## 1 Sathoshi=0.00000001 BTC 
# ### Converting satoshis to bitcoin...

# In[ ]:


#converting satoshis to Bitcoins
Grouped_df["sum_output_price"]=Grouped_df["sum_output_price"].apply(lambda x: float(x)*float(0.00000001))


# ## Now we will make a time series graph showing the total SUM of bitcoin transacted everyday.

# In[ ]:


Grouped_df_ts=Grouped_df.sort_values(by=["day"])
data = [go.Scatter(
            x=Grouped_df_ts["day"],
            y=Grouped_df_ts["sum_output_price"])]
fig = go.Figure(data=data)
print ("_"*30+ " Time series of bitcoin transacted"+"_"*30)
py.offline.iplot(fig)


# ## Most of the bitcoins were transacted in the end of 2012 and at the start of 2016
# 
# ## As we can see bitcoin started gaining significant popularity in the end of 2012,
# ## Then with a pause of 3 years, There was a spike once again in the start of 2016

# In[ ]:


layout = dict(
    title = "Timeseries from 20th Jan 2016 to 26th Jan 2016",
    xaxis = dict(
        range = ['2016-1-20','2016-1-26'])
)
fig = dict(data=data, layout=layout)
py.offline.iplot(fig)


# ## Lets find out the date in which most number of bitcoins were transacted.

# In[ ]:


top=Grouped_df[:5]
top["day"]=top["day"].apply(lambda x: str(x)[:11])
top.plot( x="day",kind="barh")


# ## 24th January 2016: On this day approximately TOTAL SUM of bitcoins transacted was 67.35 Millions.
# #### For the sake of clarification let me redefine what I mean by transacted here
# #### If A gives 4 btc to B, B gives 2 to C, C gives 1 to A. Then total amount transacted is 7 BTC.

# ## Lets find out the date in which least number of bitcoins were transacted.

# In[ ]:


bottom=Grouped_df[::-1][:5]
bottom["day"]=bottom["day"].apply(lambda x: str(x)[:11])
bottom.plot(kind="barh", x="day")


# ## 18th July 2009. Only 200 Bitcoins were transacted, the sender was quite unlucky though.

# ## There are some more concepts which were missed here such as miner fee, i.e. The difference between input and output goes to the miner.
# ## Moreover please do look at the code and give suggesstions if I missed anything, 
# # or if anywhere I am wrong in terms of logic. 

# In[ ]:


end_time=time.time()
print ("TOTAL TIME TAKEN FOR KERNEL TO RUN IS :"+ str(end_time-start_time)+" s")

