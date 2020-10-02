#!/usr/bin/env python
# coding: utf-8

# # Comparison of Bitcoin Transactions Prices and count with Exchange Rates
# #### In this notebook we will try to analyze the bitcoin data and find out how the fluctuations in exchange rate from btc to usd effects the Transactions in bitcoins. Moreover we will also look at the average increase in value of transaction with change in exchange rate.
# 
# #### Incase you want to get some knowledge about bitcoins and a bird eye view of how bitcoins transactions work and to get a better understanding of visualizations in this notebook please do look at <a href="https://www.kaggle.com/ibadia/bitcoin-101-bitcoins-and-detailed-insights">this notebook</a> to gain some prerequisite information for the visualizations presented here.
# 
# 
# 
# #### Contents of the notebook.
# 1. <a href='#1'>Timeseries plot signifying the relation between the spike in bitcoin value, the number of transactions, the total bitcoins transacted per day.</a>
# 2. <a href='#2'>Average bitcoin price in usd to transaction counts</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.1'>2.1  2012</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.2'>2.2  2013</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.3'>2.3  2014</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.4'>2.4  2015</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.5'>2.5  2016</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href='#2.6'>2.6  2017</a><br>
# 3.  <a href='#3'>Number of bitcoins transacted with the average bitcoin value in USD</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.1'>3.1  2012</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.2'>3.2  2013</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.3'>3.3  2014</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.4'>3.4  2015</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.5'>3.5  2016</a><br>
# &nbsp;&nbsp;&nbsp;&nbsp;<a href='#3.6'>3.6  2017</a><br>
# ##### Please do give your suggesstions and upvote the notebook if you find it useful

# In[ ]:


import time
start_time=time.time()
import numpy as np
import operator
from collections import Counter
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import operator
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('fivethirtyeight')
from google.cloud import bigquery
from bq_helper import BigQueryHelper
client = bigquery.Client()
bq_assistant = BigQueryHelper("bigquery-public-data", "bitcoin_blockchain")


# ## Let us look at the data of bitcoin prices from 2012 to 2018

# In[ ]:


data_f=pd.read_csv("../input/bitstampUSD_1-min_data_2012-01-01_to_2018-06-27.csv")
data_f["Timestamp"]=pd.to_datetime(data_f["Timestamp"], unit='s')
data_f.Timestamp=data_f.Timestamp.apply(lambda x: x.replace(hour=0, minute=0, second=0))
data_f=data_f[["Timestamp", "Weighted_Price"]]
data_f=data_f.drop_duplicates(keep="first")
data_f.head()


# This contains bitcoin exchange rates  from 2012 to 2018 for every two minutes, but the issue is this that many values are repeated and we intend to get daily average bitcoin to usd exchange rate.

# In[ ]:


data_f=data_f.drop_duplicates(subset='Timestamp', keep="last")


# #### Lets make a query to get a total sum of bitcoins  transacted for every day.

# In[ ]:


q = """
SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) as Timestamp, sum(o.output_satoshis) as output_price from 
    `bigquery-public-data.bitcoin_blockchain.transactions`JOIN
    UNNEST(outputs) as o group by timestamp
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))
results3=bq_assistant.query_to_pandas(q)
results3["output_price"]=results3["output_price"].apply(lambda x: float(x/100000000))
results3=results3.sort_values(by="Timestamp")
results3.head()


# ### In this query we will find the total count of transactions per day

# In[ ]:


q = """
SELECT  TIMESTAMP_MILLIS((timestamp - MOD(timestamp,
          86400000))) as Timestamp , count(Timestamp) as output_count from 
    `bigquery-public-data.bitcoin_blockchain.transactions` group by timestamp
"""
print (str(round((bq_assistant.estimate_query_size(q)),2))+str(" GB"))
transaction_count=bq_assistant.query_to_pandas(q)
transaction_count=transaction_count.sort_values(by="Timestamp")
transaction_count.head()


# <a id='1'></a>
# ### Here comes our first plot, Timeseries plot signifying the relation between the spike in bitcoin value, the number of transactions, the total bitcoins transacted per day.
# ### But there are some problems: 
# ### The number of bitcoins transacted per day and the total number of transactions are very different, so we have to somehow make the graph in such a manner that we can see the difference in these attributes with the passage of time.
# 
# ### For scaling purposes we divide the transaction counts by 6
# ### Multiply the bitcoin prize by 10
# ### and finally divide the number of bitcoins transacted per day by 500.
# I need your suggestions over here: Please do tell me if there is more elegant way to scale the values and what more techniques can I use to compare the change in trend.

# In[ ]:


import datetime
def to_unix_time(dt):
    epoch =  datetime.datetime.utcfromtimestamp(0)
    return (dt - epoch).total_seconds() * 1000
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
data = [go.Scatter(x=transaction_count.Timestamp, y=transaction_count.output_count/6, name="transaction_count/6"), go.Scatter(x=data_f.Timestamp,y=data_f.Weighted_Price*10, name="BITCOIN_PRICE*10"),
       go.Scatter(x=results3.Timestamp, y=results3.output_price/500, name="transactions price/500")
       
       ]
layout = go.Layout(
    xaxis=dict(
        range=[
        to_unix_time(datetime.datetime(2012, 1, 1)),
            to_unix_time(datetime.datetime(2018, 5, 1))]
    )
)

fig=go.Figure(data=data, layout=layout)
py.iplot(fig)


# ### We can clearly see that with increase in bitcoins prize in the end of 2017, the transaction count also increased and the suprisingly transactions prize(total number of bitcoins transacted per day) remained significantly low.
# ### This implies that although with the recent spike in bitcoins price the number of bitcoin transactions increased but the (number of bitcoins) transacted remained constant.
# ### One more insights we can find from this plot is this that in 2016 the NUMBER OF BITCOINS transacted increased.
# ### One conclusion I can make from this is that in the end of 2017, the bitcoin value spiked upto 15000 dollars, hence there were small transactions instead of bigger transactions as in 2016 and 2012

# In[ ]:


all_months=["","January","February","March","April","May","June","July","August","September","October","November","December"]
def Bitcoin_Price_avg_monthly(year):
    new_data_f=data_f[(data_f['Timestamp']>datetime.date(year,1,1)) & (data_f['Timestamp']<datetime.date(year+1,1,1))]
    new_data_f["Timestamp"]=new_data_f.Timestamp=data_f.Timestamp.apply(lambda x: x.month)
    
    month_dictionary={}
    for i,x in new_data_f.iterrows():
        if x["Timestamp"] not in month_dictionary:
            month_dictionary[int(x["Timestamp"])]=[]
            
        month_dictionary[int(x["Timestamp"])].append(x["Weighted_Price"])
        
    for i in month_dictionary.keys():
        all_sum=month_dictionary[i]
        
        all_sum=float(sum(all_sum))/float(len(all_sum))
        month_dictionary[i]=all_sum
        
    return month_dictionary
    
def Average_transaction_count_monthly(year, average=True,  mode="Price"):
    if mode=="Price":
        new_data_ff=results3[(results3['Timestamp']>datetime.date(year,1,1)) & (transaction_count['Timestamp']<datetime.date(year+1,1,1))]
    else:
        new_data_ff=transaction_count[(transaction_count['Timestamp']>datetime.date(year,1,1)) & (transaction_count['Timestamp']<datetime.date(year+1,1,1))]
    new_data_ff["Timestamp"]=new_data_ff.Timestamp=transaction_count.Timestamp.apply(lambda x: x.month)
    
    month_dictionary={}
    key="output_price"
    if mode!="Price":
        key="output_count"
    for i,x in new_data_ff.iterrows():
        if x["Timestamp"] not in month_dictionary:
            month_dictionary[int(x["Timestamp"])]=[]
        month_dictionary[int(x["Timestamp"])].append(x[key])
    
    for i in month_dictionary.keys():
        all_sum=month_dictionary[i]
        if not average:
            all_sum=int(sum(all_sum))
        else:
            all_sum=float(sum(all_sum))/float(len(all_sum))
        month_dictionary[i]=int(all_sum)  
    return month_dictionary
    


# In[ ]:


all_months=["","January","February","March","April","May","June","July","August","September","October","November","December"]
from operator import itemgetter
def Compare_Transaction_Price_Yearly(year, average=True, mode="Price",title=""):
    title=title+" "+str(year)
    new_x=Bitcoin_Price_avg_monthly(year)
    new_x2=Average_transaction_count_monthly(year, average=average, mode=mode)
    new_x=Counter(new_x).most_common()
    new_x2=Counter(new_x2).most_common()
    new_x=sorted(new_x, key=itemgetter(0))
    new_x2=sorted(new_x2, key=itemgetter(0))
    for i in range(0,len(new_x)):
        x=list(new_x[i])
        x[0]=all_months[i+1]
        new_x[i]=tuple(x)
        x=list(new_x2[i])
        x[0]=all_months[i+1]
        new_x2[i]=tuple(x)
    
    x0=[x[0] for x in new_x]
    y0=[x[1] for x in new_x]
    x1=[x[0] for x in new_x2]
    y1=[x[1] for x in new_x2]
    plt.figure(figsize=(12,8))
    plt.subplot(1, 2, 1)
    g = sns.barplot( x=x0, y=y0, palette="winter")
    plt.xticks(rotation=90)
    plt.title('Bitcoin average Price monthly '+str(year))
    plt.xlabel("Month")
    plt.ylabel("Price in USD")

    plt.subplot(1, 2, 2)
    g = sns.barplot( x=x1, y=y1, palette="winter")
    plt.xticks(rotation=90)
    plt.title(title)
    plt.ylabel("Transaction_"+mode)
    plt.xlabel("Month")
    plt.tight_layout()


# <a id='2'></a>
# ### Lets get some more indepth analysis with respect to year
# ### Lets find out the average bitcoin price in usd for 2012 and the number of transactions in 2012
# <a id='2.1'></a>

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

Compare_Transaction_Price_Yearly(2012,average=False, mode="Count", title="Transaction count of ")


# <a id='2.2'></a>
# ## Similarly for 2013

# In[ ]:


Compare_Transaction_Price_Yearly(2013,average=False, mode="Count", title="Transaction count of ")


# <a id='2.3'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2014,average=False, mode="Count", title="Transaction count of ")


# <a id='2.4'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2015,average=False, mode="Count", title="Transaction count of ")


# <a id='2.5'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2016,average=False, mode="Count", title="Transaction count of ")


# <a id='2.6'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2017,average=False, mode="Count", title="Transaction count of ")


# <a id='3'></a>
# 
# ### Now lets find out the average no of bitcoins transacted with the average bitcoin price.
# <a id='3.1'></a>

# In[ ]:



Compare_Transaction_Price_Yearly(2012, mode="Price", title="Average transaction price of ")


# <a id='3.2'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2013, mode="Price", title="Average transaction price of ")


# <a id='3.3'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2014, mode="Price", title="Average transaction price of ")


# <a id='3.4'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2015, mode="Price", title="Average transaction price of ")


# <a id='3.5'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2016, mode="Price", title="Average transaction price of ")


# <a id='3.6'></a>

# In[ ]:


Compare_Transaction_Price_Yearly(2017, mode="Price", title="Average transaction price of ")


# # Please do give your suggesstions on what I can add in this notebook/ any mistake in logic etc.
# ## Thank you :) 
