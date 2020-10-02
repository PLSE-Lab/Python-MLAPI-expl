#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib
import plotly.plotly as py
import datetime as dt
import matplotlib.dates as mdates 
plt.style.use('fivethirtyeight')
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
import seaborn as sns 
import numpy as np
import pandas as pd
import numpy as np
import random as rnd
import re
import io
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import StandardScaler
from numpy import genfromtxt
from scipy.stats import multivariate_normal
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score , average_precision_score
from sklearn.metrics import precision_score, precision_recall_curve
get_ipython().run_line_magic('matplotlib', 'inline')

from mpl_toolkits.basemap import Basemap
from matplotlib import animation, rc
from IPython.display import HTML

import warnings
warnings.filterwarnings('ignore')

import base64
from IPython.display import HTML, display
import warnings
warnings.filterwarnings('ignore')
from scipy.misc import imread
import codecs


# Welcome to the world of Cryptocurrency or Digital Currency. <br>
#              ----
# In this notebook I will take help of various datasets related to cryptocurrencies available on kaggle 

# **Datasets used for my analysis**

# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# **Lets take a look at which Cryptocurrency ruling the market.**

# In[ ]:


def readabledate (unixtime):    
    return dt.datetime.fromtimestamp(float(unixtime))

Bitcoin = pd.read_csv('../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv',parse_dates=[0], date_parser=readabledate)
EthereumPrice = pd.read_csv('../input/ethereum-historical-data/EtherMarketCapChart.csv',parse_dates=[1], date_parser=readabledate)
EthereumAddress = pd.read_csv('../input/ethereum-historical-data/EthereumUniqueAddressGrowthRate.csv',parse_dates=[1], date_parser=readabledate)
Allcrypto = pd.read_csv('../input/all-crypto-currencies/crypto-markets.csv')


# Top 10 Cryptocurrencies in the Market
#               ---
# I have listed top 10 cryptocurrencies according to their current ranking in the market.  <br>
#  (source :- all-crypto-currencies/crypto-markets.csv)

# In[ ]:


name = Allcrypto['name'].unique()

Feature = []
currency  = []
marketval = []
v_features = name[:10]
for i, cn in enumerate(v_features):
     Feature.append(str(cn)) 
     filtered = Allcrypto[(Allcrypto['name']==str(cn))]
     temp = filtered[filtered['market'] == filtered['market'].max()]['name'].values
     temp1 = temp[0]
     tempval = filtered['market'].max()
     currency.append(temp1)
     marketval.append(tempval)

f, ax = plt.subplots(figsize=(13, 8)) 
g = sns.barplot( y = Feature,
            x = marketval,
                palette="summer")
plt.title("Top 10 Cryptocurrencies in the market")
ax.set_xticklabels(ax.get_xticks())
ax.get_yaxis().set_visible(False)
for i, v in enumerate(currency): 
    ax.text(2500000000, i, v,fontsize=18,color='brown',weight='bold')
fig=plt.gcf()
plt.show()


# There are various factors which are taken into consideration to rank cryptocurrencies like Price , Market cap etc. <br>
# Based on these we have Bitcoin who is topping the chart and Ethereum comes second. <br><br>
# **Lets start our analysis with Bitcoin first then we will go over Ethereum and at the end we will again have a showdown between Bitcoin and Ethereum **

#  Bitcoin (Cryptocurrency and worldwide payment system) <br>
#    ---
# Bitcoin do not have centralize controlling authority and it was introduced by satoshi nakamoto in years 2009.  <br>
# Bitcoin trade happens between peers based on mutual trust which at first looks very suspicious and dangerous but due to use of Blockchain technology it is almost impossible to get hacked or cheated by malicious users. Bitcoin holders are using this virtual currency for trading things freely without any meddling of authority and without any fear of getting your account freeze due to some reason. <br>
# 
# Above all Bitcoin claims to have solved below issues which are very often seen in traditional banking system <br>
# 1) Fake Currency <br>
# 2)Long Banking TIme <br>
# 3)Bank Collapse <br>
# 4)Financial Crisis(2008 financial crisis) <br>
# We do have Paypal , paytm , Citrus but again they are dependant on banks 
# 
# 
# Some more benefits<br>
# 1) No transaction fees is required - All transactions are completed within minute <br>
# 2)It is decentralized in that no one person own blockchain network <br>
# 3) It can never be manipulated or hacked because of the basic structure of Blockchain <br>
# 
# ![](https://www.mojix.com/wp-content/uploads/2017/08/Blockchain04.jpg)
# 
# **What is the backbone of this revolutionary payment system **<br>
# Its blockchain technology and it again runs 3 important technologies
#      -----
# 1) **Peer to Peer network** - Blockchain helps in maintaining the consistency of the public ledger <br>
# 2) **Private key Cryptography** - Blockchain uses Private key cryptography to secure identities and hash functions to make the Blockchain immutable <br>
# 3) **Proof of work** - The mathematical principle of proof of work ensures that the nodes automatically and continuously agree to the current state of the ledger <br>
# 
# Now the real question is , will you risk it all ? <br>
#          ---
# It becomes very imperative to think and study or take advice from experts before you enter this economy. 1 Bitcoin which used to value around 600 dollar couple of years back has now touched the sky with value of 16k dollar , this trend brings more questions in your mind whether it will continue its upword trend or will it go downward in coming years. Lets analyse this trend with the data provided here.

# I will be using below dataset for my analysis on Bitcoin <br>
# (source:-bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2018-01-08.csv')

# In[ ]:


Bitcoin.tail(5)


# In[ ]:


years = np.unique(Bitcoin['Timestamp'].dt.year)
mean_open = []
mean_volume = []
mean_close = []
mean_high = []
mean_low = []
mean_BTC = []
mean_average = []
for year in years:
    mean_volume.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Volume_(Currency)'].mean())
    mean_BTC.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Volume_(BTC)'].mean())
    mean_open.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Open'].mean())
    mean_close.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Close'].mean())
    mean_high.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['High'].mean())
    mean_low.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Low'].mean())
    mean_average.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Weighted_Price'].mean())


# In[ ]:


trace0 = go.Scatter(
    x = years, 
    y = mean_average,
    #fill='tonexty',
    mode='lines',
    name='Weighted_Price of 1 Bitcoin',
    line=dict(
        color='rgb(199, 121, 093)',
    )
)

data = [trace0]

layout = go.Layout(
    xaxis=dict(title='year'),
    yaxis=dict(title='Bitcoin price in $'),
    title=' Bitcoin value over the years ',
    showlegend = True)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# In[ ]:


Months = {
1 :'January',
2 :'February',
3 :'March',
 4 :'April',
5 :'May',
 6 :'June',
 7 :'July',
8 :'August',
9 :'September',
10 :'October',
11 :'November',
12 :'December',
}

Traces = {
1 :'trace0',
2 :'trace1',
3 :'trace2',
 4 :'trace3',
5 :'trace4',
 6 :'trace5',
 7 :'trace6',
8 :'trace7',
9 :'trace8',
10 :'trace9',
11 :'trace10',
12 :'trace11',
}

DF_Bitcoin_price = {
1 : 'Bitcoin_price1',
2 : 'Bitcoin_price2',
3 : 'Bitcoin_price3',
4 : 'Bitcoin_price4',
5 : 'Bitcoin_price5',
6 : 'Bitcoin_price6',
7 : 'Bitcoin_price7',
8 : 'Bitcoin_price8',
9 : 'Bitcoin_price9',
10 : 'Bitcoin_price10',
11 : 'Bitcoin_price11',
12 : 'Bitcoin_price12'
}


# In[ ]:


Bitcoin['Century'] = Bitcoin['Timestamp'].dt.year
j = 1
data = []
for i in range(12):
     Bitcoin_month = Bitcoin[Bitcoin['Timestamp'].dt.month == j]
     DF_Bitcoin_price[j] = Bitcoin_month.groupby(['Century'])['Weighted_Price'].mean()
     Traces[j] = go.Scatter(
         x = DF_Bitcoin_price[j].index,
         y = DF_Bitcoin_price[j].values,
         mode = 'lines',
         name = Months[j]
     )
     data.append(Traces[j]) 
     j = j + 1

layout = go.Layout(
      xaxis=dict(title='year'),
      yaxis=dict(title='Bitcoin Price$$'),
      title=('Monthly distribution of Bitcoin prices'))
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# This Virtual currency has increased 15 k  times in the last eight years.  <br>
# The Year ending of 2017 has seen a sudden jump in the Bitcoin prices (see the difference between November'17 and December'17)

# In[ ]:


years = np.unique(Bitcoin['Timestamp'].dt.year)
sum_coins = []
sum_volume = []
for year in years:
    sum_volume.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Volume_(Currency)'].sum())
    sum_coins.append(Bitcoin[Bitcoin['Timestamp'].dt.year == year]['Volume_(BTC)'].sum())
trace0 = go.Scatter(
    x = years, 
    y = sum_volume,
    fill= None,
    mode='lines',
    name='Trade volume in $ yearwise',
    line=dict(
        color='rgb(0, 255, 255)',
    )
)
data = [trace0]

layout = go.Layout(
    xaxis=dict(title='year'),
    yaxis=dict(title='Trade volume in $'),
    title='Trade volume in $ yearwise',
    showlegend = True)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Trade volume along with soaring prices of Bitcoin is kept on increasing . Dont consider 2018 as a drop in volume because we have just entered 2018 and predicting further business is kind of tough task since it depends on how many bitcoins would enter in coming year thru mining process.

# In[ ]:


trace0 = go.Scatter(
    x = years, 
    y = sum_coins,
    #fill='tonexty',
    mode='lines',
    name='Bitcoin mined /used for trading yearwise',
    line=dict(
        color='rgb(199, 121, 093)',
    )
)

data = [trace0]

layout = go.Layout(
    xaxis=dict(title='year'),
    yaxis=dict(title='Bitcoins'),
    title='Bitcoins mined /used for trading yearwise',
    showlegend = True)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# The easy money was taken out long ago and the rest are hidden under the cryptographic equivalent of miles of hard rock.
# We have just entered in 2018 . So lets wait and watch how this graph developes further. But clearly Mining is not an easy task now.

# 

# **Year 2017 at glance**
#     ---

# In[ ]:


Bitcoin_2017 = Bitcoin[Bitcoin['Timestamp'].dt.year == 2017]
months = Bitcoin_2017.groupby(Bitcoin_2017['Timestamp'].dt.month)['Volume_(BTC)'].sum().keys() 
months = np.array(months)
fig = plt.figure(figsize=(12,6))
axes1 = fig.add_axes([0.1,0.1,0.9,0.9])
axes2 = fig.add_axes([0.15,0.65,0.2,0.3])
axes3 = fig.add_axes([0.45,0.65,0.2,0.3])
axes4 = fig.add_axes([0.75,0.65,0.2,0.3])
axes5 = fig.add_axes([0.15,0.2,0.2,0.3])
axes6 = fig.add_axes([0.45,0.2,0.2,0.3])
axes7 = fig.add_axes([0.75,0.2,0.2,0.3])

#axes1.plot(years,sum_volume)
axes1.set_title('')

axes2.plot(months, Bitcoin_2017.groupby(Bitcoin_2017['Timestamp'].dt.month)['Volume_(BTC)'].sum().values, color="Blue", lw=5);
axes2.set_title('bitcoins mined/used in Year 2017')

axes3.plot(Bitcoin_2017['Timestamp'].dt.month,Bitcoin_2017['Weighted_Price'],color='Gold')
axes3.set_title('bitcoin prices in Year 2017')

axes4.plot(months, Bitcoin_2017.groupby(Bitcoin_2017['Timestamp'].dt.month)['Volume_(Currency)'].sum().values, color="Red", lw=5);
axes4.set_title('Trade volume $ in Year 2017');
axes4.set_yticklabels(axes4.get_yticks());

axes5.plot(years,sum_coins , color="Blue", lw=5);
axes5.set_title('bitcoins mined/used over the years')
axes5.set_yticklabels(axes5.get_yticks())

axes6.plot(years,mean_average,color='Gold')
axes6.set_title('bitcoin prices over the years')

axes7.plot(years,sum_volume , color="Red", lw=5);
axes7.set_title('Trade volume over the years');
axes7.set_yticklabels(axes7.get_yticks());

axes1.set_xticks([])
axes1.set_yticks([]);


# Few points which has contributed in making bitcoin even more popular in year 2017
#         ----
# 
# 1. Many countries have come forward to legalize Bitcoin <br>
# 
# 2. Recent example is Japan. Over 260000 shops are ready to begin accepting Bitcoins as payment.<br>
# 
# 3. Even Russia is planning to legalize Bitcoin and other cryptocurrency in year 2018. <br>
# 
# 4. India will examine the current framework of cryptocurrency and big giant like Ambani's of Reliance has decided to launch their own Cryptocurrency Jiocoin.

# Ethereum (Ether's Blockchain)
#    ----
# We have few other 700 cryptocurrency right now but Ethereum is the second most popular digital currency after Bitcoin. Ethereum is also backed by Blockchain technology and Many large corporations like Microsoft ,J.P. Morgan and Barclays are backing Ethereum. <br>
# 
# Unlike Bitcoin's Blockchain it features key technology called the "Smart Contract".
# Smart contracts let you exchange not just money but property,stock really anything without having to go through a lawyer,notary or some other service provider. It cuts out the middle man entirely thats why investors have taken notice and whay many think ETHREUM is a stronger and potentially more lucrative technology than the one that underpins BITCOIN. Its like ultimate vending machine,you make a deposit of the cryptocurrency for the specific product you want and all the mechanics of the transaction are automated right down to the penalty if you dont hold up your end of the bargain so whereas bitcoin is all about payment technology The Ethreum blockchain technology has other real-world applications ranging from gambling to banking.
# 
# 

# I will use below datasets for my analysis <br>
# Source :- ethereum-historical-data/EtherMarketCapChart.csv  ,
#                 ethereum-historical-data/EthereumUniqueAddressGrowthRate.csv
# 

# In[ ]:


EthereumPrice.tail(5)


# Ethereum price over the years
#      ---

# In[ ]:


trace0 = go.Scatter(
    x = EthereumPrice['UnixTimeStamp'], 
    y = EthereumPrice['Price'],
    #fill='tonexty',
    mode='lines',
    name='Price of Ethereum',
    line=dict(
        color='rgb(199, 121, 093)',
    )
)

data = [trace0]

layout = go.Layout(
    xaxis=dict(title='year'),
    yaxis=dict(title='Ethereum price in $'),
    title=' Ethereum price over the years ',
    showlegend = True)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Supply of Ethereum coins over the years
#            ---

# In[ ]:


trace0 = go.Scatter(
    x = EthereumPrice['UnixTimeStamp'], 
    y = EthereumPrice['Supply'],
    #fill='tonexty',
    mode='lines',
    name='supply of Ethereum coins',
    line=dict(
        color='rgb(199, 121, 093)',
    )
)

data = [trace0]

layout = go.Layout(
    xaxis=dict(title='year'),
    yaxis=dict(title='Supply of Ethereum coins'),
    title='supply of Ethereum coins over the years ',
    showlegend = True)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Market Cap over the Years 
#         ---

# In[ ]:


trace0 = go.Scatter(
    x = EthereumPrice['UnixTimeStamp'], 
    y = EthereumPrice['MarketCap'],
    #fill='tonexty',
    mode='lines',
    name='MarketCap in Million',
    line=dict(
        color='rgb(199, 121, 093)',
    )
)

data = [trace0]

layout = go.Layout(
    xaxis=dict(title='year'),
    yaxis=dict(title='MarketCap in Million $'),
    title='MarketCap of Ethereum over the years ',
    showlegend = True)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# No of growing Addresses over the years
#             ----

# In[ ]:


trace0 = go.Scatter(
    x = EthereumAddress['UnixTimeStamp'], 
    y = EthereumAddress['Value'],
    #fill='tonexty',
    mode='lines',
    name='No of Address',
    line=dict(
        color='rgb(199, 121, 093)',
    )
)

data = [trace0]

layout = go.Layout(
    xaxis=dict(title='year'),
    yaxis=dict(title='No of Address'),
    title='No of growing Addresses over the years ',
    showlegend = True)

fig = go.Figure(data=data, layout=layout)
iplot(fig)


# Ethereum has followed the same route of Bitcoin and has seen sudden jump in Price and market cap growth in year 2017.
# No of growing Addresses shows the constant growth of people interested in Ethereum .No has gone beyond 14 million as of now 

# In[ ]:


hfmt = mdates.DateFormatter('%Y-%m')
hfmt2017 = mdates.DateFormatter('%m-%d')


# Year 2017 at glance
#      ---

# In[ ]:


EthereumPrice_2017 = EthereumPrice[EthereumPrice['UnixTimeStamp'].dt.year == 2017]

fig = plt.figure(figsize=(12,6))
axes1 = fig.add_axes([0.1,0.1,0.9,0.9])
axes2 = fig.add_axes([0.15,0.65,0.25,0.3])
axes3 = fig.add_axes([0.45,0.65,0.25,0.3])
axes4 = fig.add_axes([0.75,0.65,0.25,0.3])
axes5 = fig.add_axes([0.15,0.2,0.25,0.3])
axes6 = fig.add_axes([0.45,0.2,0.25,0.3])
axes7 = fig.add_axes([0.75,0.2,0.25,0.3])

#axes1.plot(years,sum_volume)
axes1.set_title('')

axes2.plot(EthereumPrice_2017['UnixTimeStamp'],EthereumPrice_2017['Supply'] , color="Blue", lw=5);
axes2.set_title('Ethereum coins supply in Year 2017')
axes2.set_yticklabels(axes2.get_yticks())
axes2.set_xticklabels(axes2.get_xticks(),rotation=70)
axes2.xaxis.set_major_formatter(hfmt2017)

axes3.plot(EthereumPrice_2017['UnixTimeStamp'],EthereumPrice_2017['Price'],color='Gold')
axes3.set_xticklabels(axes3.get_xticks(),rotation=70)
axes3.xaxis.set_major_formatter(hfmt2017)
axes3.set_title('Ethereum price in Year 2017')

axes4.plot(EthereumPrice_2017['UnixTimeStamp'],EthereumPrice_2017['MarketCap'], color="Red", lw=5);
axes4.set_title('Market cap in Year 2017');
axes4.set_xticklabels(axes4.get_xticks(),rotation=70);
axes4.xaxis.set_major_formatter(hfmt2017)
axes4.set_yticklabels(axes4.get_yticks());

axes5.plot(EthereumPrice['UnixTimeStamp'],EthereumPrice['Supply'], color="Blue", lw=5);
axes5.set_title('Ethereum coins supply over the years')
axes5.set_yticklabels(axes5.get_yticks())
axes5.set_xticklabels(axes5.get_xticks(),rotation=70)
axes5.xaxis.set_major_formatter(hfmt)

axes6.plot(EthereumPrice['UnixTimeStamp'],EthereumPrice['Price'],color='Gold')
axes6.set_title('Ethereum price over the years')
axes6.set_xticklabels(axes6.get_xticks(),rotation=70)
axes6.xaxis.set_major_formatter(hfmt)

axes7.plot(EthereumPrice['UnixTimeStamp'],EthereumPrice['MarketCap'], color="Red", lw=5);
axes7.set_title('Market cap over the years');
axes7.set_xticklabels(axes7.get_xticks(),rotation=70)
axes7.xaxis.set_major_formatter(hfmt)
axes7.set_yticklabels(axes7.get_yticks());

axes1.set_xticks([])
axes1.set_yticks([]);


# I will use below dataset for Candlestick Chart
#            ----

# In[ ]:


Allcrypto.head()


# **Small introduction to understand candlestick Chart which i will draw to showcase trend of Open/High/Low/Close values of Bitcoin and Ethereum over the years**
# ![](https://steemit-production-imageproxy-thumbnail.s3.amazonaws.com/U5dsCh8183wAuMwFtaqQate3iRmwsTD_1680x8400)

# ![](http://swingalpha.com/resources/Pictures/Using%20Candlestick%20Patterns.gif)

# **Candlestick OHLC graph for Bitcoin**
#       ----

# In[ ]:


Bitcoin = Allcrypto[Allcrypto['ranknow'] == 1]
Ethereum = Allcrypto[Allcrypto['ranknow'] == 2]
Ripple  = Allcrypto[Allcrypto['ranknow'] == 3]
from matplotlib.finance import candlestick_ohlc
BitcoinOHLC = Bitcoin[['date','open','high','low','close']]
EthereumOHLC = Ethereum[['date','open','high','low','close']]
RippleOHLC = Ripple[['date','open','high','low','close']]
import matplotlib.dates as mdates
BitcoinOHLC['date'] = pd.to_datetime(BitcoinOHLC['date'])
BitcoinOHLC['date'] = mdates.date2num(BitcoinOHLC['date'].astype(dt.date))
RippleOHLC['date'] = pd.to_datetime(RippleOHLC['date'])
RippleOHLC['date'] = mdates.date2num(RippleOHLC['date'].astype(dt.date))
EthereumOHLC['date'] = pd.to_datetime(EthereumOHLC['date'])
EthereumOHLC['date'] = mdates.date2num(EthereumOHLC['date'].astype(dt.date))


# In[ ]:


f,ax=plt.subplots(figsize=(15,11))
ax.xaxis_date()
plt.xlabel("Date")
candlestick_ohlc(ax,BitcoinOHLC.values,width=5, colorup='g', colordown='r',alpha=0.75)
ax.set_xticklabels(ax.get_xticks(),rotation=70)
ax.xaxis.set_major_formatter(hfmt)
plt.ylabel("Price")
plt.legend()
plt.show()


# **Candlestick OHLC graph for Ethereum**
#       ----

# In[ ]:


f,ax=plt.subplots(figsize=(15,11))
ax.xaxis_date()
plt.xlabel("Date")
candlestick_ohlc(ax,EthereumOHLC.values,width=5, colorup='g', colordown='r',alpha=0.75)
ax.set_xticklabels(ax.get_xticks(),rotation=70)
ax.xaxis.set_major_formatter(hfmt)
plt.ylabel("Price")
plt.legend()
plt.show()


# Lets compare candlestick graph of Bitcoin and Ethereum for last few months of year 2017
#              ---

# In[ ]:


EthereumOHLC = EthereumOHLC[EthereumOHLC['date'] > 736630.0]
BitcoinOHLC = BitcoinOHLC[BitcoinOHLC['date'] > 736630.0]

fig = plt.figure(figsize=(12,12))
axes1 = fig.add_axes([0.1,0.1,0.9,0.9])
axes2 = fig.add_axes([0.15,0.65,0.8,0.3])
axes3 = fig.add_axes([0.15,0.2,0.8,0.3])

axes2.xaxis_date()
candlestick_ohlc(axes2,BitcoinOHLC.values,width=2, colorup='g', colordown='r',alpha=0.70)
axes2.set_xticklabels(axes2.get_xticks(),rotation=70)
axes2.xaxis.set_major_formatter(hfmt)
axes2.set_title('Bitcoin Candlestick chart for last few months of 2017');
#axes2.scatter(736688.0, 17000,marker="o", color="white", s=50000, linewidths=0)
axes2.annotate('Bearish Engulfing', (736684.0,19000),fontsize=14,rotation=0,color='r')
axes2.annotate('.', xy=(736684.0,17000), xytext=(736688.0, 19000),
            arrowprops=dict(facecolor='Red', shrink=0.06),
            )
axes2.annotate('.', xy=(736691.0,16000), xytext=(736689.0, 19000),
            arrowprops=dict(facecolor='Red', shrink=0.06),
            )


axes3.xaxis_date()
candlestick_ohlc(axes3,EthereumOHLC.values,width=2, colorup='g', colordown='r',alpha=0.70)
axes3.set_xticklabels(axes3.get_xticks(),rotation=70)
axes3.xaxis.set_major_formatter(hfmt)
axes3.set_title('Ethereum Candlestick chart for last few months of 2017');
axes3.annotate('Bullish Engulfing', (736684.0,1300),fontsize=14,rotation=0,color='g')
axes3.annotate('.', xy=(736695.0,850), xytext=(736688.0, 1300),
            arrowprops=dict(facecolor='Green', shrink=0.06),
            )


axes1.set_xticks([])
axes1.set_yticks([]);


# **2018 seems to be a year full of surprizes** <br>
#  Entry of other cryptoassets in the market will definately change the market condition upside down  <br><br>
# Few observations <br>
# 1) We see frequency of Bearish engulfing has increased post 2017 and Bitcoin prices has strted coming down. <br>
# 2) We see frequency of Bullish engulfing has increased post 2017 and Ethereum prices are on a rise. 
# 
# 

# More to come 
#      ---     
