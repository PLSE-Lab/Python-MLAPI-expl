#!/usr/bin/env python
# coding: utf-8

# ## Strategy and Rules
# 
# - We are going to apply a trading strategy which is possibly the easiest, that is: "Moving Average Crossover Strategy".
# 
# ### Terms:
# - N_moving average : Average of any attribute calculated over last N observed values. Moving average smoothens the curve of a trend line. And, gives us better understanding of the trend.
# - Fast signal      : Smaller moving average (ex. 10 days / 20 days moving average) It is called fast signal because it moves closely to the actual trend. (We'll see this later in this notebook)
# - Slow signal      : Bigger moving average (ex. 30 dys, 50 days, 200 days) It is called slow signal because it changes less with the changes of values. (Will be clearer with the visualization.)
# 
# ### Rules:
# - When fast signal crosses over slow signal; it indicates a change in trend.
# - If fast signal has been lower than the slow signal and now has a crossover and gets higher than the slow signal; it indicates an upward trend.
# - If fast signal has been higher than the slow signal and now has a crossover and gets smaller than the slow signal; it indicates an downward trend.
# 
# ### Strategy:
# - We'll buy stock at the positive crossover, i.e. on a signal of upward trend.
# - We'll sell stocks at the begetive crossover, i.e. on a signal of negative trend.
# - This way we'll be able to buy a stock in the starting of an upward trend and sell them as soon as the trend starts going downwards.
# - We'll only nuy one share of stock at a time.

# In[ ]:


### Get all the libraries
import datetime as dt                    ## To work with date time
import pandas as pd                      ## Work with datasets
import pandas_datareader.data as web     ## Get data from web
import numpy as np                       ## Linear Algebra
import plotly.express as px              ## Graphing/Plotting- Visualization
import plotly.graph_objects as go        ## Graphing/Plotting- Visualization
pd.set_option('display.max_columns', 50) ## Display All Columns
import warnings
warnings.filterwarnings("ignore")       ## Mute all the warnings


# - Nifty is an index of 50 most actively traded companies in India from various Sectors.
# - Here, I have a list of NIFTY companies. Since, these are the companies that are traded most; I thought it will be a good idea to work on one of these big companies.

# In[ ]:


## Names of companies on NIFTY index
company_list = ['ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS', 'BAJAJFINSV.NS', 'BPCL.NS',
                'BHARTIARTL.NS', 'INFRATEL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
                'GAIL.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS',
                'HDFC.NS', 'ICICIBANK.NS', 'ITC.NS', 'IOC.NS', 'INDUSINDBK.NS', 'INFY.NS', 'JSWSTEEL.NS','KOTAKBANK.NS',
                'LT.NS', 'M&M.NS', 'MARUTI.NS', 'NTPC.NS', 'NESTLEIND.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS',
                'SHREECEM.NS', 'SBIN.NS', 'SUNPHARMA.NS', 'TCS.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TECHM.NS', 
                'TITAN.NS', 'UPL.NS', 'ULTRACEMCO.NS', 'VEDL.NS', 'WIPRO.NS', 'ZEEL.NS']
print(company_list)


# - WE'LL proceed with Bajaj Fiinserv here. But, the same approach can be applied on any other company as well.
# - Ticker for Bajaj Finserv is "BAJAJFINSV.NS". We'll use this to get historical data of the company.

# ## Getting data from the web.

# In[ ]:


start = dt.datetime(2010, 1, 1) ## Start Date of data
end = dt.datetime(2020, 5, 5)  ## End Date of data

target = 'BAJAJFINSV.NS'
df = web.DataReader(target, 'yahoo', start, end)
columns = df.columns
index = df.index

df.head()


# - As we can see; we have got historical market data for our target company here.
# 
# #### Here:
# - High: Highest price of a share of company on any perticular day
# - Low: lowest price of a share of company on any perticular day.
# - Open: Price to which the share market opened that day.
# - Close: Price on which share market closed that day.
# - Volume: The number of shares exchanged between buyers & sellers on that day.
# - Adj Close: Closing price of a stock after accounting for any corporate actions.
# 
# #### Now, to apply "Moving Average Crossover Strategy" we only need closing price so; we'll drop other attributes from the data.

# ## Generating moving averages.

# In[ ]:


## It's safe to have a copy of the dataset and work on that.
Bj_fs = df.copy()
Bj_fs = Bj_fs[['Close']]
Bj_fs['10_ma'] = Bj_fs['Close'].rolling(10).mean()  ## 10 days moving average.
Bj_fs['30_ma'] = Bj_fs['Close'].rolling(30).mean()  ## 30 days moving average.


fig = go.Figure(data=[
    go.Scatter(x = Bj_fs.index, y=Bj_fs.Close, name='Close', fillcolor='blue'),
    go.Scatter(x = Bj_fs.index, y=Bj_fs['10_ma'], name='10_ma', fillcolor='red'),
    go.Scatter(x = Bj_fs.index, y=Bj_fs['30_ma'], name='30ma', fillcolor='green'),
])
fig.update_layout(title="Moving Average 10 and Moving Average 30 with Close",
                 xaxis_title='Date', yaxis_title='Value')
fig.show()


# ### Explaination of the graph:
# - We can see that the red line(10_ma) fluctuated more and moves more closely with the blue line(Original price). This is what i was talkiing about in the starting of this notebook.
# - The green line(30_ma) is more curvy then the other two and changes less frequently.

# ## Implementing the strategy

# #### Explaination to the strategy:
# - temp == -1 menas that money is deducted from our account as we've bought shares.
# - temp == 0 means that we have not done any transection.
# - temp == +1 means that money has come into our account as we've sold our shares.

# In[ ]:


Bj_fs.reset_index(inplace=True)


# In[ ]:


## -1: share bught :: 1 = share sold
## Strategy:: if intersection goes down: Sell; if intersection goes up: Buy

temp = [0]
Shares=[0]
for i in Bj_fs.index[1:]:
    if (Bj_fs.iloc[i-1, 2]<=Bj_fs.iloc[i-1, 3]) and (Bj_fs.iloc[i, 2]>Bj_fs.iloc[i, 3]):
        temp.append(-1)
        Shares.append(Shares[-1]+1)
    elif (Bj_fs.iloc[i-1, 2]>=Bj_fs.iloc[i-1, 3]) and (Bj_fs.iloc[i, 2]<Bj_fs.iloc[i, 3]):
        if Shares[-1]>0:
            temp.append(Shares[-1])
            Shares.append(0)
        else:
            temp.append(0)
            Shares.append(0)
    else:
        temp.append(0)
        Shares.append(Shares[-1])
temp = np.array(temp)
Shares = np.array(Shares)


# ## Model Performance
# ### Now; we'll see how our model has performed. For that, let's define some new attributes:
# - Direction is same as temp: -1=shares bought, +1=shares sold.
# - Cur_Shares is No of shares that we hold on that perticular day.
# - Transection stimulates our bank account transection. i.e. money deducted for buying shares or money came in as we sold our shares.
# - Profits is just the cumulative sum of transections. It tells us how much money we have made till date.

# In[ ]:


Bj_fs['Direction'] = temp
Bj_fs['Cur_Shares'] = Shares
Bj_fs.dropna(inplace=True) ## Time when we don't have moving avergae of
Bj_fs['Transection'] = Bj_fs['Direction'] * Bj_fs['Close'] ## Bought or sold shares
Bj_fs['Profits'] = Bj_fs['Transection'].cumsum() ## Track of pfofit
Bj_fs.set_index('Date', inplace=True)


# In[ ]:


fig = go.Figure(data=[
    go.Scatter(x = Bj_fs.index, y=Bj_fs['Close'], name="Daily Close"),
    go.Scatter(x = Bj_fs.index, y=Bj_fs['10_ma'], name="10 MA"),
    go.Scatter(x = Bj_fs.index, y=Bj_fs['30_ma'], name = "30 MA"),
    go.Scatter(x = Bj_fs.index, y=Bj_fs['Transection'], name = "Transections"),
    go.Scatter(x = Bj_fs.index, y=Bj_fs['Profits'], name="Total Profit")
])
fig.update_layout(title = "Trading History:",
                 xaxis_title="Date_Time", yaxis_title="Value")
fig.show()
print("Total Profit Till Day:", Bj_fs['Profits'][-1])


# #### Here, one thing to keep in mind is that, profit going negetive does not mean loss, it is just that we are buying shares again. But after that on selling, if we don't get higher than before then it is a loss.

# ## Transection history:

# In[ ]:


## Transection history
bought = Bj_fs.loc[Bj_fs['Direction']==-1]
sold = Bj_fs.loc[Bj_fs['Direction']==1]
trans = pd.concat([bought, sold]).sort_index()
trans


# ### We've made a profit of total 8233 rupee on an investment of 350 rupee in a period of 10 years.
# 
# - Let's see what it means:

# ## Observations

# In[ ]:


investment = abs( trans['Profits'].iloc[0])
inflation = 0.055

fd_return = investment * (1+0.06)**10
trading_profit = trans['Profits'].iloc[-1]

cur_value = investment * (1-inflation)**10

print("Current value of our money if we woulld have just saved it:", cur_value)
print("Total value of our money if we woulld have deposited in bank:", fd_return)
print("Total value of our money with trading:", trading_profit)


# - Link for inflation rate: https://en.wikipedia.org/wiki/Inflation_in_India
# 
# #### We have observed that:
# - Saving money is not a good idea. It just loses its value over time.
# - Bank deposit is a better idea than just saving it.
# - Our trading strategy has given us a huge return on out meer investment.

# ## Exiting? Really?
# 
# #### We've made a huge profit and we are happy about it. But, there is more to this tradig strategy. Let's see if it'll work for another company or not? We'll take Airtel for furthur analysis.

# In[ ]:


target = 'BHARTIARTL.NS'
df = web.DataReader(target, 'yahoo', start, end)
columns = df.columns
index = df.index

airtel = df.copy()
airtel = airtel[['Close']]
airtel['10_ma'] = airtel['Close'].rolling(10).mean()
airtel['30_ma'] = airtel['Close'].rolling(30).mean()


fig = go.Figure(data=[
    go.Scatter(x = airtel.index, y=airtel.Close, name='Close'),
    go.Scatter(x = airtel.index, y=airtel['10_ma'], name='10_ma'),
    go.Scatter(x = airtel.index, y=airtel['30_ma'], name='30ma'),
])
fig.update_layout(title="Moving Average 10 and Moving Average 30 with Close",
                 xaxis_title='Date', yaxis_title='Value')
fig.show()



airtel.reset_index(inplace=True)

## -1: share bught :: 1 = share sold
## Strategy:: if intersection goes down: Sell; if intersection goes up: Buy

temp = [0]
Shares=[0]
for i in airtel.index[1:]:
    if (airtel.iloc[i-1, 2]<=airtel.iloc[i-1, 3]) and (airtel.iloc[i, 2]>airtel.iloc[i, 3]):
        temp.append(-1)
        Shares.append(Shares[-1]+1)
    elif (airtel.iloc[i-1, 2]>=airtel.iloc[i-1, 3]) and (airtel.iloc[i, 2]<airtel.iloc[i, 3]):
        if Shares[-1]>0:
            temp.append(Shares[-1])
            Shares.append(0)
        else:
            temp.append(0)
            Shares.append(0)
    else:
        temp.append(0)
        Shares.append(Shares[-1])
temp = np.array(temp)
Shares = np.array(Shares)

airtel['Direction'] = temp
airtel['Cur_Shares'] = Shares
airtel.dropna(inplace=True) ## Time when we don't have moving avergae of
airtel['Transection'] = airtel['Direction'] * airtel['Close'] ## Bought or sold shares
airtel['Profits'] = airtel['Transection'].cumsum() ## Track of pfofit
airtel.set_index('Date', inplace=True)


fig = go.Figure(data=[
    go.Scatter(x = airtel.index, y=airtel['Close'], name="Daily Close"),
    go.Scatter(x = airtel.index, y=airtel['10_ma'], name="10 MA"),
    go.Scatter(x = airtel.index, y=airtel['30_ma'], name = "30 MA"),
    go.Scatter(x = airtel.index, y=airtel['Transection'], name = "Transections"),
    go.Scatter(x = airtel.index, y=airtel['Profits'], name="Total Profit")
])
fig.update_layout(title = "Trading History:",
                 xaxis_title="Date_Time", yaxis_title="Value")
fig.show()

print("Total Profit Till Day:", airtel['Profits'][-1])
print("Maximum Profit made in history:", airtel['Profits'].max())


# ## Final Thoughts
# ### Not so promising anymore?
# - For this new company, we are ssing that our track of profits hasn't been that great. And, we have not made any profits yet.
# - The reason is that "Moving average crossover strategy" actually does not perform that great in reality. 
# - The first company that we took has an overall upward trend. That is why we were able to get such huge profit.
# - Also; we've implemented over 10 days and 30 days period. In reality, whenever this strategy is applied, it never does good for short period of times, We stake bigger mmoving_averages.
# - Even after that, this strategy does not promise good returns.
# - The main reason is behind the intuition for the strategy. This strategy is used as it tells us about the trend. And, that is exactly it fails may times.
# - It tells us about the trend when the trend has already started. Not before that.
# 
# #### I used this here to demonstrate how we can automate trading with pyhton. Which is called algorithemic trading. This perticular strategy is not actually used for trading. But this gives an idea of how algorithemic trading works. We use more advanced techniques in real which, you'll see in my upcoming notebooks.
# 
# #### Thank You! I hope this notebook gave you a fair knowledge of how we can use python for trading and of "Moving Average Crossover Strategy" .Feel free to comment for any query or suggestions. I would love to know your thoughts on this one.

# In[ ]:




