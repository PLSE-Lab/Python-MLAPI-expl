#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#General Imports go here: 
#Import Libraries
from scipy.stats import norm
import pandas as pd
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import matplotlib.mlab as mlab
import seaborn as sns
from datetime import timedelta
from matplotlib import style
import plotly.graph_objects as go
from scipy.signal import savgol_filter as smooth
from scipy.signal import argrelextrema as extrema
get_ipython().run_line_magic('matplotlib', 'inline')
#Ignore all warnings:
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


# In[ ]:


#Setup a cloud machine GCP to run :
import pandas as pd
import numpy as np
from scipy import stats
import requests
import time
from datetime import datetime
import pytz
from google.cloud import bigquery
from google.cloud import storage
import pyarrow

def cloud_machine(event, context):
    # Get the api key from cloud storage
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('')
    blob = bucket.blob('')
    api_key = blob.download_as_string()
    
    # Check if the market was open today
    today = datetime.today().astimezone(pytz.timezone("Germany/Frankfurt"))
    today_fmt = today.strftime('%Y-%m-%d')

    market_url = ''

    params = {
        'apikey': api_key,
        'date': today_fmt
        }
    
    request = requests.get(
        url=market_url,
        params=params
        ).json()
    
def run_query():
#SQL QUERY:
    client = bigquery.Client()

# Load the historical stock data from DB
    sql_hist = """
                SELECT
                  symbol,
                  closePrice,
                  date
                FROM 
                  `Stocks_Data`
                """

    df = client.query(sql_hist).to_dataframe()


# In[ ]:



#Ignore all warnings:
warnings.filterwarnings('ignore')
#list all files in the folder:
print(os.listdir("../input"))


# User Input : Define Short/Long Window 
# A.Generate Trading Signals (Buy/Sell) using a moving average for a time window

# #Functions not to be used: 
# 
# def get_rsi(df, length):
#        # Get the difference in price from previous step
#     delta = df['Close'].diff()
#        # Get rid of the first row, which is NaN since it did not have a previous
#        # row to calculate the differences
#     delta = delta[1:]
#        # Make the positive gains (up) and negative gains (down) Series
#     up, down = delta.copy(), delta.copy()
#     up[up < 0.0] = 0.0
#     down[down > 0.0] = 0.0
#        # Calculate the EWMA
#     roll_up1 = up.ewm(com=(length-1), min_periods=length).mean()
#     roll_down1 = down.abs().ewm(com=(length-1), min_periods=length).mean()
#        # Calculate the RSI based on EWMA
#     RS1 = roll_up1 / roll_down1
#     RSI1 = 100.0 - (100.0 / (1.0 + RS1))
#     
#     return RSI1
#     
#     def basic_features(df, short, long):
#     # Bundle of baseline features that were initially applied. Short and Long are windows given in days
#     def slope(y):
#         x = np.array(range(len(y)))
#         m, b = np.polyfit(x, y, 1)
#         return m
#         
#     def acc(y):
#         x = np.array(range(len(y)))
#         A, v, x0 = np.polyfit(x, y, 2)
#         g = 0.5*A
#         return g
#      # Counting consecutive candles 
# def candle_counts(df):
#     df['color'] = (df.Close >= df.Open).astype(np.uint8)
#     df['region'] = (df.color != df.color.shift()).cumsum()
# 
#     gp = df.groupby(['region', 'color']).size()
#     gp = pd.DataFrame(gp)
# 
#     gp.columns = ['region_len']
#     df = df.reset_index().merge(gp, on=['region', 'color'], how='left').set_index('index')
#     df['sgn_region_len'] = df['region_len']
#     df.loc[df.color == 0, 'sgn_region_len'] *= -1
#     return df
#     
#     # A few general rsi indicators to help indicate up/down trends
# def bindicators(df):
#     df['RSI>70'] = np.where(df['RSI']>70., 1., 0.)
#     df['RSI<30'] = np.where(df['RSI']<30., 1., 0.)
#     df['RSI<20'] = np.where(df['RSI']<20., 1., 0.)
#     
#     df['MA_s>MA_l'] = np.where(df['MA_s']>df['MA_l'], 1., 0.)
#     
#     return df

# In[ ]:


#Functions :
def get_data (file):
    df = pd.read_csv(str(file))
    #df = df.reindex(index=df.index[::-1])
    df = df.rename(columns = {'Date': 'date'})
    #df.date = pd.to_datetime(df.date)
    #df.index = df['date']
    df = df.fillna(method = 'ffill')
    #df.index = df.index.rename('index')
    return df

def buysignal(df):
            if df>0:
                return 1
            if df<0:
                return 0
            
def sellsignal(df):
            if df>0:
                return 0
            if df<0:
                return -1
            
def basic_features(df, short, long):
    # Bundle of baseline features that were initially applied. Short and Long are windows given in days
    
       # Short and long moving averages
    df['MA_s'] = df['Close'].rolling(window = short, min_periods = 1, center = False).mean()
    df['MA_l'] = df['Close'].rolling(window = long, min_periods = 1, center = False).mean()
    df['PctChange'] = df['Close'].pct_change()
    return df

# The following functions are for plotting ideal targets for buy and sell signals
def mac_target(df, short, long):
    short = short
    long = long
    signal_df = pd.DataFrame(index = df.index)
    signal_df['signal'] = 0.0
    signal_df['short_mav'] = df['Close'].rolling(window = short, min_periods = 1, center = False).mean()
    signal_df['long_mav'] = df['Close'].rolling(window = long, min_periods = 1, center = False).mean()
    signal_df['signal'][short:] = np.where(signal_df['short_mav'][short:] > signal_df['long_mav'][short:], 1.0, 0.0)
    signal_df['positions'] = signal_df['signal'].diff()
    df['MAC_TARGET'] = signal_df['positions']
    
    return (df, signal_df)


# In[ ]:


#User Input  
short_window = 30
long_window = 120
#data = pd.read_csv('../input/AMZN.csv')
data=get_data('../input/AMZN.csv')
basic_features(data,short_window,long_window)
#Here to select the stock :
#data=data[data.Stock_Name=='AMZN'] # Select Other Stock Names 
# Convert Date to datetime
data= basic_features(data,short_window,long_window)
data = data.fillna(0)


# In[ ]:


#data.info()
#data.tail()


# In[ ]:


df, signal_df = mac_target(data,short_window,long_window)
#df, smoothsig = smooth_target(df, 12)
#df.Smooth.head()
signal_df.head()
#smoothsig.head()


# In[ ]:


fig = plt.figure(figsize = (90,30))
plt1 = fig.add_subplot(111, ylabel = 'Close Price')
plt1.plot(df.index, df.Close, color = 'blue', label = "Close", linewidth = 4.0)
plt1.plot(signal_df.index, signal_df.short_mav, color = 'green',
          linewidth = 4.0, linestyle = "--", label = "Short MAV")
plt1.plot(signal_df.index, signal_df.long_mav, color = 'magenta',
          linewidth = 4.0, linestyle = ":", label = "Long MAV")


plt1.plot(signal_df.short_mav[signal_df.positions == -1.0].index, signal_df.short_mav[signal_df.positions == -1], 
          'v', markersize = 30, color = 'red',label='Sell')
plt1.plot(signal_df.short_mav[signal_df.positions == 1.0].index, signal_df.short_mav[signal_df.positions == 1], 
          '^', markersize = 30, color = 'green',label='Buy')

plt.title('Ideal Trading Signals', size = 60)
plt.xlabel('Date', size = 40)
plt.ylabel('Close Price', size = 40)
plt.legend(fontsize = 40)
plt.grid(alpha = 0.4, color = 'black')
plt.savefig('smoothie.jpg', bbox_inches = 'tight')
plt.show()


# df['target_region'] = (df.SMOOTH_TARGET != df.SMOOTH_TARGET.shift()).cumsum()
# mid_region_days = df.groupby('target_region').apply(lambda x: x.index.tolist()).apply(lambda x: x[int(len(x)//2)]).tolist()
# mid_region_days[:5]

# In[ ]:


prices = df['Adj Close']
positions =signal_df.positions
actions = ['Buy', 'Hold', 'Sell']
hist=10
n = len(prices)
n_train = int(n * 0.25)
train_prices = prices[:n_train]
test_prices = prices[n_train:]
budget = 1000
num_stocks = 0
share_value = 0
transitions = list()
history = []
portfolio_history = []


# print(len(prices),len(positions))
# range(len(prices) - hist - 1)
# ans=1 % 1000
# print('progress {:.2f}%'.format(float(100*ans) / (len(prices) - hist - 1)))
# share_value = float(prices[2+ hist])
# print (share_value)

# #Portfolio Simulation:
#     
# for i in range(len(prices) - hist - 1):
#     if i % 1000 == 0:
#         print('progress {:.2f}%'.format(float(100*i) / (len(prices) - hist - 1)))
#         current_state = np.asmatrix(np.hstack((prices[i:i+hist], budget, num_stocks)))
#         current_portfolio = budget + num_stocks * share_value
#         share_value = float(prices[i + hist])
#         # Update portfolio values based on action
#         if  positions[i] == 1 and budget >= share_value:
#             budget -= share_value
#             num_stocks += 1
#             history.append((prices[i], 'BUY'))
#             portfolio_history.append((current_portfolio, 'BUY'))
#             print('BUY -- Current Portfolio--- {:.2f} Euros' .format(float(current_portfolio)))
#             print('STOCKS  -- Current Stocks--- {:.2f}' .format(float(num_stocks)))
#         elif positions[i] == 0 and num_stocks > 0:
#             budget += share_value
#             num_stocks -= 1
#             history.append((prices[i],'SELL'))
#             portfolio_history.append((current_portfolio,'SELL'))
#             print('SELL -- Current Portfolio {:.2f} %' .format(current_portfolio))
#             print('STOCKS  -- Current Stocks--- {:.2f}' .format(float(num_stocks)))
#         else:
#             action = 'Hold'
#             history.append((prices[i], 'HOLD'))
#             portfolio_history.append((current_portfolio, 'HOLD'))
#             print('HOLD -- Current Portfolio %' .format(current_portfolio))
#             print('STOCKS  -- Current Stocks--- {:.2f}' .format(float(num_stocks)))
#         # Compute new portfolio value after taking action
#         new_portfolio = budget + num_stocks * share_value
#     # Compute final portfolio worth
# portfolio = budget + num_stocks * share_value
# print(portfolio)

# In[ ]:


window=20
data['Buy']=np.zeros(len(data))
data['Sell']=np.zeros(len(data))
data['RMax'] = signal_df['long_mav']
data['RMin'] = signal_df['short_mav']
data.loc[data['RMax'] < data['Adj Close'], 'Buy'] = 1
data.loc[data['RMin'] > data['Adj Close'], 'Sell'] = -1

fig,ax1 = plt.subplots(1,1)
ax1.plot(data['Adj Close'])
y = ax1.get_ylim()
ax1.set_ylim(y[0] - (y[1]-y[0])*0.4, y[1])
ax2 = ax1.twinx()
ax2.set_position(matplotlib.transforms.Bbox([[0.000001,0.00001],[5.05,0.99]]))
ax2.plot(data['Buy'], color='green',label='Buy')
ax2.plot(data['Sell'], color='red',label='Sell')


# In[ ]:


# Test 1 : This is with one Window Buy Signal:

#Trading Strategy (df_trade) Moving Average:
df_trade=data.copy()

fig = go.Figure(data=[
    go.Scatter(x = df_trade.index, y=df_trade['Adj Close'], name='Adj Close', fillcolor='blue'),
    go.Scatter(x = df_trade.index, y=df_trade['RMin'], name='short_ma', fillcolor='red'),
    go.Scatter(x = df_trade.index, y=df_trade['RMax'], name='long_ma', fillcolor='green'),
])
fig.update_layout(title="Moving Average SHORT and Moving Average LONG with Adj Close",
                 xaxis_title='Date', yaxis_title='Value')
fig.show()



# In[ ]:


#Test with 2 Windows 
# Trading Strategy with Short_window /Long_window 
df=data.copy()
short = short_window #See user input
long = long_window   #See user input
signal_df = pd.DataFrame(index = data.index)
signal_df['signal'] = 0.0
signal_df['short_mav'] = df['MA_s']
signal_df['long_mav'] =  df['MA_l']
signal_df['signal'][short:] = np.where(signal_df['short_mav'][short:] > signal_df['long_mav'][short:], 1.0, 0.0)
signal_df['positions'] = signal_df['signal'].diff()

#Plot the Signals Buy/Sell:
fig = plt.figure(figsize = (60,15))
plt1 = fig.add_subplot(111, ylabel = 'Adj Close Price')
plt1.plot(signal_df.index, signal_df.short_mav, color = 'green',
          linewidth = 1.0, linestyle = "--", label = "Short MAV")
plt1.plot(signal_df.index, signal_df.long_mav, color = 'blue',
          linewidth = 1.0, linestyle = ":", label = "Long MAV")
plt1.plot(df.index, df.Close, color = 'red', label = "Adj Close", linewidth = 1.0)
plt1.plot(signal_df.short_mav[signal_df.positions == -1.0].index, signal_df.short_mav[signal_df.positions == -1], 
          'v', markersize = 15, color = 'red',label='Sell')
plt1.plot(signal_df.short_mav[signal_df.positions == 1.0].index, signal_df.short_mav[signal_df.positions == 1], 
          '^', markersize = 15, color = 'green',label='Buy')
plt.title('AMZN Trading Signals', size = 30)
plt.xlabel('Date', size = 20)
plt.ylabel('Adj Close Price', size = 20)
plt.legend(fontsize = 20)
plt.grid(alpha = 0.25, color = 'gray')
plt.show()


# In[ ]:


#Portfolio Simulation with Rmax/Rmin Use Portfolio Simulation 2 Instead: OUTDATED
df_trade=data.copy()
df_trade.reset_index(inplace=True)
trade  = [0]
Stocks = [0]
for i in df_trade.index[1:]:
    
    if (df_trade['RMin'].iloc[i-1]<=df_trade['RMax'].iloc[i-1]) and (df_trade['RMin'].iloc[i]>df_trade['RMax'].iloc[i]):
        trade.append(-1)
        Stocks.append(Stocks[-1]+1)
        
        
    elif (df_trade['RMin'].iloc[i-1]>=df_trade['RMax'].iloc[i-1]) and (df_trade['RMin'].iloc[i]<df_trade['RMax'].iloc[i]):
        
        if Stocks[-1]>0:
            trade.append(Stocks[-1])
            Stocks.append(0)

        else:
            trade.append(0)
            Stocks.append(0)

    else:
        trade.append(0)
        Stocks.append(Stocks[-1])
        #print('Your Stocks are {:.2f}' .format(float(Stocks[i])))
        #print('Your Trade  are {:.2f}' .format(float(trade[i])))
    
trade = np.array(trade)
Stocks = np.array(Stocks)
df_trade['Direction'] = trade[0:]
df_trade['My Stocks'] = Stocks[0:]
df_trade.dropna(inplace=True) 
df_trade['Transaction'] = df_trade['Direction'] * df_trade['Adj Close'] ## BUY/SELL stocks
df_trade['Profits'] = df_trade['Transaction'].cumsum() ## Profit Log
fig = go.Figure(data=[
go.Scatter(x = df_trade.index, y=df_trade['Adj Close'], name="Adj Close"),
go.Scatter(x = df_trade.index, y=df_trade['RMin'], name="short MA"),
go.Scatter(x = df_trade.index, y=df_trade['RMax'], name = "long MA"),
go.Scatter(x = df_trade.index, y=df_trade['Transaction'], name = "Transactions"),
go.Scatter(x = df_trade.index, y=df_trade['Profits'], name="Total Profit")
])
fig.update_layout(title = "Trading History:",
                 xaxis_title="Date_Time", yaxis_title="Value")
fig.show()


# In[ ]:


df_trade.loc[df_trade['Direction']==1] #BUY 
df_trade.loc[df_trade['Direction']==1] #SELL
df_trade.loc[df_trade['Direction']==0] #HOLD


# In[ ]:


#Simple Return is same As percentage change:
data['Simple Return']=(data["Adj Close"]/data['Adj Close'].shift(1))-1
#Check Log returns instead
log_returns =np.log(1+data['Adj Close'].pct_change())
data['Logs']=log_returns
#Daily Return Plots for understanding the Stock
f, ax = plt.subplots(2,2,figsize = (11,8))
k1 = sns.lineplot(y = data["Logs"], x = data.index.get_level_values(0), ax = ax[0,0])
k2 = sns.lineplot(y = data["Adj Close"], x = data.index.get_level_values(0), ax = ax[1,1])
k2 = sns.lineplot(y = data["Close"], x = data.index.get_level_values(0), ax = ax[0,1])
k2 = sns.lineplot(y = data["Simple Return"], x = data.index.get_level_values(0), ax = ax[1,0])
#Visualise MAs
#Calculate moving average not for the assignment only for our understanding
ma_days = [20, 50, 200]
for ma in ma_days:
    new_columns = '%s day MA' %(str(ma))
    data[new_columns] = data['Adj Close'].rolling(ma).mean()
data[['Adj Close','20 day MA', '50 day MA', '200 day MA']].plot(subplots=False,
       figsize=(12,8), title="AMZN price/moving averages")

#Create a dataset with all the moving average values 
m = [20,50,200] # define here the windows
ma_20 = data['Adj Close'].rolling(20).mean()
ma_50 = data['Adj Close'].rolling(50).mean()
ma_200 = data['Adj Close'].rolling(200).mean()
ma_concat = pd.concat([ma_20,ma_50,ma_200], axis=1)


# In[ ]:


data['Buy']=np.zeros(len(data))
data['Sell']=np.zeros(len(data))


# In[ ]:


data['Rolling_Long']=data['Open'].rolling(long_window).max()
data['Rolling_short']=data['Open'].rolling(short_window).min()


# In[ ]:


data=data.dropna()
data['buy_signal']=data['Adj Close']-data['Rolling_Long']
data['sell_signal']=data['Adj Close']-data['Rolling_short']


# In[ ]:


data.loc[:,'buy_signal']=data.loc[:,'buy_signal'].apply(lambda x: buysignal(x))
data.loc[:,'sell_signal']=data.loc[:,'sell_signal'].apply(lambda x: sellsignal(x))
data['signal']=data['buy_signal']+data['sell_signal']


# In[ ]:


# setup counting system
holding=0        ## how many Stocks we are holding
holdingcost=0    ## prime cost for getting the stock we are holding
outflow=0       ## tracking on cash outflow (cumulative)
inflow=100         ## tracking on cash inflow (cumulative)
order={}         ## Keep track of buys/sells per date
number_stocks =1 ## Select number of stocks set to 1 to see the return of investment of one stock in the duration date


# In[ ]:


for date,row in data.iterrows():

           # if signal is openpos(signal=1), and there's no holding (holding=0) then
           if row['signal']==1 and holding==0:
               
               #Strategy: We add one Stock to our portfolio (holdings)
               #Outflow tracks the cashflow
               #BuyIn Cost
               #Buy time

               holding=holding+number_stocks
               outflow-=row['Adj Close']
               holdingcost+=row['Adj Close']*holding
               order[date]=-row['Adj Close']
               #print('Buy Stock:{buy_create}$ @ {dt}'.format(buy_create=order[date],dt=date))
               
           elif row['signal']< 0 and holding>0:
               # Cashflow before selling
               # record the Sell of Stock
               # Reset the holdings since we sell all
               # We hold no more stocks
               
               inflow+= row['Adj Close']*holding            
               order[date]=+row['Adj Close']*holding
               holding=0
               holdingcost=0
               #print('Sell Stock:{sell_create}$ @ {dt}'.format(sell_create=round(order[date],2),dt=date)) ## round

           data.loc[date,'holding']=holding
           data.loc[date,'portvalue']=data.loc[date,'holding']*row['Adj Close']
           data.loc[date,'holdingcost']=holdingcost
           data.loc[date,'profit']=inflow+outflow+data.loc[date,'portvalue']
           #---------------Print Out Result---------------#
#Return on Investment:
data['roi']=(data['portvalue']-data['holdingcost'])/data['holdingcost']
data['roi']=data['roi'].fillna(0)
print('------------------------Summary------------------------')
print('Profit(Incl Amzn Holding):{profit:} $'.format(profit=round((outflow+inflow+data.loc[data.index[-1],'portvalue'])))) ## round 
print('Maximum Holding Cost(EURs):{profit:} Euros'.format(profit=(round(data['holdingcost'].max())))) ## round
print('Duration:{timeframe:}'.format(timeframe=(data.index[-1]-data.index[0]))) 
print('Max Drawdown:{drawdown:} %'.format(drawdown=round(100*data['roi'].min())))


# In[ ]:


Buy_in_date=[]
Sell_out_date=[]

for key,value in order.items():
    if value >0:
        Sell_out_date.append(key)
    if value<0:
        Buy_in_date.append(key)

Sell_Price=data.loc[Sell_out_date,'Adj Close']
Buy_Price=data.loc[Buy_in_date,'Adj Close']


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Scatter(x=data.index,y=data['Adj Close'],name='Adj Close',line_color='Blue'))
fig.add_trace(go.Scatter(x=data.index,y=data['profit'],name='Profit',line_color='Orange'))

fig.add_trace(go.Scatter(x=Buy_in_date,y=np.abs(Buy_Price),
                                name='Buy',mode='markers',
                                marker=dict(size=15,symbol=5),
                                marker_color='green',
                                text='BUY'))

fig.add_trace(go.Scatter(x=Sell_out_date,y=np.abs(Sell_Price),
                                name='Sell',mode='markers',
                                marker=dict(size=15,symbol=6),
                                marker_color='red',
                                text='SELL'))

fig.update_layout(title_text='Signal BUY/SELL',
                        xaxis_rangeslider_visible=True,
                        template='gridon')

 # show the plot
fig.show()


# In[ ]:


#How to identify BUY/SELL Signals:
data.loc[data['Rolling_Long'] < data['Adj Close'],'Buy']=1
data.loc[data['Rolling_short']> data['Adj Close'],'Sell']=-1


# In[ ]:


data['Daily Return']=data['Adj Close'].pct_change(1)
data['Daily Return'].hist(bins=100,figsize = (12,8))


# In[ ]:


class Actions(Enum):
    Sell = 0
    Buy = 1

# Moving Average
def MA(df, n):
    MA = pd.Series(df['Close'].rolling(min_periods=1, center=True, window=n).mean(), name = 'MA_' + str(n))
    df = df.join(MA)
    return df

data['PCT_difference']=data['MA_s']-data['MA_l']/data['MA_l']
data['shorts'] = data['PCT_difference'] < 0
data['longs'] = data['PCT_difference']>0
securities_to_trade = (data['shorts']|data['longs'])

print(securities_to_trade)
print(data['PCT_difference'])


# What is VaR?
# What is the maximum loss I can expect my portfolio to have with a time horizon X and a certainty of Y%?
# 
# 

# Plot : ****

# **VAR Signals:**

# In[ ]:


def make_pipeline():
    
    


# In[ ]:





# In[ ]:





# In[ ]:


#Determine the mean and standard deviation for this returns series. These will be the inputs to the random number generator(s)
from tabulate import tabulate
data['returns'] = data['Adj Close'].pct_change()
mean = np.mean(data['returns'])
std_d = np.std(data['returns']) #standard devi


# In[ ]:


data['returns'].hist(bins=100, normed = True, histtype='stepfilled', alpha=0.5)
x1 = np.linspace(mean-2*std_d, mean+2*std_d, 100)
plt.plot(x, mlab.normpdf(x1, mean, std_d),'red') # psaxto!!!!!
plt.show()


# In[ ]:


VAR_95 = norm.ppf(1-0.95, mean, std_d)
VAR_99 = norm.ppf(1-0.99, mean, std_d)
#install tabulate
print (tabulate([['95%',VAR_95],['99%', VAR_99]], headers = ['confidence level','VAR']))
#print(VAR_95)
#print(VAR_99)


# In[ ]:


#Monte Carlo Simulation:
# Buy Signal Price new_price = (1+random_value) * signal_price
# Sell Signal Price; 
from scipy.stats import norm
log_returns =np.log(1+data['Adj Close'].pct_change())
mean = log_returns.mean() #mean of the log_return
var =log_returns.var() #Variance
trend =mean -(0.5*var) #Log trend
std_d=log_returns.std()
S_steps = 100000
R =100
V =101 #101 days 

daily_ret = np.exp(trend +std_d*norm.ppf(np.random.rand(V,R)))


# In[ ]:


#Array of zeros
List_price = np.zeros_like(daily_ret)
List_price[0] =Price_1


# In[ ]:


#Forecasts for 101 days
for t in range(1,S_steps):
    List_price[t]=List_price[t-1]*daily_ret[t]


# In[ ]:


List_price =pd.DataFrame(List_price)
List_price['Adj Close']=List_price[0]
List_price.head()


# In[ ]:


Price_1= data['Adj Close'].iloc[-1]
Price_1


# In[ ]:


Amz_Adj_Close =data['Adj Close']
Amz_Adj_Close =pd.DataFrame(Amz_Adj_Close)
Combo = [Amz_Adj_Close,List_price]
MC_FC = pd.concat(Combo)
MC_FC.head()


# In[ ]:


MC_FC.tail()


# In[ ]:


MC=MC_FC.iloc[:,:].values
import matplotlib.pyplot as plt
plt.figure(figsize=(20,9))
plt.plot(MC)
plt.show()


# In[ ]:


data['log_returns']=log_returns
data['Date']=pd.to_datetime(data['Date'])
data.head()
data=data.dropna()

