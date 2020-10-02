#!/usr/bin/env python
# coding: utf-8

# # Target
# - Construct a Backtest Tool for Bitcoin Trading
# - Using Turtle Trading Strategy with the Backtest Tool (Actually you can use any other strategy)
# - Plot the result nicely (with plotly)

# # Referrence of Turtle Trading Strategy
# 
# - current price is higher than the Highest price in the previous 20 days -> buy in
# - current price is lower than the lowest price in the previous 10 days -> sell out

# In[ ]:


# default import
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# import package we need
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta
from matplotlib import style
import plotly.graph_objects as go
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# ignore the warnings
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# define the path
path='../input/bitcoin-historical-price-1h2017820202/btc_1h.csv'


# In[ ]:


# read the data into dataframe
btc=pd.read_csv(path,index_col=0,parse_dates=True)


# In[ ]:


# have a peek on the data
btc.head()


# # Backtester Construction
# 
# We have 5 chunk seperated in the code
# 
# - Data selector
# - **Signal generator**
# - Trading Loop
# - Result
# - Plot Setup

# In[ ]:


# define the header and parameters
def backtest(dataframe,
             start_date,end_date):

#---------------Date Selector---------------#    
    
        # selecting the startdate and enddate
        # if start and enddate is not specify, then select all range
        if start_date==None:
            start_date=dataframe.index[1]
        if end_date==None:
            end_date=dataframe.index[-1]
        # slicing the range if start/end date is specified
        data=dataframe.loc[start_date:end_date,:]

#---------------Signal Generator---------------#    
        
        data['20high']=data['open'].rolling(20*24).max()
        data['10low']=data['open'].rolling(10*24).min()
        
        data=data.dropna()

        data['buysignal']=data['close']-data['20high']
        data['sellsignal']=data['close']-data['10low']

        def buysig(series):
            if series>0:
                return 1
            if series<0:
                return 0

        def sellsig(series):
            if series>0:
                return 0
            if series<0:
                return -1

        # apply the transformation function to dataframe
        data.loc[:,'buysignal']=data.loc[:,'buysignal'].apply(lambda x: buysig(x))
        data.loc[:,'sellsignal']=data.loc[:,'sellsignal'].apply(lambda x: sellsig(x))

        data['signal']=data['buysignal']+data['sellsignal']

#---------------Account Setup---------------#    
        
        # setup counting system
        holding=0        ## how many bitcoin is currently holding
        holdingcost=0    ## prime cost for getting the bitcoin we are holding
        outflow=0        ## tracking on cash outflow (cumulative)
        inflow=0         ## tracking on cash inflow (cumulative)
        order={}         ## a dict should contain buy/sell date and its price

#---------------Trading Loop---------------#

        # iterating over the dataframe by iterrows to execute the order
        for date,row in data.iterrows():

            # if signal is openpos(signal=1), and there's no holding bitcoin(holding=0) then
            if row['signal']==1 and holding==0:

                # add 1 bitcoin into holding
                holding=holding+1
                # track on all-time cash outflow
                outflow-=row['close']
                # track on the single time buyin cost
                holdingcost+=row['close']
                # recortd the cost and buytime into dict
                order[date]=-row['close']
                # print out the on-time buyin action
                print('Buy:{buy_create}$ @ {dt}'.format(buy_create=order[date],dt=date))

            # else if signal is less than 0,and holding any of bitcoin then sell out all
            elif row['signal']< 0 and holding>0:

                # track the all-time cash inflow (before selling)
                inflow+= row['close']*holding
                # record the sell action into dict             
                order[date]=+row['close']*holding
                # reset the holding number (since we have sold all)
                holding=0
                # reset the holding cost (since we don't hold anything now)
                holdingcost=0
                # print out the on-time sell out action
                print('Sell:{sell_create}$ @ {dt}'.format(sell_create=round(order[date],2),dt=date)) ## round

            # record bitcoin holding qty in dataframe 
            data.loc[date,'holding']=holding
            # record currently portfolio value in dataframe
            data.loc[date,'portvalue']=data.loc[date,'holding']*row['close']
            # record currently holding cost in dataframe
            data.loc[date,'holdingcost']=holdingcost

            # record profit in each timestamp
            data.loc[date,'profit']=inflow+outflow+data.loc[date,'portvalue']

#---------------Print Out Result---------------#

        # calculation of the ROI
        data['roi']=(data['portvalue']-data['holdingcost'])/data['holdingcost']
        data['roi']=data['roi'].fillna(0)

        # print a seperator
        print('------------------------Summary------------------------')
        # print the total profit made
        print('Profit(Incl Bitcoin Holding):{profit:} $'.format(profit=round((outflow+inflow+data.loc[data.index[-1],'portvalue'])))) ## round 
        # print the total profit made
        print('Maximum Holding Cost(USDT):{profit:} $'.format(profit=(round(data['holdingcost'].max())))) ## round
        # print the total time used
        print('Duration:{timeframe:}'.format(timeframe=(data.index[-1]-data.index[0]))) 
        # print the maximum drawdown rate
        print('Max Drawdown:{drawdown:} %'.format(drawdown=round(100*data['roi'].min())))

#---------------Plot Nicely with Plotly---------------#        

        # create empty list for saving buy_date and sell_date
        buy_date=[]
        sell_date=[]

        # loop over the order dictionary and append buy & sell date
        for key,value in order.items():
            if value>0:
                sell_date.append(key)
            if value<0:
                buy_date.append(key)

        # extract the buy price and sell price
        sp=data.loc[sell_date,'close']
        bp=data.loc[buy_date,'close']
        
        # setup plotly object figure
        fig=go.Figure()

        # adding trace of close price and profit
        fig.add_trace(go.Scatter(x=data.index,y=data['close'],name='Close',line_color='SlateBlue'))

        fig.add_trace(go.Scatter(x=data.index,y=data['profit'],name='Profit',line_color='LightPink'))

        # adding marker of Buyin and sell out 
        fig.add_trace(go.Scatter(x=buy_date,y=np.abs(bp),
                                name='Buy',mode='markers',
                                marker=dict(size=10,symbol=5),
                                marker_color='red',
                                text='BUY'))

        fig.add_trace(go.Scatter(x=sell_date,y=np.abs(sp),
                                name='Sell',mode='markers',
                                marker=dict(size=10,symbol=6),
                                marker_color='lime',
                                text='SELL'))

        # styling the plot and put text on it
        fig.update_layout(title_text='Turtle_Backtest',
                        xaxis_rangeslider_visible=True,
                        template='gridon')

        # show the plot
        fig.show()

        # return the data into dataframe
        return data


# # Parameter of backtest
# 1. dataframe : the dataframe you are going to do backtesting (should contain OHLC data, and the date is sorted by ascending)
# 2. start_date : the date of backtest started (should be a string, like '20170101')
# 3. end_date : the date of backtest ended (should be a string, like '20190101')

# In[ ]:


# run the backtester
df=backtest(
    dataframe=btc,
    start_date='20170101',
    end_date='20200210',
    )

