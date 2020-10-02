#!/usr/bin/env python
# coding: utf-8

# The data we have available is daily the SNP 500 stock data from 2010 to 2016.
# 
# This code analyses historical Amazon stock data and decides wether to buy apple stock based on the change in the previous days price fluctuation. If the price falls below a certain percentage the code decides to sell the stock and if the price rises above a certain percentage the code decides  to buy the stocks back.But if the change in price does not fullfill the benchmark of either buy or sell the code decides to hold its previous position. The deciding percentage to both buy or sell is selected by cross validation.
# 
# For convinience we decide our innitial investment to be 1000 dollars and we invest for a period of 1 year, i.e from 1/1/2016 to 12/30/2016.

# <img src="https://media.breitbart.com/media/2017/09/wi/ap/07/19d99j5-640x430.jpg" />

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Tools
from datetime import datetime as dt
from datetime import timedelta

#IPython manager
from IPython.display import display

#Graphs
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
df = pd.read_csv("../input/prices.csv", parse_dates=['date'])
df.head()
# Any results you write to the current directory are saved as output.


# In[ ]:


ticker_list = df['symbol'].unique() 
print(ticker_list)


# In[ ]:


date_list = df['date'].unique() 
date_list


# In[ ]:


df.sample(10)


# In[ ]:


year_list = df["date"].dt.year
print(year_list.unique())


# In[ ]:


#Create a new df for Apple tickers
df_AMZN = df[df['symbol'] == 'AMZN']


# In[ ]:


#Sort the dataframe by date
df_AMZN.sort_values(by=['date'], inplace=True, ascending= True)
# reset index to have sequential numbering
df_AMZN = df_AMZN.reset_index(drop=True)

df_AMZN.head()


# In[ ]:


# Create a new column to show the change in percentage between open and close
df_AMZN['change'] = (df_AMZN['close'] / df_AMZN['open']-1).fillna(0)
df_AMZN.head()


# In[ ]:


print(df_AMZN.index.max())


# In[ ]:


print(df_AMZN.loc[df_AMZN.index.max()-261]['open'])


# In[ ]:


# create a list of probable % change between open and close
rise = [.0009,.001,.002,.003,.004,.005,.008,.01]
fall = [-.01,-.011,-.012,-.014,-.015,-.016,-.02,-.04]

d = 251 # Nummber of days to invest (number of working days in year)
initial_amt = 1000 # initial amount to invest in dollars
total_profit = np.zeros([8, 8],dtype = float)


# In[ ]:


df_AMZN.shape


# ## Main Function

# Below we use a to function loop through all the rise and fall percentages and and itterate through the dates to calculate the possible profit. It also populates the profit matrix, which finally helps determine the indeal percentage settings.

# In[ ]:


# define the starting date to invest from
start_index = (df_AMZN.index.max()-d)
start_index


# In[ ]:


#Convert dollars to invest into shares based on open price of the first day
#the 2016 data starts on 4 Jan, we find the open value on that date
start_open = df_AMZN.loc[start_index]['open']
# use the first days open value to deduce number of shares
shares = initial_amt/start_open

print(shares)


# In[ ]:


df_AMZN.reset_index(inplace=True)


# Inorder to decide the ideal value for percentage change we cross validate different values of change using lists 'rise' and 'fall'. The code below loops through these two lists of percentage change tabulating total profit in a 2D array name 'total_profit'(I alway forget what my varables are meant for).
# 
# Ideally the code below should have been a function, so you could easily change out the Amazon ticker for anotherr ticker. But for now we will go with this.

# In[ ]:


x=0 # iterator for the rise loop
#loops through posible percentage increase in stock price
for b in rise: 
    
    y=0#iterator for the fall loop
    #loops through posible percentage decrease in stock price
    for s in fall:
        
        available_funds = 0.0
        temp_shares= shares
        #loop throught using index
        for i in range(start_index,df_AMZN.index.max()):
            
            daily_change = df_AMZN.loc[i]['change']
            share_price = df_AMZN.loc[i]['open']
            
            #Sell shares if price ffalls below certain percentage
            if daily_change <=s and temp_shares>0.0:
                #liquidate shares 
                available_funds = temp_shares*share_price
                temp_shares = 0
            #Buys shares if price rises above certain percentage
            elif daily_change >= b and available_funds>0:
                temp_shares = available_funds/share_price
                available_funds = 0
            else:
                continue
        
        #Total funds generated at the end of the cycle
        if temp_shares > 0:
            final_fund = temp_shares*share_price
        else:
            final_fund = available_funds
        
        # Calculate profit
        #print(final_fund)
        profit = final_fund-initial_amt
        total_profit[y,x] = profit
        y+=1
    x+=1
print(total_profit)      


# In[ ]:


max_profit = np.amax(total_profit)
# find the index of the highest profit
result = np.where(total_profit == max_profit)

final_rise = rise[int(result[1])]
final_fall = fall[int(result[0])]
print(max_profit, final_rise, final_fall)


# In[ ]:


print('The maximum profit we made Amazon stock investment of $1000 is ${:,.2f}'.format(max_profit),'/n')
print('The positive change percentage delimiter used ',final_rise)
print('The negative change percentage delimiter used ',final_fall)


# In[ ]:




