#!/usr/bin/env python
# coding: utf-8

# # Bitcoin EMA Crossover Trading Strategy

# ## Introduction

# ### Context

# Blockchain technology, first implemented by Satoshi Nakamoto in 2009 as a core component of Bitcoin, is a distributed, public ledger recording transactions.  Its usage allows secure peer-to-peer communication by linking blocks containing hash pointers to a previous block, a timestamp, and transaction data.  Bitcoin is a decentralized digital currency (cryptocurrency) which leverages the Blockchain to store transactions in a distributed manner in order to mitigate against flaws in the financial industry.  The motivation behind this project is to help those who have an interest in trading Bitcoin but perhaps are detoured by its inherent volatility by providing some predictive analytics using an exponential moving average (EMA) crossover trading strategy.

# ### Content

# This data set comes from Kaggle and features Bitstamp, a bitcoin exchange based in Luxembourg.  The time period of this data set is from January 1, 2012 to August 12, 2019, with minute to minute updates of OHLC (Open, High, Low, Close), Volume in BTC and indicated currency, and weighted bitcoin price.  Timestamps are in Unix time.  Timestamps without any trades or activity have their data fields forward filled from the last valid time period.  If a timestamp is missing, or if there are jumps, this may be because the exchange (or its API) was down, the exchange (or its API) did not exist, or some other unforseen technical error in data reporting or gathering.

# ### Column Information

# - Timestamp
# - Open
# - High
# - Low
# - Close
# - Volume_(BTC)
# - Volume_(Currency)
# - Weighted_Price

# ### Kaggle Link

# https://www.kaggle.com/mczielinski/bitcoin-historical-data/data

# ## Obtain

# ### Import Libraries

# In[ ]:


# Import Libraries.

from datetime import datetime
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
plt.style.use('seaborn-poster')
import functions as func
import warnings
warnings.filterwarnings('ignore')


# ### Load Data

# In[ ]:


# Load Bitstamp data as a DataFrame df.

df = pd.read_csv("../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")


# ### Inspect Data

# In[ ]:


# Check to see if df loaded properly.

print(df.shape)
df.head()


# In[ ]:


# Check index dtype and column dtypes, non-null values and memory usage.

df.info()


# ## Scrub

# ### Drop Misprint

# In[ ]:


# Drop misprint low price.

df.drop(df[df['Low'] == df['Low'].min()].index, inplace=True)


# ### Date Format

# In[ ]:


# Encode the date.

df['date'] = pd.to_datetime(df['Timestamp'], unit='s').dt.date


# ### Data Preprocess

# In[ ]:


# Create DataFrames grouped by date and average close price.

daily_close_12 = df.groupby('date').Close.mean().to_frame()
daily_close_26 = df.groupby('date').Close.mean().to_frame()
daily_close = df.groupby('date').Close.mean().to_frame()
daily_close_50 = df.groupby('date').Close.mean().to_frame()
daily_close_200 = df.groupby('date').Close.mean().to_frame()


# In[ ]:


# Make exponential moving averages for 12, 26, 50, and 200-day prices. 

daily_close_12['12_EMA'] = daily_close_12.ewm(12).mean()
daily_close_26['26_EMA'] = daily_close_26.ewm(26).mean()
daily_close_50['50_EMA'] = daily_close_50.ewm(50).mean()
daily_close_200['200_EMA'] = daily_close_200.ewm(200).mean()


# In[ ]:


# Reset the index for each DataFrame.

daily_close.reset_index(inplace=True)
daily_close_12.reset_index(inplace=True)
daily_close_26.reset_index(inplace=True)
daily_close_50.reset_index(inplace=True)
daily_close_200.reset_index(inplace=True)


# ## Explore

# ### Descriptive Statistics

# In[ ]:


# Descriptive statistics of df.

pd.set_option('float_format', '{:f}'.format)
df.describe()


# ### All-Time Highs (ATHs) and All-Time Lows (ATLs)

# In[ ]:


# All time low price.

ATL_price = df[df['Low'] == df['Low'].min()]
ATL_price


# In[ ]:


# All time high price.

ATH_price = df[df['High'] == df['High'].max()]
ATH_price


# ### Static Historical Daily Price Chart

# In[ ]:


# Plot df with date and close price. 

plt.figure(figsize=(18, 10))
plt.plot(df['date'], df['Close'])
plt.title("BTC/USD Daily Line Chart")
plt.xlabel("Years")
plt.ylabel("Price (USD)")
plt.grid()
plt.show()


# ### Static BTC/USD 50-Day and 200-Day EMA Chart

# In[ ]:


# Plot the 50-day EMA and 200-day EMA over actual prices.

plt.figure(figsize=(15, 10))
plt.plot(daily_close['date'], daily_close['Close'])
plt.plot(daily_close_50['date'], daily_close_50['50_EMA'],          label='50-Day EMA')
plt.plot(daily_close_200['date'], daily_close_200['200_EMA'],          label='200-Day EMA')
plt.title("BTC/USD 50-Day and 200-Day EMA Chart")
plt.xlabel("Years")
plt.ylabel("Price (USD)")
plt.grid()
plt.legend()
plt.show()


# ### Dynamic BTC/USD 50-Day and 200-Day EMA Chart

# In[ ]:


# Plot the 50-day EMA and 200-day EMA over actual prices.

fig_1 = go.Figure()
fig_1.add_trace(go.Scatter(
                x=daily_close['date'],
                y=daily_close['Close'],
                name="Close",
                line_color='blue'))

fig_1.add_trace(go.Scatter(
                x=daily_close_50['date'],
                y=daily_close_50['50_EMA'],
                name="50-Day EMA",
                line_color='orange'))

fig_1.add_trace(go.Scatter(
                x=daily_close_200['date'],
                y=daily_close_200['200_EMA'],
                name="200-Day EMA",
                line_color='green'))

fig_1.update_layout(
    title=go.layout.Title(
        text="BTC/USD 50-Day and 200-Day EMA Chart",
        xref="paper",
        x=0
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Time",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Price",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)

fig_1.update_layout(xaxis_range=['2012-01-01','2019-08-12'])

fig_1.show()


# ### Dynamic BTC/USD 12-Day and 26-Day EMA Chart

# In[ ]:


# Plot the 12-day EMA and 26-day EMA over actual prices.

fig_2 = go.Figure()
fig_2.add_trace(go.Scatter(
                x=daily_close['date'],
                y=daily_close['Close'],
                name="Close",
                line_color='blue'))

fig_2.add_trace(go.Scatter(
                x=daily_close_12['date'],
                y=daily_close_12['12_EMA'],
                name="12-Day EMA",
                line_color='orange'))

fig_2.add_trace(go.Scatter(
                x=daily_close_26['date'],
                y=daily_close_26['26_EMA'],
                name="26-Day EMA",
                line_color='green'))

fig_2.update_layout(
    title=go.layout.Title(
        text="BTC/USD 12-Day and 26-Day EMA Chart",
        xref="paper",
        x=0
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Time",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="Price",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)

fig_2.update_layout(xaxis_range=['2012-01-01','2019-08-12'])

fig_2.show()


# ## Model

# ### Long Term EMA (Golden Cross) Model

# ### Merge 50-Day EMA and 200-Day EMA

# In[ ]:


# Create a new DataFrame cross to show when 50 day EMA is greater than the 200 
# day EMA.

long_term = pd.merge(daily_close_50, daily_close_200, how='left')
long_term['golden'] = long_term['50_EMA'] > long_term['200_EMA']
long_term['golden'] = long_term.golden.astype('int')
long_term.head()


# ### 50-Day EMA and 200-Day EMA Crossover Dates and Prices

# In[ ]:


# Long term cross over dates and prices.

func.golden_cross_model(long_term)


# ### Date Differential Variables

# In[ ]:


# Create variables for time of each bull market.

first_bull_market_length = func.golden_cross_model(long_term)[1][0] - func.golden_cross_model(long_term)[0][0]
second_bull_market_length = func.golden_cross_model(long_term)[3][0] - func.golden_cross_model(long_term)[2][0]
third_bull_market_length = func.golden_cross_model(long_term)[5][0] - func.golden_cross_model(long_term)[4][0]
fourth_bull_market_start = func.golden_cross_model(long_term)[6][0]


# ### Profit Differential Variables

# In[ ]:


# Create variables for USD value gain in each bull market.

first_bull_market_profit = func.golden_cross_model(long_term)[1][1] - func.golden_cross_model(long_term)[0][1]
second_bull_market_profit = func.golden_cross_model(long_term)[3][1] - func.golden_cross_model(long_term)[2][1]
third_bull_market_profit = func.golden_cross_model(long_term)[5][1] - func.golden_cross_model(long_term)[4][1]
fourth_bull_market_price = func.golden_cross_model(long_term)[6][1]


# ### Execution Strategy

# In[ ]:


# Print bull market length and USD profits.

print(f"First Bull Market Length: {first_bull_market_length}")
print(f"Start Date: {func.golden_cross_model(long_term)[0][0]}")
print(f"End Date: {func.golden_cross_model(long_term)[1][0]}")
print(f"Buy Price: {'${:.2f}'.format(func.golden_cross_model(long_term)[0][1])}")
print(f"Sell Price: {'${:.2f}'.format(func.golden_cross_model(long_term)[1][1])}")
print(f"Profit: {'${:.2f}'.format(first_bull_market_profit)}")
print("--------------------------------------------")
print(f"Second Bull Market Length: {second_bull_market_length}")
print(f"Start Date: {func.golden_cross_model(long_term)[2][0]}")
print(f"End Date: {func.golden_cross_model(long_term)[3][0]}")
print(f"Buy Price: {'${:.2f}'.format(func.golden_cross_model(long_term)[2][1])}")
print(f"Sell Price: {'${:.2f}'.format(func.golden_cross_model(long_term)[3][1])}")
print(f"Profit: {'${:.2f}'.format(second_bull_market_profit)}")
print("--------------------------------------------")
print(f"Third Bull Market Length: {third_bull_market_length}")
print(f"Start Date: {func.golden_cross_model(long_term)[4][0]}")
print(f"End Date: {func.golden_cross_model(long_term)[5][0]}")
print(f"Buy Price: {'${:.2f}'.format(func.golden_cross_model(long_term)[4][1])}")
print(f"Sell Price: {'${:.2f}'.format(func.golden_cross_model(long_term)[5][1])}")
print(f"Profit: {'${:.2f}'.format(third_bull_market_profit)}")
print("--------------------------------------------")
print(f"Fourth Bull Market Start: {fourth_bull_market_start}")
print(f"Buy Price: {'${:.2f}'.format(fourth_bull_market_price)}")


# ### Short Term EMA Model

# ### Merge 12-Day EMA and 26-Day EMA

# In[ ]:


# Create a new DataFrame cross to show when 12 day EMA is greater than the 26 
# day EMA.

short_term = pd.merge(daily_close_12, daily_close_26, how='left')
short_term['golden'] = short_term['12_EMA'] > short_term['26_EMA']
short_term['golden'] = short_term.golden.astype('int')
short_term.head()


# ### 12-Day EMA and 26-Day EMA Crossover Dates and Prices

# In[ ]:


# Short term crossover dates and prices.

func.golden_cross_model(short_term)


# ### 12-Day EMA and 26-Day EMA Crossover Dates and Prices Segment

# In[ ]:


# Create a short term crossover segment. 

short_term_segment = func.golden_cross_model(short_term)[18:]


# ### Date Differential Variables

# In[ ]:


# Create variables for time of each short term crossover.

first_crossover_bullish_length = short_term_segment[1][0] - short_term_segment[0][0]
second_crossover_bearish_length = short_term_segment[3][0] - short_term_segment[2][0]
third_crossover_bearish_length = short_term_segment[5][0] - short_term_segment[4][0]
fourth_crossover_bullish_start = short_term_segment[6][0]


# ### Profit and Loss Differential Variables

# In[ ]:


# Create variables for USD value gain or loss in each short term crossover.

first_crossover_bullish_pnl = short_term_segment[1][1] - short_term_segment[0][1]
second_crossover_bearish_pnl = short_term_segment[3][1] - short_term_segment[2][1]
third_crossover_bearish_pnl = short_term_segment[5][1] - short_term_segment[4][1]
fourth_crossover_bullish_price = short_term_segment[6][1]


# ### Execution Strategy

# In[ ]:


# Print short term crossover length and USD profits/losses.

print(f"First Short Term Crossover Length: {first_crossover_bullish_length}")
print(f"Start Date: {short_term_segment[0][0]}")
print(f"End Date: {short_term_segment[1][0]}")
print(f"Buy Price: {'${:.2f}'.format(short_term_segment[0][1])}")
print(f"Sell Price: {'${:.2f}'.format(short_term_segment[1][1])}")
print(f"{func.PNL(first_crossover_bullish_pnl)}: {'${:.2f}'.format(first_crossover_bullish_pnl)}")
print("----------------------------------------------------")
print(f"Second Short Term Crossover Length: {second_crossover_bearish_length}")
print(f"Start Date: {short_term_segment[2][0]}")
print(f"End Date: {short_term_segment[3][0]}")
print(f"Buy Price: {'${:.2f}'.format(short_term_segment[2][1])}")
print(f"Sell Price: {'${:.2f}'.format(short_term_segment[3][1])}")
print(f"{func.PNL(second_crossover_bearish_pnl)}: {'${:.2f}'.format(second_crossover_bearish_pnl)}")
print("---------------------------------------------------")
print(f"Third Short Term Crossover Length: {third_crossover_bearish_length}")
print(f"Start Date: {short_term_segment[4][0]}")
print(f"End Date: {short_term_segment[5][0]}")
print(f"Buy Price: {'${:.2f}'.format(short_term_segment[4][1])}")
print(f"Sell Price: {'${:.2f}'.format(short_term_segment[5][1])}")
print(f"{func.PNL(third_crossover_bearish_pnl)}: {'${:.2f}'.format(third_crossover_bearish_pnl)}")
print("---------------------------------------------------")
print(f"Fourth Short Term Crossover Start: {fourth_crossover_bullish_start}")
print(f"Buy Price: {'${:.2f}'.format(fourth_crossover_bullish_price)}")

