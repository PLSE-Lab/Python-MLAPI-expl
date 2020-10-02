#!/usr/bin/env python
# coding: utf-8

# # Overview
# Can we imagine what is the behind the success of one of the darlings of Stock Exchange? Nvidia, one the large cap stocks that has grown maximum in the last five years has been having a bull run ever since it started powering the GPUs being utilized by massive computations required by AI amd ML projects! Result is continuous and unpredented growth in Earning per Share as well as revenue and hence the stock price. How could we uncover such interesting details given S&P data for last five years? Also can we get a feel of algo trading as well as event based trading? In this kernel we will try to explore if certains events do influence stock price fluctuations. If we did have an NLP engine crunching all news items and figuring out if the news would be a positive influence or a negative influence on the related stock, would it actually yield any benefits? Also what is this algo trading and associated technical indicators? Can they actually help us trade with increased probability of making money? Is that how Warren Buffet does it? That is not so easy to figure out of course but never the less this kernel will shed some light on some of these questions. So lets get started! 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import plotly.offline as pyo
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
init_notebook_mode(connected=True)

import plotly.graph_objs as go
print(os.listdir(".."))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/all_stocks_5yr.csv")


# In[ ]:


df["year"] = pd.DatetimeIndex(df["date"]).year
df["month"] = pd.DatetimeIndex(df["date"]).month
#df = df[df["date"] < '2018-01-01']


# # Data Preview
# Let us see what data we have and what are the attributes.

# In[ ]:


print(df.head(5))


# For each year, I want to plot closing price, total volume and % increment in price, Also I want to highligh the ones that slid dow in red color. 
# Steps
# 1. for closing price
# a. Get record for last day in each year. that is group by year, get max date
# b. Get data for only these dates
# 2. For total volume 
# a. Group by year and stock and get sum of volume
# b. Merge output of one with output of 2 on Year and stock name
# 3. For % increment in price,
# a. Get min date in each year
# b. Get data for these dates, rename Opening Stock Yr Open
# c Join output of 2 with output of 3
# d Calculate incr = Yr close - Yr open
# e % incr = incr * 100 / Yr open
# 
# 
# 

# In[ ]:


df_year_end_date = df.groupby(["year"]).agg({"date":{"date":["max"]}})
df_year_begin_date = df.groupby(["year"]).agg({"date":{"date":["min"]}})
df_year_end_date = df_year_end_date.reset_index()
df_year_end_date.columns = df_year_end_date.columns.map(lambda x: x[0])
df_year_begin_date = df_year_begin_date.reset_index()
df_year_begin_date.columns = df_year_begin_date.columns.map(lambda x: x[0])
df_year_end_data = pd.merge(df, df_year_end_date, on=["year","date"], how="inner")
df_year_begin_data = pd.merge(df, df_year_begin_date, on=["year","date"], how="inner")
df_year_volume_data = df.groupby(["year","Name"]).agg({"volume":{"volume":["sum"]}})
df_year_volume_data = df_year_volume_data.reset_index()
df_year_volume_data.columns = df_year_volume_data.columns.map(lambda x: x[0])
df_year_data = pd.merge(df_year_volume_data, df_year_end_data[["year","Name","close"]], on=["year","Name"], how="inner")
df_year_data = pd.merge(df_year_data, df_year_begin_data[["year","Name","open"]], on=["year","Name"], how="inner")

df_year_data["incr price"] = df_year_data["close"] - df_year_data["open"]
df_year_data["% incr price"] = (df_year_data["incr price"] * 100) / df_year_data["open"]
df_year_data.head(5)


# # Interesting Attribute No 1: Daily Volume
# First attribute that catches attention is volume. What we have is daily volume of each stock. Now it would be interesting to see if certain stocks had an abnormally high volume as compared their average volume in a certain period. We could then do a little market research to see if there was any news that triggered such a spurt. Let us first do this activity fo the entire duration that we have this data that is almost 5 years. We will then compare the findings with varying periods like 50 days or 200 days or even on an yearly basis. So let us list down the steps that we need to perform:
# ## Steps to find top stocks that had a maximum deviation in volume from their five year average
# 1. Group by stock name and get mean volume
# 2. Add a new columns in dataset for mean volume, diff from mean volume
# 3. Sort by diff from mean volume
# 4. Get top stocks from sorted data
# 5. Display these  stocks
# ## Charts to see the movement in volume
# 1. Plot a box plot for each of these stocks
# 2. Plot a histogram
# 3. Plot a line chart
# ## See if there was any news for these stocks on the days of max deviation 
# We want to see if we can locate the news item that triggered this market behaviour that is huge increment in volume
# Ok so let us get started.

# In[ ]:


df_name = df.groupby("Name").agg({"volume":{"mean volume":["mean"]},
                                  "close":{"mean close":["mean"]}})
df_name = df_name.reset_index()
df_name.columns = df_name.columns.map(lambda x: x[0])
df = pd.merge(df, df_name, on="Name")
df["diff from mean volume"] = df["volume"] - df["mean volume"]
df["diff price"] = df["close"] - df["open"]
df = df.sort_values(["diff from mean volume"])
top_10_by_diff_from_mean_volume = df.tail(10)
top_10_name_by_diff_from_mean_volume = list(top_10_by_diff_from_mean_volume["Name"])


# In[ ]:


top_10_by_diff_from_mean_volume


# Let us also add another column,  % diff from mean volume. 

# In[ ]:


df["% diff from mean volume"] = (df["diff from mean volume"] * 100 ) / df["mean volume"]
df["% diff price"] = (df["diff price"] * 100 ) / df["open"]
df = df.sort_values(["% diff from mean volume"])
top_10_by_perc_diff_from_mean_volume = df.tail(10)
top_10_by_perc_diff_from_mean_volume


# If we sort stocks by both volume and percentage increase in volume we find Verizon is the stock that falls in both the list. Also MAA is the stock that saw an abnormally high increase in volume along with a noticeable prible change. Let us plot various charts to visually see what is happening! 

# In[ ]:


for name in top_10_by_perc_diff_from_mean_volume["Name"]:
    if name in top_10_name_by_diff_from_mean_volume:
        name_data = df[df["Name"]==name]["volume"]
        data = [go.Histogram(x=name_data)]
        layout = go.Layout(title=name + " - Histogram")
        fig = go.Figure(data=data,layout=layout)
        iplot(fig)


# In[ ]:


for name in top_10_by_perc_diff_from_mean_volume["Name"]:
    if name in top_10_name_by_diff_from_mean_volume:
        name_data = df[df["Name"]==name]["volume"]
        data = [go.Box(y=name_data)]
        layout = go.Layout(title=name + " - Box Chart")
        fig = go.Figure(data=data,layout=layout)
        iplot(fig)


# In[ ]:


for name in top_10_by_perc_diff_from_mean_volume["Name"]:
    if name in top_10_name_by_diff_from_mean_volume:
        df_name = df[df["Name"]==name].sort_values(["date"])
        y_data = df_name["volume"]
        x_data = df_name["date"]
        data = [go.Scatter(x=x_data, y=y_data, mode="lines")]
        layout = go.Layout(title=name + " - Line Chart")
        fig = go.Figure(data=data,layout=layout)
        iplot(fig)


# In[ ]:


for name in ["MAA"]:
    df_name = df[df["Name"]==name].sort_values(["date"])
    y_data = df_name["volume"]
    x_data = df_name["date"]
    data = [go.Scatter(x=x_data, y=y_data, mode="lines")]
    layout = go.Layout(title=name + " - Line Chart")
    fig = go.Figure(data=data,layout=layout)
    iplot(fig)


# # What was the news?
# ## Verizon on 24th Feb 2014
# When we surf the net what we find is that on 24th Feb 2014, Vodaphone distributed Verizon shares to its shareholders as part of the joint venture anounced prior month. A number of investors then sold these shares and booked profits! So while we can see that the closing price is lower than opening it is no where close to being described as a steep fall. Was this stock a good candidate for algorithmic trading? Probably yes.. A careful analysis of the intraday movement and sentiment on the day would help us understand this better.
# ## MAA on 1 Dec 2016
# Another interesting stock is MAA. While volume was abnormally high, we can also see a decent slide in price. It turns out it is again related to merger of Mid America Aprtments with Post Properties. They announced completion of merger on the date of 1 dec 2016 and on this date Post Properties shares got converted to MAA shares. So looks Merger And Aquisition relted events involving conversion or allocaton of shares increase trading volumes, which seems natural enough. 
# 
# OK so may be it is time to add another interesting attribute!

# # Interesting Attribute No. 2: Closing Price
# Let us now pick stocks with abnormally high volume and abnormally high increment in price.
# 

# In[ ]:


num_rec = 500
df = df.sort_values(["diff from mean volume"])
list_volume = set(df.tail(num_rec)["Name"])
df = df.sort_values(["% diff from mean volume"])
list_perc_volume = set(df.tail(num_rec)["Name"])
df = df.sort_values(["diff price"])
list_close = set(df.tail(num_rec)["Name"])
df = df.sort_values(["% diff price"])
list_perc_close = set(df.tail(num_rec)["Name"])
print(list_volume & list_perc_volume & list_close & list_perc_close)


# In[ ]:


df_Stock = df[df["Name"]=="NVDA"]
print(df_Stock.sort_values(["diff from mean volume"]).tail(2))
print(df_Stock.sort_values(["% diff from mean volume"]).tail(2))
print(df_Stock.sort_values(["diff price"]).tail(2))
print(df_Stock.sort_values(["% diff price"]).tail(2))


# In[ ]:


for name in ["NVDA"]:
    df_name = df[df["Name"]==name].sort_values(["date"])
    y_data1 = df_name["volume"]/500000
    y_data2 = df_name["close"]
    x_data = df_name["date"]
    data = [go.Scatter(x=x_data, y=y_data1, mode="lines", name="Volume"), go.Scatter(x=x_data, y=y_data2, mode="lines", name="Closing Price")]
    layout = go.Layout(title=name + " - Line Chart")
    fig = go.Figure(data=data,layout=layout)
    iplot(fig)


# # What happened to NVDA on 11 Nov 2016
# So on evening prior to 11 Nov 2016 NVDA report earnings of 0.94 as against prediction of 0.69. The revenue was also much higher than expected at $2.0 billion against a prediction of $1.69 billion. They reported a breakout quarter -- record revenue, record margins and record earnings were driven by strength across all product lines. They reported, "Our Pascal GPUs are fully ramped and enjoying great success in gaming, [virtual reality], self-driving cars and datacenter AI computing." So that makes complete sense, great quarter, stock should have a great ride. And yeah, the stock continues to be great! And they seem to be cashing on the GPUs that we are currently using to progrm our ML algorithms. What a finding, ML & AI are so hot that hardware that are facilitating these massive computations are doing good business!
# 
# Let us now look a little more closely at Netflix story.

# In[ ]:


df_Stock = df[df["Name"]=="NFLX"][["Name","open","close","diff from mean volume","% diff from mean volume","diff price","% diff price","date"]]
print(df_Stock.sort_values(["diff from mean volume"]).tail(2))
print(df_Stock.sort_values(["% diff from mean volume"]).tail(2))
print(df_Stock.sort_values(["diff price"]).tail(2))
print(df_Stock.sort_values(["% diff price"]).tail(2))


# In[ ]:


for name in ["NFLX"]:
    df_name = df[df["Name"]==name].sort_values(["date"])
    y_data1 = df_name["volume"]/500000
    y_data2 = df_name["close"]
    x_data = df_name["date"]
    data = [go.Scatter(x=x_data, y=y_data1, mode="lines", name="Volume"), go.Scatter(x=x_data, y=y_data2, mode="lines", name="Closing Price")]
    layout = go.Layout(title=name + " - Line Chart")
    fig = go.Figure(data=data,layout=layout)
    iplot(fig)


# # What happened to NFLX on 11 Nov 2016
# So Netflix announced launch of service to 130 more countries. As we can see this led to increase in price but not very high increase in volume. 
# 
# Ok so now let us plot some technical stats and see if we can draw some predictions for future based on those.

# # Moving Average Convergence Divergence - MACD
# Let us first identify stocks that we would like to analyze. For this let us calculate volume * average price for the day(Close + Open / 2). Sort desc on this value and see what we get. We are trying to locate large cap stocks. We find the candidate stocks as AAPL, FB, AMZN, MSFT, BAC.  That is preety close to the current set of top 5 companies where in we do have AAPL, FB, AMZN and MSFT. Instead of BAC we have Alphabet. 
# 

# In[ ]:


df["tran value"] = df["volume"] * ((df["close"] + df["open"])/2)
df_tran_value = df.groupby(["Name"]).agg({"tran value":{"tot tran value":["sum"]}})
df_tran_value = df_tran_value.reset_index()
df_tran_value.columns = df_tran_value.columns.map(lambda x: x[0])
df_tran_value = df_tran_value.sort_values(["tot tran value"])
print(df_tran_value.tail(5))


# Let us plot MACD for these companies and get our concepts clear.

# In[ ]:


def plot_macd(name):
    df_stock = df[df["Name"] == name].sort_values(["date"])
    df_stock['26 ema'] = df_stock["close"].ewm(span=26, adjust=False).mean()
    df_stock['12 ema'] = df_stock["close"].ewm(span=12, adjust=False).mean()
    df_stock['MACD'] = df_stock['12 ema'] - df_stock['26 ema']
    df_stock['9 ema'] = df_stock["MACD"].ewm(span=9, adjust=False).mean()
    y_data1 = df_stock["MACD"]
    y_data2 = df_stock["9 ema"]
    x_data = df_stock["date"]
    data = [go.Scatter(x=x_data, y=y_data1, mode="lines", name="MACD"), go.Scatter(x=x_data, y=y_data2, mode="lines", name="9 day EMA")]
    layout = go.Layout(title=name + " - Moving Average Convergance Divergence")
    fig = go.Figure(data=data,layout=layout)
    iplot(fig)


# In[ ]:


plot_macd("AAPL")


# In[ ]:


plot_macd("FB")


# In[ ]:


plot_macd("AMZN")


# In[ ]:


plot_macd("MSFT")


# In[ ]:


plot_macd("BAC")


# # What is the 'Moving Average Convergence Divergence - MACD'
# 
# Moving average convergence divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of prices. The MACD is calculated by subtracting the 26-day exponential moving average (EMA) from the 12-day EMA. A nine-day EMA of the MACD, called the "signal line", is then plotted on top of the MACD, functioning as a trigger for buy and sell signals.
# 
# Read more: Moving Average Convergence Divergence (MACD) https://www.investopedia.com/terms/m/macd.asp#ixzz5SmuJbrAH
# 
# As shown in the charts above, when the MACD falls below the signal line, it is a bearish signal, which indicates that it may be time to sell. Conversely, when the MACD rises above the signal line, the indicator gives a bullish signal, which suggests that the price of the asset is likely to experience upward momentum. Many traders wait for a confirmed cross above the signal line before entering into a position to avoid getting "faked out" or entering into a position too early, as shown by the first arrow.
# 

# We can see if we had taken actions on crossover points, we would have made money.
# Now this for sure looks to be super exciting! Will release a v2 of kernel with Technical Indicators like Alpha & Beta. Till then have fun trading....Naah we are not there yet....

# In[ ]:




