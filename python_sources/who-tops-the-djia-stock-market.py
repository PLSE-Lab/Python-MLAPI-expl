#!/usr/bin/env python
# coding: utf-8

# **King of DJIA**

# ![](https://i0.wp.com/swingalpha.com/wp-content/uploads/2018/02/Bull-Vs-Bear-Market-Characteristics-Banner-Image-1-1024x603.jpg)

# **Note:**  
# Kindly upvote the kernel if you find it useful. Suggestions are always welome. Let me know your thoughts in the comment if any.

# **Context**
# 
# Stock market data can be interesting to analyze and as a further incentive, strong predictive models can have large financial payoff. The amount of financial data on the web is seemingly endless. A large and well structured dataset on a wide array of companies can be hard to come by. Here provided a dataset with historical stock prices (last 12 years) for 29 of 30 DJIA companies (excluding 'V' because it does not have the whole 12 years data).
# 
# **Content**
# The data is presented in a couple of formats to suit different individual's needs or computational limitations. I have included files containing 13 years of stock data (in the all_stocks_2006-01-01_to_2018-01-01.csv and corresponding folder) and a smaller version of the dataset (all_stocks_2017-01-01_to_2018-01-01.csv) with only the past year's stock data for those wishing to use something more manageable in size.
# 
# The folder individual_stocks_2006-01-01_to_2018-01-01 contains files of data for individual stocks, labelled by their stock ticker name. The all_stocks_2006-01-01_to_2018-01-01.csv and all_stocks_2017-01-01_to_2018-01-01.csv contain this same data, presented in merged .csv files. Depending on the intended use (graphing, modelling etc.) the user may prefer one of these given formats.
# 
# All the files have the following columns: Date - in format: yy-mm-dd
# 
# * Open - price of the stock at market open (this is NYSE data so all in USD)
# * High - Highest price reached in the day
# * Low Close - Lowest price reached in the day
# * Volume - Number of shares traded
# * Name - the stock's ticker name
# 
# **Inspiration**
# This dataset lends itself to a some very interesting visualizations. One can look at simple things like how prices change over time, graph an compare multiple stocks at once, or generate and graph new metrics from the data provided. From these data informative stock stats such as volatility and moving averages can be easily calculated. The million dollar question is: can you develop a model that can beat the market and allow you to make statistically informed trades!
# 
# **Acknowledgement**
# This Data description is adapted from the dataset named 'S&P 500 Stock data'. This data is scrapped from Google finance using the python library 'pandas_datareader'. Special thanks to Kaggle, Github and the Market.  
# 
# **Note:**  
# Kindly upvote the kernel if you find it useful.  

# **0. Loading required Packages & Global options**

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from pylab import rcParams
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
# Above is a special style template for matplotlib, highly useful for visualizing time series data
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import datetime as dt
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
np.set_printoptions(suppress=True)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# **1. Reading the Dataset**

# In[ ]:


all_stock = pd.read_csv("../input/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv")


# **2. Taking Top 10 stocks w.r.t to Average Volume traded for analysis**

# In[ ]:


top10_query = """SELECT *
                 FROM (SELECT Name, AVG(Volume) as Avg
                       FROM all_stock
                       GROUP BY Name
                       ORDER BY AVG(Volume) DESC )
                 LIMIT 10;"""


top10 = pysqldf(top10_query)


# **3. Selecting Top 10 Companies stocks from all stocks data**

# In[ ]:


stock_10_query = """SELECT * FROM all_stock
                    where Name in ('AAPL', 'GE', 'MSFT', 'INTC', 'CSCO',
                                   'PFE', 'JPM', 'AABA', 'XOM', 'KO')"""

stock_10 = pysqldf(stock_10_query)


# **4. Stats on the numeric columns in the dataset**

# In[ ]:


stock_10.describe()


# **5. Function to find the missing values in the dataset**

# In[ ]:


# Function to calculate missing values by column# Funct 
def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns


# In[ ]:


#Missing value proportion in the dataset
missing_values_table(stock_10)


# As the number of missing values is very less which are barely not even a percent, I am dropping all the missing values along their row values as well.

# In[ ]:


#Dropping all the rows which has NA values
stock_10.dropna(inplace=True)


# **6. Printing the Dtype of the columns**

# In[ ]:


stock_10.dtypes


# Date column seems to have "object" as datatype. So, converting the "Date" column to Date format. 

# In[ ]:


#Converting Object to Date format for Date column
stock_10['Date'] = pd.to_datetime(stock_10['Date'])


# **7. Creating Year and Month-Year columns for analysis**

# In[ ]:


stock_10['Year'] = stock_10['Date'].apply(lambda x: dt.datetime.strftime(x,'%Y'))
stock_10['Mon'] = stock_10['Date'].apply(lambda x: dt.datetime.strftime(x,'%b'))
stock_10['Mon-Year'] = stock_10['Date'].apply(lambda x: dt.datetime.strftime(x,'%b-%Y'))


# **8. Average Stock Volume Trend - Top 10 Companies stock over 2006 - 2017**

# In[ ]:


year_trend_query = """SELECT Name, Year, AVG(Volume) as Avg
                      from stock_10
                      GROUP BY Name, Year
                      ORDER BY Name, Year;"""

year_trend = pysqldf(year_trend_query)

AAPL_trend = year_trend[year_trend.Name == 'AAPL']
GE_trend = year_trend[year_trend.Name == 'GE']
MSFT_trend = year_trend[year_trend.Name == 'MSFT']
INTC_trend = year_trend[year_trend.Name == 'INTC']
CSCO_trend = year_trend[year_trend.Name == 'CSCO']
PFE_trend = year_trend[year_trend.Name == 'PFE']
JPM_trend = year_trend[year_trend.Name == 'JPM']
AABA_trend = year_trend[year_trend.Name == 'AABA']
XOM_trend = year_trend[year_trend.Name == 'XOM']
KO_trend = year_trend[year_trend.Name == 'KO']


# In[ ]:


data = [
    go.Scatter(
        x=AAPL_trend['Year'], 
        y=AAPL_trend['Avg'],
        name='Apple'
    ),
    go.Scatter(
        x=GE_trend['Year'], 
        y=GE_trend['Avg'],
        name='GE'
    ),
        go.Scatter(
        x=MSFT_trend['Year'], 
        y=MSFT_trend['Avg'],
        name='Microsoft'
    ),
    go.Scatter(
        x=INTC_trend['Year'], 
        y=INTC_trend['Avg'],
        name='Intel'
    ),
        go.Scatter(
        x=CSCO_trend['Year'], 
        y=CSCO_trend['Avg'],
        name='Cisco'
    ),
    go.Scatter(
        x=PFE_trend['Year'], 
        y=PFE_trend['Avg'],
        name='Pfizer'
    ),
        go.Scatter(
        x=JPM_trend['Year'], 
        y=JPM_trend['Avg'],
        name='JPMorgan'
    ),
    go.Scatter(
        x=AABA_trend['Year'], 
        y=AABA_trend['Avg'],
        name='Altaba'
    ),
        go.Scatter(
        x=XOM_trend['Year'], 
        y=XOM_trend['Avg'],
        name='Exxon Mobil'
    ),
    go.Scatter(
        x=KO_trend['Year'], 
        y=KO_trend['Avg'],
        name='Coca-Cola'
    )
]

layout = go.Layout(
    xaxis=dict(type='category', title='Year'),
    yaxis=dict(title='Average Volume of Stocks Traded'),
    title="Average Stock Volume Trend - Top 10 Companies stock over 2006 - 2017"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='line-chart')


# Boom! Boom! year 2008 - 2009 around the globe. Due to recession and followed by financial meltdown every company took a hit and as a result over the world people reduced their investment due to dry cash flow which can be seen as a downward trend from 2009 w.r.t to volume of stocks traded by top 10 companies around the world.
# 
# Let's see how far this has impacted on the stock prices of these top 10 companies.

# **9. Average Stock Price based on Close Price - Top 10 Companies stock over 2006 - 2017**

# In[ ]:


year_trend_query = """SELECT Name, Year, AVG(Close) as Avg
                      from stock_10
                      GROUP BY Name, Year
                      ORDER BY Name, Year;"""

year_trend = pysqldf(year_trend_query)

AAPL_trend = year_trend[year_trend.Name == 'AAPL']
GE_trend = year_trend[year_trend.Name == 'GE']
MSFT_trend = year_trend[year_trend.Name == 'MSFT']
INTC_trend = year_trend[year_trend.Name == 'INTC']
CSCO_trend = year_trend[year_trend.Name == 'CSCO']
PFE_trend = year_trend[year_trend.Name == 'PFE']
JPM_trend = year_trend[year_trend.Name == 'JPM']
AABA_trend = year_trend[year_trend.Name == 'AABA']
XOM_trend = year_trend[year_trend.Name == 'XOM']
KO_trend = year_trend[year_trend.Name == 'KO']


# In[ ]:


data = [
    go.Scatter(
        x=AAPL_trend['Year'], 
        y=AAPL_trend['Avg'],
        name='Apple'
    ),
    go.Scatter(
        x=GE_trend['Year'], 
        y=GE_trend['Avg'],
        name='GE'
    ),
        go.Scatter(
        x=MSFT_trend['Year'], 
        y=MSFT_trend['Avg'],
        name='Microsoft'
    ),
    go.Scatter(
        x=INTC_trend['Year'], 
        y=INTC_trend['Avg'],
        name='Intel'
    ),
        go.Scatter(
        x=CSCO_trend['Year'], 
        y=CSCO_trend['Avg'],
        name='Cisco'
    ),
    go.Scatter(
        x=PFE_trend['Year'], 
        y=PFE_trend['Avg'],
        name='Pfizer'
    ),
        go.Scatter(
        x=JPM_trend['Year'], 
        y=JPM_trend['Avg'],
        name='JPMorgan'
    ),
    go.Scatter(
        x=AABA_trend['Year'], 
        y=AABA_trend['Avg'],
        name='Altaba'
    ),
        go.Scatter(
        x=XOM_trend['Year'], 
        y=XOM_trend['Avg'],
        name='Exxon Mobil'
    ),
    go.Scatter(
        x=KO_trend['Year'], 
        y=KO_trend['Avg'],
        name='Coca-Cola'
    )
]

layout = go.Layout(
    xaxis=dict(type='category', title='Year'),
    yaxis=dict(title='Average Close Price of the Stocks'),
    title="Average Stock Price based on Close Price - Top 10 Companies stock over 2006 - 2017"
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='line-chart')


# Wow! Here comes the sweet spot. Hit the bull's eye. As we can clearly see that Apple stock seems to follow a trend  every 2 to 3 years. If we remove the "Apple" and "Exxon Mobil" stock from the legend by clicking the corresponding legends one at a time, we can see a some repeating trends in "JPMorgan" and "Microsoft".
# 
# Further, When we additionally remove "Coca-Cola", "Altaba", "JPMorgan" and "Microsoft", we can clearly see that "GE" is the one which got hit heavely due recession because during the year 2009, the price of "GE" share alomost got reduced to half of its value.

# **10. Apple Stock for Analysis**

# For the analysis purpose I am taking the Apple stock as it tops the list in both the number of shares traded and with respect to price of the share.

# In[ ]:


stk = pd.read_csv("../input/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv", index_col='Date', 
                  parse_dates=['Date'])

app_stk = stk.query('Name == "AAPL"')


# **11. Apple Stock Trend w.r.t Open, High, Low, Close & Volume from 2006 to 2017**

# In[ ]:


app_stk['2006':'2017'].plot(subplots=True, figsize=(10,12))
plt.title('Apple stock trend from 2006 to 2017')
plt.savefig('app_stk.png')
plt.show()


# **12. Percentage change based on Close price**

# In[ ]:


app_stk['Change'] = app_stk.Close.div(app_stk.Close.shift())
app_stk['Change'].plot(figsize=(20,8))


# **13. Stock return based on Close price**

# In[ ]:


app_stk['Return'] = app_stk.Change.sub(1).mul(100)
app_stk['Return'].plot(figsize=(20,8))


# **14. Absolute change in successive rows of Close price**

# In[ ]:


app_stk.Close.diff().plot(figsize=(20,6))


# **15. Comparing more than two time series**

# For showing the comparison of more than two time series, I am considering the companies of "Apple", "Microsoft" and "Intel".

# In[ ]:


stk = pd.read_csv("../input/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv", index_col='Date', 
                  parse_dates=['Date'])
ms_stk = stk.query('Name == "MSFT"')
itl_stk = stk.query('Name == "INTC"')


# In[ ]:


# Plotting before normalization
app_stk.Close.plot()
ms_stk.Close.plot()
itl_stk.Close.plot()
plt.legend(['Apple','Microsoft', 'Intel'])
plt.show()


# In[ ]:


# Normalizing and comparison
# Both stocks start from 100
norm_app_stk = app_stk.Close.div(app_stk.Close.iloc[0]).mul(100)
norm_ms_stk_stk = ms_stk.Close.div(ms_stk.Close.iloc[0]).mul(100)
norm_itl_stk_stk = itl_stk.Close.div(itl_stk.Close.iloc[0]).mul(100)
norm_app_stk.plot()
norm_ms_stk_stk.plot()
norm_itl_stk_stk.plot()
plt.legend(['Apple','Microsoft', 'Intel'])
plt.show()


# **16. Windows function in Time series**

# Window functions are used to identify sub periods, calculates sub-metrics of sub-periods.
# 
# * **Rolling** - Same size and sliding
# * **Expanding** - Contains all prior values

# In[ ]:


# Rolling window functions
rolling_app = app_stk.Close.rolling('90D').mean()
app_stk.Close.plot()
rolling_app.plot()
plt.legend(['Close','Rolling Mean'])
# Plotting a rolling mean of 90 day window with original Close attribute of Apple stocks
plt.show()


# In[ ]:


# Expanding window functions
app_stk_mean = app_stk.Close.expanding().mean()
app_stk_std = app_stk.Close.expanding().std()
app_stk.Close.plot()
app_stk_mean.plot()
app_stk_std.plot()
plt.legend(['Close','Expanding Mean','Expanding Standard Deviation'])
plt.show()


# **17. OHLC charts**

# An OHLC chart is any type of price chart that shows the open, high, low and close price of a certain time period. Open-high-low-close Charts (or OHLC Charts) are used as a trading tool to visualise and analyse the price changes over time for securities, currencies, stocks, bonds, commodities, etc. OHLC Charts are useful for interpreting the day-to-day sentiment of the market and forecasting any future price changes through the patterns produced.
# 
# The y-axis on an OHLC Chart is used for the price scale, while the x-axis is the timescale. On each single time period, an OHLC Charts plots a symbol that represents two ranges: the highest and lowest prices traded, and also the opening and closing price on that single time period (for example in a day). On the range symbol, the high and low price ranges are represented by the length of the main vertical line. The open and close prices are represented by the vertical positioning of tick-marks that appear on the left (representing the open price) and on right (representing the close price) sides of the high-low vertical line.
# 
# Colour can be assigned to each OHLC Chart symbol, to distinguish whether the market is "bullish" (the closing price is higher then it opened) or "bearish" (the closing price is lower then it opened).

# In[ ]:


# OHLC chart of Apple for December 2016
trace = go.Ohlc(x=app_stk['12-2016'].index,
                open=app_stk['12-2016'].Open,
                high=app_stk['12-2016'].High,
                low=app_stk['12-2016'].Low,
                close=app_stk['12-2016'].Close)
data = [trace]
iplot(data, filename='simple_ohlc')


# In[ ]:


# OHLC chart of Apple stock for 2016
trace = go.Ohlc(x=app_stk['2016'].index,
                open=app_stk['2016'].Open,
                high=app_stk['2016'].High,
                low=app_stk['2016'].Low,
                close=app_stk['2016'].Close)
data = [trace]
iplot(data, filename='simple_ohlc')


# In[ ]:


# OHLC chart of Apple stock 2006 - 2017
trace = go.Ohlc(x=app_stk.index,
                open=app_stk.Open,
                high=app_stk.High,
                low=app_stk.Low,
                close=app_stk.Close)
data = [trace]
iplot(data, filename='simple_ohlc')


# **18. Candlestick Charts**

# This type of chart is used as a trading tool to visualise and analyse the price movements over time for securities, derivatives, currencies, stocks, bonds, commodities, etc. Although the symbols used in Candlestick Charts resemble a Box Plot, they function differently and therefore, are not to be confused with one another.
# 
# Candlestick Charts display multiple bits of price information such as the open price, close price, highest price and lowest price through the use of candlestick-like symbols. Each symbol represents the compressed trading activity for a single time period (a minute, hour, day, month, etc). Each Candlestick symbol is plotted along a time scale on the x-axis, to show the trading activity over time.
# 
# The main rectangle in the symbol is known as the real body, which is used to display the range between the open and close price of that time period. While the lines extending from the bottom and top of the real body is known as the lower and upper shadows (or wick). Each shadow represents the highest or lowest price traded during the time period represented. When the market is Bullish (the closing price is higher than it opened), then the body is coloured typically white or green. But when the market is Bearish (the closing price is lower than it opened), then the body is usually coloured either black or red.

# In[ ]:


# Candlestick chart of Apple for December 2016
trace = go.Candlestick(x=app_stk['12-2016'].index,
                open=app_stk['12-2016'].Open,
                high=app_stk['12-2016'].High,
                low=app_stk['12-2016'].Low,
                close=app_stk['12-2016'].Close)
data = [trace]
iplot(data, filename='simple_candlestick')


# In[ ]:


# Candlestick chart of Apple for 2016
trace = go.Candlestick(x=app_stk['2016'].index,
                       open=app_stk['2016'].Open,
                       high=app_stk['2016'].High,
                       low=app_stk['2016'].Low,
                       close=app_stk['2016'].Close)
data = [trace]
iplot(data, filename='simple_candlestick')


# In[ ]:


# Candlestick chart of Apple stock 2006 - 2017
trace = go.Candlestick(x=app_stk.index,
                       open=app_stk.Open,
                       high=app_stk.High,
                       low=app_stk.Low,
                       close=app_stk.Close)
data = [trace]
iplot(data, filename='simple_candlestick')


# **19. Time Series Decomposition**

# **19.1 Trend, Seasonality and Noise**
# These are the components of a time series
# * **Trend** - Consistent upwards or downwards slope of a time series
# * **Seasonality** - Clear periodic pattern of a time series(like sine funtion)
# * **Noise** - Outliers or missing values

# In[ ]:


# Consider the Apple stock w.r.t Close price
app_stk["Close"].plot(figsize=(16,8))


# In[ ]:


# Decomposition of Apple Stock based on Close price
rcParams['figure.figsize'] = 11, 9
decomposed_app_stk = sm.tsa.seasonal_decompose(app_stk["Close"],freq=360) # The frequncy is annual
figure = decomposed_app_stk.plot()
plt.show()


# **Observations:**
# * Clearly an upward trend is seen
# * Uniform seasonal change is seen
# * From Residual, the data seems to have a Non-uniform noise

# **19.2 White Noise**

# A time series may be white noise.
# 
# A time series is white noise if the variables are independent and identically distributed with a mean of zero.
# 
# This means that all variables have the same variance (sigma^2) and each value has a zero correlation with all other values in the series.
# 
# If the variables in the series are drawn from a Gaussian distribution, the series is called Gaussian white noise.

# **Why Does it Matter?**
# 
# White noise is an important concept in time series analysis and forecasting.
# 
# **It is important for two main reasons:**
# 
# * **Predictability:** If your time series is white noise, then, by definition, it is random. You cannot reasonably model it and make predictions.
# * **Model Diagnostics:** The series of errors from a time series forecast model should ideally be white noise.
# Model Diagnostics is an important area of time series forecasting.
# 
# Time series data are expected to contain some white noise component on top of the signal generated by the underlying process.
# 
# Once predictions have been made by a time series forecast model, they can be collected and analyzed. The series of forecast errors should ideally be white noise.
# 
# When forecast errors are white noise, it means that all of the signal information in the time series has been harnessed by the model in order to make predictions. All that is left is the random fluctuations that cannot be modeled.
# 
# A sign that model predictions are not white noise is an indication that further improvements to the forecast model may be possible.
# 
# **Reference:** [https://machinelearningmastery.com/white-noise-time-series-python/](http://)

# In[ ]:


# Plotting white noise
rcParams['figure.figsize'] = 16, 6
white_noise = np.random.normal(loc=0, scale=1, size=1000)
# loc is mean, scale is variance
plt.plot(white_noise)


# In[ ]:


# Plotting autocorrelation of white noise
plot_acf(white_noise,lags=20)
plt.show()


# **20. Stationary **

# A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time.
# 
# * **Strong stationarity:** is a stochastic process whose unconditional joint probability distribution does not change when shifted in time. Consequently, parameters such as mean and variance also do not change over time.
# * **Weak stationarity:** is a process where mean, variance, autocorrelation are constant throughout the time
# 
# Stationarity is important as non-stationary series that depend on time have too many parameters to account for when modelling the time series. diff() method can easily convert a non-stationary series to a stationary series.

# In[ ]:


# The original non-stationary plot
rcParams['figure.figsize'] = 16, 6
decomposed_app_stk.trend.plot()


# In[ ]:


# The new stationary plot
decomposed_app_stk.trend.diff().plot()

