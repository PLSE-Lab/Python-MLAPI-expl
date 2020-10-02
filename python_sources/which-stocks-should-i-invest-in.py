#!/usr/bin/env python
# coding: utf-8

# The Nasdaq Stock Market is the second-largest stock exchange in the world by market capitalization (USD 10 trillion). It has around 1000 listed companies with a Market Cap of atleast USD 1 billion. 
# 
# Of these companies can we find how the ones that will **outperform the broader market**?

# In[ ]:


import numpy as np 
import pandas as pd 
from os import listdir

import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


# Make the default figures a bit bigger
plt.rcParams['figure.figsize'] = (5,3) 
plt.rcParams["figure.dpi"] = 120 

sns.set(style="ticks")
sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 5})
greek_salad = ['#D0D3C5', '#56B1BF', '#08708A', '#D73A31', '#032B2F']
sns.set_palette(greek_salad)


# # 1. Data extraction

# ## Get all tickers for NASDAQ will a market cap of atleast $10 billion

# In[ ]:


# https://www.nasdaq.com/screening/company-list.aspx
nasdaq = pd.read_csv('../input/nasdaq-company-list/companylist.csv')
cols = ['Symbol', 'Name', 'MarketCap', 'Sector']
nasdaq = nasdaq[cols]
nasdaq = nasdaq.drop_duplicates(subset=['Name'], keep='first')
nasdaq = nasdaq[nasdaq['MarketCap'] >= 1e9]
print(nasdaq.shape)
nasdaq.sort_values(by='MarketCap', ascending=False).head(10)


# ### Get filenames for all tickers

# In[ ]:


def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

path_to_dir = '../input/amex-nyse-nasdaq-stock-histories/fh_20190217/full_history/'
filenames = find_csv_filenames(path_to_dir)


# ### Read data for all tickers into a dataframe

# In[ ]:


get_ipython().run_cell_magic('time', '', '# Create Empty Dataframe\nstock_final = pd.DataFrame()\n\nfor i in range(len(list(nasdaq[\'Symbol\']))): #filenames\n    #print(i)    \n    try:\n        stock=[]\n        stock = pd.read_csv(path_to_dir+list(nasdaq[\'Symbol\'])[i]+\'.csv\')\n        stock[\'name\'] = list(nasdaq[\'Symbol\'])[i] #filenames[i].replace(".csv", "")\n        # Data starting from 2015\n        stock[\'date\'] = pd.to_datetime(stock[\'date\'])\n        stock = stock[stock.date >= \'2016-01-01\']\n        stock_final = pd.DataFrame.append(stock_final, stock, sort=False)\n    \n    except Exception:\n        i = i+1     ')


# In[ ]:


print("Available tickers", stock_final.name.nunique())
display(stock_final.sample(3))


# ## Extract relevant data
# 
# Will be working with only **Adjusted Close** and **Volume**

# In[ ]:


cols = ['date', 'adjclose', 'name']
df_close = stock_final[cols].pivot(index='date', columns='name', values='adjclose')

cols = ['date', 'volume', 'name']
df_volume = stock_final[cols].pivot(index='date', columns='name', values='volume')

print('Dataset shape:',df_close.shape)
display(df_close.tail(3))

print('Dataset shape:',df_volume.shape)
display(df_volume.tail(3))


# ## Missing data

# In[ ]:


percent_missing = pd.DataFrame(df_close.isnull().sum() * 100 / len(df_close))
percent_missing.columns = ['percent_missing']
percent_missing.sort_values('percent_missing', inplace=True, ascending=False)

percent_missing_plot = pd.DataFrame(percent_missing.reset_index().groupby('percent_missing').size())
percent_missing_plot.reset_index(inplace=True)
percent_missing_plot.columns = ['percent_missing', 'count']

ax = sns.scatterplot(x='percent_missing', y='count', data=percent_missing_plot, color=greek_salad[2])
ax.set_yscale('log')
ax.set_ylabel('Number of tickers')
ax.set_xlabel('Missing Data (%)')
sns.despine()


# ## Remove columns with any missing data

# In[ ]:


complete_data_tickers = percent_missing[percent_missing['percent_missing'] == 0].index
df = df_close[complete_data_tickers].head()

print("Available tickers", df.shape[1])
display(df.sample(3))


# ## This is the final dataset we will be working with: Daily percentage change in **price** and **volume**

# In[ ]:


df_pct_change = df.pct_change()
df_pct_change.head(3)


# In[ ]:


complete_data_tickers = percent_missing[percent_missing['percent_missing'] == 0].index
df_volume = df_volume[complete_data_tickers].head()

df_vol_change = df_volume.pct_change()
df_vol_change.head(3)


# # 2. Correlation of percent change

# In[ ]:


plt.figure(figsize=(6,6))
corr = df_pct_change.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, cmap=sns.diverging_palette(20,220, n=11), center=0)
    ax.set_yticklabels('')
    ax.set_xticklabels('')
    ax.set_ylabel('')
    ax.set_xlabel('')
    sns.despine()
    plt.tight_layout()


# In[ ]:


plt.figure(figsize=(6,6))
corr = df_vol_change.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True, cmap=sns.diverging_palette(20,220, n=11), center=0)
    ax.set_yticklabels('')
    ax.set_xticklabels('')
    ax.set_ylabel('')
    ax.set_xlabel('')
    sns.despine()
    plt.tight_layout()


# # 3. How good a bet is Microsoft? **`MSFT`**

# In[ ]:


MSFT = stock_final[stock_final.name == 'MSFT'].copy()
MSFT.set_index('date', inplace=True)
MSFT.head()


# In[ ]:


from plotly import tools
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.tools as tls
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

trace = go.Ohlc(x=MSFT.index,
                open=MSFT['open'],
                high=MSFT['high'],
                low=MSFT['low'],
                close=MSFT['close'],
               increasing=dict(line=dict(color= '#58FA58')),
                decreasing=dict(line=dict(color= '#FA5858')))

layout = {
    'title': 'MSFT Historical Price',
    'xaxis': {'title': 'Date',
             'rangeslider': {'visible': False}},
    'yaxis': {'title': 'Stock Price (USD$)'},
    'shapes': [{
        'x0': '2018-12-31', 'x1': '2018-12-31',
        'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',
        'line': {'color': 'rgb(30,30,30)', 'width': 1}
    }],
    'annotations': [{
        'x': '2019-01-01', 'y': 0.05, 'xref': 'x', 'yref': 'paper',
        'showarrow': False, 'xanchor': 'left',
        'text': '2019 <br> starts'
    }]
}

data = [trace]

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='simple_ohlc')


# ## Forecast stock price

# In[ ]:


# Drop the columns
ph_df = MSFT.drop(['open', 'high', 'low','volume', 'adjclose', 'name'], axis=1)
ph_df.reset_index(inplace=True)
ph_df.rename(columns={'close': 'y', 'date': 'ds'}, inplace=True)
ph_df['ds'] = pd.to_datetime(ph_df['ds'])
ph_df['y'] = np.log1p(ph_df['y'])
ph_df.head()


# In[ ]:


get_ipython().system('pip3 uninstall --yes fbprophet')
get_ipython().system('pip3 install fbprophet --no-cache-dir --no-binary :all:')


# In[ ]:


from fbprophet import Prophet
m = Prophet()
m.fit(ph_df) 

# Create Future dates
future_prices = m.make_future_dataframe(periods=365)

# Predict Prices
forecast = m.predict(future_prices)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# ## Daily predictions

# In[ ]:


fig = m.plot(forecast)
ax1 = fig.add_subplot(111)
ax1.set_title("MSFT Stock Price Forecast", fontsize=16)
ax1.set_xlabel("Date", fontsize=12)
ax1.set_ylabel("$log(1 + Close Price)$", fontsize=12)
sns.despine()
plt.tight_layout()


# In[ ]:


# fig2 = m.plot_components(forecast)
# plt.show()


# ## Monthly predictions

# In[ ]:


# Monthly Data Predictions
m = Prophet(changepoint_prior_scale=0.01).fit(ph_df)
future = m.make_future_dataframe(periods=12, freq='M')
fcst = m.predict(future)
fig = m.plot(fcst)
plt.title("Monthly Prediction \n 1 year time frame", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("$log(1+Close Price)$", fontsize=12)
sns.despine()
plt.tight_layout()


# In[ ]:


stock_max = np.round(np.expm1(fcst.tail(12)['yhat'].max()), 2)
stock_min = np.round(np.expm1(fcst.tail(12)['yhat'].min()), 2)
stock_current = np.expm1(ph_df.sort_values(by='ds').tail(1)['y'].values)

gain = (stock_max - stock_current) / stock_current
loss = (stock_current - stock_min) / stock_current

print('Current price:', np.round(stock_current,2), '$')
print('Expected High:', np.round(stock_max,2), '$')
print('Expected Low:', np.round(stock_min,2), '$')
print('Expected profit:', np.round(gain*100,2), '%')
print('Expected loss:', np.round(loss*100,2), '%')


# # 4. Forecasting Dow Jones Industrial Average
# 
# Can we find a ticker which will outperform the benchmark?

# In[ ]:


get_ipython().system('pip3 install fix_yahoo_finance --upgrade --no-cache-dir')


# In[ ]:


from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
from datetime import date

yf.pdr_override() 

end = date.today()
DJI = pdr.get_data_yahoo("^DJI", start="2016-01-01", end=end)


# In[ ]:


DJI.tail()


# In[ ]:


from plotly import tools
import plotly.plotly as py
import plotly.figure_factory as ff
import plotly.tools as tls
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

trace = go.Ohlc(x=DJI.index,
                open=DJI['Open'],
                high=DJI['High'],
                low=DJI['Low'],
                close=DJI['Close'],
               increasing=dict(line=dict(color= '#58FA58')),
                decreasing=dict(line=dict(color= '#FA5858')))

layout = {
    'title': 'DJI Historical Price',
    'xaxis': {'title': 'Date',
             'rangeslider': {'visible': False}},
    'yaxis': {'title': 'Stock Price (USD$)'},
    'shapes': [{
        'x0': '2018-12-31', 'x1': '2018-12-31',
        'y0': 0, 'y1': 1, 'xref': 'x', 'yref': 'paper',
        'line': {'color': 'rgb(30,30,30)', 'width': 1}
    }],
    'annotations': [{
        'x': '2019-01-01', 'y': 0.05, 'xref': 'x', 'yref': 'paper',
        'showarrow': False, 'xanchor': 'left',
        'text': '2019 <br> starts'
    }]
}

data = [trace]

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='simple_ohlc')


# In[ ]:


# Drop the columns
ph_df = DJI.drop(['Open', 'High', 'Low','Volume', 'Adj Close'], axis=1)
ph_df.reset_index(inplace=True)
ph_df.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
ph_df['ds'] = pd.to_datetime(ph_df['ds'])
ph_df['y'] = np.log1p(ph_df['y'])
ph_df.head()

# Monthly Data Predictions
m = Prophet(changepoint_prior_scale=0.01).fit(ph_df)
future = m.make_future_dataframe(periods=12, freq='M')
fcst = m.predict(future)
fig = m.plot(fcst)
plt.title("Monthly Prediction for DJI Index \n 1 year time frame", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("$log(1+Close)$", fontsize=12)
sns.despine()
plt.tight_layout()


# In[ ]:


stock_max = np.round(np.expm1(fcst.tail(12)['yhat'].max()), 2)
stock_min = np.round(np.expm1(fcst.tail(12)['yhat'].min()), 2)
stock_current = np.expm1(ph_df.sort_values(by='ds').tail(1)['y'].values)

DJI_gain = (stock_max - stock_current) / stock_current
DJI_loss = (stock_current - stock_min) / stock_current

print('Current :', np.round(stock_current,2))
print('Expected High:', np.round(stock_max,2))
print('Expected Low:', np.round(stock_min,2))
print('Expected rise:', np.round(DJI_gain*100,2), '%')
print('Expected fall:', np.round(DJI_loss*100,2), '%')


# # 5. Predicting expected gain and loss for each ticker

# In[ ]:


get_ipython().run_cell_magic('time', '', "df_gains = pd.DataFrame()\ni = 0\nfor ticker in df_close.columns:\n    tmp = pd.DataFrame()\n    ticker = df_close.columns[i]\n    ph_df = pd.DataFrame(df_close[ticker].copy())\n    ph_df.reset_index(inplace=True)\n    ph_df.rename(columns={ticker: 'y', 'date': 'ds'}, inplace=True)\n    ph_df['ds'] = pd.to_datetime(ph_df['ds'])\n    ph_df['y'] = np.log1p(ph_df['y'])\n\n    m = Prophet(changepoint_prior_scale=0.01).fit(ph_df)\n    future = m.make_future_dataframe(periods=12, freq='M')\n    fcst = m.predict(future)\n    \n    stock_max = np.round(np.expm1(fcst.tail(12)['yhat'].max()), 2)\n    stock_min = np.round(np.expm1(fcst.tail(12)['yhat'].min()), 2)\n    stock_current = np.expm1(ph_df.sort_values(by='ds').tail(1)['y'].values)\n\n    gain = (stock_max - stock_current) / stock_current\n    loss = (stock_current - stock_min) / stock_current\n    tmp = pd.DataFrame([ticker, gain, loss]).T\n    t = [('ticker', ticker),\n         ('gain', gain),\n         ('loss', loss)]\n    tmp = pd.DataFrame.from_items(t)\n    df_gains = df_gains.append(tmp)\n    i = i+1\n    ")


# # Tickers that may beat the benchmark

# In[ ]:


df_gains = df_gains.loc[(df_gains['gain'] >= DJI_gain[0])]
df_gains = df_gains.loc[(df_gains['loss'] <= DJI_loss[0])]
df_gains.sample(5)


# ## Distribution of expected gain 

# In[ ]:


fig = figure(num=None, figsize=(12, 4), dpi=120, facecolor='w', edgecolor='k')

plt.subplot(1, 1, 1)
ax1 = sns.distplot(df_gains['gain'].dropna()*100, bins=50, color=greek_salad[2]);
#ax1.set_xlim(0, 400)
ax1.set_xlabel('Gain (%)', weight='bold')
ax1.set_ylabel('Density', weight = 'bold')
ax1.set_title('Distribution of expected 1 year gain')
sns.despine()
plt.tight_layout();


# In[ ]:


# ## Distribution of expected loss
# fig = figure(num=None, figsize=(12, 4), dpi=120, facecolor='w', edgecolor='k')

# plt.subplot(1, 1, 1)
# ax1 = sns.distplot(df_gains['loss'].dropna()*100, bins=50, color=greek_salad[3]);
# #ax1.set_xlim(0, 400)
# ax1.set_xlabel('Loss (%)', weight='bold')
# ax1.set_ylabel('Density', weight = 'bold')
# ax1.set_title('Distribution of expected 1 year loss')
# sns.despine()
# plt.tight_layout();


# ### These are the tickers that atleast show a higher growth trend and may outperform the market

# In[ ]:


df_selected_stocks = pd.merge(df_gains, nasdaq, how='inner', left_on='ticker', right_on='Symbol')
cols = ['ticker', 'gain', 'Name', 'MarketCap', 'Sector']

df_selected_stocks = df_selected_stocks[cols]
df_selected_stocks.to_csv('selected_stocks.csv', sep=',', encoding='utf-8')
df_selected_stocks.sample(5)


# In[ ]:


f = {'gain':['median'], 'MarketCap':['sum'], 'Name':['count']}

ratios = df_selected_stocks.groupby('Sector').agg(f)
ratios.columns = ratios.columns.get_level_values(0)
ratios = ratios.reset_index()
ratios = ratios.sort_values('gain', ascending=False)

fig = figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(1, 3, 1)
ax1 = sns.barplot(x="Name", y="Sector", data=ratios, palette=("Greys_d"))
ax1.set_xlabel('Number of companies', weight='bold')
ax1.set_ylabel('Sector', weight = 'bold')
ax1.set_title('Sector breakdown\n')

plt.subplot(1, 3, 2)
ax2 = sns.barplot(x="MarketCap", y="Sector", data=ratios, palette=("Greens_d"))
ax2.set_xlabel('Total Market Cap', weight='bold')
ax2.set_ylabel('')
ax2.set_yticks([])

plt.subplot(1, 3, 3)
ax2 = sns.barplot(x="gain", y="Sector", data=ratios, palette=("Greens_d"))
ax2.set_xlabel('Median Gain', weight='bold')
ax2.set_ylabel('')
ax2.set_yticks([])

sns.despine()
plt.tight_layout();


# In[ ]:




