#!/usr/bin/env python
# coding: utf-8

# # Beginner's Exploratory Data Analysis for Crypto Market

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import cross_decomposition
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

sns.set()

# Make charts a bit bolder
#sns.set_context("talk")

get_ipython().run_line_magic('matplotlib', 'inline')

# Default figure size
sns.set(rc={"figure.figsize": (12, 6)})

# This actually makes autocomplete WAY faster ...
get_ipython().run_line_magic('config', 'Completer.use_jedi = False')

# Show only 2 decimals for floating point numbers
pd.options.display.float_format = "{:.2f}".format

sns.set_style('whitegrid')


# In[ ]:


data = pd.read_csv('../input/crypto-markets.csv')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.shape, data.info()


# Here are the descriptions for some of the columns that I wasn't really sure about:
# 
# * volume - Transactions volume
# * market - Market Cap
# * ranknow - Currency rank
# * spread - Spread between high and low
# 
# Also, one thing I noticed -- market caps are quite huge. For ease of observing, let's introduce a new column - *market_billion*, which will represent currencies Market Cap in billion

# ## Data Wrangle & Cleanup

# In[ ]:


# Convert date to real date
data['date'] = pd.to_datetime(data['date'])
data['market_billion'] = data['market'] / 1000000000
data['volume_million'] = data['volume'] / 1000000000
data['volume_billion'] = data['volume']


# In[ ]:


# Let's prepare one dataframe where we will observe closing prices for each currency
wide_format = data.groupby(['date', 'name'])['close'].last().unstack()
wide_format.head(3)


# In[ ]:


wide_format.shape


# In[ ]:


wide_format.describe()


# ## Data Exploration
# 
# ### Top 10 cryptocurrencies in 2018

# In[ ]:


ax = data.groupby(['name'])['market_billion'].last().sort_values(ascending=False).head(10).sort_values().plot(kind='barh');
ax.set_xlabel("Market cap (in billion USD)");
plt.title("Top 10 Currencies by Market Cap");


# In[ ]:


ax = data.groupby(['name'])['volume_million'].last().sort_values(ascending=False).head(10).sort_values().plot(kind='barh');
ax.set_xlabel("Transaction Volume (in million)");
plt.title("Top 10 Currencies by Transaction Volume");


# In[ ]:


# For sake of convenience, let's define the top 5 currencies

top_5_currency_names = data.groupby(['name'])['market'].last().sort_values(ascending=False).head(5).index
data_top_5_currencies = data[data['name'].isin(top_5_currency_names)]
data_top_5_currencies.head(5)


# In[ ]:


data_top_5_currencies.describe()


# ## Trend Charts

# In[ ]:


ax = data_top_5_currencies.groupby(['date', 'name'])['close'].mean().unstack().plot();
ax.set_ylabel("Price per 1 unit (in USD)");
plt.title("Price per unit of currency");


# In[ ]:


ax = data_top_5_currencies.groupby(['date', 'name'])['market_billion'].mean().unstack().plot();
ax.set_ylabel("Market Cap (in billion USD)");
plt.title("Market cap per Currency");


# In[ ]:


ax = data_top_5_currencies.groupby(['date', 'name'])['volume_million'].mean().unstack().plot();
ax.set_ylabel("Transaction Volume (in million)");
plt.title("Transaction Volume per Currency");


# ## Trend Charts in 2017

# In[ ]:


ax = data_top_5_currencies[data_top_5_currencies.date.dt.year >= 2017].groupby(['date', 'name'])['close'].mean().unstack().plot();
ax.set_ylabel("Price per 1 unit (in USD)");
plt.title("Price per unit of currency (from 2017th)");


# In[ ]:


ax = data_top_5_currencies[data_top_5_currencies.date.dt.year >= 2017].groupby(['date', 'name'])['market_billion'].mean().unstack().plot();
ax.set_ylabel("Market Cap (in billion USD)");
plt.title("Market cap per Currency (from 2017th)");


# In[ ]:


ax = data_top_5_currencies[data_top_5_currencies.date.dt.year >= 2017].groupby(['date', 'name'])['volume_million'].mean().unstack().plot();
ax.set_ylabel("Transaction Volume (in million)");
plt.title("Transaction Volume per Currency (from 2017th)");


# ## Trend Charts in 2018

# In[ ]:


ax = data_top_5_currencies[data_top_5_currencies.date.dt.year >= 2018].groupby(['date', 'name'])['close'].mean().unstack().plot();
ax.set_ylabel("Price per 1 unit (in USD)");
plt.title("Price per unit of currency (from 2018th)");


# In[ ]:


ax = data_top_5_currencies[data_top_5_currencies.date.dt.year >= 2018].groupby(['date', 'name'])['market_billion'].mean().unstack().plot();
ax.set_ylabel("Market Cap (in billion USD)");
plt.title("Market cap per Currency (from 2018th)");


# In[ ]:


ax = data_top_5_currencies[data_top_5_currencies.date.dt.year >= 2018].groupby(['date', 'name'])['volume_million'].mean().unstack().plot();
ax.set_ylabel("Transaction Volume (in million)");
plt.title("Transaction Volume per Currency (from 2018th)");


# ## Correlation

# In[ ]:


plt.figure(figsize=(14,8))
sns.heatmap(wide_format[top_5_currency_names].corr(),vmin=0, vmax=1, cmap='coolwarm', annot=True);


# ## Experiments
# 
# Small experiment - let's assume that we invested some amount (say - 1000 USD) at some point. Let's see what ROI would we have.

# In[ ]:


def plot_roi(amount, df):
    ((amount / df.iloc[0]) * df).plot(figsize=(12,8))


# In[ ]:


plot_roi(1000, wide_format[['Bitcoin']])


# In[ ]:


wide_format_2017th = wide_format[(wide_format.index.year >= 2017)]
plot_roi(1000, wide_format_2017th[top_5_currency_names])


# In[ ]:


wide_format_late_2017th = wide_format[(wide_format.index.year >= 2017) & (wide_format.index.month >= 10)]
plot_roi(1000, wide_format_late_2017th[top_5_currency_names])


# In[ ]:


wide_format_2018th = wide_format[(wide_format.index.year >= 2018)]
plot_roi(1000, wide_format_2018th[top_5_currency_names])


# In[ ]:


len(data.slug.unique())


# In[ ]:


# Some common filters that we might be using
is_bitcoin = data['symbol'] == 'BTC'
is_ethereum = data['symbol'] == 'ETH'
is_ripple  = data['symbol'] == 'XRP'

# Pull out a part of dataset that only has the most interesting currencies
data_top_currencies = data[is_bitcoin | is_ethereum | is_ripple]


# Let's chart out Top cryptocurrencies according to latest reported Market Cap

# In[ ]:


top10Currencies = data.groupby('name')['market_billion'].last().sort_values(ascending=False).head(10)


# In[ ]:


ax = top10Currencies.sort_values().plot(kind='barh')
ax.set_xlabel("Market cap in Billion");
ax.set_ylabel("Currency");


# As we can see, and as it was expected, Bitcoin has the highest market cap. Let's see the trend for couple of top currencies.

# In[ ]:


ax = data_top_currencies.groupby(['date', 'name'])['close'].mean().unstack().plot()
ax.set_ylabel("Price per 1 unit (in USD)")


# That's rather amusing. Let's see focus on trend starting in 2018th.

# In[ ]:


data_top_currencies[data_top_currencies.date.dt.year >= 2018].groupby(['date', 'name'])['close'].mean().unstack().plot()
ax.set_ylabel("Price per 1 unit (in USD)")


# We can see that prices have jumped enormously in start and then decreases monotonically with a sharp increase in between Feb and March of 2018th. The cause? Apparently, there are lots of causes. From people's awareness about crypto currencies, to introduction of other currencies that increased the overal need.
# 
# ## Let's see a trend of Trading Volume for top currencies now

# In[ ]:


ax = data_top_currencies[data_top_currencies.date.dt.year >= 2018].groupby(['date', 'name'])['volume_billion'].mean().unstack().plot()
ax.set_ylabel("Trading volume (in billion)");


# There seems to be a correlation in trading between currencies. Which probably makes sense as, if I understood correctly, most of the currencies are actually traded using Bitcoin (i.e. you have to purchase Bitcoin in order to purchase Ripple). For sake of visibility, I'll plot Bitcoin and other currencies separately. Thing is that Bitcoin prices are actually masking other currencies.

# # Experiments
# 
# 
# Let's do a small experiment. Let's say that we invested 1000$ in each crypto currency 5 years ago. Let's see how much money would you have now.
# 
# First, let's start by drawing a diagram of closing prices for each year for each currency.

# In[ ]:


def plot_with_textvalue(df):
    ax = df.plot(kind='bar')
    
    ax.set_ylabel("Yearly closing prices (in USD)")

    for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d$' % int(height), ha='center', va='bottom')


# In[ ]:


top10Currencies


# In[ ]:


closing_prices_bitcoin_and_ethereum = data[is_bitcoin | is_ethereum].groupby(['date','name'])['close'].last().unstack().reset_index()
closing_prices_other_currencies = data[data['name'].isin(top10Currencies.index) & ~is_bitcoin & ~is_ethereum].groupby(['date','name'])['close'].last().unstack().reset_index()


# In[ ]:


yearly_closing_prices_bitcoin_and_ethereum = closing_prices_bitcoin_and_ethereum.groupby(closing_prices_bitcoin_and_ethereum.date.dt.year).last()
yearly_closing_prices_bitcoin_and_ethereum.drop(columns='date', inplace=True)
plot_with_textvalue(yearly_closing_prices_bitcoin_and_ethereum)


# In[ ]:


yearly_closing_prices_other_currencies = closing_prices_other_currencies.groupby(closing_prices_other_currencies.date.dt.year).last()
yearly_closing_prices_other_currencies.drop(columns='date', inplace=True)
yearly_closing_prices_other_currencies.plot(kind='bar')


# In[ ]:


closing_prices_other_currencies.head()


# Let's plot the closing prices.

# In[ ]:


closing_prices_bitcoin_and_ethereum.head()


# In[ ]:


closing_prices_other_currencies.head()


# In[ ]:


def calc_earnings(currency_name, df):
    #print("Displaying stats for "+currency_name)

    closing_prices = df[(df['name'] == currency_name) & (~df['close'].isnull())][['date', 'close']]

    # Num. currency purchased for 1000$
    #print("Closing price at the beginning: " + str(closing_prices.iloc[0]['close']))

    num_units_purchased = 1000 / closing_prices.iloc[0]['close']
    num_units_purchased

    #print("Num. units purchased: " + str(num_units_purchased))

    # Current value
    last_price = closing_prices.iloc[-1]['close']
    #print("Last price: " + str(last_price))

    amount_earned = (num_units_purchased * last_price) - 1000

    #print("Amount you would have earned: " + str(amount_earned) + "$")
    
    return amount_earned
    
# Borrow the index :-)
top_10_currencies_earnings = top10Currencies

for currency in top10Currencies.index:
    top_10_currencies_earnings[currency] = calc_earnings(currency, data)
    
ax = top_10_currencies_earnings.sort_values(ascending=False).plot(kind='bar')
for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d$' % int(height), ha='center', va='bottom')


# In[ ]:


# Borrow the index :-)
top_10_currencies_earnings_2018 = top10Currencies

for currency in top10Currencies.index:
    top_10_currencies_earnings_2018[currency] = calc_earnings(currency, data[data.date.dt.year >= 2018])
    
top_10_currencies_earnings_2018

ax = top_10_currencies_earnings_2018.sort_values(ascending=False).plot(kind='bar')
for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d$' % int(height), ha='center', va='bottom')


# In[ ]:


# Borrow the index :-)
top_10_currencies_earnings_2018 = top10Currencies

for currency in top10Currencies.index:
    top_10_currencies_earnings_2018[currency] = calc_earnings(currency, data[data.date.dt.year >= 2018])
    
top_10_currencies_earnings_2018

ax = top_10_currencies_earnings_2018.sort_values(ascending=False).plot(kind='bar')
for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d$' % int(height), ha='center', va='bottom')


# In[ ]:


top_10_currencies_earnings_without_nem = top_10_currencies_earnings[top_10_currencies_earnings.index != 'NEM']

ax = top_10_currencies_earnings_without_nem.sort_values(ascending=False).plot(kind='bar')
for rect in ax.patches:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%d$' % int(height), ha='center', va='bottom')


# In[ ]:


top10Currencies = data.groupby('name')['market_billion'].last().sort_values(ascending=False).head(5)
closing_prices_top10 = data[data['name'].isin(top10Currencies.index)].groupby(['date', 'name'])['close'].mean().unstack()
closing_prices_top10.corr()

plt.figure(figsize=(12,6))
sns.heatmap(closing_prices_top10.corr(),vmin=0, vmax=1, cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap between Bitcoin and other top 5 Crypto')


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(closing_prices_top10.corr(),vmin=0, vmax=1, cmap='coolwarm', annot=True)
plt.title('Correlation Heatmap between Bitcoin and other top 4 Crypto')


# ## Predictions
# 
# 
# Let's try something -- let's take values from previos 3 days and predict whether the price is going to go up or down

# In[ ]:


test = data[data['name'] == 'Bitcoin'].copy()


# In[ ]:


test['price_diff_d1'] = 0 # 1-day ago
test['price_diff_d2'] = 0 # 2-days ago
test['price_diff_d3'] = 0 # 3-days ago

test['trend'] = 0 # 0 = no change, -1 = price dropped, 1 = price increased


# In[ ]:


for i, row in test.iterrows():
    for j in range(1, 4):
        if ((i-j) < 0):
            # Skip rows at the beginning
            continue
   
        current_price  = row['close']
        prev_price = test.iloc[(i-j)]['close']
        
        column = 'price_diff_d'+str(j)
        
        test.ix[i, column] = (current_price - prev_price)
        
    if (i > 0):
        test.ix[i, 'trend'] = 1 if current_price > test.loc[(i-1)]['close'] else -1


# In[ ]:


X = test[['close', 'price_diff_d1', 'price_diff_d2', 'price_diff_d3']]
y = test['trend']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[ ]:


X_train.head(10)


# In[ ]:


X_test.shape, y_train.shape


# In[ ]:


model = RandomForestClassifier(n_estimators=10, max_depth=None, random_state=1234, oob_score=True)

model.fit(X_train, y_train)


# In[ ]:


#model?


# In[ ]:


scores = cross_val_score(model, X, y)
scores.mean()


# In[ ]:


model.oob_score_


# In[ ]:


y_pred = model.predict(X_test)


# In[ ]:


y_pred[-5:]


# In[ ]:


test['predicted'] = model.predict(X)


# In[ ]:


test[['date','predicted', 'trend']]


# In[ ]:


test.set_index('date')


# In[ ]:


test['datetime'] = pd.to_datetime(test['date'])


# In[ ]:


test = test.set_index('datetime')


# In[ ]:


test['close'].plot()


# **Copyright** by [Quanonblocks](https://www.kaggle.com/quanonblocks)
# 
# 
# Released under the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0) open source license.
