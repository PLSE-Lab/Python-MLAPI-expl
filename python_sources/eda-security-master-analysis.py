#!/usr/bin/env python
# coding: utf-8

# # Security Master Analysis
# by @marketneutral

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotly.plotly as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import cufflinks as cf
init_notebook_mode(connected=False)
cf.set_config_file(offline=True, world_readable=True, theme='polar')


# In[ ]:


plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = 12, 7


# # Why Sec Master Analysis?
# 
# Before you do anything exciting in financial data science, you **need to understand the nature of the universe of assets you are working with** and **how the data is presented**; otherwise, garbage in, garbage out. A "security master" refers to reference data about the lifetime of a particular asset, tracking ticker changes, name changes, etc over time. In finance, raw information is typically uninteresting and uninformative and you need to do substantial feature engineering and create either or both of time series and cross-sectional features.  However **to do that without error requires that you deeply understand the nature of the asset universe.** This is not exciting fancy data science, but absolutely essential. Kaggle competitions are usually won in the third or fourth decimal place of the score so every detail matters.
# 
# ### What are some questions we want to answer?
# 
# **Is `assetCode` a unique and permanent identifier?**
# 
# If you group by `assetCode` and make time-series features are you assured to be referencing the same instrument? In the real world, the ticker symbol is not guaranteed to refer to the same company over time. Data providers usually provide a "permanent ID" so that you can keep track of this over time. This is not provided here (although in fact both Intrinio and Reuters provide this in the for sale version of the data used in this competition).
# 
# The rules state:
# 
# > Each asset is identified by an assetCode (note that a single company may have multiple assetCodes). Depending on what you wish to do, you may use the assetCode, assetName, or time as a way to join the market data to news data.
# 
# >assetCode(object) - a unique id of an asset
# 
# So is it unique or not and can we join time series features **always** over time on `assetCode`?
# 
# **What about `assetName`?  Is that unique or do names change over time?**
# 
# >assetName(category) - the name that corresponds to a group of assetCodes. These may be "Unknown" if the corresponding assetCode does not have any rows in the news data.
# 
# **What is the nature of missing data? What does it mean when data is missing?**
# 
# 
# Let's explore and see.
# 
# 
# 
# 

# In[ ]:


# Make environment and get data
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()


# [](http://)Let's define a valid "has_data" day for each asset if there is reported trading `volume` for the day.

# In[ ]:


df = market_train_df
df['has_data'] = df.volume.notnull().astype('int')


# And let's see how long an asset is "alive" by the
# - the distance betwen the first reported data point and last
# - and the number of days in that distance that actually has data

# In[ ]:


lifetimes_df = df.groupby(
        by='assetCode'
    ).agg(
        {'time': [np.min, np.max],
         'has_data': 'sum'
        }
)
lifetimes_df.columns = lifetimes_df.columns.droplevel()
lifetimes_df.rename(columns={'sum': 'has_data_sum'}, inplace=True)
lifetimes_df['days_alive'] = np.busday_count(
    lifetimes_df.amin.values.astype('datetime64[D]'),
    lifetimes_df.amax.values.astype('datetime64[D]')
)


# In[ ]:


#plt.hist(lifetimes_df.days_alive.astype('int'), bins=25);
#plt.title('Histogram of Asset Lifetimes (business days)');
data = [go.Histogram(x=lifetimes_df.days_alive.astype('int'))]
layout = dict(title='Histogram of Asset Lifetimes (business days)',
              xaxis=dict(title='Business Days'),
              yaxis=dict(title='Asset Count')
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# This was shocking to me. There are very many assets that only exist for, say, 50 days or less. When we look at the amount of data in these spans, it is even more surprising. Let's compare the asset lifetimes with the amout of data in those lifetime. Here I calculate the difference between the number of business days in each span and the count of valid days; sorted by most "missing data".

# In[ ]:


lifetimes_df['alive_no_data'] = np.maximum(lifetimes_df['days_alive'] - lifetimes_df['has_data_sum'],0)
lifetimes_df.sort_values('alive_no_data', ascending=False ).head(10)


# For example, ticker VNDA.O has its first data point on 2007-02-23, and its last on 2016-12-22 for a span of 2556 business days. However in that 2556 days, there are only 115 days that actually have data!

# In[ ]:





# In[ ]:


df.set_index('time').query('assetCode=="VNDA.O"').returnsOpenNextMktres10.iplot(kind='scatter',mode='markers', title='VNDA.O');


# **It's not the case that VNDA.O didn't exist during those times; we just don't have data.**
# Looking across the entire dataset, however, things look a little better.

# In[ ]:


#plt.hist(lifetimes_df['alive_no_data'], bins=25);
#plt.ylabel('Count of Assets');
#plt.xlabel('Count of missing days');
#plt.title('Missing Days in Asset Lifetime Spans');

data = [go.Histogram(x=lifetimes_df['alive_no_data'])]
layout = dict(title='Missing Days in Asset Lifetime Spans',
              xaxis=dict(title='Count of missing days'),
              yaxis=dict(title='Asset Count')
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# Now let's look at whether tickers change over time. **Is either `assetCode` or `assetName` unique?**

# In[ ]:


df.groupby('assetName')['assetCode'].nunique().sort_values(ascending=False).head(20)


# **So there are a number of companies that have more than 1 `assetCode` over their lifetime. ** For example, 'T-Mobile US Inc':

# In[ ]:


df[df.assetName=='T-Mobile US Inc'].assetCode.unique()


# And we can trace the lifetime of this company over multiple `assetCodes`. 

# In[ ]:


lifetimes_df.loc[['PCS.N', 'TMUS.N', 'TMUS.O']]


# The company started its life as PCS.N, was merged with TMUS.N (NYSE-listed) and then became Nasdaq-listed.
# 
# In this case, if you want to make long-horizon time-based features, you need to join on `assetName`.
# 

# In[ ]:


(1+df[df.assetName=='T-Mobile US Inc'].set_index('time').returnsClosePrevRaw1).cumprod().plot(title='Time joined cumulative return');


# **One gotcha I see is that  don't think that `assetName` is correct "point-in-time" .** This is hard to verify without proper commercial security master data, but:
# 
# - I don't think that the actual name of this company in 2007 was **T-Mobile** it was something like **Metro PCS**. T-Mobile acquired MetroPCS on May 1, 2013 (google search "when did t-mobile acquire MetroPCS"). You can see this data matches with the lifetimes dataframe subset above.
# - Therefore, the `assetName` must **not be point-in-time**, rather it looks like `assetName` is the name of the company when this dataset was created for Kaggle recently, and then backfilled.
# - However, it would be very odd for the Reuters News Data to **not be point-in-time.** Let's see if we can find any news on this company back in 2007.
# 

# In[ ]:


news_train_df[news_train_df.assetName=='T-Mobile US Inc'].T


# What's fascinating here is that you can see in the article headlines, that the company is named correctly, point-in-time, as "MetroPCS Communications Inc", however the `assetName` is listed as "T-Mobile US Inc.". So the organizers have also backfilled today's `assetName` into the news history.
# 
# This implies that **you cannot use NLP on the `headline` field in any way to join or infer asset clustering.** However, `assetName` continues to look like a consistent choice over time for a perm ID.
# 
# What about the other way around? Is `assetName` a unique identifier? In the real world, companies change their names all the time (a hilarious example of this is [here](https://www.businessinsider.com/long-blockchain-company-iced-tea-sec-stock-2018-8)). What about in this dataset?

# In[ ]:


df.groupby('assetCode')['assetName'].nunique().sort_values(ascending=False).head(20)


# **YES!** We can conclude that since no `assetCode` has ever been linked to more than `assetName`, that `assetName`  could be a good choice for a permanent identifier. It is possible that a company changed its ticker *and* it's name on the same day and therefore we would not be able to catch this, but let's assume this doesn't happen.
# 
# However, here is **a major gotcha**: dual class stock. Though not very common, some companies issue more than one class of stock at the same time. Likely the most well know is Google (called Alphabet Inc for its full life in this dataset); another is Comcast Corp.

# In[ ]:


df[df.assetName=='Alphabet Inc'].assetCode.unique()


# In[ ]:


lifetimes_df.loc[['GOOG.O', 'GOOGL.O']]


# Because of this overlapping data, there is no way to be sure about how to link assets over time. You are stuck with one of two bad choices: link on `assetCode` and miss ticker changes and corporate actions, or link on `assetName` but get bad output in the case of dual-class shares.

# ## Making time-series features when rows dates are missing
# Let's say you want to make rolling window time-series feature, like a moving average on volume. As we saw above, it is not possible to do this 100% without error because we don't know the permanent identifier; we must make a tradeoff between the error of using `assetCode` or `assetName`. Given that `assetCode` will never overlap on time (and therefore allows using time as an index), I choose that here. 
# 
# To make a rolling feature, it was my initial inclination to try something like:

# In[ ]:


df = market_train_df.reset_index().sort_values(['assetCode', 'time']).set_index(['assetCode','time'])
grp = df.groupby('assetCode')
df['volume_avg20'] = (
    grp.apply(lambda x: x.volume.rolling(20).mean())
    .reset_index(0, drop=True)
)


# Let's see what we got:

# In[ ]:


(df.reset_index().set_index('time')
 .query('assetCode=="VNDA.O"').loc['2007-03-15':'2009-06', ['volume', 'volume_avg20']]
)


# Look at the time index...the result makes no sense... the rolling average of 20 days spans **the missing period of >2007-03-20 and <2009-06-26 which is not right in the context of financial time series.** Instead we need to account for business days rolling. This will not be 100% accurate becuase we don't know exchange holidays, but it should be very close. **To do this correctly, you need to roll on business days**. However, pandas doesn't like to roll on business days (freq tag 'B') and will throw: `ValueError: <20 * BusinessDays> is a non-fixed frequency`. The next best thing is to roll on calendar days (freq tag 'D').
# 
# It took me awhile to get this to work as pandas complains a lot on multi-idexes (this [issue](https://github.com/pandas-dev/pandas/issues/15584) helped a lot).

# In[ ]:


df = df.reset_index().sort_values(['assetCode', 'time']).reset_index(drop=True)
df['volume_avg20d'] = (df
    .groupby('assetCode')
    .rolling('20D', on='time')     # Note the 'D' and on='time'
    .volume
    .mean()
    .reset_index(drop=True)
)


# In[ ]:


df.reset_index().set_index('time').query('assetCode=="VNDA.O"').loc['2007-03-15':'2009-06', ['volume', 'volume_avg20', 'volume_avg20d']]


# This is much better! Note that the default `min_periods` is 1 when you use a freq tag (i.e., '20D') to roll on. So even though we asked for a 20-day window, as long as there is at least 1 data point, we will get a windowed average. The result makes sense: if you look at 2009-06-26, you will see that the rolling average does **not** include any information from the year 2007, rather it is time-aware and since there are 19+ missing rows before, give the 1-day windowed average.
# 
# # Takeaways
# - Security master issues are critical.
# - You have to be very careful with time-based features because of missing data. Long-horizon features like, say, 12m momentum, may not produce sufficient asset coverage to be useful becuase so much data is missing.
# - The fact that an asset is missing data *is not informative in itself*; it is an artifact of the data collection and delivery process. For example, you cannot calcuate a true asset "age" (e.g., hypothesizing that days since IPO is a valid feature) and use that as a factor. This is unfortunate becuase you may hypothesize that news impact is a bigger driver of return variance during the early part of an asset's life due to lack of analyst coverage, lack of participation by quants, etc.
# - `assetCode` is not consistent across time; the same economic entity can, and in many cases does, have a different `assetCode`; `assetCode` is not a permanent identifier.
# - `assetName` while consistent across time, can refer to more than once stock *at the same time* and therefore cannot be used to make time series features; `assetName` is not a unique permanent identifier.
# - Missing time series data does not show up as `NaN` on the trading calendar; rather the rows are just missing. As such, to make time series features, you have to be careful with pandas rolling calculations and roll on calendar days, not naively on the count of rows.
# 
# 

# In[ ]:




