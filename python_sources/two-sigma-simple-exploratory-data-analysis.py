#!/usr/bin/env python
# coding: utf-8

# ### If you like this kernel greatly appreciate an  UP VOTE 
# 
# # Two Sigma Stock Prediction
# 
# ## Introduction
# 
#  <img src="http://i65.tinypic.com/2im5eno.jpg">
# Can we use the content of news analytics to predict stock price performance? The ubiquity of data today enables investors at any scale to make better investment decisions. The challenge is ingesting and interpreting the data to determine which data is useful, finding the signal in this sea of information. Two Sigma is passionate about this challenge and is excited to share it with the Kaggle community.
# 
# As a scientifically driven investment manager, Two Sigma has been applying technology and data science to financial forecasts for over 17 years. Their pioneering advances in big data, AI, and machine learning have pushed the investment industry forward. Now, they're eager to engage with Kagglers in this continuing pursuit of innovation.
# 
# By analyzing news data to predict stock prices, Kagglers have a unique opportunity to advance the state of research in understanding the predictive power of the news. This power, if harnessed, could help predict financial outcomes and generate significant economic impact all over the world.
# 
# ## Data Description
# 
# In this competition, we need to predict future stock price returns based on two sources of data:
# 
# **Market data (2007 to present) provided by Intrinio **
# 
# It contains financial market information such as opening price, closing price, trading volume, calculated returns, etc.
# 
# **News data (2007 to present) Source: Thomson Reuters**
# 
# It  contains information about news articles/alerts published about assets, such as article details, sentiment, and other commentary.
# 
# - Each asset is identified by an assetCode (note that a single company may have multiple assetCodes). Depending on what you wish to do, you may use the assetCode, assetName, or time as a way to join the market data to news data.
# 
# ## Data Mining
# 
# **Important Note : **
# 
# Before we actually go into the details of market data and news data we will start with an data mining section to fetch the data from the Kaggle DataFrames provided as below 
# 
# ### Import Libraries 

# In[ ]:


import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

warnings.filterwarnings('ignore')

import numpy as np 

import pandas as pd 


# Data mine both market data and news data from Kaggle .

# In[ ]:


from kaggle.competitions import twosigmanews

#   You  can  only    call    make_env() once, so don't lose it!

env = twosigmanews.make_env()

print('Done!')


# Now let us call  **get_training_data** function from the new environment  which returns market training data and news training data dataFrames containing all market and news data from February 2007 to December 2016.

# In[ ]:


(market_train_data, news_train_data) = env.get_training_data()


# Now let us explore the data in detail .Firstly lets look at Market Data .
# 
# ## Market data

# In[ ]:


market_train_data.head()


# In[ ]:


market_train_data.shape


# The data includes a subset of US-listed instruments. The set of included instruments changes daily and is determined based on the amount traded and the availability of information. This means that there may be instruments that enter and leave this subset of data. There may therefore be gaps in the data provided, and this does not necessarily imply that that data does not exist (those rows are likely not included due to the selection criteria).
# 
# The marketdata contains a variety of returns calculated over different timespans. All of the returns in this set of marketdata have these properties:
# 
# - Returns are always calculated either open-to-open (from the opening time of one trading day to the open of another) or close-to-close (from the closing time of one trading day to the open of another).
# - Returns are either raw, meaning that the data is not adjusted against any benchmark, or market-residualized (Mktres), meaning that the movement of the market as a whole has been accounted for, leaving only movements inherent to the instrument.
# - Returns can be calculated over any arbitrary interval. Provided here are 1 day and 10 day horizons.
# - Returns are tagged with 'Prev' if they are backwards looking in time, or 'Next' if forwards looking.
# 
# Within the marketdata, you will find the following columns:
# 
# - time(datetime64[ns, UTC]) - the current time (in marketdata, all rows are taken at 22:00 UTC)
# - assetCode(object) - a unique id of an asset
# - assetName(category) - the name that corresponds to a group of assetCodes. These may be "Unknown" if the corresponding - assetCode does not have any rows in the news data.
# - universe(float64) - a boolean indicating whether or not the instrument on that day will be included in scoring. This value is not provided outside of the training data time period. The trading universe on a given date is the set of instruments that are avilable for trading (the scoring function will not consider instruments that are not in the trading universe). The trading universe changes daily.
# - volume(float64) - trading volume in shares for the day
# - close(float64) - the close price for the day (not adjusted for splits or dividends)
# - open(float64) - the open price for the day (not adjusted for splits or dividends)
# - returnsClosePrevRaw1(float64) - see returns explanation above
# - returnsOpenPrevRaw1(float64) - see returns explanation above
# - returnsClosePrevMktres1(float64) - see returns explanation above
# - returnsOpenPrevMktres1(float64) - see returns explanation above
# - returnsClosePrevRaw10(float64) - see returns explanation above
# - returnsOpenPrevRaw10(float64) - see returns explanation above
# - returnsClosePrevMktres10(float64) - see returns explanation above
# - returnsOpenPrevMktres10(float64) - see returns explanation above
# - returnsOpenNextMktres10(float64) - 10 day, market-residualized return. This is the target variable used in competition scoring. The market data has been filtered such that returnsOpenNextMktres10 is always not null.
# 
# 
# Now let us examine the market dataset to see if there are any null values by using a function

# In[ ]:


def missing_value_graph(data):
    data = [
    go.Bar(
        x = data.columns,
        y = data.isnull().sum(),
        name = 'Unknown Asset Values',
        textfont=dict(size=12),
        marker=dict(
        line=dict(
            color='red',
            width=1,
        ), opacity = 0.55
    )
    ),
    ]
    layout= go.Layout(
        title= '"Total Missing Value By Feature"',
        xaxis= dict(title='Features', ticklen=5, zeroline=False, gridwidth=2),
        yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),
        showlegend=True
    )
    fig= go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='skin')


#     Let us now plot the missing values in market train dataset.

# In[ ]:


missing_value_graph(market_train_data)


# Now let us explore the news train dataset in detail .
# # News Data

# In[ ]:


news_train_data.shape


# In[ ]:


news_train_data.head()


# The news data contains information at both the news article level and asset level (in other words, the table is intentionally not normalized).
# 
# - time(datetime64[ns, UTC]) - UTC timestamp showing when the data was available on the feed (second precision)
# - sourceTimestamp(datetime64[ns, UTC]) - UTC timestamp of this news item when it was created
# - firstCreated(datetime64[ns, UTC]) - UTC timestamp for the first version of the item
# - sourceId(object) - an Id for each news item
# - headline(object) - the item's headline
# - urgency(int8) - differentiates story types (1: alert, 3: article)
# - takeSequence(int16) - the take sequence number of the news item, starting at 1. For a given story, alerts and articles have separate sequences.
# - provider(category) - identifier for the organization which provided the news item (e.g. RTRS for Reuters News, BSW for Business Wire)
# - subjects(category) - topic codes and company identifiers that relate to this news item. Topic codes describe the news item's subject matter. These can cover asset classes, geographies, events, industries/sectors, and other types.
# - audiences(category) - identifies which desktop news product(s) the news item belongs to. They are typically tailored to specific audiences. (e.g. "M" for Money International News Service and "FB" for French General News Service)
# - bodySize(int32) - the size of the current version of the story body in characters
# - companyCount(int8) - the number of companies explicitly listed in the news item in the subjects field
# - headlineTag(object) - the Thomson Reuters headline tag for the news item
# - marketCommentary(bool) - boolean indicator that the item is discussing general market conditions, such as "After the Bell" summaries
# - sentenceCount(int16) - the total number of sentences in the news item. Can be used in conjunction with firstMentionSentence to determine the relative position of the first mention in the item.
# - wordCount(int32) - the total number of lexical tokens (words and punctuation) in the news item
# - assetCodes(category) - list of assets mentioned in the item
# - assetName(category) - name of the asset
# - firstMentionSentence(int16) - the first sentence, starting with the headline, in which the scored asset is mentioned.
#     - 1: headline
#     - 2: first sentence of the story body
#     - 3: second sentence of the body, etc
#     - 0: the asset being scored was not found in the news item's headline or body text. As a result, the entire news item's text (headline + body) will be used to determine the sentiment score.
# - relevance(float32) - a decimal number indicating the relevance of the news item to the asset. It ranges from 0 to 1. If the asset is mentioned in the headline, the relevance is set to 1. When the item is an alert (urgency == 1), relevance should be gauged by firstMentionSentence instead.
# - sentimentClass(int8) - indicates the predominant sentiment class for this news item with respect to the asset. The indicated class is the one with the highest probability.
# - sentimentNegative(float32) - probability that the sentiment of the news item was negative for the asset
# - sentimentNeutral(float32) - probability that the sentiment of the news item was neutral for the asset
# - sentimentPositive(float32) - probability that the sentiment of the news item was positive for the asset
# - sentimentWordCount(int32) - the number of lexical tokens in the sections of the item text that are deemed relevant to the asset. This can be used in conjunction with wordCount to determine the proportion of the news item discussing the asset.
# - noveltyCount12H(int16) - The 12 hour novelty of the content within a news item on a particular asset. It is calculated by comparing it with the asset-specific text over a cache of previous news items that contain the asset.
# - noveltyCount24H(int16) - same as above, but for 24 hours
# - noveltyCount3D(int16) - same as above, but for 3 days
# - noveltyCount5D(int16) - same as above, but for 5 days
# - noveltyCount7D(int16) - same as above, but for 7 days
# - volumeCounts12H(int16) - the 12 hour volume of news for each asset. A cache of previous news items is maintained and the number of news items that mention the asset within each of five historical periods is calculated.
# - volumeCounts24H(int16) - same as above, but for 24 hours
# - volumeCounts3D(int16) - same as above, but for 3 days
# - volumeCounts5D(int16) - same as above, but for 5 days
# - volumeCounts7D(int16) - same as above, but for 7 days
# 
# 
# Now let us examine the news dataset to see if there are any null values 

# In[ ]:


missing_value_graph(news_train_data)


# It appears there are no null values in the news dataset.
# 
# Now let us do some pre processing to impute the missing values by using a function as follows

# In[ ]:


def missing_value_impute(data):
    for i in data.columns:
        if data[i].dtype == "object":
            data[i] = data[i].fillna("other")
        elif (data[i].dtype == "int64" or data[i].dtype == "float64"):
            data[i] = data[i].fillna(data[i].mean())
        else:
            pass
    return data


# Let us now impute the missing values in the market train dataset.

# In[ ]:


market_train_data = missing_value_impute(market_train_data)


# ### Top 10 Performing Stock Assets by Volume

# In[ ]:


asset_by_volume = market_train_data.groupby("assetCode")["close"].count().to_frame().sort_values(by=['close'],ascending= False)

asset_by_volume = asset_by_volume.sort_values(by=['close'])

top10_asset_by_volume = list(asset_by_volume.nlargest(10, ['close']).index)

top10_asset_by_volume


# Now let us visualise the top 10 performing assets

# In[ ]:


import random
def generate_color():
    color = '#{:02x}{:02x}{:02x}'.format(*map(lambda x: random.randint(0, 255), range(10)))
    return color

for i in top10_asset_by_volume:
    asset1_df = market_train_data[(market_train_data['assetCode'] == i) & (market_train_data['time'] > '2007-02-01') & (market_train_data['time'] < '2017-01-01')]
    # Create a trace
    trace1 = go.Scatter(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset1_df['close'].values,
        line = dict(color = generate_color()),opacity = 0.8
    )

    layout = dict(title = "Closing Price of {}".format(i),
                  xaxis = dict(title = 'Year'),
                  yaxis = dict(title = 'Price in USD'),
                  )
    data = [trace1]
    py.iplot(dict(data=data, layout=layout), filename='basic-line')


# ### Top 10 Performing Stock Assets by Open & Close Values

# In[ ]:


for i in top10_asset_by_volume:

    asset1_df['high'] = asset1_df['open']
    asset1_df['low'] = asset1_df['close']

    for ind, row in asset1_df.iterrows():
        if row['close'] > row['open']:
            
            asset1_df.loc[ind, 'high'] = row['close']
            asset1_df.loc[ind, 'low'] = row['open']

    trace1 = go.Candlestick(
        x = asset1_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        Open = asset1_df['open'].values,
        Low = asset1_df['low'].values,
        High = asset1_df['high'].values,
        Close = asset1_df['close'].values,
        increasing=dict(line=dict(color= generate_color())),
        decreasing=dict(line=dict(color= generate_color())))

    layout = dict(title = "Candlestick chart for {}".format(i),
                  xaxis = dict(
                      title = 'Date',
                      rangeslider = dict(visible = False)
                  ),
                  yaxis = dict(title = 'Price in USD')
                 )
    data = [trace1]

    py.iplot(dict(data=data, layout=layout), filename='basic-line')


# ### Top 100 Assets Detailed Analysis

# In[ ]:


for i in range(1,100,10):
    Volume_By_Assets = market_train_data.groupby(market_train_data['assetCode'])['volume'].sum()
    Highest_Volumes = Volume_By_Assets.sort_values(ascending=False)[i:i+9]
    # Create a trace
    colors = ['#E1396C','#FEBFB3','#D0F9B1','#96D38C']
    trace1 = go.Pie(
        labels = Highest_Volumes.index,
        values = Highest_Volumes.values,
        textfont=dict(size=20),
        marker=dict(colors=colors,line=dict(color='#000000', width=2)), hole = 0.45)
    layout = dict(title = "Highest Trading Volumes in the range of {} to {}".format(i, i+9))
    data = [trace1]
    py.iplot(dict(data=data, layout=layout), filename='basic-line')


# **Top 20 Unknown Assets by Asset Code **

# In[ ]:


unknown_asset_name= market_train_data[market_train_data['assetName'] == 'Unknown'].groupby('assetCode')

unknown_assets = unknown_asset_name.size().reset_index('assetCode')

unknown_assets.columns = ['assetCode',"value"]

unknown_assets = unknown_assets.sort_values("value", ascending= False)

unknown_assets.head(5)

colors = []

for i in range(len(unknown_assets)):
     colors.append(generate_color())

        
data = [
    go.Bar(
        x = unknown_assets.assetCode.head(20),
        y = unknown_assets.value.head(20),
        name = 'Unknown Assets',
        textfont=dict(size=20),
        marker=dict(
        color= colors,
        line=dict(
            color='#000000',
            width=2,
        ), opacity = 0.65
    )
    ),
    ]
layout= go.Layout(
    title= 'Unknown Assets by Asset code',
    xaxis= dict(title='Asset Codes', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Value Count', ticklen=5, gridwidth=2),
    showlegend=False
)
fig= go.Figure(data=data, layout=layout)
py.iplot(fig, filename='skin')


# 
# ## If you like this kernel greatly appreciate an  UP VOTE 
