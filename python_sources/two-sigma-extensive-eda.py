#!/usr/bin/env python
# coding: utf-8

# # Two Sigma: Extended EDA 
# **Contents**  
# [1. Introduction](#1)    
# &nbsp;&nbsp;&nbsp;&nbsp; [1.1 End-to-End Usage Example](#1.1)  
# &nbsp;&nbsp;&nbsp;&nbsp; [1.2 *get_training_data* Function](#1.2)  
# &nbsp;&nbsp;&nbsp;&nbsp; [1.3 Understanding the target](#1.3)  
# [2. Exploration - Market Train](#2)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.0 Snapshot - Market Train](#2.0)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.1 *time* Variable](#2.1)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.2 *Asset Code* Variable](#2.2)    
# &nbsp;&nbsp;&nbsp;&nbsp; [2.3 *Asset Name*   Variable](#2.3)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.4 *volume* Variable](#2.4)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.5 *Close & Open* Variable](#2.5)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.6 *returnsClose & returnsOpen*  Variable](#2.6)   
# &nbsp;&nbsp;&nbsp;&nbsp; [2.7 *universe* Variable](#2.7)   
# [3. Exploration - News Train](#3)    
# &nbsp;&nbsp;&nbsp;&nbsp; [3.0 Snapshot - News Train](#3.0)    
# &nbsp;&nbsp;&nbsp;&nbsp; [3.1 *time* Variable](#3.1)    
# &nbsp;&nbsp;&nbsp;&nbsp; [3.2 *firstCreated* Variable](#3.2)    
# &nbsp;&nbsp;&nbsp;&nbsp; [3.3 *sourceId*   Variable](#3.3)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.4 *headline* Variable](#3.4)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.5 *urgency* Variable](#3.5)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.6 *takeSequence*  Variable](#3.6)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.7 *provider* Variable](#3.7)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.8 *subjects* Variable](#3.8)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.9 *audiences* Variable](#3.9)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.10 *bodySize* Variable](#3.10)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.11 *companyCount* Variable](#3.11)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.12 *headlineTag* Variable](#3.12)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.13 *marketCommentary* Variable](#3.13)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.14 *sentenceCount* Variable](#3.14)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.15 *wordCount* Variable](#3.15)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.16 *assetCodes* Variable](#3.16)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.17 *assetName* Variable](#3.17)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.18 *firstMentionSentence* Variable](#3.18)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.19 *relevance* Variable](#3.19)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.20 *sentiment* Variable](#3.20)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.21 *sentimentPositive* Variable](#3.21)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.22 *sentimentWordCount* Variable](#3.22)   
# &nbsp;&nbsp;&nbsp;&nbsp; [3.23 *noveltyCountXXY* Variable](#3.23)   
# [4. Workflow strcuture](#4)    
# &nbsp;&nbsp;&nbsp;&nbsp; [4.1 *get_prediction_days* Function](#4.1)   
# &nbsp;&nbsp;&nbsp;&nbsp; [4.2 *predict* Function](#4.2)   
# &nbsp;&nbsp;&nbsp;&nbsp; [4.3 Main Loop Structure](#4.3)   
# &nbsp;&nbsp;&nbsp;&nbsp; [4.4 *write_submission_file* Function](#4.3)    
# &nbsp;&nbsp;&nbsp;&nbsp; [4.5 Restart the Kernel to run your code again](#4.5)   
# &nbsp;&nbsp;&nbsp;&nbsp; [4.6 the1owl's LGBM](#4.6)  
# 
# -----
# ## <a id="1">1. Introduction </a>
# In this competition we will predict how stocks change based on market state and news articles.
# 
# We will loop through a long series of trading days; for each day, we will receive an updated state of the market, and a series of news articles which were published since the last trading day, along with impacted stocks and sentiment analysis.  We . will use this information to predict whether each stock will have increased or decreased ten trading days into the future.  Once you make these predictions, you can move on to the next trading day. 
# 
# This competition is different from most Kaggle Competitions in that:
# 1. We can **only** submit from **Kaggle Kernels**, and we may **not** use **other data sources**, GPU, or internet access.
# 1.**two-stage competition**.  
# &nbsp;&nbsp;&nbsp;&nbsp;1. In **Stage One** we can **edit** our Kernel and **improve** our model, where **Public Leaderboard** scores are based on their predictions relative to past market data.   
# &nbsp;&nbsp;&nbsp;&nbsp;2. At the beginning of **Stage Two**, Kernels are locked, and our Kernels will then be **re-run over the next six months**, scoring them based on their predictions relative to live data as those six months unfold.
# 1. We must use our custom **`kaggle.competitions.twosigmanews`** Python module.  The purpose of this module is to control the flow of information to ensure that you are not using future data to make predictions for the current trading day.    
# 
# **Note that there are no limitations regarding model development.**
# 
# ### <a id="1.1">1.1 End-to-End Usage Example</a>  
# 
# In this Starter Kernel, we'll show how to use the **`twosigmanews`** module to get the training data, get test features and make predictions, and write the submission file.  
# 
# ```
# from kaggle.competitions import twosigmanews
# env = twosigmanews.make_env()
# 
# (market_train_df, news_train_df) = env.get_training_data()
# train_my_model(market_train_df, news_train_df)
# 
# for (market_obs_df, news_obs_df, predictions_template_df) in env.get_prediction_days():
#   predictions_df = make_my_predictions(market_obs_df, news_obs_df, predictions_template_df)
#   env.predict(predictions_df)
#   
# env.write_submission_file()
# ```  
# 
# Note that `train_my_model` and `make_my_predictions` are functions you need to write for the above example to work.

# In[ ]:


# import all libraries

import warnings # adds, removes or modifies python library behavior 
warnings.simplefilter(action='ignore', category=FutureWarning) # turn off future warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) # turn off deprecation warnings
# Deprecation Warnings: cross_validation, weight_boosting, grid_search,learning_curve

import numpy as np # linear algebra
import pandas as pd # data processing
import logging # tracking events
import datetime # classes for dates
import time # time definitions
import os # operating system

import scipy.stats as stats # stats contains probability distributions
import pylab as pl # combines pyplot and numpy

import seaborn as sns # statistical data visualization
import matplotlib.pyplot as plt # 2D plotting library

# tools for data mining and data analysis
from sklearn import *

from xgboost import XGBClassifier # high performance gradient boosting
import lightgbm as lgb # fast, distributed, high performance gradient boosting

import plotly.offline as py # graphing library
py.init_notebook_mode(connected=True) # plot your graphs offline inside a Jupyter Notebook 
import plotly.graph_objs as go # web-service for hosting graphs
import plotly.tools as tls # web-service for hosting graphs


# In[ ]:


from kaggle.competitions import twosigmanews # imports kaggle module and create an environment
env = twosigmanews.make_env()


# ### <a id="1.2">1.2 *get_training_data* Function</a>
# 
# Returns the training data DataFrames as a tuple of:
# * `market_train_df`: DataFrame with market training data (In this kernet `mt_df` is used for convinience)
# * `news_train_df`: DataFrame with news training data (In this kernet `mt_df` is used for convinience)
# 
# These DataFrames contain all market and news data from February 2007 to December 2016.  See the [competition's Data tab](https://www.kaggle.com/c/two-sigma-financial-news/data) for more information on what columns are included in each DataFrame.

# # Load train data

# In[ ]:


logging.info('Load data in 2 dataframes: mt_df (market_train_df) & nt_df (news_train_df)')
(mt_df, nt_df) = env.get_training_data()


# # Load test data

# In[ ]:


days = env.get_prediction_days()
(mt_obs_df, nt_obs_df, predictions_template_df) = next(days)


# ## <a id="2">2. Exploration - Market Train </a>
# ### <a id="2.0">2.0 Snapshot - Application Train </a>

# In[ ]:


print("market_train_df's shape:",mt_df.shape)


# Our market data has a total of 4.072.956 rows and 16 columns. We now look at datatypes.

# In[ ]:


mt_df.dtypes


# All types seem to be float64 except for:
# 
# * *time* -> Datetime
# * *AssetCode* -> object
# * *assetName* -> category  
# We now look at the number of **NaN** in each field

# In[ ]:


mt_df.isna().sum()


# In[ ]:


percent1 = (100 * mt_df.isnull().sum() / mt_df.shape[0]).sort_values(ascending=False)
percent1.plot(kind="bar", figsize = (20,10), fontsize = 20)
plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Value Percent(%)", fontsize = 20)
plt.title("Total Missing Value by market_obs_df", fontsize = 20)


# We now look at **number of unique values**

# In[ ]:


mt_df.nunique()


# We now look at the some value samples

# In[ ]:


mt_df.head(5)


# In[ ]:


mt_df.tail(5)


# In[ ]:


mt_df.select_dtypes(include=['float64']).describe()


# In[ ]:


mt_obs_df.head(5)


# In[ ]:


mt_obs_df.tail(5)


# ### <a id="2.1">2.1 *time* Variable </a>
# Data Description: time(datetime64[ns, UTC]) - the current time (in marketdata, all rows are taken at 22:00 UTC)  
# We first look at the oldest and most recent dates and at the number of unique dates.

# In[ ]:


print('Oldest date:', mt_df['time'].min().strftime('%Y-%m-%d'))
print('Most recent date:', mt_df['time'].max().strftime('%Y-%m-%d'))
print('Total number of different dates:', mt_df['time'].nunique())


# According to the competition's data description:
# > all rows are taken at 22:00 UTC  

# In[ ]:


mt_df['time'].dt.time.describe()


# As there is only 1 unique value the above statement proves to be true  
# We then look at the distribution of rows by date:

# In[ ]:


mt_df["time"].groupby([mt_df["time"].dt.year, mt_df["time"].dt.month]).count().plot(kind="bar",figsize=(21,5))


# ### <a id="2.2">2.2 *Asset Code* Variable </a>
# Data Description: assetCode(object) - a unique id of an asset  
# We first check the total number of unique *asset Codes* 

# In[ ]:


print('Total number of unique assetCodes:', mt_df['assetCode'].nunique())


# By multiplying total number of unique days (2.498) and unique asset codes (3.780) we obtain: **9.442.440 combinations vs dataset size of 4.072.956** which means that not all assets have been reported for all dates.  We now check some assetCode values to find out what they look like. Note that we already check that there are no nulls in this field.

# In[ ]:


print(mt_df['assetCode'].values)


# ### <a id="2.3">2.3 *Asset Name* Variable </a>
# Data Description: assetName(category) - the name that corresponds to a group of assetCodes. These may be "Unknown" if the corresponding assetCode does not have any rows in the news data.  
# We first check the total number of unique *asset Name*

# In[ ]:


print('Total number of unique assetNames:', mt_df['assetName'].nunique())


# So, number of number of unique *asset Names* (3.511) < unique *asset Codes* (3.780). We now check how many combinations *asset Names* & *asset Codes*  in order to know if the same asset code can have different asset Names

# In[ ]:


print('Total number of unique assetCode & assetNames:', mt_df[['assetName','assetCode']].nunique())


# Since the Total number of unique assetCode & assetNames is 3.780 which is equal to the number of unique *asset Codes* (3.780), we can safely assume that **1 asset code can only have 1 asset name**.  
# We now check some assetCode values to find out what they look like. Note that we already check that there are no nulls in this field.

# In[ ]:


print("There are {:,} records with assetName = `Unknown` in the training set".format(mt_df[mt_df['assetName'] == 'Unknown'].size))
assetNameGB = mt_df[mt_df['assetName'] == 'Unknown'].groupby('assetCode')
unknownAssets = assetNameGB.size().reset_index('assetCode')
print("There are {} unique assets without assetName in the training set".format(unknownAssets.shape[0]))
unknownAssets.columns = ['assetCode','unknowns']
unknownAssets.set_index('assetCode')
unknownAssets.loc[:15,['assetCode','unknowns']].sort_values(by='unknowns', ascending=False).head(10)


# In[ ]:


print(mt_df['assetName'].values)
mt_df['assetName'].iloc[0]


#  ### <a id="2.4">2.4 *Volume* Variable </a>
#  Data Description: volume(float64) - trading volume in shares for the day  

# In[ ]:


print('Min:', round(mt_df['volume'].min(),0))
print('Max:', round(mt_df['volume'].max(),0))
print('Mean:', round(mt_df['volume'].mean(),0))
print('Median:', round(mt_df['volume'].median(),0))


# We then look at volumes distribution

# In[ ]:


mt_df['volume'].plot(kind='hist', bins=[0,200000,400000,600000,800000,1000000]) 


#  ### <a id="2.5">2.5 *Close & Open* Variable </a>
#   Data Description:   
# * close(float64) - the close price for the day (not adjusted for splits or dividends)  
# * open(float64) - the open price for the day (not adjusted for splits or dividends)  
# ---
# We now look at both, close and open values in more detail

# In[ ]:


mt_df['close'].describe().apply(lambda x: format(x, 'f'))


# In[ ]:


mt_df['open'].describe().apply(lambda x: format(x, 'f'))


# Both distributions seem normal and no fruther analysis seems required

#  ### <a id="2.6">2.6 *returnsClose & returnsOpen* Variable </a>
#  Data Description:  The marketdata contains a variety of returns calculated over different timespans. All of the returns in this set of marketdata have these properties:
# 
# * Returns are always calculated either:
#     * **open-to-open** (from the opening time of one trading day to the open of another) or 
#     * **close-to-close** (from the closing time of one trading day to the open of another).  
# * Returns are either:
#     * **raw**, meaning that the data is not adjusted against any benchmark, or
#     * **market-residualized** (Mktres), meaning that the movement of the market as a whole has been accounted for, leaving only movements inherent to the instrument.  
# * Returns can be calculated over any arbitrary interval. Provided here are 1 day and 10 day horizons.  
# * Returns are tagged with 'Prev' if they are backwards looking in time, or 'Next' if forwards looking.  
# 
# ---
# * returnsClosePrevRaw1
# * returnsOpenPrevRaw1
# * returnsClosePrevMktres1
# * returnsOpenPrevMktres1
# * returnsClosePrevRaw10
# * returnsOpenPrevRaw10
# * returnsClosePrevMktres10
# * returnsOpenPrevMktres10
# * returnsOpenNextMktres10
# 
# Let's see...

# In[ ]:


mt_df['open']


# ### <a id="2.7">2.7 *universe* Variable </a>
# Data Description: universe(float64) - a boolean indicating whether or not the instrument on that day will be included in scoring. This value is not provided outside of the training data time period. The trading universe on a given date is the set of instruments that are avilable for trading (the scoring function will not consider instruments that are not in the trading universe). The trading universe changes daily.  
# 

# In[ ]:


mt_df['universe'].describe().apply(lambda x: format(x, 'f'))


# ### 1.8 Sample -  Apple Inc

# In[ ]:


# plotAsset plots assetCode1 from date1 to date2
def plotAsset(assetCode1,date1,date2):
    asset_df = mt_df[(mt_df['assetCode'] == assetCode1) 
                      & (mt_df['time'] > date1) 
                      & (mt_df['time'] < date2)]
    # Create a trace
    trace1 = go.Scatter(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values)

    layout = dict(title = "Closing prices of {}".format(assetCode1),
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),)
    
    data = [trace1]

    py.iplot(dict(data=data, layout=layout), filename='basic-line')


# In[ ]:


plotAsset('AAPL.O','2015-01-01','2017-01-01')


# In[ ]:


def Candlestick(assetCode1,date1,date2):

    asset_df = mt_df[(mt_df['assetCode'] == assetCode1) 
                  & (mt_df['time'] > date1) 
                  & (mt_df['time'] < date2)]
    
    asset_df['high'] = asset_df['open']
    asset_df['low'] = asset_df['close']

    for ind, row in asset_df.iterrows():
        if row['close'] > row['open']:
            asset_df.loc[ind, 'high'] = row['close']
            asset_df.loc[ind, 'low'] = row['open']

    trace1 = go.Candlestick(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        open = asset_df['open'].values,
        low = asset_df['low'].values,
        high = asset_df['high'].values,
        close = asset_df['close'].values
    )

    layout = dict(title = "Candlestick chart for {}".format(assetCode1),
                  xaxis = dict(
                      title = 'Month',
                      rangeslider = dict(visible = False)
                  ),
                  yaxis = dict(title = 'Price (USD)'))
    
    data = [trace1]

    py.iplot(dict(data=data, layout=layout), filename='basic-line')   


# In[ ]:


Candlestick('AAPL.O','2015-01-01','2017-01-01')


# **Types of Returns**
# *For example we'll look for the first stock in assetNames*
# 1. Returns calculated close-to-close (from the closing time of one trading day to the closing time of anotherc & not adjusted) for 1 day. *returnsClosePrevRaw1*
# 2. Returns calculated open-to-open (from the opening time of one trading day to the opening time of another & not adjusted) for 1 day. *returnsOpenPrevRaw1*
# ![image.png](attachment:image.png)

# # <a id="1">3. Exploration - News Train </a>
# The news data contains information at both the news article level and asset level (in other words, the table is intentionally not normalized).

# In[ ]:


print("news_train_df's shape:",nt_df.shape) 


# In[ ]:


nt_df.dtypes


# In[ ]:


nt_df.isna().sum()


# There are no nulls in the news train data

# In[ ]:


nt_df.nunique()


# In[ ]:


nt_df.head(5)


# In[ ]:


nt_df.tail(5)


# In[ ]:


nt_df.describe(include='all')


# ### <a id="3.1">3.1 *time* & *sourceTimestamp* Variables </a> 
# **Data Description:**   
# * time(datetime64[ns, UTC]) - UTC timestamp showing when the data was available on the feed (second precision)  
# * sourceTimestamp(datetime64[ns, UTC]) - UTC timestamp of this news item when it was created

# In[ ]:


print('Oldest date:', nt_df['time'].min().strftime('%Y-%m-%d'))
print('Most recent date:', nt_df['time'].max().strftime('%Y-%m-%d'))
print("There are {} missing values in the `time` column".format(nt_df['time'].isna().sum()))
nt_df['time'].dt.date.describe()


# In[ ]:


print('Oldest date:', nt_df['sourceTimestamp'].min().strftime('%Y-%m-%d'))
print('Most recent date:', nt_df['sourceTimestamp'].max().strftime('%Y-%m-%d'))
print("There are {} missing values in the `sourceTimestamp` column".format(nt_df['sourceTimestamp'].isna().sum()))
nt_df['sourceTimestamp'].dt.date.describe()


# In[ ]:


print(nt_df.loc[nt_df['time'] == nt_df['sourceTimestamp']].shape[0])


# It seems that in about 80% of the cases both time and sourceTimestamp match

# ### <a id="3.2">3.2 *firstCreated* Variable </a> 
# **Data Description:** firstCreated(datetime64[ns, UTC]) - UTC timestamp for the first version of the item

# In[ ]:


print('Oldest date:', nt_df['firstCreated'].min().strftime('%Y-%m-%d'))
print('Most recent date:', nt_df['firstCreated'].max().strftime('%Y-%m-%d'))
print("There are {} missing values in the `firstCreated` column".format(nt_df['firstCreated'].isna().sum()))
nt_df['firstCreated'].dt.date.describe()


# ### <a id="3.3">3.3 *sourceId* Variable </a> 
# **Data Description:** sourceId(object) - an Id for each news item

# In[ ]:


print("There are {} missing values in the `sourceId` column".format(nt_df.sourceId.isna().sum()))
print("There are {} unique values in the `sourceId` column".format(nt_df.sourceId.nunique()))
print("There are {} unique values in the `sourceId` column".format(nt_df.sourceId.count()))
print(nt_df.sourceId.describe())


# Looking at the data: 6.340.206 unique source ids vs 9.328.750 total source ids we notice that some news appear repeated.  
# We will now look at the one that appears repeated the most 'd7ad319ee02edea0'    

# In[ ]:


nt_df[nt_df['sourceId']=='d7ad319ee02edea0']


# 

# In[ ]:


nt_df[nt_df.duplicated(keep=False)].shape[0]


# ### <a id="3.4">3.4 *headline* Variables </a> 
# **Data Description:** headline(object) - the item's headline

# In[ ]:


print("There are {} missing values in the `headline` column".format(nt_df['headline'].isna().sum()))
print("There are {} unique values in the `headline` column".format(nt_df.headline.nunique()))
print("There are {} unique values in the `headline` column".format(nt_df.headline.count()))


# We now look at a few headlines:

# In[ ]:


for i in range(0,20):
    print(nt_df['headline'].iloc[i])


# 

# ### <a id="3.5">3.5 *urgency* Variable </a> 
# **Data Description:** urgency(int8) - differentiates story types (1: alert, 3: article)

# In[ ]:


print(nt_df['urgency'].describe())
print("Unique values in the `urgency` column: {}".format(nt_df['urgency'].unique()))


# We now look at the distribution

# In[ ]:


print(nt_df['urgency'].head(5))
print(nt_df.groupby(['urgency']).count())
sns.distplot(nt_df['urgency'])


# Only 25 news have been reported as urgency 2. Does it make sence to keep this category? 

# ### <a id="3.6">3.6 *takeSequence* Variable </a> 
# **Data Description:** takeSequence(int16) - the take sequence number of the news item, starting at 1. For a given story, alerts and articles have separate sequences.

# In[ ]:


print(nt_df['takeSequence'].head(5))
sns.distplot(nt_df['takeSequence'])


# ### <a id="3.7">3.7 *provider* Variable </a> 
# **Data Description:** provider(category) - identifier for the organization which provided the news item (e.g. RTRS for Reuters News, BSW for Business Wire)

# In[ ]:


print(nt_df['provider'].head(5))


# ### <a id="3.8">3.8 *subjects* Variable </a> 
# **Data Description:** subjects(category) - topic codes and company identifiers that relate to this news item. Topic codes describe the news item's subject matter. These can cover asset classes, geographies, events, industries/sectors, and other types.

# In[ ]:


print(nt_df['subjects'].head(5))


# In[ ]:


for i in list(range(5)):
    print(nt_df['subjects'].iloc[i])


# ### <a id="3.9">3.9 *audiences* Variable </a> 
# **Data Description:** audiences(category) - identifies which desktop news product(s) the news item belongs to. They are typically tailored to specific audiences. (e.g. "M" for Money International News Service and "FB" for French General News Service)

# In[ ]:


for i in list(range(5)):
    print(nt_df['audiences'].iloc[i])


# ### <a id="3.10">3.10 *bodySize* Variable </a> 
# **Data Description:** bodySize(int32) - the size of the current version of the story body in characters

# In[ ]:


print(nt_df['bodySize'].head(5))
print(nt_df['bodySize'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.companyCount)


# ### <a id="3.11">3.11 *companyCount* Variable </a> 
# **Data Description:** companyCount(int8) - the number of companies explicitly listed in the news item in the subjects field

# In[ ]:


print(nt_df['companyCount'].head(5))
sns.distplot(nt_df.companyCount)


# ### <a id="3.12">3.12 *headlineTag* Variable </a> 
# **Data Description:** headlineTag(object) - the Thomson Reuters headline tag for the news item

# In[ ]:


print("There are {} missing values in the `headlineTag` column".format(nt_df['headlineTag'].isna().sum()))
print("There are {} unique values in the `headlineTag` column".format(nt_df.headlineTag.nunique()))
print("There are {} unique values in the `headlineTag` column".format(nt_df.headlineTag.count()))
print(nt_df['headlineTag'].unique())


# ### <a id="3.13">3.13 *marketCommentary* Variable </a> 
# **Data Description:** marketCommentary(bool) - boolean indicator that the item is discussing general market conditions, such as "After the Bell" summaries[](http://)

# In[ ]:


nt_df['marketCommentary'].unique()


# ### <a id="3.14">3.14 *sentenceCount* Variable </a> 
# **Data Description:** sentenceCount(int16) - the total number of sentences in the news item. Can be used in conjunction with firstMentionSentence to determine the relative position of the first mention in the item.

# In[ ]:


print(nt_df['sentenceCount'].head(5))
print(nt_df['sentenceCount'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.sentenceCount)


# ### <a id="3.15">3.15 *wordCount* Variable </a> 
# **Data Description:** wordCount(int32) - the total number of lexical tokens (words and punctuation) in the news item

# In[ ]:


print(nt_df['wordCount'].head(5))
print(nt_df['wordCount'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.wordCount)


# ### <a id="3.16">3.16 *assetCodes* Variable </a> 
# **Data Description:** assetCodes(category) - list of assets mentioned in the item

# In[ ]:


print(nt_df['assetCodes'].head(5))


# ### <a id="3.17">3.17 *assetName* Variable </a> 
# **Data Description:** assetName(category) - name of the asset

# In[ ]:


print(nt_df['assetName'].head(5))


# ### <a id="3.18">3.18 *firstMentionSentence* Variable </a> 
# **Data Description:** firstMentionSentence(int16) - the first sentence, starting with the headline, in which the scored asset is mentioned.
# 1: headline
# 2: first sentence of the story body
# 3: second sentence of the body, etc
# 0: the asset being scored was not found in the news item's headline or body text. As a result, the entire news item's text (headline + body) will be used to determine the sentiment score.

# In[ ]:


print(nt_df['firstMentionSentence'].head(5))
print(nt_df['firstMentionSentence'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.firstMentionSentence)


# ### <a id="3.19">3.19 *relevance* Variable </a> 
# **Data Description:** relevance(float32) - a decimal number indicating the relevance of the news item to the asset. It ranges from 0 to 1. If the asset is mentioned in the headline, the relevance is set to 1. When the item is an alert (urgency == 1), relevance should be gauged by firstMentionSentence instead.

# In[ ]:


print(nt_df['relevance'].head(5))
print(nt_df['relevance'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.relevance)


# ### <a id="3.20">3.20 *sentiment* Variable </a> 
# **Data Description:**  
# * sentimentClass(int8) - indicates the predominant sentiment class for this news item with respect to the asset. The indicated class is the one with the highest probability.
# * sentimentNegative(float32) - probability that the sentiment of the news item was negative for the asset
# * sentimentNeutral(float32) - probability that the sentiment of the news item was neutral for the asset

# In[ ]:


print(nt_df['sentimentClass'].unique())
print(nt_df['sentimentClass'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.sentimentClass)


# In[ ]:


print(nt_df['sentimentNegative'].unique())
print(nt_df['sentimentNegative'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.sentimentNegative)


# In[ ]:


print(nt_df['sentimentNeutral'].unique())
print(nt_df['sentimentNeutral'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.sentimentNeutral)


# [](http://)### <a id="3.21">3.21 *sentimentPositive* Variable </a> 
# **Data Description:** sentimentPositive(float32) - probability that the sentiment of the news item was positive for the asset

# In[ ]:


print(nt_df['sentimentPositive'].head(5))
print(nt_df['sentimentPositive'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.sentimentNeutral)


# ### <a id="3.22">3.22 *sentimentWordCount* Variable </a> 
# **Data Description:** sentimentWordCount(int32) - the number of lexical tokens in the sections of the item text that are deemed relevant to the asset. This can be used in conjunction with wordCount to determine the proportion of the news item discussing the asset.  

# In[ ]:


print(nt_df['sentimentWordCount'].head(5))
print(nt_df['sentimentWordCount'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.sentimentWordCount)


# ### <a id="3.23">3.23 *noveltyCountXXY* Variable </a> 
# **Data Description:**   
# * noveltyCount12H(int16) - The 12 hour novelty of the content within a news item on a particular asset. It is calculated by comparing it with the asset-specific text over a cache of previous news items that contain the asset.
# * noveltyCount24H(int16) - same as above, but for 24 hours
# * noveltyCount3D(int16) - same as above, but for 3 days
# * noveltyCount5D(int16) - same as above, but for 5 days
# * noveltyCount7D(int16) - same as above, but for 7 days
# * volumeCounts12H(int16) - the 12 hour volume of news for each asset. A cache of previous news items is maintained and the number of news items that mention the asset within each of five historical periods is calculated.
# * volumeCounts24H(int16) - same as above, but for 24 hours
# * volumeCounts3D(int16) - same as above, but for 3 days
# * volumeCounts5D(int16) - same as above, but for 5 days
# * volumeCounts7D(int16) - same as above, but for 7 days

# In[ ]:


print(nt_df['noveltyCount12H'].head(5))
print(nt_df['noveltyCount12H'].describe().apply(lambda x: format(x, '5.2f')))
sns.distplot(nt_df.noveltyCount12H)


# ## <a id="4">4. Workflow </a>
# ### <a id="4.1">4.1 `get_prediction_days` function </a> 
# 
# Generator which loops through each "prediction day" (trading day) and provides all market and news observations which occurred since the last data you've received.  Once you call **`predict`** to make your future predictions, you can continue on to the next prediction day.
# 
# Yields:
# * While there are more prediction day(s) and `predict` was called successfully since the last yield, yields a tuple of:
#     * `market_observations_df`: DataFrame with market observations for the next prediction day.
#     * `news_observations_df`: DataFrame with news observations for the next prediction day.
#     * `predictions_template_df`: DataFrame with `assetCode` and `confidenceValue` columns, prefilled with `confidenceValue = 0`, to be filled in and passed back to the `predict` function.
# * If `predict` has not been called since the last yield, yields `None`.

# In[ ]:


# You can only iterate through a result from `get_prediction_days()` once
# so be careful not to lose it once you start iterating.
# days = env.get_prediction_days()


# In[ ]:


# (market_obs_df, news_obs_df, predictions_template_df) = next(days)
# predictions_template_df.head()


# ### <a id="4.2">4.2 **`predict`** function </a> 
# Stores your predictions for the current prediction day.  Expects the same format as you saw in `predictions_template_df` returned from `get_prediction_days`.
# 
# Args:
# * `predictions_df`: DataFrame which must have the following columns:
#     * `assetCode`: The market asset.
#     * `confidenceValue`: Your confidence whether the asset will increase or decrease in 10 trading days.  All values must be in the range `[-1.0, 1.0]`.
# 
# The `predictions_df` you send **must** contain the exact set of rows which were given to you in the `predictions_template_df` returned from `get_prediction_days`.  The `predict` function does not validate this, but if you are missing any `assetCode`s or add any extraneous `assetCode`s, then your submission will fail.

# In[ ]:


def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0
make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)


# ### <a id="4.3">4.3 Main Loop </a>
# Let's loop through all the days and make our random predictions.  The `days` generator (returned from `get_prediction_days`) will simply stop returning values once you've reached the end.

# In[ ]:


for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_random_predictions(predictions_template_df)
    env.predict(predictions_template_df)
print('Done!')


# ### <a id="4.4">4.4 *write_submission_file* Function </a>
# Writes your predictions to a CSV file (`submission.csv`) in the current working directory.

# In[ ]:


env.write_submission_file()


# **Important:** As indicated by the helper message, calling `write_submission_file` on its own does **not** make a submission to the competition.  It merely tells the module to write the `submission.csv` file as part of the Kernel's output.  To make a submission to the competition, you'll have to **Commit** your Kernel and find the generated `submission.csv` file in that Kernel Version's Output tab (note this is _outside_ of the Kernel Editor), then click "Submit to Competition".  When we re-run your Kernel during Stage Two, we will run the Kernel Version (generated when you hit "Commit") linked to your chosen Submission.

# ### <a id="4.5">4.5 Restart the Kernel to run your code again </a>
# In order to combat cheating, you are only allowed to call `make_env` or iterate through `get_prediction_days` once per Kernel run.  However, while you're iterating on your model it's reasonable to try something out, change the model a bit, and try it again.  Unfortunately, if you try to simply re-run the code, or even refresh the browser page, you'll still be running on the same Kernel execution session you had been running before, and the `twosigmanews` module will still throw errors.  To get around this, you need to explicitly restart your Kernel execution session, which you can do by pressing the Restart button in the Kernel Editor's bottom Console tab:
# ![Restart button](https://i.imgur.com/hudu8jF.png)

# ### <a id="4.6">4.6 the1owl's LGBM</a>
# As a baseline I would recommend [the1owl's my-two-sigma-cents-only](https://www.kaggle.com/the1owl/my-two-sigma-cents-only)

# In[ ]:


market_train = mt_df
news_train = nt_df


# In[ ]:


market_train.time = market_train.time.dt.date
news_train.time = news_train.time.dt.hour
news_train.sourceTimestamp= news_train.sourceTimestamp.dt.hour
news_train.firstCreated = news_train.firstCreated.dt.date
news_train['assetCodesLen'] = news_train['assetCodes'].map(lambda x: len(eval(x)))
news_train['assetCodes'] = news_train['assetCodes'].map(lambda x: list(eval(x))[0])
kcol = ['firstCreated', 'assetCodes']
news_train = news_train.groupby(kcol, as_index=False).mean()

market_train = pd.merge(market_train, news_train, 
                        how='left', 
                        left_on=['time', 'assetCode'], 
                        right_on=['firstCreated', 'assetCodes'])

lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}

market_train['assetCodeT'] = market_train['assetCode'].map(lbl)

fcol = [c for c in market_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 
                                             'assetName', 'audiences', 'firstCreated', 
                                             'headline', 'headlineTag', 'marketCommentary', 
                                             'provider', 'returnsOpenNextMktres10', 'sourceId',
                                             'subjects', 'time', 'time_x', 'universe']]


# In[ ]:


x1, x2, y1, y2 = model_selection.train_test_split(market_train[fcol], 
                                                  market_train['returnsOpenNextMktres10'], 
                                                  test_size=0.25, 
                                                  random_state=99)

def lgb_rmse(preds, y): # update to Competition Metric
    y = np.array(list(y.get_label()))
    score = np.sqrt(metrics.mean_squared_error(y, preds))
    return 'RMSE', score, False

params = {'learning_rate': 0.2, 
          'max_depth': 6, 
          'boosting': 'gbdt',
          'objective': 'regression',
          'seed': 2018}

lgb_model = lgb.train(params, 
                      lgb.Dataset(x1, label=y1), 
                      500, 
                      lgb.Dataset(x2, label=y2), 
                      verbose_eval=10,
                      early_stopping_rounds=20)


# In[ ]:


df = pd.DataFrame({'imp': lgb_model.feature_importance(importance_type='gain'), 'col':fcol})
df = df.sort_values(['imp','col'], ascending=[True, False])
_ = df.plot(kind='barh', x='col', y='imp', figsize=(7,12))
#plt.savefig('lgb_gain.png')

df = pd.DataFrame({'imp': lgb_model.feature_importance(importance_type='split'), 'col':fcol})
df = df.sort_values(['imp','col'], ascending=[True, False])
_ = df.plot(kind='barh', x='col', y='imp', figsize=(7,12))
# plt.savefig('lgb_split.png')


# In[ ]:


env.write_submission_file()


# In[ ]:




