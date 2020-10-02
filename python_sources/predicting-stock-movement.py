#!/usr/bin/env python
# coding: utf-8

# ![](https://www.liteforex.com/uploads/article/949_Forex-Trading-Basic-Candlestick-Definitions.jpg)

# # <div style="text-align: center">This is mine comprehensive tutorial of 2sigma competition. Basic premise is predicting the stock market using news and market data
# 

#  <a id="top"></a> <br>
# ## Notebook  Content
# 1. [Introduction](#1)
# 1. [EDA (market data)](#2)
# 1. [Pre Processing (market data)](#3)
# 1. [Feature Engineering (market data)](#4)
# 1. [News data](#5)
# 1. [Joining the datasets and some additional feature engineering](#6)
# 1. [Model and theoretical explanation of xgboost](#7)
# 

# <a id="1"></a> <br>
# #  1-Introduction
# I will be walking the reader trough some of the concepts of data science as well as theory behind tree based algorithms (xgboost in particular). This can be found in seperate papers. This will happen while I try to get predict the stock market.
# Goal of this competition is pretty straight foreward. Given the market (open, close prices, returns etc...) and news data (headlines of stories, reports concerning different companies & markets etc...) we are supposed to predict the 10-day return of a given asset (stock). Detailed description can be found at [Outline](https://www.kaggle.com/c/two-sigma-financial-news). Reader should also read rules and the framework of this competition before proceeding. There are basically two different datasets. We ought to first make ourself familier with it ->>>> **EDA**, and we will first handle market data.

# <a id="2"></a> <br>
# 
# #  2-EDA (market data)
# ### Exploratory data analysis. Goal of this step is to get ourself more familiar with data. There different libraries for plotting in python we will be using plotly (interactive one).
# 
# ## What can we expect from such analysis? We can foresee different trends, errors, correlations and generally get a feeling in what direction should we move in order to get to our goal (building a model usually)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import gc

import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
stop = set(stopwords.words('english'))
from datetime import datetime, timedelta
import time
import fancyimpute
from itertools import chain

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls


# There is a specific way to load the data in this competition (give that we have certain constraints and goals that we want to accomplish later on) and that is:

# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('Data is loaded')


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()


# # Market Data
# 
# We have data set of different companies and their performance measures from 2007 until 2017 on a stock market. More specifically we have 14 performance measures, and 4072956 (these are not all of the companies, since index is time based!)

# In[ ]:


print(f'{market_train_df.shape[0]} samples and {market_train_df.shape[1]} features ')


# **Assumption**
# Data before 2010 will be droped. There is simply to much noise concerning the 2008 crisis. **ALSO** Since this kernel can not submit anymore and it has a learning character, in order to have sufficient memory, we will take an even smaller subsample of data.
# 

# In[ ]:


start = datetime(2012, 1, 1, 0, 0, 0).date()
market_train_df = market_train_df.loc[market_train_df['time'].dt.date >= start].reset_index(drop=True)
news_train_df = news_train_df.loc[news_train_df['time'].dt.date >= start].reset_index(drop=True)
del start

#collect residual garbage
gc.collect()


# Memory saving function:

# In[ ]:



def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    return df


# In[ ]:


reduce_mem_usage(market_train_df)
reduce_mem_usage(news_train_df)


# In[ ]:


market_train_df.head()


# In[ ]:


market_train_df.dtypes


# **NaN values?**

# In[ ]:


market_train_df.isna().sum()


# Unique values?

# In[ ]:


market_train_df.nunique()


# What are the most frequent assets traded (remember not all are present at all times, bankruptcy, IPO etc)

# In[ ]:


volumeByAssets = market_train_df.groupby(market_train_df['assetCode'])['volume'].sum()
highestVolumes = volumeByAssets.sort_values(ascending=False)[0:10]

trace1 = go.Pie(
    labels = highestVolumes.index,
    values = highestVolumes.values
)

layout = dict(title = "Highest trading volumes")
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')


# 2 Variables that are significant and important (for later joining with the news dataset) are **assetcode** and **assetname** assetcode "Each asset is identified by an **assetCode** (note that a single company may have multiple assetCodes)" assetname " the name that corresponds to a group of assetCodes. These may be "Unknown" if the corresponding assetCode does not have any rows in the news data."

# In[ ]:


market_train_df['assetName'].describe()


# In[ ]:


print("There are {:,} records with assetName = `Unknown` in the training set".format(market_train_df[market_train_df['assetName'] == 'Unknown'].size))


# But that does not say us much, we want to know exactly what asset codes are not to be found in the news data set!

# In[ ]:


assetNameGB = market_train_df[market_train_df['assetName'] == 'Unknown'].groupby('assetCode')
unknownAssets = assetNameGB.size().reset_index('assetCode')
print("There are {} unique assets without assetName in the training set".format(unknownAssets.shape[0]))


# In[ ]:


unknownAssets.head()


# Now let us take 5 random assets and plot them. Note that not all assets start measurament from 2007

# In[ ]:


data = []
for asset in np.random.choice(market_train_df['assetName'].unique(), 5):
    asset_df = market_train_df[(market_train_df['assetName'] == asset)]

    data.append(go.Scatter(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Closing prices of 5 random assets",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"))
py.iplot(dict(data=data, layout=layout), filename='basic-line')


# Those were just a 5 random companies, let us have a look now at the general trend of the whole stock market, i.e. the whole data set along with some significant dates (crashes)

# In[ ]:


data = []
for i in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    price_df = market_train_df.groupby('time')['close'].quantile(i).reset_index()

    data.append(go.Scatter(
        x = price_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = price_df['close'].values,
        name = f'{i} quantile'
    ))
layout = go.Layout(dict(title = "Trends of closing prices by quantiles",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"),
    annotations=[
        dict(
            x='2008-09-01 22:00:00+0000',
            y=82,
            xref='x',
            yref='y',
            text='Collapse of Lehman Brothers',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        ),
        dict(
            x='2011-08-01 22:00:00+0000',
            y=85,
            xref='x',
            yref='y',
            text='Black Monday',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        ),
        dict(
            x='2014-10-01 22:00:00+0000',
            y=120,
            xref='x',
            yref='y',
            text='Another crisis',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=-20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        ),
        dict(
            x='2016-01-01 22:00:00+0000',
            y=120,
            xref='x',
            yref='y',
            text='Oil prices crash',
            showarrow=True,
            font=dict(
                family='Courier New, monospace',
                size=16,
                color='#ffffff'
            ),
            align='center',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='#636363',
            ax=20,
            ay=-30,
            bordercolor='#c7c7c7',
            borderwidth=2,
            borderpad=4,
            bgcolor='#ff7f0e',
            opacity=0.8
        )
    ])
py.iplot(dict(data=data, layout=layout), filename='basic-line')


# I told that we could also detect errors/outliers in the measurament of the data. If we think about potential errors concerning stack data, that would (besides NaN-missing values) be values that have sudden huge drop or rise. More than normal even in the time of crisis. We can quantify that with standard deviation of the difference between open and close prices. And then plot it to see where standard deviation is highly unusual (different than the average around 1 point std)

# In[ ]:


market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']
grouped = market_train_df.groupby('time').agg({'price_diff': ['std', 'min']}).reset_index()


# In[ ]:


print(f"Average standard deviation of price change within a day in {grouped['price_diff']['std'].mean():.4f}.")


# Now let us see which time span(month) has the most unusual behavior. And does it coincide with financial crisis.

# In[ ]:


g = grouped.sort_values(('price_diff', 'std'), ascending=False)[:10]
g['min_text'] = 'Maximum price drop: ' + (-1 * g['price_diff']['min']).astype(str)
trace = go.Scatter(
    x = g['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = g['price_diff']['std'].values,
    mode='markers',
    marker=dict(
        size = g['price_diff']['std'].values,
        color = g['price_diff']['std'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = g['min_text'].values
    #text = f"Maximum price drop: {g['price_diff']['min'].values}"
    #g['time'].dt.strftime(date_format='%Y-%m-%d').values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Top 10 months by standard deviation of price change within a day',
    hovermode= 'closest',
    yaxis=dict(
        title= 'price_diff',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


# "Maximum price drop almost 10 000" Something must be wrong here.

# In[ ]:


market_train_df.sort_values('price_diff')[:10]


# We can see that for certain assets opening and closing prices are way off, even in a time of crisis. But to be sure we can consult some other service provider and check that these are actually measurament errors. (bloomberg, yahoo finance etc). Now that we are sure that there are errors. We need to know which ones. But how do you define it? In other words whats the value (falling or rising one) the next day that can be classified as an error. Its a bit subjective, but I suppose that consensus can be reached at lets say +- 100% difference. There will be times when higher values are considered normal, hence a further analysis of this cut-off value should be performed. First of a "crisis cut-off" value could be 20%, where we can say that turbulance is big

# In[ ]:


market_train_df['close_to_open'] =  np.abs(market_train_df['close'] / market_train_df['open'])


# In[ ]:


print(f"In {(market_train_df['close_to_open'] >= 1.2).sum()} lines price increased by 20% or more.")
print(f"In {(market_train_df['close_to_open'] <= 0.80).sum()} lines price decreased by 20% or more.")


# Thats strange, but as already stated, 100% change in either direction is definately an outlier. Out of 4 milion entries there is not that many (40,20)

# In[ ]:


print(f"In {(market_train_df['close_to_open'] >= 2).sum()} lines price increased by 100% or more.")
print(f"In {(market_train_df['close_to_open'] <= 0.5).sum()} lines price decreased by 100% or more.")


# What do we do with outliers? there are some pretty sophisticated imputation techniques that ought to be considered with a larger number of outliers. Here we can be satisfied with mean or median prices to imputat the outliers with. Carefull, we do decrease volatility of our data with it. Best solution would be to build a ML model that imputates the values based on other independent variables. We are going to use simple technique here -> mean imputation

# In[ ]:


market_train_df['assetName_mean_open'] = market_train_df.groupby('assetName')['open'].transform('mean')
market_train_df['assetName_mean_close'] = market_train_df.groupby('assetName')['close'].transform('mean')

# if open price is too far from mean open price for this company, replace it. Otherwise replace close price.
for i, row in market_train_df.loc[market_train_df['close_to_open'] >= 2].iterrows():
    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):
        market_train_df.iloc[i,5] = row['assetName_mean_open']
    else:
        market_train_df.iloc[i,4] = row['assetName_mean_close']
        
for i, row in market_train_df.loc[market_train_df['close_to_open'] <= 0.5].iterrows():
    if np.abs(row['assetName_mean_open'] - row['open']) > np.abs(row['assetName_mean_close'] - row['close']):
        market_train_df.iloc[i,5] = row['assetName_mean_open']
    else:
        market_train_df.iloc[i,4] = row['assetName_mean_close']


# After we impute it, we can observe standard fluctuation:

# In[ ]:


market_train_df['price_diff'] = market_train_df['close'] - market_train_df['open']
grouped = market_train_df.groupby(['time']).agg({'price_diff': ['std', 'min']}).reset_index()
g = grouped.sort_values(('price_diff', 'std'), ascending=False)[:10]
g['min_text'] = 'Maximum price drop: ' + (-1 * np.round(g['price_diff']['min'], 2)).astype(str)
trace = go.Scatter(
    x = g['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = g['price_diff']['std'].values,
    mode='markers',
    marker=dict(
        size = g['price_diff']['std'].values * 5,
        color = g['price_diff']['std'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = g['min_text'].values
    #text = f"Maximum price drop: {g['price_diff']['min'].values}"
    #g['time'].dt.strftime(date_format='%Y-%m-%d').values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Top 10 months by standard deviation of price change within a day',
    hovermode= 'closest',
    yaxis=dict(
        title= 'price_diff',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter2010')


# There are other raw and market residualized (transformed) variables, 1 and 10 day span. How they are residualized is described here [mktres](https://www.kaggle.com/marketneutral/eda-what-does-mktres-mean)

# I want to inspect the targer varible** returnsOpenNextMktres10** Market-residualized open-to-open returns in the next 10 days.

# In[ ]:


market_train_df['returnsOpenNextMktres10'].describe()


# Assumption is that (intuition) this variable should be standard normally distributed. But some of the values (min max) are problematic, let us get rid of them.

# In[ ]:


noOutliers = market_train_df[(market_train_df['returnsOpenNextMktres10'] < 1) &  (market_train_df['returnsOpenNextMktres10'] > -1)]


# In[ ]:


trace1 = go.Histogram(
    x = noOutliers.sample(n=10000)['returnsOpenNextMktres10'].values
)

layout = dict(title = "returnsOpenNextMktres10 (random 10.000 sample; without outliers)")
data = [trace1]

py.iplot(dict(data=data, layout=layout), filename='basic-line')


# As a final EDA analysis we are going to plot mean values of all of the return variables

# In[ ]:


data = []
for col in ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
       'returnsOpenNextMktres10']:
    df = market_train_df.groupby('time')[col].mean().reset_index()
    data.append(go.Scatter(
        x = df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = df[col].values,
        name = col
    ))
    
layout = go.Layout(dict(title = "Treand of mean values",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"),)
py.iplot(dict(data=data, layout=layout), filename='basic-line')


# # Conclusion, **no** ARBITRAGE.... or ? :)

# <a id="3"></a> <br>
# #  3. Pre Processing (market data)
# 
# Now we already tackled some of it (outliers that we noticed with the between closing and opening prices <-> 100% increase is not plausible! hence we imputed with mean value). Now we ought to do it more concisely. NaN, outliers, scaling...

# In[ ]:


market_train_df.isna().sum().to_frame()


# **MICE** MICE has become an industry standard way of dealing with null values while preprocessing data. It is argued that by simply using fill values such as the mean or mode we are throwing information away that is present in other variables that might give insight into what the null values might be. With that thought in mind, we predict the null values from the other features present in the data. Thus preserving as much information as possible. If the data is not missing at random (MAR) then this method is inappropriate.

# In[ ]:


num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                   'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                   'returnsOpenPrevMktres10']


# In[ ]:


from fancyimpute import IterativeImputer


# In[ ]:


imp_cols = market_train_df[num_cols].columns.values
market_train_df[num_cols] = pd.DataFrame(IterativeImputer(verbose=True).fit_transform(market_train_df[num_cols]),columns= imp_cols)


# In[ ]:


market_train_df.isna().sum().to_frame()


# In[ ]:


scaler = StandardScaler()
market_train_df[num_cols] = scaler.fit_transform(market_train_df[num_cols])


# In[ ]:


market_train_df.isna().sum().to_frame()


# <a id="4"></a> <br>
# #  4- Feature Engineering(market data)
# 
# Feature engineering of market (numerical) data. We will do one more Feature Engineering on the time column on the joined dataset underneath

# In[ ]:


def generate_lag_features(df):
    df['MA7MA'] = df['close'].rolling(window=7).mean()
    df['MA_15MA'] = df['close'].rolling(window=15).mean()
    df['MA_30MA'] = df['close'].rolling(window=30).mean()
    df['MA_60MA'] = df['close'].rolling(window=60).mean()
    ewma = pd.Series.ewm
    df['close_30EMA'] = ewma(df["close"], span=30).mean()
    df['close_26EMA'] = ewma(df["close"], span=26).mean()
    df['close_12EMA'] = ewma(df["close"], span=12).mean()

    df['MACD'] = df['close_12EMA'] - df['close_26EMA']

    no_of_std = 2

    df['MA7MA'] = df['close'].rolling(window=7).mean()
    df['MA_7MA_std'] = df['close'].rolling(window=7).std() 
    df['MA_7MA_BB_high'] = df['MA7MA'] + no_of_std * df['MA_7MA_std']
    df['MA_7MA_BB_low'] = df['MA7MA'] - no_of_std * df['MA_7MA_std']


    df['VMA_7MA'] = df['volume'].rolling(window=7).mean()
    df['VMA_15MA'] = df['volume'].rolling(window=15).mean()
    df['VMA_30MA'] = df['volume'].rolling(window=30).mean()
    df['VMA_60MA'] = df['volume'].rolling(window=60).mean()
    
    new_col = df["close"] - df["open"]
    df.insert(loc=6, column="daily_diff", value=new_col)
    df['close_to_open'] =  np.abs(df['close'] / df['open'])
   
    return df


# In[ ]:


generate_lag_features(market_train_df)


# In[ ]:


num_cols = ['volume', 'close', 'open', 'returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1',
                   'returnsOpenPrevMktres1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10',
                   'returnsOpenPrevMktres10','MA7MA','MA_15MA','MA_30MA','MA_60MA','close_30EMA','close_26EMA','close_12EMA','MACD','MA_7MA','MA_7MA_std','MA_7MA_BB_high','MA_7MA_BB_low','close_to_open','VMA_7MA','VMA_15MA','VMA_30MA','VMA_60MA']


# In[ ]:


market_train_df.head()


# 
# **Skewness and outliers**

# **Skewness** I ommited, these are all returns. We plotted all of them together (above no arbitrage!) and we saw all of them are around 0, likely normally distributed.
# 
# **Outliers** We already handled them in the EDA section, we also made an assumption that 2008 and 2009 were the outlier years, and we proceede with our analysis further. I did check and there were single digits outliers still to be found but these fall into normally abnormal behaviour of every distribution function.

# <a id="5"></a> <br>
# #  **5-News data**
# I think this is the **main point** of this competition. To predict (mainly) using news data with suplementation of market data stock movement. Hence we will focus more on this aspect
# 
# There is one interesting (general) consensus among finance community. **Under the efficient market hypothesis (EMH)**, the price of a asset should react almost instantaneously to relevant news. Consider the example where we (or our model) is surveying the data immediately before the market open. There has been very impactful news in the last 24 hours, and we hope our model will predict a price shift because of this.
# 
# The problem, though, is that everyone else in the market will have also seen this news and adjusted their buys and sells accordingly. This will change the open price, and absent of any drift (which generally there should be none if markets are efficient), the price will not change further. After this point, any further shifts in price over the next 10 days will happen for reasons unrelated to the original news article. In short, by the time the news article comes out, we're already too late.
# 
# Of course company 2sigma must know this but they still believe that there are some smart way to use news data in order to predict stock movement.

# #  Main objection: News articles are just backward looking commentary ('company xxx jumped 15% today due to yyy')
# ### Goal is to refute that, [J.P. Morgan did it](https://www.ravenpack.com/research/reversal-news-neutralization-japan/)

# In[ ]:


news_train_df.head()


# Good news is we already have some information. Like the severity of the news, region etc.. (Mostly categories) But what we really want to analyse and extract is the headlines column

# **TF_IDF** Score. In short the most important words (not too common not to rare) That could make a difference:

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords


# In[ ]:


vectorizer = CountVectorizer(max_features=1000, stop_words={"english"})

X = vectorizer.fit_transform(news_train_df['headline'].values)
tf_transformer = TfidfTransformer(use_idf=False).fit(X)
X_train_tf = tf_transformer.transform(X)
X_train_vals = X_train_tf.mean(axis=1)


del vectorizer
del X
del X_train_tf

#mean tf-idf score for news article.
d = pd.DataFrame(data=X_train_vals)
news_train_df['tf_score'] = d


# In[ ]:


from wordcloud import WordCloud, STOPWORDS


# LEts do a higher analysis of the words that could be interesting, not just guess them. For that we will need to analyse the "headline column" a bit deeper, wordcloud. More specifically a world cloud that stands for positive return and world cloud for negative return. That will help us build additional features. First we need to filtrate it.
# 
# 
# What could be the indicator that tells us that a sentiment was positive. There are 3 columns in news dataframe: sentimentNegative	sentimentNeutral	sentimentPositive. These are are all** probabilites** that sentiment will have a certain infulence on the stock movement. Let us use that to subset the headline column and plot these 2 wordclouds.

# In[ ]:


news_train_df_positive = news_train_df[news_train_df.sentimentPositive> 0.5]


# In[ ]:


news_train_df_negative = news_train_df[(news_train_df.sentimentNegative + news_train_df.sentimentNeutral) > 0.5]


# In[ ]:


plt.figure(figsize=(16,13))
wc = WordCloud(background_color="black", max_words=10000, stopwords=STOPWORDS, max_font_size= 40)
wc.generate(" ".join(news_train_df_positive['headline']))
plt.title("HP Lovecraft (Cthulhu-Squidy)", fontsize=20)
# plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)
plt.imshow(wc.recolor( colormap= 'pink_r' , random_state=17), alpha=0.98)
plt.axis('off')


# Now we know what words are crucial, and only one (rise_fall) column can capture that.

# In[ ]:


gc.collect()


# In[ ]:


news_train_df["rise_fall"]=(news_train_df.headline.str.lower().str.contains('research|roundup|raises|public offering|target|second quarter')).astype(int)
news_train_df.head()


# Using the same analogy, one can make following hypothesis (for example), test it and if it is plausible implement it. Eventually using feature importance of lgbm for example one can also confirm that this new variable did have an impact. Another idea: "wordCount" column. I assume that shorter headlines/news are more groundbreaking and will have more power. Longer ones are like reports that already capture the market behaviour. Additionally we can combine it with urgency columns were 1 is an alert and 3 is an article. So using these 2 columns, we can plot (histogram to see the distribution) and than plot it against sentiment Pos/Neg, alongt with rise_fall column to see if there is any connection. Also we can build a wordcloud of the short headlines to confirm our suspition etc...

# **Aggregations** This approach can be applied almost to any feature engineering, and it is very often useful. Just create columns/predictors using aggregations of some of the canonic columns/values. Now in this case we are going to group by date and than asset, and than only apply aggregations. We have to think about what makes sence, since (for example) aggregations only for date wont make much since since there will be much fluctation between different assets and no prediction power can be found (except ofcourse if we have a meltdown whic we excluded from the dataset).

# Following dictionary will be used for aggregations (after we merged the datasets further down below)

# In[ ]:


news_agg_cols = [f for f in news_train_df.columns if 'novelty' in f or
                'volume' in f or
                'sentiment' in f or
                'bodySize' in f or
                'Count' in f or
                'marketCommentary' in f or
                'tf_score' in f or
                'rise_fall' in f or
                'relevance' in f]
news_agg_dict = {}
for col in news_agg_cols:
    news_agg_dict[col] = ['mean', 'sum', 'max', 'min']
news_agg_dict['urgency'] = ['min', 'count']
news_agg_dict['takeSequence'] = ['max']


# <a id="6"></a> <br>
# #  **6. Joining the datasets and some additional feature engineering**
# 
# 
# In the following lines of code we are first joining the news and market dataon two columns. Since the assetCode in the news data entails actually a dictionary of assetCodes we have to take care of that before we apply merge. When we are done with that (a function) we will be using the merged dataset to create aggregations on columns specified above

# In[ ]:


# update market dataframe to only contain the specific rows with matching indecies.
def check_index(index, indecies):
    if index in indecies:
        return True
    else:
        return False

# note to self: fill int/float columns with 0
def fillnulls(X):
    
    # fill headlines with the string null
    X['headline'] = X['headline'].fillna('null')
    
def generalize_time(X):
    # convert time to string and get rid of Hours, Minutes, and seconds
    X['time'] = X['time'].dt.strftime('%Y-%m-%d %H:%M:%S').str.slice(0,16) #(0,10) for Y-m-d, (0,13) for Y-m-d H

# this function checks for potential nulls after grouping by only grouping the time and assetcode dataframe
# returns valid news indecies for the next if statement.
def partial_groupby(market_df, news_df, df_assetCodes):
    
    # get new dataframe
    temp_news_df_expanded = pd.merge(df_assetCodes, news_df[['time', 'assetCodes']], left_on='level_0', right_index=True, suffixes=(['','_old']))

    # groupby dataframes
    temp_news_df = temp_news_df_expanded.copy()[['time', 'assetCode']]
    temp_market_df = market_df.copy()[['time', 'assetCode']]

    # get indecies on both dataframes
    temp_news_df['news_index'] = temp_news_df.index.values
    temp_market_df['market_index'] = temp_market_df.index.values

    # set multiindex and join the two
    temp_news_df.set_index(['time', 'assetCode'], inplace=True)

    # join the two
    temp_market_df_2 = temp_market_df.join(temp_news_df, on=['time', 'assetCode'])
    del temp_market_df, temp_news_df

    # drop nulls in any columns
    temp_market_df_2 = temp_market_df_2.dropna()

    # get indecies
    market_valid_indecies = temp_market_df_2['market_index'].tolist()
    news_valid_indecies = temp_market_df_2['news_index'].tolist()
    del temp_market_df_2

    # get index column
    market_df = market_df.loc[market_valid_indecies]
    
    return news_valid_indecies

def join_market_news(market_df, news_df, nulls=False):
    
    # convert time to string
    generalize_time(market_df)
    generalize_time(news_df)
    
    # Fix asset codes (str -> list)
    news_df['assetCodes'] = news_df['assetCodes'].str.findall(f"'([\w\./]+)'")

    # Expand assetCodes
    assetCodes_expanded = list(chain(*news_df['assetCodes']))
    assetCodes_index = news_df.index.repeat( news_df['assetCodes'].apply(len) )
    
    assert len(assetCodes_index) == len(assetCodes_expanded)
    df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})
    
    if not nulls:
        news_valid_indecies = partial_groupby(market_df, news_df, df_assetCodes)
    
    # create dataframe based on groupby
    news_col = ['time', 'assetCodes', 'headline'] + sorted(list(news_agg_dict.keys()))
    news_df_expanded = pd.merge(df_assetCodes, news_df[news_col], left_on='level_0', right_index=True, suffixes=(['','_old']))
    
    # check if the columns are in the index
    if not nulls:
        news_df_expanded = news_df_expanded.loc[news_valid_indecies]

    def news_df_feats(x):
        if x.name == 'headline':
            return list(x)
    
    # groupby time and assetcode
    news_df_expanded = news_df_expanded.reset_index()
    news_groupby = news_df_expanded.groupby(['time', 'assetCode'])
    
    # get aggregated df
    news_df_aggregated = news_groupby.agg(news_agg_dict).apply(np.float32).reset_index()
    news_df_aggregated.columns = ['_'.join(col).strip() for col in news_df_aggregated.columns.values]
    
    # get any important string dataframes
    news_df_cat = news_groupby.transform(lambda x: news_df_feats(x))['headline'].to_frame()
    new_news_df = pd.concat([news_df_aggregated, news_df_cat], axis=1)
    
    # cleanup
    del news_df_aggregated
    del news_df_cat
    del news_df
    
    # rename columns
    new_news_df.rename(columns={'time_': 'time', 'assetCode_': 'assetCode'}, inplace=True)
    new_news_df.set_index(['time', 'assetCode'], inplace=True)
    
    # Join with train
    market_df = market_df.join(new_news_df, on=['time', 'assetCode'])

    # cleanup
    fillnulls(market_df)

    return market_df


# In[ ]:


X_train = join_market_news(market_train_df, news_train_df, nulls=False)


# In[ ]:


X_train.head()


# **Feature-engienering on time data** We created a lot of features but none with time. Now we are going to use a function to create this additional features from "time" columns using datetime and slicing.

# In[ ]:


# first get dates
def split_time(df):
    # split date_time into categories
    df['time_day'] = df['time'].str.slice(8,10)
    df['time_month'] = df['time'].str.slice(5,7)
    df['time_year'] = df['time'].str.slice(0,4)
    df['time_hour'] = df['time'].str.slice(11,13)
    df['time_minute'] = df['time'].str.slice(14,16)
    
    # source: https://www.kaggle.com/nicapotato/taxi-rides-time-analysis-and-oof-lgbm
    df['temp_time'] = df['time'].str.replace(" UTC", "")
    df['temp_time'] = pd.to_datetime(df['temp_time'], format='%Y-%m-%d %H')
    
    df['time_day_of_year'] = df.temp_time.dt.dayofyear
    df['time_week_of_year'] = df.temp_time.dt.weekofyear
    df["time_weekday"] = df.temp_time.dt.weekday
    df["time_quarter"] = df.temp_time.dt.quarter
    
    del df['temp_time']
    gc.collect()
    
    # convert to non-object columns
    time_feats = ['time_day', 'time_month', 'time_year','time_hour','time_minute','time_day_of_year','time_week_of_year',"time_weekday","time_quarter"]
    df[time_feats] = df[time_feats].apply(pd.to_numeric)
    df['time'] = pd.to_datetime(df['time'])
    
    del time_feats
    gc.collect()


# In[ ]:


split_time(market_train_df)


# Now lets drop all non-numeric and non-usable features:

# In[ ]:


def remove_cols(X):
    del_cols = [f for f in X.columns if X[f].dtype == 'object']
    for f in del_cols:
        del X[f]


# In[ ]:


gc.collect()


# In[ ]:


remove_cols(X_train)


# Now we have a model-ready dataset:

# In[ ]:


X_train.head()


# <a id="7"></a> <br>
# #  **7. Model and theoretical explanation of xgboost**
# 

# Since the submission deadline is over and it does not make much sence building a model. Much of the survays convey that model building is a small part of the data-science project and that right data wins over the right model. 
# 
# Nevertheless if one were to make a model often times (I am guilty of this) user treats it as a **black-box**. Without really understanding what is happening behind it. 
# 
# Hence :
# 
# 
# a) Not selecting the right model
# 
# b) Not selecting the right parameters
# 
# 

# If one were to explore possibilities of **eXtreme Gradient Boosting** I did compile a short paper concerning tree based methods (foundation of xgboost) and xgboost in particular. Just to get a sence of what is happening behind the scenes.
# 
# 
# 
# 
# [Tree based methods](https://drive.google.com/file/d/1ce1_gCIZ6-j-9Da5ekbxv_8gBE6bnf7u/view?usp=sharing)
# 
# 
# 
# [eXtreme Gradient boosting](https://drive.google.com/file/d/1jIkbJGB869bXrJeIxxZiSDPM49vsPVod/view?usp=sharing)

# Now that we have a model, another important aspect is (Hyper)parameters, and their optimisation. I did it already (on lgbm also) in this [project](https://www.kaggle.com/zikazika/useful-new-features-and-a-optimised-model). There one can found full xbg model along with Bayesian optimisation.

# 

# 
