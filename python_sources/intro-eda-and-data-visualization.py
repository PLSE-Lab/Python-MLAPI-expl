#!/usr/bin/env python
# coding: utf-8

# I made this kernel as a bit of an introduction to the  challenge and the data and an example of how you should go about investigating your data. I have tried to be as thorough as possible. If you have questions leave them in the comments and I will do my best to answer.
# 
# For this challenge we need to predict what direction a stock is going to move, and also weight our guess with a number between -1 and 1. The more confident you are, the more weight you should give it. E.g. if you are certain the stock will be up in ten days you assign a 1, if you're certain it will be down, -1, and for everything else something in between.
# 
# The  stock price can only go up or down on any given day so their probabilities must sum to 1. So P<sub>up</sub>+P<sub>down</sub>=1
# 
# The stock can also stay the same price, but the probability that it is exactly the same price is infinitesimal so we can neglect it. We can make a simple method for producing $\hat{y}$ by subtracting P<sub>down</sub>  from P<sub>up</sub>: 
# 
# $\hat{y}$=P<sub>up</sub>-P<sub>down</sub>=P<sub>up</sub>-(1-P<sub>up</sub>)=2P<sub>up</sub>-1
# 
# We can check if this makes sense. If we are certain the stock will go up P<sub>up</sub>=1 then $\hat{y}$=1, likewise if we are certain it will go down P<sub>down</sub>=1 implies P<sub>up</sub>=0 and $\hat{y}$=-1. Conveniently we are left predicting P<sub>up</sub> as a "simple" binary classification problem(which maps to the space [0,1]), for which there are a number of good tools.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import *
import matplotlib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import time
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()


# The options below change how pandas displays things so that the dataframes we print will not omit rows/columns,  it can be a bit unwieldly but I think it's important to know what the data looks like and understand what exactly we are doing to it. Also a tip: it is much easier to look at printed dataframes in the console window.

# In[ ]:


pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 25)
pd.set_option('display.max_columns', 50)


# Let's take a look at the market data first:

# In[ ]:


market_train.head()


# Even in the first 5 rows we see some NaN values, let's investigate a little bit. The code below will tell us how many NaN's are in each column of the dataframe:

# In[ ]:


market_train.isna().sum()


# So we see that there are ~16,000 entries without market residualized returns from the previous day, and ~93,000 without market residualized returns from the previous ten days. Let's see what fraction of the total dataset they are and display it in a nice bar chart.

# In[ ]:


(market_train.isna().sum()/market_train.shape[0]).plot.bar()


# So we can see it is about 0.5% and 2% of the data missing for the different entries. What should we do about it? We could just drop them all together(2% is not too much of our dataset), or we can fill in some reasonable value. They are returns, so if the asset price didn't change over the 1 or 10 days respectively the return would just be zero. Furthermore these are market residualized returns so the movement of the market is adjusted for and hence these numbers tend to be quite small. Let's see what typical values are for them and see if zero would be a reasonable thing to fill in:

# In[ ]:


market_train.describe()


# We can see for returnsClosePrevMktres1 the mean is ~0.002 and for returnsOpenPrevMktres1 mean value is ~0.01. Looking at the quartiles(25% and 75%) the distribution appears to be centered around zero. The means for returnsOpenPrevRaw10 and returnsClosePrevMktres10 are similarly small and the distributions approximately centered around zero. So it looks like zero is a completely legitimate value to fill in for our missing entries, let's do it and check that the values have been filled in:

# In[ ]:


for i in ['returnsClosePrevMktres1','returnsOpenPrevMktres1','returnsClosePrevMktres10','returnsOpenPrevMktres10']:
    market_train[i].fillna(0, inplace=True)
market_train.head()


# Now let's take a look at the distribution of our data and see if we see any major outliers.

# In[ ]:


f = plt.figure(figsize=(24,12))
for i,k in enumerate(market_train.columns):
    if 2 < i < 15:
        f.add_subplot(3,4,i-2)
        sns.boxplot(x=market_train[k])


# For most of these boxplots the boxes are so squished you can't even see them. All the plots have pretty serious outliers. Let's just chop off the highest and lowest 0.05% for each variable and take another look:

# In[ ]:


f = plt.figure(figsize=(24,12))
for i,k in enumerate(market_train.columns):
    if 2 < i < 15:
        f.add_subplot(3,4,i-2)
        sns.boxplot(x=np.clip(market_train[k], np.percentile(market_train[k],0.05), np.percentile(market_train[k],99.95)))


# These plots look much better. Note that all the return variables are centered around zero which is what we expect, since for any given 1-10 day period the returns for a stock are almost random. The first three variables, volume, close, and open are not random variables at all, and so their plot has a different structure. A boxplot is likely not the best tool for visualizing them, but for a first look EDA such as this it is fine to just get an idea how the values are distributed. For instance we can tell most stocks open and close in an approximate range of \$20-\$50.  Ultimately in the competition we are concerned with predicting the stock price ten days in the future which is the variable returnsOpenNextMktres10. Let's take a look and see if there are any obvious correlations:

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10)) 
correlation = market_train.corr()
sns.heatmap(correlation, ax=ax)


# Nothing really stands out, in fact it looks like there is almost no correlation. One thing we can do is change the colormap to choose our max and min so that we get maximum color contrast on our target variable:

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10)) 
correlation = market_train.corr()
sns.heatmap(correlation, vmin=correlation.returnsOpenNextMktres10.min(), vmax = correlation.returnsOpenNextMktres10.nlargest(2)[1], ax=ax)


# We can see the highest correlation is returnsOpenPrevRaw10 and it's only 0.016 so not a great predictor.
# 
# Maybe the news data will have a better predictor, let's move on and see what it looks like:

# In[ ]:


news_train.head()


# Let's check for empty cells and get a sense of what the news data variables look like:

# In[ ]:


news_train.isna().sum()


# In[ ]:


news_train.describe()


# So there are no empty cells which is good, let's take a look at the distribution of some of the variables. Below we'll plot 14 of the 35 columns as histograms and plot them on a log scale so that the scale of the largest bins doesn't make the shorter bins impossible to see. This will take a second to run:

# In[ ]:


cols_to_plot = ['urgency','bodySize','sentenceCount','firstMentionSentence','relevance',
                'sentimentClass','sentimentNegative','sentimentNeutral','sentimentPositive','sentimentWordCount','noveltyCount24H',
                'noveltyCount7D','volumeCounts24H','volumeCounts7D']
q = plt.figure(figsize=(24,30))
for i,k in enumerate(cols_to_plot):
    q.add_subplot(4,4,i+1)
    #sns.countplot(x=news_train[k])
    news_train[k].plot.hist(log=True)
    plt.title(cols_to_plot[i])


# We can immediately pull some useful information out of the data here. First we note that articles are generally urgent or not,  there are significantly less (10^-4) articles with a medium urgency, so this is almost a binary variable. 
# 
# bodySize, sentenceCount, and firstMentionSentence all have exponential distributions with significantly more shorter articles/earlier mentions. We should note though that the bins here are pretty large so if we break it down even finer we may find something different about the distribution.
# 
# relevance is clearly peaked at a value near 1 and then decays from 0 up suggesting articles are either directly relevant, or not really. This could be a good variable to cut on when deciding what articles to use for predicting asset results.
# 
# sentimentClass shows a clear trend towards the positive and does a nice job of categorizing article sentiment in broad strokes. The categories sentimentNegative/Neutral/Positive break it down even further. The sentimentNegative plot appears fairly binary which is nice for us because it simplifies things. The sentimentNeutral plot is fairly flat apart from the first bin which is about what you'd expect, the first bin is very large which implies that the article is either strongly negative or positive. The sentimentPositive plot is a bit strange. We see a large spike near zero which is the negative articles, and a spike again near one for the unambiguously positive articles, but there's a strange spike around 0.5-0.6 that I am not sure how to interpret. Maybe these are tangentially related topics that could potentially be good for the business, cutting on the relevance criteria and replotting could maybe shed some light on that.
# 
# sentimentWordCount, defined as "the number of lexical tokens in the sections of the item text that are deemed relevant to the asset", exhibits another exponential distribution where most articles have a significant amount of relevant material with less and less articles having less and less relevant "lexical tokens."
# 
# The noveltyCount variable is defined a bit strangely it is: "The 12 hour novelty of the content within a news item on a particular asset". I'm interpreting this to mean how unique the article is, on a scale from 0-500. for the 24H plot it looks almost binary, that is articles are either very unique with a score ~500 or not very unique at all ~0. Comparing to the 7D variable we see a large amount of articles pop up on the lower end, so it looks like there are maybe new details about the news event coming out or new interpretations that are novel, but not completely unique coming out over the 7days after the news breaks.
# 
# The volumeCounts variables don't tell us too much interesting here. We see the 7D variable basically just a bit more beefed up so news breaks and then is covered a bit more in the following 6 days. This is what we would expect though so it is a good sanity check.
# 
# So now that we have a good handle on all our variables and have a decent understanding of what they do we may naturally want to merge our datasets. Unfortunately this is a bit tricky, since the datasets are organized a little differently. Our market data has a clear date, time, and assetcode associated with it, unfortunately our news data has multiple asset codes. In the first kernel I forked and started working with here: https://www.kaggle.com/jannesklaas/lb-0-63-xgboost-baseline the data is joined using the first entry from a list of the assetCodes. Unfortunately the assetCodes are stored as a category and have an ordering such that if you pull the first one, it does not necessarily match the assetcode used for the marketdata. Let's clean up the dates and times and take a look at matching the assetcodes by making a new column with the first entry from the assetCode list:

# In[ ]:


market_train.time = market_train.time.dt.date
news_train.time = news_train.time.dt.hour
news_train.sourceTimestamp= news_train.sourceTimestamp.dt.hour
news_train.firstCreated = news_train.firstCreated.dt.date


# In[ ]:


news_train['firstAssetCode'] = news_train['assetCodes'].map(lambda x: list(eval(x))[0])
news_train.head()


# From what I can tell the above mapping function seems to be returning a rotating assetCode whenever the session is refreshed. Looking at the fourth entry of the news data we can see that first assetCode is not GOOG.O, so it won't find it's matching counterparts in the market data. There are three different tickers for Google in the news data but there is only one that the market data uses so we need to be a little careful about choosing the right one. To illustrate:

# In[ ]:


goog = market_train[market_train.assetCode=='GOOG.O']
print(goog.shape)
goog = market_train[market_train.assetCode=='GOOGa.DE']
print(goog.shape)
goog = market_train[market_train.assetCode=='GOOG.OQ']
print(goog.shape)


# Unfortunately the other method I've prepared to match the data will not fit in this kernel so I will have to save it for my next kernel. In short my solution for joining the dataframes was to duplicate the news rows for each assetCode present, and then let them drop out when we merge with the market data. There's also an issue about what to do with multiple news articles on the same day. One way to handle it is to simply aggregate it, perhaps by taking averages of the news variables, although we will lose some data if we do that. The downside to not aggregating is that if we have multiple entries for an assetCode on a single day, certain assetCodes or dates might make up a disproportionate amount of our training data, and we will have multiple predictions which we will have to turn in to one(which could increase or decrease our accuracy.)  I hope this was helpful I tried to make it as accessible as I could.
