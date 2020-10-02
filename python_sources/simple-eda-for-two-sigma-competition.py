#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import datetime
import gc
import time
import warnings
from itertools import chain

import lightgbm as lgb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest, VotingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
warnings.filterwarnings("ignore")
import os


# **EDA**
# 
# This kernel explores the data in Two Sigma Competition. There is no model provided for prediction. 

# First, we start with importing the data, as explained in the competition. There are two datasets, 
# 1. Market Data, stock prices and relative information per day.
# 1. News Data, Reuters news data that we need to combine with stock prices for predictive analysis.

# In[ ]:


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()


# In[ ]:


(marketdata, news) = env.get_training_data()


# Let's investigate the data first. What are the variables, how does the data look at first glance, how many observations are there, etc. An essential thing we have to figure out before visually observing the data is N/A values. 

# In[ ]:


marketdata.head(1)


# Let's fix the time format as date to perform operations easier.

# In[ ]:


marketdata["time"] = marketdata["time"].dt.date
marketdata.rename(columns={"time": "date"}, inplace=True)


# In[ ]:


marketdata.shape


# In[ ]:


len(marketdata[marketdata.assetCode=='A.N']) #Number of Days


# In[ ]:


len(news.time) #number of news


# In[ ]:


marketdata.describe()


# In[ ]:


marketdata.isna().sum()


# As we can see, there are NA's in the market raw return data,which can be replaced with adjusted return data. By applying this replacement, we will get rid of N/A's, which will give freedom to us in EDA. (Exploratory Data Analysis)

# In[ ]:


marketdata["returnsClosePrevMktres1"].fillna(marketdata["returnsClosePrevRaw1"], inplace=True)
marketdata["returnsOpenPrevMktres1"].fillna(marketdata["returnsOpenPrevRaw1"], inplace=True)
marketdata["returnsClosePrevMktres10"].fillna(marketdata["returnsClosePrevRaw10"], inplace=True)
marketdata["returnsOpenPrevMktres10"].fillna(marketdata["returnsOpenPrevRaw10"], inplace=True)
print(marketdata.isna().sum())


# Now, we can see that there are no null values in the marketdata. As we looked at the describe function above, there is one thing that was obvious about the dataset. The max returns for the data, was way more higher than expected. For instance, the max value for the variable "returnsOpenPrevRaw1", which is the 1 day raw return open-open, 9209. Clearly, there is some discrepancy there, and we have to understand the size of how many outliers are there, and remove them if it is necessary.

# In[ ]:


returns = marketdata["close"].values / marketdata["open"].values
outliers = ((returns > 1.5).astype(int) + (returns < 0.5).astype(int)).astype(bool)
marketdata = marketdata.loc[~outliers, :]


# In[ ]:


marketdata.shape


# In[ ]:


marketdata.describe()


# In[ ]:


marketdata.sort_values('returnsOpenPrevRaw1',ascending=False)[:5]


# In[ ]:


#We can check for the stocks above and the days above
marketdata[(marketdata.assetCode=='EXH.N') & (marketdata.date==pd.to_datetime("2007-8-23").date())]


# If we dig into data deeper, where the variable returnsOpenPrevRaw1 has extremely large values, yesterday's data is not available. So, we need to remove those data which is extremely large, which has excessive returns for 1 and 10 day returns. 

# In[ ]:


return_columns = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
for i in return_columns:
    returns = marketdata[i].values
    outliers = ((returns > 1).astype(int) + (returns < -0.7).astype(int)).astype(bool)
    marketdata = marketdata.loc[~outliers, :]
marketdata.shape


# In[ ]:


del returns
del outliers


# In[ ]:


marketdata.describe()


# Also, let's drop the rows which has unknown Asset Name, which is not related with the news data.

# In[ ]:


marketdata = marketdata.loc[marketdata["assetName"] != "Unknown", :]


# Let's plot the data and see if it is clean, are there outliers or any other things to fix. Also, plotting for some individual assets will make us realize the market trends too. (PLOT THESE)

# This plot is inspired from Andrew Lukyanenko's "EDA, feature engineering and everything" notebook. In this plot, we can see the effect of crisis happened prior to 2010.
# 

# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(marketdata.groupby('date').returnsOpenPrevRaw10.mean().index,marketdata.groupby('date').returnsOpenPrevRaw10.mean().values,color='green')
plt.title("Mean 10 Day Returns")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(marketdata.groupby('date').returnsOpenPrevRaw1.mean().index,marketdata.groupby('date').returnsOpenPrevRaw1.mean().values,color='brown')
plt.title("Mean 1 Day Returns")
plt.show()


# In[ ]:


for asset in np.random.choice(marketdata['assetName'].unique(), 5):
    asset_df = marketdata[(marketdata['assetName'] == asset)]
    plt.figure(figsize=(10,5))
    plt.plot(asset_df.date,asset_df.returnsOpenPrevRaw1,color="blue")
    plt.title(asset)
    plt.show()


# In[ ]:


del asset_df
gc.collect()


# Let's move on to News Data.

# In[ ]:


news.isna().sum() # no NA


# In[ ]:


news.sample(2)


# To clean the news data, we have to convert time to date again. Also, each sentence and paragraph has different length. The position of an asset in the given news can not be analyzed by the given method. We have to normalize it to understand the exact position of the mention of asset in the news.

# In[ ]:


news["sourceTimestamp"] = news["sourceTimestamp"].dt.date #Convert time to date
news.rename(columns={"sourceTimestamp": "date"}, inplace=True) #Rename accurately
#Normalize the location of the mentioning in word and sentence counts
news["realfirstMentionPos"] = news["firstMentionSentence"].values / news["sentenceCount"].values
news["realSentimentWordCount"] = news["sentimentWordCount"].values / news["wordCount"].values
#Normalization Continues
news["realSentenceCount"] = news.groupby(["date"])["sentenceCount"].transform(lambda x: (x - x.mean()) / x.std())
news["realWordCount"] = news.groupby(["date"])["wordCount"].transform(lambda x: (x - x.mean()) / x.std())
news["realBodySize"] = news.groupby(["date"])["bodySize"].transform(lambda x: (x - x.mean()) / x.std())


# By this way, we come to an end to exploratory data analysis. We cleaned the data, added necessary variables for analysis and looked at the insights from data. A more detailed analysis can be made easily, but this kernel aims to provide a simple EDA for everyone to understand!

# 
