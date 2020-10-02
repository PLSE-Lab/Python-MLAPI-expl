#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from itertools import chain
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
print('Done!')


# In[ ]:


(market_train_df, news_train_df) = env.get_training_data()
print(f'market train df shape: {market_train_df.shape}')
print(f'news train df shape: {news_train_df.shape}')


# ## Let's take a look at market data

# In[ ]:


market_train_df.head()


# In[ ]:


market_train_df.info()


# ### missing values:
# - There are missing values in 4 returns columns spreadding out over lots of trading days. The reason is pointed out in the data description: "The set of included instruments changes daily and is determined based on the amount traded and the availability of information. This means that there may be instruments that enter and leave this subset of data. There may therefore be gaps in the data provided, and this does not necessarily imply that that data does not exist."

# In[ ]:


missing_count = market_train_df.isna().sum()
missing_count


# In[ ]:


plt.figure(figsize=(12,8))
plt.bar(missing_count.index, missing_count.values)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


# show number of missing values over time

missing_col = ['returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 
               'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
df_na = market_train_df[market_train_df.isnull().any(axis=1)]
missing_day = df_na.loc[:, missing_col].isnull().groupby(df_na.time).sum()

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, figsize=(12,8))
ax1.plot(missing_day.index, missing_day.returnsOpenPrevMktres1)
ax1.set_ylim(0,100)
ax1.set_title('returnsClosePrevMktres1')
ax2.plot(missing_day.index, missing_day.returnsOpenPrevMktres1)
ax2.set_ylim(0,100)
ax2.set_title('returnsOpenPrevMktres1')
ax3.plot(missing_day.index, missing_day.returnsClosePrevMktres10)
ax3.set_ylim(0,200)
ax3.set_title('returnsClosePrevMktres10')
ax4.plot(missing_day.index, missing_day.returnsOpenPrevMktres10)
ax4.set_ylim(0,200)
ax4.set_title('returnsOpenPrevMktres10')
plt.show()


# There are little more unique asset codes than names. It means there are cases that multiple asset codes correspond to the same asset name(there is an "unknown" asset name). Also note that the predictions are based on asset codes. 

# In[ ]:


print(f'number of unique asset Codes: {market_train_df.assetCode.unique().shape[0]}')
print(f'number of unique asset Names: {market_train_df.assetName.unique().shape[0]}')


# ### continuous variables

# In[ ]:


market_train_df.describe()


# In[ ]:


# histograme for log10(volume)

plt.hist(market_train_df.volume.apply(lambda x: np.log10(x) if x!=0 else 0), bins=50)
plt.title('volume')
plt.show()


# In[ ]:


# there are some very high open(10k) and close value(1.5k) 
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,4))
ax1.hist(market_train_df.open, bins=50, range=(0,500))
ax1.set_title('open')
ax2.hist(market_train_df.close, bins=50, range=(0,500))
ax2.set_title('close')
plt.show()


# - Most of the returns values are close to 0, although in extreme cases the number can be quite large. Also, the distributions of 10 day return are wider than 1day return, since it's over longer period.
# - Given the range of next 10 day return and the range of y (-1,1) the submission wants, consider using tanh(next 10 day return) as target for prediction. 

# In[ ]:


f, axes = plt.subplots(3, 3, sharey=True, figsize=(15,15))
for i in range(3):
    for j in range(3):
        axes[i,j].hist(market_train_df.iloc[:, 6+i*3+j], range=(-0.5,0.5), bins=50)
        axes[i,j].set_title(market_train_df.columns[6+i*3+j])
plt.show()


# ## Next,  news data

# In[ ]:


news_train_df.head()


# In[ ]:


news_train_df.info()


# In[ ]:


# no missing value here
news_train_df.isna().sum().sum()


# In[ ]:


news_train_df.describe()


# In[ ]:


# plot histogram of all the numeric columns
news_train_df.select_dtypes(include=[np.number]).hist(figsize=(15,15))
plt.show()


# It appears that there are much more asset codes and names in the news dataframe than the market dataframe. So there are assets in the news that are not included in market set. 

# In[ ]:


n_codes = len(set(chain(*news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'"))))
print(f'number of unique asset Codes in news set: {n_codes}')
print(f'number of unique asset Names in news set: {news_train_df.assetName.unique().shape[0]}')
print('*'*50)
print(f'number of unique asset Codes in market set: {market_train_df.assetCode.unique().shape[0]}')
print(f'number of unique asset Names in market set: {market_train_df.assetName.unique().shape[0]}')


# ## I haven't seen anyone doing EDA on test set yet, but there is something that may have been overlooked.
# 
# Each time we call env.get_prediction_days(),  the Two Sigma method will spit out a set of market data in which each row is the data for a asset that we will need to make prediction. It also spit out the news **since the last trading day**. So as you can see, for 2017-1-3, the news set acutally contains news from 2016-12-30 to 2017-1-3(weekends and holidays). Also note that Two Sigma split days at 22:00, time after 22:00 and before 0:00 is considered the next day. So it looks like attention need to be paid when joining market and news set. Simply joining by date and assetCode may throw away useful information, or put 'future'(arguably) information in training data. 

# In[ ]:


days = env.get_prediction_days()
(market_obs_df, news_obs_df, predictions_template_df) = next(days)
print(f'market_obs_df shape: {market_obs_df.shape}')
print(f'news_obs_df shape: {news_obs_df.shape}')
print(f'predictions_template_df shape: {predictions_template_df.shape}')


# In[ ]:


market_obs_df.head()


# In[ ]:


news_obs_df.head()


# In[ ]:


print(f'date in market set: {market_obs_df.time.dt.date.unique().tolist()}')
print(f'date in news set: {news_obs_df.time.dt.date.unique().tolist()}')


# Thanks!
