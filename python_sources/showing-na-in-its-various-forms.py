#!/usr/bin/env python
# coding: utf-8

# This dataset first appears to be pretty complete without many missing values. But I noticed from one of the EDA kernels that there are quite a few missing values once you unpack the JSON columns into separate features. Looking more closely, there are also several values present that are the verbal equivalent of NA. Here I'll look at the NAs/NaNs in their various forms and look for patterns in the missing data. 

# In[ ]:


import numpy as np
import pandas as pd
import json
import missingno as msno
import hvplot.pandas


# ### Data Prep
# First some basic data prep. For this data we have the added fun of four JSON columns with one of them having some nested JSON within (or a dictionary, I'm not sure). I'll expand the keys into their own columns as others have done.
# 
# Several of the values in the data sound like NA and we can replace them. Also, Misha Losovyi found several interesting things which you can read in [this thread](https://www.quantamagazine.org/to-build-truly-intelligent-machines-teach-them-cause-and-effect-20180515/). There's a column for instance that contains only False and NaN. It seems that NaN really means True for this feature. I'll replace these values in an attempt to get the dataset to reflect typical expectations for missing values.

# In[ ]:


json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']

nan_list = ["not available in demo dataset",
            "unknown.unknown",
            "(not provided)",
            "(not set)", 
            "Not Socially Engaged"] # single value for the feature
nan_dict = {nl:np.nan for nl in nan_list}

def df_prep(file):
    df = pd.read_csv(file, dtype={'fullVisitorId': str, 'date': str}, 
            parse_dates=['date'], nrows=None)
    df.drop('sessionId', axis=1, inplace=True)
    for jc in json_cols:  # parse json
        flat_df = pd.DataFrame(df.pop(jc).apply(pd.io.json.loads).values.tolist())
        flat_df.columns = ['{}_{}'.format(jc, c) for c in flat_df.columns]
        df = df.join(flat_df)
    ad_df = df.pop('trafficSource_adwordsClickInfo').apply(pd.Series) # handle dict column
    ad_df.columns = ['tS_adwords_{}'.format(c) for c in ad_df.columns]
    df = df.join(ad_df)
    df.replace(nan_dict, inplace=True) # convert disguised NaNs
    df.tS_adwords_isVideoAd.fillna(value=True, inplace=True) # strange column
    df.set_index(['fullVisitorId', 'visitId'], inplace=True)
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = df_prep('../input/train.csv')\ntest = df_prep('../input/test.csv')\ndisplay(train.head(10))")


# ### Patterns
# 
# I'll use the [missingno](https://github.com/ResidentMario/missingno) package by Aleksey Bilogur to show the patterns within and across the two datasets. 

# In[ ]:


display(msno.matrix(train.sample(1000), inline=True, sparkline=True, figsize=(24,8), 
                    sort='ascending', labels=True, fontsize=12))
display(msno.matrix(test.sample(1000), inline=True, sparkline=True, figsize=(24,8), 
                    sort='ascending', labels=True, fontsize=12, color=(0.25, 0.45, 0.6)))


# Maxwell found an apparent relationship between Transaction Revenue and Bounces as discussed in [this thread](https://www.kaggle.com/c/google-analytics-customer-revenue-prediction/discussion/65989). Zooming in on those two columns we can see that Bounces (which is null about half the time) is always null when Transaction Revenue is > 0, but not vice-versa.

# In[ ]:


display(msno.matrix(train[['totals_bounces', 'totals_transactionRevenue']].sample(1000, random_state=32), inline=True, sparkline=False, figsize=(8,6), 
                    sort='ascending', labels=True, fontsize=12))


# Here's the dendrogram view of the missing values showing which columns are most correlated.

# In[ ]:


msno.dendrogram(train, inline=True, fontsize=12, figsize=(20,20))


# ### Transaction Revenue
# 
# Our target column, transactionRevenue, looks especially sparse. This has been confirmed in the Discussion Forum. Also, it might be the case that revenue is listed in microdollars so I'll do some scaling here. Let's look closer...

# In[ ]:


train['totals_transactionRevenue'] = train.totals_transactionRevenue.astype(np.float32)/1_000_000
train.totals_transactionRevenue.isnull().sum()/train.shape[0]


# In[ ]:


train.fillna(0, inplace=True)
train.hvplot.hist('totals_transactionRevenue', bins=48)
# train.totals_transactionRevenue.min()


# Well, OK...there's quite a bit of 0s here.  Zooming in on the 1%  greater than 0 shows that there are some buyers here. You can further zoom in segments of the histogram.

# In[ ]:


train.replace(0, np.NaN, inplace=True)
train.hvplot.hist('totals_transactionRevenue', bins=48)


# These aren't grouped by UserID but it gives a good idea of the visits and potential challenges with this data. Hope it helps!
