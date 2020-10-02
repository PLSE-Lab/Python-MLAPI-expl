#!/usr/bin/env python
# coding: utf-8

# <h2>1. Introduction</h2>
# 
# There are 34 usefull features in this dataset, of these 12 are about traffic source. Most missing values are also from traffic source, but there is a pattern on the data as we shall see. In this notebook I will try to explain how traffic source features are related and plot their values. If you find any wrong information please leave a comment.
# 
# For a complete exploration on this dataset: [Complete exploratory analysis](https://www.kaggle.com/jsaguiar/complete-exploratory-analysis-all-columns)

# In[ ]:


import os
import numpy as np
import pandas as pd
# Seaborn and matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
# Plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
init_notebook_mode(connected=True)

path = "../input/python-quick-start-clean-and-pickle-data"
source_cols = ['trafficSource_adContent', 'trafficSource_adwordsClickInfo.adNetworkType',
               'trafficSource_adwordsClickInfo.gclId', 'trafficSource_adwordsClickInfo.isVideoAd', 
               'trafficSource_adwordsClickInfo.page', 'trafficSource_adwordsClickInfo.slot', 
               'trafficSource_campaign', 'trafficSource_isTrueDirect',
               'trafficSource_keyword', 'trafficSource_medium', 'trafficSource_referralPath', 'trafficSource_source']


# In[ ]:


train = pd.read_pickle(os.path.join(path, 'train_clean.pkl'))
test = pd.read_pickle(os.path.join(path, 'test_clean.pkl'))
train[source_cols].head()    # Head of traffic source columns


# <h2>2. Features</h2>

# <h3>Ads</h3>
# 
# There are six features related to Ads. The first one is adContent and the other five are informations about AdWords:
# * adContent
# 
# <b>AdWords</b>
# * adNetworkType
# * gclId
# > Gclid is a globally unique tracking parameter (Google Click Identifier) used by Google to pass information back and forth between Google AdWords and Google Analytics. If you enable URL auto tagging in Google AdWords, Google will append a unique ?gclid parameter on your destination URLs at run-time. Because it is a redirect, you won't see any gclid parameters on your ad words text ad destination url's, but it will show up in your Web server log files. Auto tagging was introduced in 2004 and is on by default in any Google AdWords accounts. 
# The gclid parameter enables data sharing between Google Analytics and Google AdWords. For example, the "Traffic Sources" feature in Google Analytics to segment your campaigns based on ad text and other PPC-related campaign metrics and dimensions, relies on gclid.
# 
# * isVideoAd: If the Ad is a video (boolean)
# 
# * page	
# * slot

# <h3>Other features</h3>
# 
# There are six more features:
# * campaign
# * isTrueDirect
# * keyword
# * medium
# * referralPath
# * source

# <h2>3. medium</h2>
# 
# The most important feature to understand traffic source is medium, which tells how did the user arrived at the store.

# In[ ]:


def plotbar(df, col, title, top=None):
    frame = pd.DataFrame()
    frame['totals_transactionRevenue'] = df['totals_transactionRevenue'].copy()
    frame[col] = df[col].fillna('missing')
    # Percentage of revenue
    tmp_rev = frame.groupby(col)['totals_transactionRevenue'].sum().to_frame().reset_index()
    tmp_rev = tmp_rev.sort_values('totals_transactionRevenue', ascending=False)
    tmp_rev = tmp_rev.rename({'totals_transactionRevenue': 'Revenue percentage'},axis=1)
    tmp_rev['Revenue percentage'] = 100*tmp_rev['Revenue percentage']/df['totals_transactionRevenue'].sum()
    # Percentage of visits
    tmp = frame[col].value_counts().to_frame().reset_index()
    tmp.sort_values(col, ascending=False)
    tmp = tmp.rename({'index': col, col: 'Percentage of Visits'},axis=1)
    tmp['Percentage of Visits'] = 100*tmp['Percentage of Visits']/len(df)
    tmp = pd.merge(tmp, tmp_rev, on=col, how='left')
    if top:
        tmp = tmp.head(top)
    # Barplot
    trace1 = go.Bar(x=tmp[col], y=tmp['Percentage of Visits'],
                    name='Visits', marker=dict(color='rgb(55, 83, 109)'))
    trace2 = go.Bar(x=tmp[col], y=tmp['Revenue percentage'],
                    name='Revenue', marker=dict(color='rgb(26, 118, 255)'))

    layout = go.Layout(
        barmode='group',
        title=title,
    )
    
    layout = go.Layout(
        title=title,
        xaxis=dict(tickfont=dict(size=14, color='rgb(107, 107, 107)')),
        yaxis=dict(
            title='Percentage',
            titlefont=dict(size=16, color='rgb(107, 107, 107)'),
            tickfont=dict(size=14, color='rgb(107, 107, 107)')
        ),
        legend=dict(x=0.95, y=1.0, bgcolor='rgba(255, 255, 255, 0)',
                    bordercolor='rgba(255, 255, 255, 0)'),
        barmode='group',
        bargap=0.15,
        bargroupgap=0.1
    )
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    iplot(fig)
    
plotbar(train, 'trafficSource_medium', 'Train set - visits and revenue by medium')


# There are seven labels (and no missing values) for this feature. We will go through all values and relate to the other 11 columns:

# <h3> Organic</h3>
# 
# Organic refers to searchs performed in search mechanisms like Google and Baidu. Almost all searchs are from google. The plot below shows the percentage of visits and revenue w.r.t. organic searches for each source. 
# 
# Note: the percentage is relative to the total revenue/visits for organic search, not for the total.

# In[ ]:


plotbar(train[train['trafficSource_medium'] == 'organic'], 'trafficSource_source', 'Visits and revenue by source for Organic search')


# This category:
# * Has no <b>referralPath</b> (missing)
# * Has no <b>Ad variables</b> (missing)
# * All <b>Campaigns</b> are '(not set)'
# * channelGrouping is 'Organic Search'
# 
# As you can see here:

# In[ ]:


def print_value_counts(category):
    cols = ['trafficSource_adContent', 'trafficSource_adwordsClickInfo.adNetworkType',
           'trafficSource_adwordsClickInfo.gclId', 'trafficSource_adwordsClickInfo.isVideoAd',
           'trafficSource_adwordsClickInfo.page', 'trafficSource_adwordsClickInfo.slot',
           'trafficSource_referralPath', 'trafficSource_campaign']
    for c in cols:
        nunique = train[train['trafficSource_medium'] == category][c].nunique()
        if nunique < 5:
            print(train[train['trafficSource_medium'] == category][c].value_counts(dropna=False))
        else:
            print(train[train['trafficSource_medium'] == category][c].describe())
            
print_value_counts('organic')


# <h3>Referral</h3>
# 
# Users that clicked in a link or were redirected from another webpage. The feature trafficSource_referralPath is exclusive for this category. It's interesting that youtube is responsible for more than 60% of the visits coming from referral, but 93% of the revenue is from mail.googleplex.com.
# 
# The plot below shows the top 8 sources for referral. As before, percentages are based on the totals for this category.

# In[ ]:


plotbar(train[train['trafficSource_medium'] == 'referral'], 'trafficSource_source', 'Visits and revenue by source for Referral', top=8)


# This category:
# * Has all <b>referral paths</b> that are not missing
# * Has no <b>ad variables</b> (all missing)
# * All <b>campaigns</b> are '(not set)'
# * channelGrouping is 'Referral' or 'Social'

# In[ ]:


print_value_counts('referral')


# 

# The feature channelGrouping, which is not included in traffic source, can have two values for referral medium. 

# In[ ]:


plotbar(train[train['trafficSource_medium'] == 'referral'], 'channelGrouping', 'Visits and revenue by channel for REFERRAL MEDIUM')


# <h3>cpc</h3>
# 
# When channel is 'Paid Search', the medium is cpc and it's basically google paid search results:

# In[ ]:


plotbar(train[train['trafficSource_medium'] == 'cpc'], 'trafficSource_source', 'Visits and revenue by source for CPC', top=None)


# Details:
# * <b>Ad Network</b> is mostly 'Google Search'
# * Ad features are usually not missing
# * referral path is always missing
# * Campaign has very few missing values

# In[ ]:


print_value_counts('cpc')


# There are 8 possible campaigns for paid searches:

# In[ ]:


plotbar(train[train['trafficSource_medium'] == 'cpc'], 'trafficSource_campaign', 'Visits and revenue by campaign for CPC')


# <h3>cpm</h3>
# 
# Possible sources for CPM:

# In[ ]:


plotbar(train[train['trafficSource_medium'] == 'cpm'], 'trafficSource_source', 'Visits and revenue by source for CPM')


# I don't know what this category means, but we can conclude that in the train set:
# * Most sources are dfa
# * Channel grouping is always 'Display'
# * Ad variables and referral path are all missing
# * Campaign is always (not set)

# In[ ]:


print_value_counts('cpm')


# <h3>Affiliate</h3>
# 
# * All traffic sources are 'Partners'
# * Channel grouping is always 'Affiliates'
# * All Campaigns are 'Data Share Promo'
# * Ad variables are missing

# In[ ]:


#print(train[train['trafficSource_medium'] == 'affiliate']['trafficSource_campaign'].value_counts())
print_value_counts('affiliate')


# <h3>(none)</h3>
# 
# Medium (none) are direct acesses to the store
# * traffic source is always '(direct)'
# * channel grouping is always 'Direct'
# * Ad features are missing
# * Campaign is missing
# * referral path is '(not set)'

# In[ ]:


print_value_counts('(none)')


# <h3>4. Work in progress...</h3>
