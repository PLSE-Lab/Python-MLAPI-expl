#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/1-step-by-step-format-data-to-columnar-format/"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
idir = "../input/1-step-by-step-format-data-to-columnar-format/"
train_df = pd.read_csv(idir + "train_flat.csv", 
    dtype={'fullVisitorId': str, 'date': str, 'visitId':str, 'visitNumber':str, 'visitStartTime':str, 'sessionId': str  },
    nrows=1000000, 
    low_memory=False)

test_df = pd.read_csv(idir + "test_flat.csv", 
    dtype={'fullVisitorId': str, 'date': str, 'visitId':str, 'visitNumber':str, 'visitStartTime':str, 'sessionId': str  },
    nrows=1000000, 
    low_memory=False)

train_df.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'cdf = pd.concat([train_df.count(), train_df.nunique(), train_df.isna().sum(), train_df.dtypes], axis = 1)\ncdf = cdf.reset_index()\ncdf.columns = ["Column_Name", "Total_Records", "Unique_Values", "Null Values", "data_types"]')


# In[ ]:


cdf[cdf.Total_Records > 0].reset_index()


# ## Drop columns in train not in test 
# Except for total revenue

# In[ ]:


print("Variables not in test but in train : ", set(train_df.columns).difference(set(test_df.columns)))


# In[ ]:


train_df.drop(columns = ['trafficSource_campaignCode'], inplace = True)


# ### Drop columns that has 0 values

# In[ ]:


get_ipython().run_cell_magic('time', '', 'for df in [train_df, test_df]: \n    df.drop(columns = cdf[cdf.Total_Records == 0]["Column_Name"], inplace=True)')


# In[ ]:


train_df.select_dtypes('object').describe()


# In[ ]:


train_df.select_dtypes(exclude = 'object').describe()


# ## Fill in 0's for null numeric values

# In[ ]:


for df in [train_df, test_df]: 
    df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].fillna(0)
    df['totals_newVisits'] = train_df['totals_newVisits'].fillna(0)
    df['totals_bounces'] = train_df['totals_newVisits'].fillna(0)
    df['trafficSource_adwordsClickInfo.page'] = train_df['trafficSource_adwordsClickInfo.page'].fillna(0)

train_df.select_dtypes(exclude = 'object').describe()


# ## Handle the Text Boolean columns

# In[ ]:


for df in ([train_df, test_df]): 
    df['trafficSource_adwordsClickInfo.isVideoAd'] = df['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True)
    df['trafficSource_isTrueDirect'] = df['trafficSource_isTrueDirect'].fillna(False)
    df['trafficSource_isVideoAd'] = df['trafficSource_adwordsClickInfo.isVideoAd'].map({True: 1, False: 0})
    df['trafficSource_isDirect'] = df['trafficSource_isTrueDirect'].map({True: 1, False: 0})
    df.drop(columns = ['trafficSource_adwordsClickInfo.isVideoAd', 'trafficSource_isTrueDirect'])


# ## Handle the null character columns
# 
# ### TODO: Hierarchy to be  be check  if data low level data is avialable
# -> If country is available, then we should populate continent and sub-continent  
# -> If Device is available, OS should be guessed

# In[ ]:


get_ipython().run_cell_magic('time', '', "for df in [train_df, test_df]: \n    for col in df.select_dtypes('object').columns: \n        df[col] = df[col].fillna('DEF')")


# In[ ]:


train_df.select_dtypes('object').describe()


# In[ ]:


train_df.select_dtypes(exclude = 'object').describe()


# ## Group values with few rows and 0 revenue into a single value
# 
# Some columns has huge list of values and we have to split them into batches of 1000's

# In[ ]:


get_ipython().run_cell_magic('time', '', '# attributes = [\'device_browser\']\nattributes = [\'device_browser\', \'device_deviceCategory\', \'device_operatingSystem\', \'geoNetwork_city\', \'geoNetwork_metro\', \'geoNetwork_region\', \'geoNetwork_country\', \'geoNetwork_subContinent\', \'geoNetwork_continent\', \'geoNetwork_networkDomain\', \'trafficSource_adContent\', \'trafficSource_adwordsClickInfo.adNetworkType\', \'trafficSource_adwordsClickInfo.page\', \'trafficSource_adwordsClickInfo.slot\', \'trafficSource_campaign\', \'trafficSource_keyword\', \'trafficSource_medium\', \'trafficSource_referralPath\', \'trafficSource_source\', \'trafficSource_adwordsClickInfo.gclId\']\nfor att in attributes: \n    print(att)\n    c = train_df[att].value_counts()\n    s = train_df.groupby(att)[\'totals_transactionRevenue\'].sum()\n    df = pd.concat([c, s], axis = 1, sort=True)\n    df.columns = [\'recs\', \'tr\']\n    df = df.reset_index()\n    df.columns = [\'idx\', \'recs\', \'tr\']\n    val_to_keep = list(df[ (df.recs > 10000) | (df.tr > 0 ) ].idx)\n    val_to_keep.append(\'DEF\')\n    d = dict()\n    counter = 0\n    for i in c.index:\n        if i not in val_to_keep: \n            d[i] = "DEF"\n            counter = counter + 1\n            if ( counter % 2000 == 0 ): \n                print("Iter : " + str(round(counter/2000)))\n                train_df[att].replace(d, inplace=True)\n                test_df[att].replace(d, inplace=True)\n                d = dict()\n    train_df[att].replace(d, inplace=True)\n    test_df[att].replace(d, inplace=True)')


# In[ ]:


f = './train_flat_cln1.csv'
train_df.to_csv(f, index = False)
s = os.stat(f)
num_lines = sum(1 for line in open(f))
print(f + ":" + str(round(s.st_size / (1024 * 1024)) ) + " MB : " + str(num_lines) + " lines")


# In[ ]:


f = './test_flat_cln1.csv'
test_df.to_csv(f, index = False)
s = os.stat(f)
num_lines = sum(1 for line in open(f))
print(f + ":" + str(round(s.st_size / (1024 * 1024)) ) + " MB : " + str(num_lines) + " lines")


# ## Correlation analysis

# In[ ]:


c = train_df.select_dtypes(exclude = 'object').drop(columns = ['totals_visits', 'trafficSource_isVideoAd', 'trafficSource_isDirect']).corr()
c


# In[ ]:


from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools


# In[ ]:


init_notebook_mode(connected=True)
trace = go.Heatmap(z = c.values.tolist(), x = c.columns, y = c.columns, colorscale='RdBu')
data=[trace]
layout = go.Layout(
    autosize=False,
    width=600,
    height=600,
)

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='basic-heatmap')


# From the above figure we see revenue is correlated with total_hits and page_views

# ## Understanding the variables
# We have 4 JSON objects in the source - Device, Location, Traffic and Totals source.  Lets analyze these 4 in detail.

# ## Program to create 

# In[ ]:


def create_count_rev_plot(bar_series, scatter_series): 
    trace1 = go.Bar(
        x = bar_series.index, 
        y = bar_series.values, 
        name = 'No of Records'
    )
    trace2 = go.Scatter(
        x = scatter_series.index, 
        y = scatter_series['totals_transactionRevenue'],
        yaxis = 'y2', 
        mode='markers+text', 
        name = 'TR'
    )
    trace3 = go.Scatter(
        x = scatter_series.index, 
        y = scatter_series['totals_transactionRevenue'] / scatter_series['totals_pageviews'],
        yaxis = 'y3', 
        mode='markers+text', 
        name = 'TR / PV'
    )
    trace4 = go.Scatter(
        x = scatter_series.index, 
        y = scatter_series['totals_transactionRevenue'] / scatter_series['totals_hits'],
        yaxis = 'y4', 
        mode='markers+text', 
        name = 'TR / Hit'
    )
    layout = go.Layout(
        title=sc.name + ' & ' + rs.name,
        yaxis=dict(
            title='No. of Records', 
            titlefont=dict(
                color='#1f77b4'
            ),
            tickfont=dict(
                color='#1f77b4'
            )
        ),
        yaxis2=dict(
            title='Total Revenue',
            titlefont=dict(
                color='#ff7f0e'
            ),
            tickfont=dict(
                color='#ff7f0e'
            ),
            anchor='free',
            overlaying='y',
            side='left',
            position=0.02
        ),
        yaxis3=dict(
            title='Total Revenue Per PageView',
            titlefont=dict(
                color='#d62728'
            ),
            tickfont=dict(
                color='#d62728'
            ),
            anchor='x',
            overlaying='y',
            side='right'        
        ),
        yaxis4=dict(
            title='Total Revenue Per Hit',
            titlefont=dict(
                color='#9467bd'
            ),
            tickfont=dict(
                color='#9467bd'
            ),
            anchor='free',
            overlaying='y',
            side='right',
            position=0.98        
        )
    )
    data = [trace1, trace2, trace3, trace4]
    fig = go.Figure(data=data, layout=layout)
    fig['layout'].update(showlegend = False, height = 600, width = 800)
    return iplot(fig, filename='multiple-axes-multiple')


# In[ ]:


for att in ["device_browser", 'device_deviceCategory', 'device_operatingSystem']:
    sc = train_df[att].value_counts()
    sc.name = att
    rs = train_df.groupby(att)['totals_transactionRevenue', 'totals_hits', 'totals_pageviews'].sum()
    rs.name = "Revenues"
    create_count_rev_plot(sc,rs)


# In[ ]:


# Skip: 'geoNetwork_networkDomain'
for att in ["geoNetwork_city", 'geoNetwork_metro', 'geoNetwork_region', 'geoNetwork_country', 'geoNetwork_subContinent', 'geoNetwork_continent', 'geoNetwork_networkDomain']:
    sc = train_df[att].value_counts()
    sc.name = att
    rs = train_df.groupby(att)['totals_transactionRevenue', 'totals_hits', 'totals_pageviews'].sum()
    rs.name = "Revenues"
    create_count_rev_plot(sc,rs)


# In[ ]:


# Skip: 'trafficSource_adwordsClickInfo.gclId'
for att in ["trafficSource_adContent", 'trafficSource_adwordsClickInfo.adNetworkType', 'trafficSource_adwordsClickInfo.isVideoAd', 'trafficSource_adwordsClickInfo.page', 'trafficSource_adwordsClickInfo.slot', 'trafficSource_campaign', 'trafficSource_isTrueDirect', 'trafficSource_keyword', 'trafficSource_medium', 'trafficSource_referralPath', 'trafficSource_source', 'trafficSource_adwordsClickInfo.gclId']:
    sc = train_df[att].value_counts()
    sc.name = att
    rs = train_df.groupby(att)['totals_transactionRevenue', 'totals_hits', 'totals_pageviews'].sum()
    rs.name = "Revenues"
    create_count_rev_plot(sc,rs)

