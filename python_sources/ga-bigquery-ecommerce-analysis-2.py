#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("../input"))


# In[ ]:


import bq_helper

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
bq_assistant = bq_helper.BigQueryHelper("bigquery-public-data", "google_analytics_sample")
print(bq_assistant.list_tables(a)[:5])
print(bq_assistant.list_tables()[-5:])
table_names = bq_assistant.list_tables()
# a table for each day : 2016-08-01 - 2017-08-01
# a year's worth of data


# In[ ]:


bq_assistant.table_schema("ga_sessions_20170801").head()


# In[ ]:


def inspect(query, nrows=15, sample=False):
    """Display response from given query but don't save. 
    query: str, raw SQL query
    nrows: int, number of rows to display, default 15
    sample: bool, use df.sample instead of df.head, default False """
    response = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=10)
    if sample:
        return response.sample(nrows)
    return response.head(nrows) 

def retrieve(query, nrows=10):
    """Save response from given query and print a preview. 
    query: str, raw SQL query
    nrows: int, number of rows to display"""
    response = bq_assistant.query_to_pandas_safe(query, max_gb_scanned=10)
    print(response.head(nrows))
    return response


def plot_metric_by_month(vc_df, metric, group):
    '''Convert date from string and plot metric by group
    vc_df: df, SQL value counts output grouped by month and group
    metric: str, column name for metric of interest
    group: str, column name of group'''
    df = vc_df.copy()
    df['ym'] = pd.to_datetime( df['ym'], format='%Y%m')
    sns.lineplot(x='ym', y=metric, hue=group, data=df, marker='o')
    plt.title(f'Monthly {metric}')
    plt.legend(loc="upper left", bbox_to_anchor=(1,1))


# # Preface
# From the [previous notebook](https://www.kaggle.com/douglasl/ga-bigquery-ecommerce-analysis/code?scriptVersionId=13494397), we've started exploring what data is available and what the different columns can tell us. We know the data set contains roughly 900,000 sessions across a 1 year time period from August 2016 to August 2017. We've also answered ad hoc questions such as average page views for different groups of users. 
# 
# In this notebook, I'll try to answer questions that might come up in a marketing setting.
# 
# 1. What is our revenue by traffic source?
# 2. What is the Return on Ad Spend (ROAS) for our Paid Search campaigns?
# 

# # Revenue by traffic source - channel

# In[ ]:


# Calculate revenue across full time period available
query = """
SELECT channelGrouping
    , SUM(totals.totalTransactionRevenue) / 1000000 AS sum_revenue
    , SUM(totals.totalTransactionRevenue) / 
        (SELECT SUM(totals.totalTransactionRevenue) 
             FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`)
         * 100 AS pct_total_rev
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
    GROUP BY ROLLUP(channelGrouping) -- get all grps total
    ORDER BY sum_revenue DESC
 
"""
inspect(query)


# In[ ]:


# Mmonthly revenue 
query = """
SELECT 
    substr(date, 1, 6) as ym
    , SUM(totals.totalTransactionRevenue) month_revenue
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
    GROUP BY ym
    ORDER BY ym
 """ 
monthly_revenue_totals = retrieve(query)


# In[ ]:


# Monthly revenue with percentage of total 
query = """


SELECT 
    a.ym
    , a.channelGrouping
    , a.sum_revenue / 1000000 as sum_rev
    , b.month_revenue / 1000000 as mo_rev
    , a.sum_revenue / b.month_revenue * 100 AS pct_mo_rev
FROM 
    (SELECT substr(date, 1, 6) as ym
            , channelGrouping
            , SUM(totals.totalTransactionRevenue) AS sum_revenue
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
        GROUP BY channelGrouping, ym
        ORDER BY channelGrouping, ym ) a
JOIN 
    (SELECT substr(date, 1, 6) as ym
            , SUM(totals.totalTransactionRevenue) month_revenue
    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
        GROUP BY ym
        ORDER BY ym) b
 ON a.ym = b.ym
 ORDER BY a.channelGrouping, a.ym

 
"""
monthly_channel_rev = retrieve(query)


# In[ ]:


plot_metric_by_month(monthly_channel_rev, 'sum_rev', 'channelGrouping')
plt.xlabel('Date')
plt.ylabel('Sum of Revenue')
# Very big spike in revenue from Display in April 2017. Why?


# In[ ]:


plot_metric_by_month(monthly_channel_rev, 'pct_mo_rev', 'channelGrouping')
plt.xlabel('Date')
plt.ylabel('Percent of Overall Revenue')


# Referrals, direct (url or bookmark), and organic search are the top 3 channels for generating revenue, combining to make up 89% of the total revenue generated by the store. There is one unusual spike in revenue from a paid channel, Display (banner ads?) in April 2017.  

# In[ ]:


# Quick look at the sources of referrals - direct referral links to the store. 
# Makes sense since a lot of the customers for Google merchandise would be connected to Googlers. 

query = """SELECT trafficSource.campaign
                , channelGrouping
                , trafficSource.source
                , count(*) as cnt
            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
                WHERE channelGrouping = 'Referral'
              GROUP BY trafficSource.campaign, channelGrouping, trafficSource.source
              ORDER BY cnt desc
            
            """
inspect(query)
# Mostly direct traffic and google referrals 


# ### Investigate the spike
# What happened in April 2017?

# In[ ]:


# Review the numbers for just the Display channel
monthly_channel_rev[monthly_channel_rev['channelGrouping'] == 'Display']


# In[ ]:


# Get all data from April 2017 
query = """SELECT date
                , channelGrouping
                , trafficSource.source
                , totals.totalTransactionRevenue, totals.transactions
                , trafficSource.campaign
                , trafficSource.medium
            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201704*`
            
            """
apr2017 = retrieve(query)


# In[ ]:


apr2017.head()


# In[ ]:


apr2017.groupby(['channelGrouping','source']).agg({'totalTransactionRevenue': 'sum', 'source':'count'}).sort_values('totalTransactionRevenue', ascending=False).head(10)
# Display:dfa (Doubleclick for Advertiser) generated the highest revenue even with only 209 visits  compared to the thousands other sources had


# In[ ]:


apr2017[apr2017['source'] == 'dfa'].sort_values('totalTransactionRevenue', ascending=False).head(10)
# Only 8 sessions with revenue, and 2 involved large transactions that propel Display to the top


# In[ ]:


# look at more details from these sessions
query = """SELECT date
                , fullVisitorId
                , visitId
                , visitNumber
                , totals.hits
                , totals.transactions
                , totals.totalTransactionRevenue / 1000000 as totalTransactionRevenue
            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201704*`
            WHERE trafficSource.source = 'dfa' 
                AND totals.totalTransactionRevenue > 0 
            
            """
inspect(query)
# It was a single visitor (1957458976293878100) responsible for 5 sessions, including the 2 highest revenue sessions
# So it was an outlier that disrupted the overall trend 


# In[ ]:


# Aggregate totals for the month 
query = """SELECT fullVisitorId
                  , channelGrouping
                  , SUM(totals.totalTransactionRevenue) / 1000000 as total_rev
                  , COUNT(visitId) as total_sessions
                  , SUM(totals.totalTransactionRevenue) /  COUNT(visitId) / 1000000 as rev_per_session
            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201704*`
            WHERE trafficSource.source = 'dfa' 
                AND totals.totalTransactionRevenue > 0 
            GROUP BY fullVisitorId, channelGrouping
            """
inspect(query)


# In[ ]:


# Retrieve full records from visitos who were customers in April 2017 and came from the dfa channel
query = """
SELECT  fullVisitorId
      , channelGrouping
      , SUM(totals.totalTransactionRevenue) / 1000000 as total_rev
      , COUNT(visitId) as total_sessions
      , SUM(totals.totalTransactionRevenue) /  COUNT(visitId) / 1000000 as rev_per_session
      --, array_agg(DISTINCT channelGrouping) channels
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
 WHERE fullVisitorId IN (SELECT fullVisitorId
                        FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201704*`
                        WHERE trafficSource.source = 'dfa' AND totals.totalTransactionRevenue > 0)
GROUP BY fullVisitorId, channelGrouping
ORDER BY fullVisitorId
            
            """
apr_customer_records = retrieve(query)
# divide by 1000000 to get unit back to USD 


# In[ ]:


apr_customer_records.sort_values(['fullVisitorId', 'rev_per_session'], ascending=[True, False])
# This customer made the bulk of its spending in April 2017 but did spend nearly $30k in other months through the Display channel. They also made purchases through the Direct channel. 
# Interestingly, they may not have had as much success finding their desired item through Organic Search. Overall, they are on a different scale from the rest of the April 2017 customers and are an outlier.


# In[ ]:


# #  unused unnesting
# query = """SELECT date, visitNumber, totals.*, trafficSource.*, product.v2ProductName
#             FROM `bigquery-public-data.google_analytics_sample.ga_sessions_201704*`,
#                 UNNEST(hits) as hits, UNNEST(hits.product) as product
#             WHERE trafficSource.source = 'dfa' AND totals.totalTransactionRevenue > 0 
            
#             """
# inspect(query)


# Conclusion of investigation: the revenue spike in April 2017 was an outlier caused by a single enthusiastic visitor, so not likely to be something the marketing team can build on. 

# # What is the ROAS on campaigns? 
# Paid search is a highly touted channel for ecommerce businesses. Although it seems to play a minor role for the Google Merchandise store, let's look at how well paid search campaigns do through the ROAS metric.
# 
# ROAS = gross revenue from ad campaign / cost of ad campaign
# 
# Since I don't know the details of the other costs, I'll use a simplified version where I estimate the costs based only on the number of visits through that channel.

# In[ ]:


# Get a quick summary on how long campaigns last and which channels they belong to
query = """SELECT trafficSource.campaign
                , channelGrouping
                , MIN(date) as start_date
                , MAX(date) as end_date
                , COUNT(*) cnt_visits
            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
                GROUP BY trafficSource.campaign, channelGrouping
                ORDER BY cnt_visits DESC
            
            """
campaigns = retrieve(query)

# side note: column trafficSource.campaignCode is not useful


# In[ ]:


# Since date is a stored as a string, let's calculate the duration of theh campaign in pandas instead
campaigns['duration'] = pd.to_datetime(campaigns['end_date'], format='%Y%m%d') -  pd.to_datetime(campaigns['start_date'], format='%Y%m%d')


# In[ ]:


campaigns
# again, the volume from other channels overshadows paid search for the google merchandise store


# In[ ]:


# filter down to paid search campaigns
campaigns[campaigns['channelGrouping'] == 'Paid Search']
#Adwords - focus on paid search


# In[ ]:


# Get more details on campaigns
query = """SELECT trafficSource.campaign
                , channelGrouping
                , trafficSource.source
                , trafficSource.medium
                , count(*) as cnt_visits
                , SUM( CASE WHEN totals.transactions > 0 THEN 1 ELSE 0 END) AS cnt_transactions
                , SUM (totals.totalTransactionRevenue) / 1000000 AS total_rev
                , SUM (totals.totalTransactionRevenue) / COUNT(*) / 1000000 as revenue_per_visit
            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
                WHERE channelGrouping = 'Paid Search'
                  AND trafficSource.campaign != '(not set)'
            GROUP BY trafficSource.campaign, channelGrouping, trafficSource.source, trafficSource.medium
            ORDER BY revenue_per_visit desc
            
            """
inspect(query)

# all on cost per click model


# If each click costs $1, then the revenue_per_visit column represents the ROAS. At only 3:1 and 2:1 ratios for the two more popular campaigns, these are pretty low ROAS. Again, this is a simplifcation as cost per click may be determined by by auction and there are additional factors like personnel expenses and other costs. 
# 
# With the low ROAS and considering the audience of the Google merchandise store, I would recommend focusing on the other more successful channels. 
# 
# https://www.bigcommerce.com/ecommerce-answers/what-is-roas-calculating-return-on-ad-spend/

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




