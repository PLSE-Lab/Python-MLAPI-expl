#!/usr/bin/env python
# coding: utf-8

# This notebook generates an automatized report for an ecommerce site during a user selected time period. This is done through a function calling SQL queries to the [sample Google Analytics dataset](https://www.kaggle.com/bigquery/google-analytics-sample), via the integration of BigQuery in python. The visualizations use the pandas dataframes produced by BigQuery processed by matplotlib and seaborn. The dataset timeframe goes from 2016-08-01 to 2017-08-01.

# In[ ]:


from google.cloud import bigquery
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

client = bigquery.Client()
dataset_ref = client.dataset("google_analytics_sample", project="bigquery-public-data")
dataset = client.get_dataset(dataset_ref)


# In[ ]:


def ecommerce_report(date_start, date_end):
    #visits per day    
    query = '''
            SELECT date, count(1) as visits
            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
            WHERE _TABLE_SUFFIX BETWEEN <start> AND <end>
            GROUP BY date
            ORDER by date
            '''
    query = query.replace('<start>', "'"+date_start+"'").replace('<end>', "'"+date_end+"'")
    query_job = client.query(query) 
    
    visits_day = query_job.to_dataframe().reset_index()
    
    #visits by device and region
    query = '''
            WITH table_aux AS
                (SELECT device.deviceCategory as device
                ,geoNetwork.subcontinent as subcontinent
                ,count(1) as visits
                FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
                WHERE _TABLE_SUFFIX BETWEEN <start> AND <end>
                GROUP BY subcontinent, device
                ORDER by subcontinent, device)
            SELECT CASE 
                    WHEN subcontinent = '(not set)' THEN 'others'
                    ELSE subcontinent
                    END as region
                    ,device
                    ,visits
            FROM table_aux
            '''
    query = query.replace('<start>', "'"+date_start+"'").replace('<end>', "'"+date_end+"'")
    query_job = client.query(query) 
    
    visits_device_region = query_job.to_dataframe().reset_index()
    
    visits_device = visits_device_region.groupby(
        by = 'device').agg(
        visits = pd.NamedAgg(column = 'visits', aggfunc = 'sum')).sort_values(
        by = 'visits', ascending = False).reset_index()
    
    visits_region = visits_device_region.groupby(
        by = 'region').agg(
        visits = pd.NamedAgg(column = 'visits', aggfunc = 'sum')).sort_values(
        by = 'visits', ascending = False).reset_index()
    
    #average statistics by day (with missing values)
    query = '''
            WITH avg_daily_nulls AS
            (SELECT date
            ,CASE 
                WHEN totals.timeonsite  is not null THEN totals.timeonsite 
                ELSE 0 
                END as timeonsite_
            ,CASE 
                WHEN totals.pageviews  is not null THEN totals.pageviews 
                ELSE 0 
                END as pageviews_
            ,CASE 
                WHEN totals.transactions  is not null THEN 1 
                ELSE 0 
                END as transactions_
            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
            WHERE _TABLE_SUFFIX BETWEEN <start> AND <end>)
            SELECT date
            ,ROUND(AVG(timeonsite_)/60,2) as avg_minutes_on_site
            ,ROUND(AVG(pageviews_),2) as avg_page_views
            ,ROUND(AVG(transactions_)*100,2) as percent_transactions_visit
            FROM avg_daily_nulls
            GROUP BY date
            ORDER by date
            '''
    query = query.replace('<start>', "'"+date_start+"'").replace('<end>', "'"+date_end+"'")
    query_job = client.query(query) 
    
    visits_day_avg_nulls = query_job.to_dataframe().reset_index() 
    
    #average statistics by day (without missing values)
    query = '''
            SELECT date
            ,ROUND(AVG(totals.totalTransactionRevenue)/1000000,2) as avg_total_transaction_revenue
            FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*`
            WHERE _TABLE_SUFFIX BETWEEN <start> AND <end>
            GROUP BY date
            ORDER by date
            '''
    query = query.replace('<start>', "'"+date_start+"'").replace('<end>', "'"+date_end+"'")
    safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**8) #100M
    query_job = client.query(query, job_config=safe_config) 
    
    visits_day_avg_nonulls = query_job.to_dataframe().reset_index() 
    
    #total visits plot
    fig1 = plt.figure(constrained_layout=True, figsize = (10,10))
    
    gs = fig1.add_gridspec(2, 5)
    f1_ax1 = fig1.add_subplot(gs[0, :])
    f1_ax2 = fig1.add_subplot(gs[1, :-2])
    f1_ax3 = fig1.add_subplot(gs[1, -2:])
    
    #ax1
    data = visits_day
    sns.lineplot(data = data, ax = f1_ax1, x="date", y="visits", marker = 'o', color = 'darkorange')
    f1_ax1.set_title('Daily Visits')
    f1_ax1.set_xlabel('Date')
    f1_ax1.set_ylabel('Total Visits')
    f1_ax1.grid(True)
    l = f1_ax1.lines[0]
    x1 = l.get_xydata()[:,0]
    y1 = l.get_xydata()[:,1]
    f1_ax1.fill_between(x1,y1, color = 'darkorange', alpha=0.5)
    fill_min = 0.9*min(data['visits'])
    fill_max = 1.1*max(data['visits'])
    f1_ax1.set_ylim(fill_min,fill_max)
    f1_ax1.set_xticklabels(labels = data['date'], rotation=90)
    
    #ax2
    data = visits_region
    sns.barplot(data = data, ax = f1_ax2, x = 'region', y='visits', color = 'green')
    f1_ax2.set_title('Total Visits by Region')
    f1_ax2.set_xlabel('Region')
    f1_ax2.set_ylabel('Total Visits')
    f1_ax2.grid(True)
    f1_ax2.set_xticklabels(labels = data['region'], rotation=90)
    
    #ax3
    data = visits_device
    sns.barplot(data = data, ax = f1_ax3, x = 'device', y='visits', color = 'darkred')
    f1_ax3.set_title('Total Visits by Device')
    f1_ax3.set_xlabel('Device')
    f1_ax3.set_ylabel('Total Visits')
    f1_ax3.grid(True)
    
    fig1.suptitle('Total Visits. ' + date_start + ' to ' + date_end, fontsize = 14, fontweight = 'bold')
    plt.show()
    
    #visit activity plot
    fig2 = plt.figure(constrained_layout=True, figsize = (10,10))
    
    gs = fig2.add_gridspec(2, 1)
    f2_ax1 = fig2.add_subplot(gs[0, :])
    f2_ax2 = fig2.add_subplot(gs[1, :])
    
    #ax1
    data = visits_day_avg_nulls
    sns.lineplot(data = data, ax = f2_ax1, x="date", y="avg_minutes_on_site", marker = 'o', color = 'darkorange')
    f2_ax1.set_title('Average Time per Visit')
    f2_ax1.set_xlabel('Date')
    f2_ax1.set_ylabel('Minutes')
    f2_ax1.grid(True)
    l = f2_ax1.lines[0]
    x1 = l.get_xydata()[:,0]
    y1 = l.get_xydata()[:,1]
    f2_ax1.fill_between(x1,y1, color = 'darkorange', alpha=0.5)
    fill_min = 0.9*min(data['avg_minutes_on_site'])
    fill_max = 1.1*max(data['avg_minutes_on_site'])
    f2_ax1.set_ylim(fill_min, fill_max)
    f2_ax1.set_xticklabels(labels = data['date'], rotation=90)
    
    #ax2
    sns.lineplot(data = data, ax = f2_ax2, x="date", y="avg_page_views", marker = 'o', color = 'green')
    f2_ax2.set_title('Average Pages Seen for Visit')
    f2_ax2.set_xlabel('Date')
    f2_ax2.set_ylabel('Pages')
    f2_ax2.grid(True)
    l = f2_ax2.lines[0]
    x1 = l.get_xydata()[:,0]
    y1 = l.get_xydata()[:,1]
    f2_ax2.fill_between(x1,y1, color = 'green', alpha=0.2)
    fill_min = 0.9*min(data['avg_page_views'])
    fill_max = 1.1*max(data['avg_page_views'])
    f2_ax2.set_ylim(fill_min, fill_max)
    f2_ax2.set_xticklabels(labels = data['date'], rotation=90)
    
    fig2.suptitle('Visit Activity. '  + date_start + ' to ' + date_end, fontsize = 14, fontweight = 'bold')
    plt.show()
    
    #transactions plot
    fig3 = plt.figure(constrained_layout=True, figsize = (10,10))
    
    gs = fig3.add_gridspec(2, 1)
    f3_ax1 = fig3.add_subplot(gs[0, :])
    f3_ax2 = fig3.add_subplot(gs[1, :])
    
    #ax1
    data = visits_day_avg_nulls
    sns.lineplot(data = data, ax = f3_ax1, x="date", y="percent_transactions_visit", marker = 'o', color = 'darkorange')
    f3_ax1.set_title('% of Visits Incurring in Transactions')
    f3_ax1.set_xlabel('Date')
    f3_ax1.set_ylabel('%')
    f3_ax1.grid(True)
    l = f3_ax1.lines[0]
    x1 = l.get_xydata()[:,0]
    y1 = l.get_xydata()[:,1]
    f3_ax1.fill_between(x1,y1, color = 'darkorange', alpha=0.5)
    fill_min = 0.9*min(data['percent_transactions_visit'])
    fill_max = 1.1*max(data['percent_transactions_visit'])
    f3_ax1.set_ylim(fill_min, fill_max)
    f3_ax1.set_xticklabels(labels = data['date'], rotation=90)
    
    #ax2
    data = visits_day_avg_nonulls
    sns.lineplot(data = data, ax = f3_ax2, x="date", y="avg_total_transaction_revenue", marker = 'o', color = 'green')
    f3_ax2.set_title('Mean Total Amount ($) per Visit Incurring in Transactions')
    f3_ax2.set_xlabel('Date')
    f3_ax2.set_ylabel('Amount')
    f3_ax2.grid(True)
    l = f3_ax2.lines[0]
    x1 = l.get_xydata()[:,0]
    y1 = l.get_xydata()[:,1]
    f3_ax2.fill_between(x1,y1, color = 'green', alpha=0.2)
    fill_min = 0.9*min(data['avg_total_transaction_revenue'])
    fill_max = 1.1*max(data['avg_total_transaction_revenue'])
    f3_ax2.set_ylim(fill_min, fill_max)
    f3_ax2.set_xticklabels(labels = data['date'], rotation=90)
    
    fig3.suptitle('Transactions Activity. '  + date_start + ' to ' + date_end, fontsize = 14, fontweight = 'bold')
    plt.show()


# In[ ]:


ecommerce_report('20170401', '20170430')

