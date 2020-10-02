#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.gridspec as gridspec


# In[ ]:


sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
sample_sub= pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
sales_train_val = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')


# In[ ]:


calendar.info()


# In[ ]:


calendar.head()


# In[ ]:


calendar['date'] = pd.to_datetime(calendar['date'])
for col in ['event_name_1','event_type_1','event_name_2','event_type_2']:
    calendar[col].fillna('unknown',inplace=True)


# In[ ]:


total_sale = []
for i in sales_train_val.index:
    total_sale.append(sales_train_val.loc[i]['d_1':].sum())
sales_train_val['total_sale'] = total_sale 


# In[ ]:


fig = plt.figure(figsize=(18,13))
gs = gridspec.GridSpec(5,10)
ax1 = fig.add_subplot(gs[0:2,:4])
ax2 = fig.add_subplot(gs[0:2,5:])
ax3 = fig.add_subplot(gs[3:5,2:7])
for cat,ax in zip(sales_train_val['cat_id'].unique(),[ax1,ax2,ax3]):
    
    sales_train_val[sales_train_val['cat_id'] == cat]                                                .sort_values(by='total_sale',ascending=False)[:11]                                                .plot(x='id',y='total_sale',kind='bar',legend=False,ax=ax)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_title('Top selling items(id) in ' + cat + ' category',fontsize=15)
    ax.grid(axis='y',alpha=0.5)
    ylabel = 'Total Sale'
    xlabel = 'Id'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# In[ ]:


fig = plt.figure(figsize=(18,13))
gs = gridspec.GridSpec(2,10)
ax1 = fig.add_subplot(gs[0,:5])
ax2 = fig.add_subplot(gs[0,5:])
ax3 = fig.add_subplot(gs[1,2:7])
for cat,ax in zip(sales_train_val['cat_id'].unique(),[ax1,ax2,ax3]):
    sales_train_val[sales_train_val['cat_id'] == cat]                                        [['item_id','total_sale']]                                        .groupby('item_id')                                        .sum()                                        .sort_values(by='total_sale',ascending=False)[:11]                                        .plot(kind='bar',ax = ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.set_title('Top 10 selling ' + cat + ' items',fontsize=20)
    ax.grid(axis='y',alpha=0.5)
plt.tight_layout()


# ## Helper functions for Data analysis and visualization

# In[ ]:


def create_feature_list(feature):
    feature_list = [feature]
    for col in sales_train_val.columns:
        if col.startswith('d_'):
            feature_list.append(col)
    return feature_list


def groupby_sale(key,feature_list):
    total_daily_sale = sales_train_val[feature_list]                             .groupby(key)                             .sum()                             .T                             .set_index(calendar[:1913]['date'])
    return total_daily_sale



def daily_sale_plot(total_daily_sale,label):
    fig = plt.figure(figsize=(18,15))
    gs = gridspec.GridSpec(5,3)
    ax1 = fig.add_subplot(gs[0:2,0])
    ax2 = fig.add_subplot(gs[0:2,1:])
    ax3 = fig.add_subplot(gs[2:,:])

    ylabel = 'Daily Sale'
    xlabel = 'Date'

    total_daily_sale.sum().sort_values(ascending=False).plot(kind='bar',legend=False,ax=ax1)
    ax1.set(xlabel=label, ylabel='Total Sale')
    ax1.set_title('Total Sale for each ' + label,fontsize=15)


    total_daily_sale[:31].plot(marker='o',ax=ax2)
    ax2.autoscale(axis='x',tight=True)
    ax2.legend(bbox_to_anchor=(1.2, 1), loc='upper right',fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax2.set_title('Daily Sale for each ' + label +' (one month snippet)',fontsize=15)
    ax2.set(xlabel=xlabel, ylabel=ylabel)

    total_daily_sale.plot(ax=ax3)
    ax3.autoscale(axis='x',tight=True)
    ax3.legend(bbox_to_anchor=(1.12, 1), loc='upper right',fancybox=True, framealpha=1, shadow=True, borderpad=1)
    ax3.set_title('Daily Sale for each '+ label,fontsize=20)
    ax3.set(xlabel=xlabel, ylabel=ylabel)

    plt.tight_layout()
    plt.show()
    


def Weekday_month_sales_plot(total_daily_sale_Copy,label):
    fig = plt.figure(figsize=(18,6))
    gs = gridspec.GridSpec(1,19)
    ax1 = fig.add_subplot(gs[0,:8])
    ax2 = fig.add_subplot(gs[0,8:])

    for freq,ax in zip(['weekday','month'],[ax1,ax2]):
        total_daily_sale = total_daily_sale_Copy.copy()
        total_daily_sale = total_daily_sale.reset_index(drop=True)
        total_daily_sale[freq] = calendar[:1913][freq]
        total_daily_sale.set_index(calendar[:1913]['date'],inplace=True)

        total_daily_sale.groupby(freq,sort=False).mean().plot(marker = 'o',linewidth=3,ax=ax,legend=False)
        ax.set(xlabel=freq, ylabel='Average Sale')
        ax.set_title('Average Sale vs ' + freq +' (Per ' + label +')',fontsize=15)

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels,bbox_to_anchor=(1.1, 0.8),fancybox=True, framealpha=1, shadow=True, borderpad=1)
    plt.tight_layout()

 


def Average_Sale_plot(total_daily_sale,label):
    fig = plt.figure(figsize=(18,15))
    gs = gridspec.GridSpec(2,3)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1:])
    ax3 = fig.add_subplot(gs[1,:])

    total_daily_sale.resample('Y').mean().plot(marker='o',ax=ax1,legend=False,linewidth=3.0)
    ax1.autoscale(axis='x',tight=True)
    ax1.set_title('Average Yearly Sale for each ' + label,fontsize=15)
    ax1.set(ylabel = 'Average Sale')

    total_daily_sale.resample('M').mean().plot(marker='o',ax=ax2,legend=False)
    ax2.autoscale(axis='x',tight=True)
    ax2.set_title('Average Monthly Sale for each ' + label,fontsize=20)
    ax2.set(ylabel = 'Average Sale')

    total_daily_sale.resample('W').mean().plot(marker='o',ax=ax3,legend=False)
    ax3.autoscale(axis='x',tight=True)
    ax3.set_title('Average Weekly Sale for each ' + label,fontsize=20)
    ax3.set(ylabel = 'Average Sale')

    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels,bbox_to_anchor=(1.08, 0.9),fancybox=True, framealpha=1, shadow=True, borderpad=1)

    plt.tight_layout()
    plt.show()

    
    
def rolling_average_plot(total_daily_sale,label):
    for cat in total_daily_sale.columns:
        total_daily_sale[cat].rolling(window=90)                                                 .mean()                                                 .plot(figsize=(15,8),label= cat + ': 90 days mean',legend=True,linewidth=3)
    plt.legend(bbox_to_anchor=(1.3, 1), loc='upper right',fancybox=True, framealpha=1, shadow=True, borderpad=1,fontsize='large')
    plt.autoscale(axis='x',tight=True)
    plt.title('Rolling Average(90 days) Sales for each category',fontsize=20)
    plt.show()
    
    
    
def sell_prices_plot(cat):
    top_selling_items = sales_train_val[sales_train_val['cat_id'] == cat]                                            [['item_id','total_sale']]                                            .groupby('item_id')                                            .sum()                                            .sort_values(by='total_sale',ascending=False)[:7].index
    stores = sell_prices['store_id'].unique()

    fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(18,12))
    fig.suptitle(cat + ' category Sell Prices vs Year (Per Store)',fontsize=20)

    for item,ax in zip(top_selling_items,[ax1,ax2,ax3,ax4,ax5,ax6]):

        for store in stores:
            sell_price_per_store[store] = sell_prices_merged[(sell_prices_merged['item_id']==item) & (sell_prices_merged['store_id']==store)]['sell_price']
        sell_price_per_store.resample('A').mean().plot(ax=ax)
        ax.autoscale(axis='x',tight=True)
        ax.set_title(item +' Average Sell Prices vs Year (Per Store)',fontsize=10)
        ax.set(ylabel = 'Average Sell Price',xlabel='Year')
        ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)


# ##  EDA ( GroupBy Category)

# In[ ]:


feature_list = create_feature_list('cat_id')
total_daily_sale_Copy = groupby_sale('cat_id',feature_list)


# In[ ]:


total_daily_sale = total_daily_sale_Copy.copy()
daily_sale_plot(total_daily_sale,label = 'Category')


# In[ ]:


total_daily_sale = total_daily_sale_Copy.copy()
Weekday_month_sales_plot(total_daily_sale,label='Category')


# In[ ]:


total_daily_sale = total_daily_sale_Copy.copy()
Average_Sale_plot(total_daily_sale,label='Category')


# In[ ]:


total_daily_sale = total_daily_sale_Copy.copy()
rolling_average_plot(total_daily_sale,label='Category')


# ## EDA ( GroupBy Department)

# In[ ]:


feature_list = create_feature_list('dept_id')
total_daily_sale_Copy = groupby_sale('dept_id',feature_list)


# In[ ]:


total_daily_sale = total_daily_sale_Copy.copy()
daily_sale_plot(total_daily_sale,label = 'Department')


# In[ ]:


total_daily_sale = total_daily_sale_Copy.copy()
Weekday_month_sales_plot(total_daily_sale,label='Department')


# In[ ]:


total_daily_sale = total_daily_sale_Copy.copy()
Average_Sale_plot(total_daily_sale,label='Department')


# In[ ]:


total_daily_sale = total_daily_sale_Copy.copy()
rolling_average_plot(total_daily_sale,label='Department')


# ## EDA ( GroupBy Store) 

# In[ ]:


feature_list = create_feature_list('store_id')
total_daily_sale_Copy = groupby_sale('store_id',feature_list)


# In[ ]:


total_daily_sale = total_daily_sale_Copy.copy()
daily_sale_plot(total_daily_sale,label = 'Store')


# In[ ]:


total_daily_sale = total_daily_sale_Copy.copy()
Weekday_month_sales_plot(total_daily_sale,label='Store')


# In[ ]:


total_daily_sale = total_daily_sale_Copy.copy()
Average_Sale_plot(total_daily_sale,label='Store')


# In[ ]:


total_daily_sale = total_daily_sale_Copy.copy()
rolling_average_plot(total_daily_sale,label='Store')


# ## Sell Prices Data Visualization

# In[ ]:


sell_prices.info()


# In[ ]:


sell_prices.head()


# In[ ]:


sell_prices_merged = sell_prices.merge(calendar[['date','wm_yr_wk']],on='wm_yr_wk').sort_values(by=['date','wm_yr_wk','item_id','store_id'])
sell_prices_merged.set_index('date',inplace=True)

sell_price_per_store = pd.DataFrame(index = calendar['date'],columns=sell_prices_merged['store_id'].unique())


# In[ ]:


sell_prices_plot(cat='FOODS')


# In[ ]:


sell_prices_plot(cat='HOUSEHOLD')


# In[ ]:


sell_prices_plot(cat='HOBBIES')

