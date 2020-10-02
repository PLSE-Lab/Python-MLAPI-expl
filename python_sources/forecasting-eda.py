#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# Getting the required packages for our analysis

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import datetime
import seaborn as sns # seaborn package for visualising
import plotly.express as px # plotly visualisation
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Reading the csv files from input files-
sell_prices=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
calendar=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
sales_train_validation=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')


# In[ ]:


# Checking initial data for their behaviour
sell_prices.head()


# The calendar gives out the details for the days and giving them codes. The dataset also contains for any other events that may have occured.Some of the examples that I can think of like Black Friday or the 4th of July since these dataset is based for US based regions

# In[ ]:


#pd.set_option('max_rows',None)
calendar


# In[ ]:


sales_train_validation.head()


# M5 Forecasting-**Accuracy**
#         
#         How much camping gear will one store sell each month in a year?
# 
# M5 Forecasting-**Uncertainty**
# 
# Error check - Weighted Root Mean Squared Scaled Error (RMSSE)
# 
# https://en.wikipedia.org/wiki/Makridakis_Competitions

# The goal: We have been challenged to predict sales data provided by the retail giant **Walmart 28 days** into the future. This competition will run in 2 tracks: In addition to **forecasting the values** themselves in the Forecasting competition, we are simultaneously tasked to estimate the **uncertainty of our predictions** in the Uncertainty Distribution competition. Both competitions will have the same 28 day forecast horizon.
# 
# Hierarchical data is divided into the sub-levels that get divided till it reaches an atomic level.
# 
# Example: Understanding from above data present in these data sets, I am creating the below flow(will update it as I understand the data more)
# 
#                                                     state_id
#                                                        ||
#                                                        ||
#                                                       \\//
#                                                        \/ 
#                                                     store_id
#                                                        ||
#                                                        ||
#                                                       \\//
#                                                        \/ 
#                                                     cat_id
#                                                        ||
#                                                        ||
#                                                       \\//
#                                                        \/ 
#                                                     dept_id
#                                                        ||
#                                                        ||
#                                                       \\//
#                                                        \/ 
#                                                     item_id

# ![Hierarchial Structure](https://drive.google.com/file/d/1klagyIsWIQaOiCIxThVFq9XZ22NxMy7k/view?usp=sharing)
# 

# In[ ]:


plot=sns.countplot(data=sell_prices,x=sell_prices['store_id'])
# These are just value counts and data that is there contains time-series data so IT is possible,store_id would be repeating again
# and thus we need to investigate further.


# Since the dataset is a time series plot we should divide the set based on week/year/day depending on our needs.If we just do a simple count it would be repetitive.

# In[ ]:


#sell_prices[(sell_prices==max(sell_prices['wm_yr_wk']))]
print('The dataset contains {0} rows and {1} columns'.format(sell_prices.shape[0],sell_prices.shape[1]))


# In[ ]:


#Converting column names from object to category
sell_prices['store_id']=sell_prices['store_id'].astype('category')
sell_prices['item_id']=sell_prices['item_id'].astype('category')
sell_prices.info()


# In[ ]:


#Checking the combination of min(wm_yr_wk) and max(wm_yr_wk)
print('Data is currently available for minimum wm_yr_wk-{0} and maximum wm_yr_wk-{1}'.format(min(sell_prices['wm_yr_wk']),max(sell_prices['wm_yr_wk'])))
# 11101 --> first week ID
# 11621 --> last week ID 


# According to the guidelines wm_yr_wk is an unique ID given to each record.But with every week change this ID gets a new ID therefore subtracting the maximum of the week and the minimum week will help me get the no. of weeks the data has been collected for.

# In[ ]:


#Converting to specified data
calendar['date']=pd.to_datetime(calendar['date'])
calendar['weekday']=calendar['weekday'].astype('category')
calendar['wday']=calendar['wday'].astype('category')
calendar['month']=calendar['month'].astype('category')
calendar['year']=calendar['year'].astype('category')


# In[ ]:


calendar.info()


# In[ ]:


print('Minimum of the date collected-{0} and the maximum of the date collected-{1}'.format(min(calendar['date']),max(calendar['date'])))


# In[ ]:


no_of_days=max(calendar['date'])-min(calendar['date'])
print('Total no. of weeks for data has been collected -',no_of_days.days//7)


# In[ ]:


mapping_id_date=calendar[['date','wm_yr_wk','weekday','wday','month','year']]
sell_prices_date=sell_prices.join(mapping_id_date.set_index('wm_yr_wk'),on='wm_yr_wk')


# I joined the sell_prices dataset and calendar data to get the corresponding date or year values after which I can perform EDA based on my requirements.This would also help in deducting our analysis a little better.

# In[ ]:


sell_prices_date.head()


# In[ ]:


#Dividing based on state
CA_data=sell_prices_date[sell_prices_date['store_id'].str.match('CA')]
TX_data=sell_prices_date[sell_prices_date['store_id'].str.match('TX')]
WI_data=sell_prices_date[sell_prices_date['store_id'].str.match('WI')]


# Based on my initial analysis,I first divided the dataset based on regions and creating subsets of data based on 3 regions - **CA-California** , **TX-Texas** and **WI-Wisconsin**.I would further divide this on a store code level.

# In[ ]:


CA_data.tail()


# In[ ]:


#Joining the id with the corresponding dates
CA_data_group=pd.DataFrame(CA_data.groupby(by=['year','store_id','item_id']).mean())
TX_data_group=pd.DataFrame(TX_data.groupby(by=['year','store_id','item_id']).mean())
WI_data_group=pd.DataFrame(WI_data.groupby(by=['year','store_id','item_id']).mean())


# I am using the mean to calculate on the basis by year,store_id and then item_id.For example :
#     
#         Year|Store_id|item_id|sell_price
#         2012|CA_1|Hobbies_1|2
#         2012|CA_1|Hobbies_1|4
#         2013|CA_1|Hobbies_1|8
#         2014|CA_1|Hobbies_1|4
#         2014|CA_1|Hobbies_1|4
#         
#  The above example would result in :
#         
#         Year|Store_id|item_id|sell_price
#         2012|CA_1|Hobbies_1|3
#         2013|CA_1|Hobbies_1|8
#         2014|CA_1|Hobbies_1|4
# With this I can check the trend in a yearly level for each store_id    

# In[ ]:


CA_data_group.tail()


# In[ ]:


CA_data_group=CA_data_group.reset_index()
TX_data_group=TX_data_group.reset_index()
WI_data_group=WI_data_group.reset_index()


# In[ ]:


CA_data_group=CA_data_group[['year', 'store_id', 'item_id','sell_price']]
TX_data_group=TX_data_group[['year', 'store_id', 'item_id','sell_price']]
WI_data_group=WI_data_group[['year', 'store_id', 'item_id','sell_price']]


# In[ ]:


#3 item groups - HOUSEHOLD,FOODS,HOBBIES
CA_data_group_HOUSEHOLD=CA_data_group[(CA_data_group['item_id'].str.match('HOUSEHOLD'))&(CA_data_group['store_id'].str.match('CA'))]
CA_data_group_FOODS=CA_data_group[(CA_data_group['item_id'].str.match('FOODS'))&(CA_data_group['store_id'].str.match('CA'))]
CA_data_group_HOBBIES=CA_data_group[(CA_data_group['item_id'].str.match('HOBBIES'))&(CA_data_group['store_id'].str.match('CA'))]

TX_data_group_HOUSEHOLD=TX_data_group[(TX_data_group['item_id'].str.match('HOUSEHOLD'))&(TX_data_group['store_id'].str.match('TX'))]
TX_data_group_FOODS=TX_data_group[(TX_data_group['item_id'].str.match('FOODS'))&(TX_data_group['store_id'].str.match('TX'))]
TX_data_group_HOBBIES=TX_data_group[(TX_data_group['item_id'].str.match('HOBBIES'))&(TX_data_group['store_id'].str.match('TX'))]

WI_data_group_HOUSEHOLD=WI_data_group[(WI_data_group['item_id'].str.match('HOUSEHOLD'))&(WI_data_group['store_id'].str.match('WI'))]
WI_data_group_FOODS=WI_data_group[(WI_data_group['item_id'].str.match('FOODS'))&(WI_data_group['store_id'].str.match('WI'))]
WI_data_group_HOBBIES=WI_data_group[(WI_data_group['item_id'].str.match('HOBBIES'))&(WI_data_group['store_id'].str.match('WI'))]


# In[ ]:


sell_price_avg = pd.concat([CA_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_CA_Household'}).groupby(['year']).mean(),
                  CA_data_group_FOODS.rename(columns={'sell_price':'sell_price_CA_Foods'}).groupby(['year']).mean(),
                 CA_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_CA_Hobbies'}).groupby(['year']).mean(),
                 TX_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_TX_Household'}).groupby(['year']).mean(),
                 TX_data_group_FOODS.rename(columns={'sell_price':'sell_price_TX_Foods'}).groupby(['year']).mean(),
                 TX_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_TX_Hobbies'}).groupby(['year']).mean(),
                 WI_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_WI_Household'}).groupby(['year']).mean(),
                 WI_data_group_FOODS.rename(columns={'sell_price':'sell_price_WI_Foods'}).groupby(['year']).mean(),
                 WI_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_WI_Hobbies'}).groupby(['year']).mean()], axis=1)


# In[ ]:


sell_price_avg=sell_price_avg.reset_index()


# In[ ]:


sell_price_avg


# In[ ]:


sell_price_avg.columns


# In[ ]:


sns.lineplot(x='year',y='sell_price_CA_Household',data=sell_price_avg)
sns.lineplot(x='year',y='sell_price_CA_Foods',data=sell_price_avg)
sns.lineplot(x='year',y='sell_price_CA_Hobbies',data=sell_price_avg)
sns.lineplot(x='year',y='sell_price_TX_Household',data=sell_price_avg)
sns.lineplot(x='year',y='sell_price_TX_Foods',data=sell_price_avg)
sns.lineplot(x='year',y='sell_price_TX_Hobbies',data=sell_price_avg)
sns.lineplot(x='year',y='sell_price_WI_Household',data=sell_price_avg)
sns.lineplot(x='year',y='sell_price_WI_Foods',data=sell_price_avg)
sns.lineplot(x='year',y='sell_price_WI_Hobbies',data=sell_price_avg)


#sell_price_avg.info()


# Sns plot was created at a regional level and products that are getting sold in those regions.I have created a more interactive plot using plotly and based on year I am checking the trend for the category of those products.

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_CA_Household'],name='sell_price_CA_Household',line_shape='linear'))
fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_CA_Foods'], name="sell_price_CA_Foods",line_shape='linear'))
fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_CA_Hobbies'],name='sell_price_CA_Hobbies',line_shape='linear'))
fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_TX_Household'],name='sell_price_TX_Household',line_shape='linear'))
fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_TX_Foods'],name='sell_price_TX_Foods',line_shape='linear'))
fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_TX_Hobbies'],name='sell_price_TX_Hobbies',line_shape='linear'))
fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_WI_Household'],name='sell_price_WI_Household',line_shape='linear'))
fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_WI_Foods'],name='sell_price_WI_Foods',line_shape='linear'))
fig.add_trace(go.Scatter(x=sell_price_avg['year'], y=sell_price_avg['sell_price_WI_Hobbies'],name='sell_price_WI_Hobbies',line_shape='linear'))

fig.update_traces(hoverinfo='text+name', mode='lines+markers')
fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))

fig.show()


# After checking the values: The foods section is within the range or 3.2 dollars.It has seen a comparative increase over the years.I have a doubt in year 2013 as it is showing up a comparative decrease in each of the products except for Wisonsin food products.
# 
# There has also seen an increase in hobbies section from 2011 to 2016.And each of the trend that is showing up is same across all regions.
# 
# 2012 has also seen an increase in the sell prices across all the regions of products.

# In[ ]:


CA_data_group_HOUSEHOLD.info()


# In[ ]:


CA_data_group_HOUSEHOLD['item_id'].unique()


# In[ ]:


# CA_data_group_HOUSEHOLD['item_id'].unique()
# CA_data_group_FOODS['item_id'].unique() 
# CA_data_group_HOBBIES['item_id'].unique() 
# TX_data_group_HOUSEHOLD['item_id'].unique() 
# TX_data_group_FOODS['item_id'].unique() 
# TX_data_group_HOBBIES['item_id'].unique() 
# WI_data_group_HOUSEHOLD['item_id'].unique() 
# WI_data_group_FOODS['item_id'].unique() 
# WI_data_group_HOBBIES['item_id'].unique() 


# In[ ]:


#Analysis at store_cd level
# def store_level_graphs(store_id,data):
CA_data_group_HOUSEHOLD.head()


# In[ ]:


sell_price_avg_store_id_CA = pd.concat([CA_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_CA_Household'}).groupby(['year','store_id']).mean(),
                  CA_data_group_FOODS.rename(columns={'sell_price':'sell_price_CA_Foods'}).groupby(['year','store_id']).mean(),
                 CA_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_CA_Hobbies'}).groupby(['year','store_id']).mean()], axis=1)
                 
sell_price_avg_store_id_TX=pd.concat([TX_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_TX_Household'}).groupby(['year','store_id']).mean(),
                 TX_data_group_FOODS.rename(columns={'sell_price':'sell_price_TX_Foods'}).groupby(['year','store_id']).mean(),
                 TX_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_TX_Hobbies'}).groupby(['year','store_id']).mean()], axis=1)
sell_price_avg_store_id_WI=pd.concat([
                 WI_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_WI_Household'}).groupby(['year','store_id']).mean(),
                 WI_data_group_FOODS.rename(columns={'sell_price':'sell_price_WI_Foods'}).groupby(['year','store_id']).mean(),
                 WI_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_WI_Hobbies'}).groupby(['year','store_id']).mean()], axis=1)


# In[ ]:


sell_price_avg_store_id_CA=sell_price_avg_store_id_CA.dropna().reset_index()
sell_price_avg_store_id_TX=sell_price_avg_store_id_TX.dropna().reset_index()
sell_price_avg_store_id_WI=sell_price_avg_store_id_WI.dropna().reset_index()


# In[ ]:


fig=px.line(sell_price_avg_store_id_CA, x="year", y="sell_price_CA_Household", color='store_id',title='Average of sell price of Household products across California')
fig.show()
fig=px.line(sell_price_avg_store_id_CA, x="year", y="sell_price_CA_Foods", color='store_id',title='Average of sell price of Food products across California')
fig.show()
fig=px.line(sell_price_avg_store_id_CA, x="year", y="sell_price_CA_Hobbies", color='store_id',title='Average of sell price of Hobbies across California')
fig.show()


# ## Breaking down of analysis on the basis of store_id level
# 
# ### Average of sell price of Household products across California
# 
# As suspected,2012 has seen increase in sell_prices and 2013 has seen a decrease in sales.**CA_4** is comparatively costlier than the rest of store_id.**CA_2** store faired comparatively from 2013 to 2014.**CA_3** had lesser no. of sales during 2012-2013.
# 
# ### Average of sell price of Food products across California
# 
# What interesting in this graph is that **CA_1,CA_3,CA_4** have similar sale numbers.**CA_2** initially is cheaper but from 2013-2015 it has seen an steep increase in sell_prices.
# 
# ### Average of sell price of Hobbies across California
# 
# The trends that I am seeing in other products I don't see the same in Hobbies,It looks like people got interested in hobbies related products over time and for this their sell_prices were increased.

# In[ ]:


fig=px.line(sell_price_avg_store_id_TX, x="year", y="sell_price_TX_Household", color='store_id',title='Average of sell price of Household products across Texas')
fig.show()
fig=px.line(sell_price_avg_store_id_TX, x="year", y="sell_price_TX_Foods", color='store_id',title='Average of sell price of Food products across Texas')
fig.show()
fig=px.line(sell_price_avg_store_id_TX, x="year", y="sell_price_TX_Hobbies", color='store_id',title='Average of sell price of Hobbies across Texas')
fig.show()


# ### Average of sell price of Household products across Texas
# 
# In the end i.e. year 2016 is similar to other values for household products.**TX_2** saw decrease in sell prices from 2011-2013.It has seen then an increase to 2014.
# 
# ### Average of sell price of Food products across Texas
# 
# Fairly increase of food product sell prices over the years.Based on the graphs, all the three stores have similar trends through 5 years of data.
# 
# ### Average of sell price of Hobbies across Texas
# 
# Similar to California, there is similar trend and no decrease during 2012 or 2013 year.

# In[ ]:



fig=px.line(sell_price_avg_store_id_WI, x="year", y="sell_price_WI_Household", color='store_id',title='Average of sell price of Household products across Wisconsin')
fig.show()
fig=px.line(sell_price_avg_store_id_WI, x="year", y="sell_price_WI_Foods", color='store_id',title='Average of sell price of Food products across Wisconsin')
fig.show()
fig=px.line(sell_price_avg_store_id_WI, x="year", y="sell_price_WI_Hobbies", color='store_id',title='Average of sell price of Hobbies across Wisconsin')
fig.show()


# ### Average of sell price of Household products across Wisconsin
# 
# Sell prices of household products saw a steep decrease till 2013 and increased over the years with similar values for products of other regions.
# 
# ### Average of sell price of Food products across Wisconsin
# 
# A linear increase is seen across Wisconsin for all food-based products as compared to California and Texas
# 
# ### Average of sell price of Hobbies across Wisconsin
# 
# Similar trend is same for California,Texas and as compared to Household and Food products it has seen an increase during the years 2012 and 2013.

# In[ ]:


sell_price_avg_item_id_Household = pd.concat([CA_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_CA_Household'}).groupby(['year','item_id']).mean(),
                  TX_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_TX_Household'}).groupby(['year','item_id']).mean(),
                 WI_data_group_HOUSEHOLD.rename(columns={'sell_price':'sell_price_WI_Household'}).groupby(['year','item_id']).mean()], axis=1)

                 
sell_price_avg_item_id_Foods=pd.concat([CA_data_group_FOODS.rename(columns={'sell_price':'sell_price_CA_Foods'}).groupby(['year','item_id']).mean(),
                 TX_data_group_FOODS.rename(columns={'sell_price':'sell_price_TX_Foods'}).groupby(['year','item_id']).mean(),
                 WI_data_group_FOODS.rename(columns={'sell_price':'sell_price_WI_Foods'}).groupby(['year','item_id']).mean()], axis=1)
sell_price_avg_item_id_Hobbies=pd.concat([
                 CA_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_CA_Hobbies'}).groupby(['year','item_id']).mean(),
                 TX_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_TX_Hobbies'}).groupby(['year','item_id']).mean(),
                 WI_data_group_HOBBIES.rename(columns={'sell_price':'sell_price_WI_Hobbies'}).groupby(['year','item_id']).mean()], axis=1)


# In[ ]:


sell_price_avg_item_id_Household=sell_price_avg_item_id_Household.dropna().reset_index()
sell_price_avg_item_id_Foods=sell_price_avg_item_id_Foods.dropna().reset_index()
sell_price_avg_item_id_Hobbies=sell_price_avg_item_id_Hobbies.dropna().reset_index()


# In[ ]:


sell_price_avg_item_id_Household.columns


# In[ ]:


# Taking 3 examples of each of products which have high sell price and low sell price
# sell_price_avg_item_id_Household.sort_values(by=['sell_price_CA_Household','item_id'],ascending=False).head(15)
## Examples : HOUSEHOLD_1_060,HOUSEHOLD_2_446,HOUSEHOLD_1_378
#sell_price_avg_item_id_Household.sort_values(by=['sell_price_CA_Household','item_id']).head(15)
#Examples : HOUSEHOLD_2_371,HOUSEHOLD_1_151,HOUSEHOLD_1_517
#sell_price_avg_item_id_Household.sort_values(by=['sell_price_TX_Household','item_id'],ascending=False).head(15)
# Examples : HOUSEHOLD_1_060, HOUSEHOLD_2_446,HOUSEHOLD_1_378
#sell_price_avg_item_id_Household.sort_values(by=['sell_price_TX_Household','item_id']).head(15)
# Examples : HOUSEHOLD_2_371,HOUSEHOLD_1_151,HOUSEHOLD_1_503
# sell_price_avg_item_id_Household.sort_values(by=['sell_price_WI_Household','item_id'],ascending=False).head(15)
# Examples : HOUSEHOLD_1_060,HOUSEHOLD_2_446,HOUSEHOLD_1_378
#sell_price_avg_item_id_Household.sort_values(by=['sell_price_WI_Household','item_id']).head(15)
# Examples : HOUSEHOLD_2_371,HOUSEHOLD_1_151,HOUSEHOLD_1_517


# Taking 3 examples of each of products which have high sell price and low sell price
# sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_CA_Foods','item_id'],ascending=False).head(15)
#Examples: FOODS_3_298,FOODS_3_083,FOODS_2_239
# sell_price_avg_item_id_Foods.sort_values(by=['sell_price_CA_Foods','item_id']).head(15)
#Examples: FOODS_3_070,FOODS_3_580,FOODS_3_007
#sell_price_avg_item_id_Foods.sort_values(by=['sell_price_TX_Foods','item_id'],ascending=False).head(15)
#Examples: FOODS_3_298,FOODS_3_083,FOODS_2_389
# sell_price_avg_item_id_Foods.sort_values(by=['sell_price_TX_Foods','item_id']).head(15)
#Examples: FOODS_3_454,FOODS_3_007,FOODS_3_580\
# sell_price_avg_item_id_Foods.sort_values(by=['sell_price_WI_Foods','item_id'],ascending=False).head(15)
#Examples: FOODS_3_298,FOODS_3_083,FOODS_2_389
# sell_price_avg_item_id_Foods.sort_values(by=['sell_price_WI_Foods','item_id']).head(15)
#Examples: FOODS_3_547,FOODS_3_547,FOODS_3_547


# Taking 3 examples of each of products which have high sell price and low sell price
# sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_CA_Hobbies','item_id'],ascending=False).head(15)
#Examples: HOBBIES_1_361,HOBBIES_1_225,HOBBIES_1_060
# sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_CA_Hobbies','item_id']).head(15)
#Examples: HOBBIES_2_059,HOBBIES_2_142,HOBBIES_2_124
# sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_TX_Hobbies','item_id'],ascending=False).head(15)
#Examples: HOBBIES_1_410,HOBBIES_1_060,HOBBIES_1_361
#sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_TX_Hobbies','item_id']).head(15)
#Examples: HOBBIES_2_142,HOBBIES_2_129,HOBBIES_2_026
#sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_WI_Hobbies','item_id'],ascending=False).head(15)
#Examples: HOBBIES_1_361,HOBBIES_1_225,HOBBIES_1_060
#sell_price_avg_item_id_Hobbies.sort_values(by=['sell_price_WI_Hobbies','item_id']).head(15)
#Examples: HOBBIES_2_142,HOBBIES_2_129,HOBBIES_2_059


# Because of a large no. of items I took few items which have high sell prices and items which have low sell prices and consolidated them in a graph.Seeing from those examples I was looking for trends.

# In[ ]:


sell_price_avg_item_id_Household_eg=sell_price_avg_item_id_Household[(sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_1_060')|
                                (sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_2_446')|
                                (sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_1_378')|
                                (sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_2_371')|
                                (sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_1_151')|
                                (sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_1_517')|
                                (sell_price_avg_item_id_Household['item_id']=='HOUSEHOLD_1_503')]
sell_price_avg_item_id_Foods_eg=sell_price_avg_item_id_Foods[
    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_298')|
    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_083')|
    (sell_price_avg_item_id_Foods['item_id']=='FOODS_2_239')|
    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_070')|
    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_580')|
    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_007')|
    (sell_price_avg_item_id_Foods['item_id']=='FOODS_2_389')|
    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_454')|
    (sell_price_avg_item_id_Foods['item_id']=='FOODS_3_547')]

sell_price_avg_item_id_Hobbies_eg=sell_price_avg_item_id_Hobbies[
    (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_1_361')|
     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_1_225')|
     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_1_060')|
     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_2_059')|
     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_2_142')|
     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_2_124')|
     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_1_410')|
     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_2_129')|
     (sell_price_avg_item_id_Hobbies['item_id']=='HOBBIES_2_026')
]


# In[ ]:


fig=px.line(sell_price_avg_item_id_Household_eg, x="year", y="sell_price_CA_Household", color='item_id',title='Average of sell price of Household products across California')
fig.show()
fig=px.line(sell_price_avg_item_id_Household_eg, x="year", y="sell_price_TX_Household", color='item_id',title='Average of sell price of Household products across Texas')
fig.show()
fig=px.line(sell_price_avg_item_id_Household_eg, x="year", y="sell_price_WI_Household", color='item_id',title='Average of sell price of Household products across Wisconsin')
fig.show()


# In[ ]:


fig=px.line(sell_price_avg_item_id_Foods_eg, x="year", y="sell_price_CA_Foods", color='item_id',title='Average of sell price of Food products across California')
fig.show()
fig=px.line(sell_price_avg_item_id_Foods_eg, x="year", y="sell_price_TX_Foods", color='item_id',title='Average of sell price of Food products across Texas')
fig.show()
fig=px.line(sell_price_avg_item_id_Foods_eg, x="year", y="sell_price_WI_Foods", color='item_id',title='Average of sell price of Food products across Wisconsin')
fig.show()


# In[ ]:


fig=px.line(sell_price_avg_item_id_Hobbies_eg, x="year", y="sell_price_CA_Hobbies", color='item_id',title='Average of sell price of Hobbies across California')
fig.show()
fig=px.line(sell_price_avg_item_id_Hobbies_eg, x="year", y="sell_price_TX_Hobbies", color='item_id',title='Average of sell price of Hobbies across Texas')
fig.show()
fig=px.line(sell_price_avg_item_id_Hobbies_eg, x="year", y="sell_price_WI_Hobbies", color='item_id',title='Average of sell price of Hobbies across Wisconsin')
fig.show()


# ## Monthly basis analysis
# 
# In this, I will try to find out if the month of the year had any effect on the no. of sales over the years.

# In[ ]:


#Not on basis of region but dividing based on year and then on first on monthly basis
print('Minimum of year {0} and maximum of year {1}'.format(min(sell_prices_date['year']),max(sell_prices_date['year'])))
sell_prices_date_2011=sell_prices_date[sell_prices_date['year']==2011]
sell_prices_date_2012=sell_prices_date[sell_prices_date['year']==2012]
sell_prices_date_2013=sell_prices_date[sell_prices_date['year']==2013]
sell_prices_date_2014=sell_prices_date[sell_prices_date['year']==2014]
sell_prices_date_2015=sell_prices_date[sell_prices_date['year']==2015]
sell_prices_date_2016=sell_prices_date[sell_prices_date['year']==2016]
sell_prices_date_2017=sell_prices_date[sell_prices_date['year']==2017]


# In[ ]:


sell_prices_date_2011_month=sell_prices_date_2011.groupby(by=['month','item_id']).mean().reset_index()
sell_prices_date_2012_month=sell_prices_date_2012.groupby(by=['month','item_id']).mean().reset_index()
sell_prices_date_2013_month=sell_prices_date_2013.groupby(by=['month','item_id']).mean().reset_index()
sell_prices_date_2014_month=sell_prices_date_2014.groupby(by=['month','item_id']).mean().reset_index()
sell_prices_date_2015_month=sell_prices_date_2015.groupby(by=['month','item_id']).mean().reset_index()
sell_prices_date_2016_month=sell_prices_date_2016.groupby(by=['month','item_id']).mean().reset_index()


# In[ ]:


sell_prices_date_2011_month=sell_prices_date_2011_month[['month','item_id','sell_price']]
sell_prices_date_2012_month=sell_prices_date_2012_month[['month','item_id','sell_price']]
sell_prices_date_2013_month=sell_prices_date_2013_month[['month','item_id','sell_price']]
sell_prices_date_2014_month=sell_prices_date_2014_month[['month','item_id','sell_price']]
sell_prices_date_2015_month=sell_prices_date_2015_month[['month','item_id','sell_price']]
sell_prices_date_2016_month=sell_prices_date_2016_month[['month','item_id','sell_price']]


# In[ ]:


def build_graph(data_set,title=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_set['month'], y=data_set['sell_price_Household'],name='sell_price_Household',line_shape='linear'))
    fig.add_trace(go.Scatter(x=data_set['month'], y=data_set['sell_price_Foods'], name="sell_price_Foods",line_shape='linear'))
    fig.add_trace(go.Scatter(x=data_set['month'], y=data_set['sell_price_Hobbies'],name='sell_price_Hobbies',line_shape='linear'))
    fig.update_traces(hoverinfo='text+name', mode='lines+markers')
    fig.update_layout(title=title,xaxis_title='Month',yaxis_title='Sell price',legend=dict(y=0.5, traceorder='reversed', font_size=16))
    
    fig.show()


# ### Year 2011
# 
# Divided on basis of Household,Food and Hobbies for the year 2011

# In[ ]:


sell_prices_date_2011_month_Household=sell_prices_date_2011_month[sell_prices_date_2011_month['item_id'].str.match('HOUSEHOLD')]
sell_prices_date_2011_month_Foods=sell_prices_date_2011_month[sell_prices_date_2011_month['item_id'].str.match('FOODS')]
sell_prices_date_2011_month_Hobbies=sell_prices_date_2011_month[sell_prices_date_2011_month['item_id'].str.match('HOBBIES')]


# In[ ]:


sell_prices_date_2011_month=pd.concat([sell_prices_date_2011_month_Household.rename(columns={'sell_price':'sell_price_Household'}).groupby(['month']).mean(),
                                      sell_prices_date_2011_month_Foods.rename(columns={'sell_price':'sell_price_Foods'}).groupby(['month']).mean(),
                                      sell_prices_date_2011_month_Hobbies.rename(columns={'sell_price':'sell_price_Hobbies'}).groupby(['month']).mean()],axis=1).reset_index()


# In[ ]:


build_graph(sell_prices_date_2011_month,'Sell prices for Year 2011')


# Points to Note -
#     
#        1.Foods are cheaper than any other amenities like Household and Hobbies.
#        2.April Month has seen an increase in hobbies prices so I am thinking people would be interested in taking up their hobbies during this period.

# ### Year 2012
# 
# Divided on basis of Household,Food and Hobbies for the year 2012

# In[ ]:


sell_prices_date_2012_month_Household=sell_prices_date_2012_month[sell_prices_date_2012_month['item_id'].str.match('HOUSEHOLD')]
sell_prices_date_2012_month_Foods=sell_prices_date_2012_month[sell_prices_date_2012_month['item_id'].str.match('FOODS')]
sell_prices_date_2012_month_Hobbies=sell_prices_date_2012_month[sell_prices_date_2012_month['item_id'].str.match('HOBBIES')]


# In[ ]:


sell_prices_date_2012_month=pd.concat([sell_prices_date_2012_month_Household.rename(columns={'sell_price':'sell_price_Household'}).groupby(['month']).mean(),
                                      sell_prices_date_2012_month_Foods.rename(columns={'sell_price':'sell_price_Foods'}).groupby(['month']).mean(),
                                      sell_prices_date_2012_month_Hobbies.rename(columns={'sell_price':'sell_price_Hobbies'}).groupby(['month']).mean()],axis=1).reset_index()


# In[ ]:


build_graph(sell_prices_date_2012_month,'Sell prices for Year 2012')


# Points to Note:
#     
#     1. Food still remains the cheaper item than Hobbies or household items
#     2. Hobbies see an increase over their sales from May-June.

# ### Year 2013
# 
# Trend of sales for the year 2013 across all items

# In[ ]:


sell_prices_date_2013_month_Household=sell_prices_date_2013_month[sell_prices_date_2013_month['item_id'].str.match('HOUSEHOLD')]
sell_prices_date_2013_month_Foods=sell_prices_date_2013_month[sell_prices_date_2013_month['item_id'].str.match('FOODS')]
sell_prices_date_2013_month_Hobbies=sell_prices_date_2013_month[sell_prices_date_2013_month['item_id'].str.match('HOBBIES')]


# In[ ]:


sell_prices_date_2013_month=pd.concat([sell_prices_date_2013_month_Household.rename(columns={'sell_price':'sell_price_Household'}).groupby(['month']).mean(),
                                      sell_prices_date_2013_month_Foods.rename(columns={'sell_price':'sell_price_Foods'}).groupby(['month']).mean(),
                                      sell_prices_date_2013_month_Hobbies.rename(columns={'sell_price':'sell_price_Hobbies'}).groupby(['month']).mean()],axis=1).reset_index()


# In[ ]:


build_graph(sell_prices_date_2013_month,'Sell prices for Year 2013')


# Points to Note:
# 
#     1. Food sales are quite constant for the year 2013.
#     2. Whats interesting in this graph is the Hobbies trend sell price increased for the month from 7-8.

# In[ ]:


sell_prices_date_2014_month_Household=sell_prices_date_2014_month[sell_prices_date_2014_month['item_id'].str.match('HOUSEHOLD')]
sell_prices_date_2014_month_Foods=sell_prices_date_2014_month[sell_prices_date_2014_month['item_id'].str.match('FOODS')]
sell_prices_date_2014_month_Hobbies=sell_prices_date_2014_month[sell_prices_date_2014_month['item_id'].str.match('HOBBIES')]


# In[ ]:


sell_prices_date_2014_month=pd.concat([sell_prices_date_2014_month_Household.rename(columns={'sell_price':'sell_price_Household'}).groupby(['month']).mean(),
                                      sell_prices_date_2014_month_Foods.rename(columns={'sell_price':'sell_price_Foods'}).groupby(['month']).mean(),
                                      sell_prices_date_2014_month_Hobbies.rename(columns={'sell_price':'sell_price_Hobbies'}).groupby(['month']).mean()],axis=1).reset_index()


# In[ ]:


build_graph(sell_prices_date_2014_month,'Sell prices for Year 2014')


# Points to Note:
# 
#     1.Not a trending feature for the year 2014
#     2.Only thing we can infer that the hobbies price has increased than a household items.We can understand that people are preferring Hobbies now.

# ### Year 2015
# 
# 1. Trends to check for the year 2015

# In[ ]:


sell_prices_date_2015_month_Household=sell_prices_date_2015_month[sell_prices_date_2015_month['item_id'].str.match('HOUSEHOLD')]
sell_prices_date_2015_month_Foods=sell_prices_date_2015_month[sell_prices_date_2015_month['item_id'].str.match('FOODS')]
sell_prices_date_2015_month_Hobbies=sell_prices_date_2015_month[sell_prices_date_2015_month['item_id'].str.match('HOBBIES')]

sell_prices_date_2015_month=pd.concat([sell_prices_date_2015_month_Household.rename(columns={'sell_price':'sell_price_Household'}).groupby(['month']).mean(),
                                      sell_prices_date_2015_month_Foods.rename(columns={'sell_price':'sell_price_Foods'}).groupby(['month']).mean(),
                                      sell_prices_date_2015_month_Hobbies.rename(columns={'sell_price':'sell_price_Hobbies'}).groupby(['month']).mean()],axis=1).reset_index()



build_graph(sell_prices_date_2015_month,'Sell prices for Year 2015')


# Points to Note:
# 
#     1. No exciting trends .It seems that all the prices have reached a stagnant version of each other.Or I prefer to call it a saturated version.

# ### Year 2016
# 
# Checking trends for year 2016

# In[ ]:


sell_prices_date_2016_month_Household=sell_prices_date_2016_month[sell_prices_date_2016_month['item_id'].str.match('HOUSEHOLD')]
sell_prices_date_2016_month_Foods=sell_prices_date_2016_month[sell_prices_date_2016_month['item_id'].str.match('FOODS')]
sell_prices_date_2016_month_Hobbies=sell_prices_date_2016_month[sell_prices_date_2016_month['item_id'].str.match('HOBBIES')]

sell_prices_date_2016_month=pd.concat([sell_prices_date_2016_month_Household.rename(columns={'sell_price':'sell_price_Household'}).groupby(['month']).mean(),
                                      sell_prices_date_2016_month_Foods.rename(columns={'sell_price':'sell_price_Foods'}).groupby(['month']).mean(),
                                      sell_prices_date_2016_month_Hobbies.rename(columns={'sell_price':'sell_price_Hobbies'}).groupby(['month']).mean()],axis=1).reset_index()



build_graph(sell_prices_date_2016_month,'Sell prices for Year 2016')


# Points to Note: 
# 
#     1. Data is available till June month.
#     2. Not much trend to look out for.

# In[ ]:


submission=pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')
submission.shape


# In[ ]:


submission.tail()


# In[ ]:




