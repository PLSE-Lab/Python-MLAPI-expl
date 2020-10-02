#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from fbprophet import Prophet
import operator

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Let us assume that we would like to invest on avacados in certain region.Assuming profit for conventional avacodos is 15% while profit on organic is 45% which region is best place to invest in next 18 months(78 weeks). The decision to invest based on total profitable market in that region.
# 
# In the process look at the following 
# Which is the best region to invest and what type ?
# Do a comparision between total US vs the top region on following terms.
# 1. Is saleprice related to total volume ?
# 1.  Is there a general trend in increase in avacodos across US ?
# 1. Is total Volume seasional 

# Get to know more about data from following terms.
# 
#     * Is there null data ?
#     * What are various columns ?
#     * Drop any columns that are not necessary for now.
#     * Data in days/weeks ?. How many years ?
#     * How many regions and what are they ?
#     * How many unique type and what are they ?

# In[ ]:


df = pd.read_csv("../input/avocado.csv", parse_dates=['Date'])
train = df.copy() #Make a copy for a safe case
train.isnull().sum()


# There is no null in the data. We will explore other aspects such as
# 1. For how many years does the data is available ?
# 1. What are the various types of avacados ?
# 1. For what regions ?

# In[ ]:


print(train.type.unique())
print(train.year.unique())
print(train.region.unique())


#     * There are 2 types of avacado's: Organic and Conventional.
#     * Data is available for 4 yrs.
#     * There are many regions. Let's checkout the length.

# In[ ]:


len(train.region.unique())


# * Before getting on to doing our predictions. Let us add 2 new columns. 
# * **Revenue:** It is TotalVolume * AveragePrice
# * **Profit:** Divided in 2 parts. We have assumed that conventional fetches 15% profit of revenue while organic fetchs 45% of revenue.

# In[ ]:


train['revenue'] = train['Total Volume'] * train['AveragePrice']
train.loc[train.type == "conventional", 'profit'] = (train["revenue"] * 15 ) / 100
train.loc[train.type == "organic", 'profit'] = (train["revenue"] * 45 ) / 100


#     prophet() is helper function that is used across this kernel that uses FB's Prophet library to predict

# In[ ]:


def prophet(df_formatted, periods, draw=False):
    prop = Prophet()
    prop.fit(df_formatted)
    future_prop = prop.make_future_dataframe(periods=periods)
    forecast_prop = prop.predict(future_prop)
    if (draw == True):
        fig1_prop = prop.plot(forecast_prop)
        fig2_prop = prop.plot_components(forecast_prop)
    return forecast_prop 


#     First let's look at how avacado trends and forecast look for Total US

# In[ ]:


df_TotalUS = train[train.region == 'TotalUS']
for type in ("organic", "conventional"):
    df_type = df_TotalUS[df_TotalUS.type == type]
    df_profit  = df_type[['Date', 'profit']]
    formatted_profit = df_profit.rename(columns={'Date':'ds', 'profit':'y'})
    forecast_profit = prophet(formatted_profit, 78, draw=True)
    
plt.show()


#     For Total US case both Organic and Conventional shows upward trend.
#     Let's predict which region having types(Organic or Conventional) provides max profit for next 18 months.
#     Get top 5 region's with type.

# In[ ]:


profit_by_region_and_type = {}

def get_profit_by_region_and_type(df, region):
    df_region = df[df.region == region]
    for type in ("organic", "conventional"):
        df_type = df_region[df_region.type == type]
        df_profit  = df_type[['Date', 'profit']]
        formatted_profit = df_profit.rename(columns={'Date':'ds', 'profit':'y'})
        forecast_profit = prophet(formatted_profit, 78)
        yhat_sum = forecast_profit.tail(78).yhat.sum()
        region_type_str = region + "_" + type
        profit_by_region_and_type[region_type_str] = yhat_sum

for region in train.region.unique():
    get_profit_by_region_and_type(train, region)

value_key = ((value, key) for (key,value) in profit_by_region_and_type.items())
sorted_value_key = sorted(value_key, reverse=True)
df_profit_net = pd.DataFrame(sorted_value_key, columns=["Total Profit", "RegionAndType"])


#     Dropping the first two because they are for US Total for each Organic and conventional.
#     Coventional avacados in California region has largest market in US. So, investing in California would make sense (Note: There would be already existing players)

# In[ ]:


df_top_five = df_profit_net[2:7]
df_top_five


#     Let's start answering other questions
#     Comparision between total US vs the top region (which is California) from following aspects
#         - Is saleprice related to total volume ?
#         - Is there a general trend in increase in avacodos consumption (total volume) across US for both conventional and organic ?
#         - Is total Volume seasional 

#     Is Saleprice related with total volume for both Total US and California ?.
#     No. The correlation values does tell much.

# In[ ]:


df_totalUS_Cal = df_profit_net.loc[0:2]
df_totalUS_Cal = df_totalUS_Cal.drop(axis=0, index=1)

for regionType in df_totalUS_Cal["RegionAndType"]:
    region = regionType.split("_")[0]
    type = regionType.split("_")[1]
    df_region_type = train[(train.type == type) & (train.region == region)] 
    df_corr = df_region_type.corr()
    df_corr_total_volume = df_corr['Total Volume'][0]
    print(regionType, df_corr_total_volume)


#     Is there similar trend in between Total US and California for Conventional Avacados ?
#     No. Interestingly, The trend for Total US is upward while for California is dropping down.

# In[ ]:


df_California = train[(train.region == "California")]

for dataset in (df_TotalUS, df_California):
    df_type = dataset[dataset.type == "conventional"]
    df_vol  = df_type[['Date', 'Total Volume']]
    formatted_vol = df_vol.rename(columns={'Date':'ds', 'Total Volume':'y'})
    forecast_vol = prophet(formatted_vol, 78, draw=True)


#     Since we have decided on California. Let's dig more
#     First let us check how profit are distributed for Organic and Conventional in California.
#     Following boxplot confirms that profit predicted for conventional avacados in california is on higher side.

# In[ ]:


sns.boxplot(y="type", x="profit", data=df_California)


#     We have not answered one more question: Is volume seasional ?. Yes the column [35] shows volume is seasional for both Total US and California.

# This is first Kernel. Any feedback is appreciated.
