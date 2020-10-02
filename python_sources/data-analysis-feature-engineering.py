#!/usr/bin/env python
# coding: utf-8

# # Data Visualization Notebook
# 
# 
# Here is table of contents in this notebook:
# - [Import Libraries and Data Input](#Import-Libraries-and-Data-Input)
# - [Data Cleaning](#Data-Cleaning)
# - [Data Visualization](#Data-Visualization)
#    -  [Total Item Sold Transition](#Total-Item-Sold-Transition)
#    -  [Item Sold in each day type](#Item-Sold-in-each-day-type)
#    -  [Item sold in each State and Store](#Item-sold-in-each-State-and-Store)
#    -  [Item Sold relation Analysis](#Item-Sold-relation-Analysis)
#    -  [Store Analysis](#Store-Analysis)
#    -  [Snap Purchase Analysis](#Snap-Purchase-Analysis)
#    -  [Event Pattern Analysis](#Event-Pattern-Analysis)
#    -  [One Item Features Analysis](#One-Item-Features-Analysis)
#    -  [Sell Price Analysis](#Sell-Price-Analysis)
#    -  [Sell price and value relationship](#Sell-price-and-value-relationship)
#    -  [Relationship of Lag Variables](#Relationship-of-Lag-Variables)
#    -  [PCA Trial](#PCA-Trial)
#   
# - [Summary](#Summary)
# - [Future Work](#Future-Work)
# - [References](#References)

# # Import Libraries and Data Input

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import gc
import lightgbm as lgb
import time
# import datetime
# import xgboost as xgb
# import time
# import itertools
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[ ]:


INPUT_DIR = '/kaggle/input/m5-forecasting-accuracy'

calendar_df = pd.read_csv(f"{INPUT_DIR}/calendar.csv")
sell_prices_df = pd.read_csv(f"{INPUT_DIR}/sell_prices.csv")
sales_train_validation_df = pd.read_csv(f"{INPUT_DIR}/sales_train_validation.csv")
sample_submission_df = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")


# In[ ]:


# Calendar data type cast -> Memory Usage Reduction
calendar_df[["month", "snap_CA", "snap_TX", "snap_WI", "wday"]] = calendar_df[["month", "snap_CA", "snap_TX", "snap_WI", "wday"]].astype("int8")
calendar_df[["wm_yr_wk", "year"]] = calendar_df[["wm_yr_wk", "year"]].astype("int16") 
calendar_df["date"] = calendar_df["date"].astype("datetime64")

nan_features = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
for feature in nan_features:
    calendar_df[feature].fillna('unknown', inplace = True)

calendar_df[["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]] = calendar_df[["weekday", "event_name_1", "event_type_1", "event_name_2", "event_type_2"]] .astype("category")


# In[ ]:


calendar_df.head()


# In[ ]:


sales_train_validation_df.head()


# In[ ]:


# Sales Training dataset cast -> Memory Usage Reduction
sales_train_validation_df.loc[:, "d_1":] = sales_train_validation_df.loc[:, "d_1":].astype("int16")


# In[ ]:


# Make ID column to sell_price dataframe
sell_prices_df.loc[:, "id"] = sell_prices_df.loc[:, "item_id"] + "_" + sell_prices_df.loc[:, "store_id"] + "_validation"


# In[ ]:


sell_prices_df = pd.concat([sell_prices_df, sell_prices_df["item_id"].str.split("_", expand=True)], axis=1)
sell_prices_df = sell_prices_df.rename(columns={0:"cat_id", 1:"dept_id"})
sell_prices_df[["store_id", "item_id", "cat_id", "dept_id"]] = sell_prices_df[["store_id","item_id", "cat_id", "dept_id"]].astype("category")
sell_prices_df = sell_prices_df.drop(columns=2)


# # Data Cleaning
# First, let's combine all three dataframe.  
# The important thing is changing data format from wide to long to make prediction model easier  
# (Though this notebook doesn't dive into predicition model itself.)
# 
# 

# In[ ]:


def make_dataframe():
    # Wide format dataset 
    df_wide_train = sales_train_validation_df.drop(columns=["item_id", "dept_id", "cat_id", "state_id","store_id", "id"]).T
    df_wide_train.index = calendar_df["date"][:1913]
    df_wide_train.columns = sales_train_validation_df["id"]
    
    # Making test label dataset
    df_wide_test = pd.DataFrame(np.zeros(shape=(56, len(df_wide_train.columns))), index=calendar_df.date[1913:], columns=df_wide_train.columns)
    df_wide = pd.concat([df_wide_train, df_wide_test])

    # Convert wide format to long format
    df_long = df_wide.stack().reset_index(1)
    df_long.columns = ["id", "value"]

    del df_wide_train, df_wide_test, df_wide
    gc.collect()
    
    df = pd.merge(pd.merge(df_long.reset_index(), calendar_df, on="date"), sell_prices_df, on=["id", "wm_yr_wk"])
    df = df.drop(columns=["d"])
#     df[["cat_id", "store_id", "item_id", "id", "dept_id"]] = df[["cat_id"", store_id", "item_id", "id", "dept_id"]].astype("category")
    df["sell_price"] = df["sell_price"].astype("float16")   
    df["value"] = df["value"].astype("int32")
    df["state_id"] = df["store_id"].str[:2].astype("category")


    del df_long
    gc.collect()

    return df

df = make_dataframe()


# In[ ]:


df.dtypes


# In[ ]:


def add_date_feature(df):
    df["year"] = df["date"].dt.year.astype("int16")
    df["month"] = df["date"].dt.month.astype("int8")
    df["week"] = df["date"].dt.week.astype("int8")
    df["day"] = df["date"].dt.day.astype("int8")
    df["quarter"]  = df["date"].dt.quarter.astype("int8")
    return df


# In[ ]:


df = add_date_feature(df)
df.head()


# # Data Visualization
# ## Total Item Sold Transition

# In[ ]:


temp_series = df.groupby(["cat_id", "date"])["value"].sum()
temp_series


# In[ ]:


plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")
plt.xlabel("Year")
plt.ylabel("# of sold items")
plt.title("Total Item Sold Transition of each Category")
plt.legend()


# **Point of the graph**
# 1. FOODS is the most sold item category of these three categories.  
#    HOUSEHOLD is the 2nd one, and HOBBIES are the least sold one.  
# 
# 
# 2. FOODS category appearently has some periodical feature.   
#    During one year, it seems more items are sold in summer than in winter, however, we have to verify this.  
#    As for more short time interval, it seems the trend has monthly or weekly features. (Let's take a look below)
# 
# 3. HOUSEHOLD category items sold is gradually increasing from 2011.  
#    However, it may be because some items are not in the store in 2011.  
#    So we have to take the total item in the store into account.
#    Periodical Features are not so clear in this category compared to FOODS.
#   
# 4. In HOBBIES category, periodical features are less appearent like HOUSEHOLD category.
# 
# 5. In some point (around the end of year), all categories don't have any sold.  So I think we have to consider whether we take these days into account when training models.
# 
# So let's take a look at the latest year, 2015!

# In[ ]:


temp_series = temp_series.loc[temp_series.index.get_level_values("date") >= "2015-01-01"]
plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")
plt.xlabel("Year-Month")
plt.ylabel("# of sold items")
plt.title("Total Item Sold Transition of each Category from 2015")
plt.legend()


# 1. In all categories, the periodical trends is seemed weekly.  
#    In previous graph, we can't easily recognize that HOUSEHOLD and HOBBIES have weekly features, but in this graph we can.
#    
# 2. The day when all item sold is 0 is seemed to be Christmas Day, not new year's day, confirm it below.

# In[ ]:


# Plot only December, 2015
temp_series = temp_series.loc[(temp_series.index.get_level_values("date") >= "2015-12-01") & (temp_series.index.get_level_values("date") <= "2015-12-31")]
plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")
plt.plot(temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")
plt.xlabel("Year")
plt.ylabel("# of sold items")
plt.title("Total sold item per day in December, 2015")
plt.legend()


# On Christmas Day, the items sold are seemed to be 0, let's check it with *.loc* method

# In[ ]:


temp_series.loc[(temp_series.index.get_level_values("date") >= "2015-12-24") & (temp_series.index.get_level_values("date") <= "2015-12-26")]


# Some items are sold even on Christmas Day, but I think these are completely noisy values.   
# 
# Until now, we can find the items sold have something weekly fetures. So let's think this:   
# **Next Question: Which day of the week is the items sold most?**

# ## Item Sold in each day type

# In[ ]:


temp_series = df.groupby(["cat_id", "wday"])["value"].sum()
temp_series


# In[ ]:


plt.figure(figsize=(6, 4))
left = np.arange(1,8) 
width = 0.3
weeklabel = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]    # Please Confirm df


plt.bar(left, temp_series[temp_series.index.get_level_values("cat_id") == "FOODS"].values, width=width, label="FOODS")
plt.bar(left + width, temp_series[temp_series.index.get_level_values("cat_id") == "HOUSEHOLD"].values, width=width, label="HOUSEHOLD")
plt.bar(left + width + width, temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].values, width=width, label="HOBBIES")
plt.legend(bbox_to_anchor=(1.01, 1.01))
plt.xticks(left, weeklabel, rotation=60)
plt.xlabel("day of week")
plt.ylabel("# of sold items")
plt.title("Total sold item in each daytype")


# 1. As we can probraly guess, Saturday or Sunday is the day which the items are most sold.  
#    Tuesday or Wednesday is the least sold days.  
#    -> Later, we visualize these correlation factors with heatmap. Looking forward to it!
# 
# 2. HOBBIES are not so day dependent compared to FOODS or HOUSEHOLD.  

# ## Item sold in each State and Store

# In[ ]:


temp_series = df.groupby(["state_id", "date"])["value"].sum()
temp_series


# In[ ]:


plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("state_id") == "CA"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("state_id") == "CA"].values, label="CA")
plt.plot(temp_series[temp_series.index.get_level_values("state_id") == "TX"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("state_id") == "TX"].values, label="TX")
plt.plot(temp_series[temp_series.index.get_level_values("state_id") == "WI"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("state_id") == "WI"].values, label="WI")
plt.xlabel("Year")
plt.ylabel("# of sold items")
plt.title("Total Item Sold Transition of each State")
plt.legend()


# 1. CA is the most sold state of these three states.  
#    TX and WI are not so different except for the year 2011 and 2012.
#   
# 2. All three states have some periodical features as we've already seen in category-based item sold graph. 
# 
# First, let's focus on stores in CA.

# In[ ]:


temp_series = df.groupby(["store_id", "date"])["value"].sum()

plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_1"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_1"].values, label="CA_1")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_2"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_2"].values, label="CA_2")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_3"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_3"].values, label="CA_3")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_4"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_4"].values, label="CA_4")

plt.xlabel("Year")
plt.ylabel("# of sold items")
plt.title("Total Item Sold Transition of each Store in CA")
plt.legend()


# 1. Three stores in CA have similar amount of item sold record.  
#    CA_3 has more item sold a little bit compared to others.  
# 
# 2. The standard deviation of each store seems different, confirm it later.
# 
# 3. From around 2015 Spring or Summer, CA_2 increased its sold record rapidly. We have to investigate the reasons.
# 
# 

# In[ ]:


temp_series = df.groupby(["store_id", "date"])["item_id"].count()

plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_1"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_1"].values, label="CA_1")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_2"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_2"].values, label="CA_2")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_3"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_3"].values, label="CA_3")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_4"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_4"].values, label="CA_4")
plt.xlabel("Year")
plt.ylabel("# of item entries")
plt.title("Total item entries in each CA stores")
plt.legend()


# 3. From around 2015 Spring or Summer, CA_2 increased its sold record rapidly. We have to investigate the reasons.
# 
# -> It is because item registered in CA_2 increased rapidly.  
# After summer in 2015, all stores in CA have similar registered item count

# In[ ]:


temp_series = df.groupby(["store_id", "date"])["value"].std()
temp_series


# In[ ]:


plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_1"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_1"].values, label="CA_1")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_2"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_2"].values, label="CA_2")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_3"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_3"].values, label="CA_3")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "CA_4"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "CA_4"].values, label="CA_4")

plt.xlabel("Year")
plt.ylabel("Standard deviation of sold items")
plt.title("Standard deviation of sold items in CA stores")
plt.legend()


# 1. Since CA_3 is the most sold store in CA, standard deviation of this store is also higher than others.  
#    Expecially, around the end of 2011, sold item deviation gets higher than usual. 
#    
#  
# Let's check other state, WI next!

# In[ ]:


temp_series = df.groupby(["store_id", "date"])["value"].sum()

plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "WI_1"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "WI_1"].values, label="WI_1")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "WI_2"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "WI_2"].values, label="WI_2")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "WI_3"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "WI_3"].values, label="WI_3")
plt.xlabel("Year")
plt.ylabel("# of sold items")
plt.title("Total item sold in each WI stores")
plt.legend()


# 1. Stores in WI have similar item sold count.  
#    Before 2013, WI_3 is the most sold store in WI, but WI_2 gradually increases its proportion. (Especially around summer in 2012)
#    
# 2. In some point, WI_1 rapidly increase its sold item count. (Around on November 2012)
# 
# -> Total Sold out count depends on the number of entries at that day.  So Now we check the total item entries in each store as we did in CA stores.

# In[ ]:


temp_series = df.groupby(["store_id", "date"])["item_id"].count()

plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "WI_1"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "WI_1"].values, label="WI_1")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "WI_2"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "WI_2"].values, label="WI_2")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "WI_3"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "WI_3"].values, label="WI_3")
plt.xlabel("Year")
plt.ylabel("# of item entries")
plt.title("Total item entries in each WI stores")
plt.legend()


# As we've already seen above, the registered item count trends are different in each store.  
# WI_2 increased its item register around summer in 2012, then WI_3 increased around November of that year.  
# 
# From 2013, all stores have similar trend.  Next, stores in TX states!

# In[ ]:


temp_series = df.groupby(["store_id", "date"])["value"].sum()

plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "TX_1"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "TX_1"].values, label="TX_1")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "TX_2"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "TX_2"].values, label="TX_2")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "TX_3"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "TX_3"].values, label="TX_3")
plt.xlabel("Year")
plt.ylabel("Total sold item per day")
plt.title("Total item sold in each TX stores")
plt.legend()


# 1. Oops, in 2015, it seems some extreme points exist.  
#    For exmaple, around Febrary, TX_2 has almost 0 item sold. (I assume this store is closed exceptionally.)
#    In contrast, in one summer day of that year, TX_3 increased its total item sold exprosively.
#    
# 2. TX_2 has most item sold especially before 2014.

# In[ ]:


temp_series = df.groupby(["store_id", "date"])["item_id"].count()

plt.figure(figsize=(12, 4))
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "TX_1"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "TX_1"].values, label="TX_1")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "TX_2"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "TX_2"].values, label="TX_2")
plt.plot(temp_series[temp_series.index.get_level_values("store_id") == "TX_3"].index.get_level_values("date"), temp_series[temp_series.index.get_level_values("store_id") == "TX_3"].values, label="TX_3")
plt.xlabel("Year")
plt.ylabel("Total item entries")
plt.title("Total item entries in each TX stores")
plt.legend()


# Compared to other states, TX stores have similar tendency regarding registered entries.

# In[ ]:


temp_series = df.groupby(["store_id", "date"])["value"].sum()
temp_series


# In[ ]:


# Find the day when items are sold less than 1000 of each store
# Let's take a look at TX_2 for example
temp_series.loc[(temp_series.values < 1000) & (temp_series.index.get_level_values("date") <= "2016-04-22")].loc["TX_2"]


# 1. Oops, in 2015, it seems some extreme points exist.  
#    For exmaple, around Febrary, TX_2 has almost 0 item sold. (I assume this store is closed exceptionally.)
#    
#    -> On 2015-03-24, TX_2 has very little item sold. 

# In[ ]:


# Find the day when items are sold most of each store
temp_series.groupby(["store_id"]).idxmax()


# In[ ]:


temp_series = temp_series.reset_index()
temp_series


# In[ ]:


plt.plot(temp_series[(temp_series["store_id"] == "CA_1") & ((temp_series["date"] >= "2013-07-15") & (temp_series["date"] <= "2013-10-15"))]["date"],
         temp_series[(temp_series["store_id"] == "CA_1") & ((temp_series["date"] >= "2013-07-15") & (temp_series["date"] <= "2013-10-15"))]["value"])
plt.xticks(rotation=60)
plt.ylabel("# of sold items")
plt.xlabel("date")
plt.title("Item sold transition around its most sold day in CA_1 store")


# # Item Sold relation Analysis
# Under Construction...  
# (I tried to apply Dynamic Factor Analysis and execute the codes of this tutorial, but it seems I couldn't get informative outcome:
# https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_dfm_coincident.html
# 
# To Whoever can understand this method more specifically, I appreciate your comments.

# ## Apply Dynamic Factor Analysis Trial

# In[ ]:


# import statsmodels.api as sm


# In[ ]:


# item_id = "HOBBIES_1_008"
# temp_df = df.loc[df.item_id == item_id, ["date","store_id", "value"]]


# In[ ]:


# store_list = ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2"]

# # temp_df.drop(columns="sell_price", inplace=True)
# temp_df_wide = pd.pivot_table(temp_df, index='date', columns='store_id', values="value")
# temp_df_wide.plot(figsize=(12, 4))
# plt.legend(bbox_to_anchor=(1.01, 1.01))


# In[ ]:


# diff_cols = ["diff_" + store for store in store_list]

# for store in store_list:
#     col = "diff_" + store
#     temp_df_wide.columns = temp_df_wide.columns.add_categories(col)
#     temp_df_wide[col] = np.log(temp_df_wide[store] + 0.1).diff() * 100
    
#     std_col = "std_" + col
    
#     temp_df_wide.columns = temp_df_wide.columns.add_categories(std_col)
#     temp_df_wide[std_col] = (temp_df_wide[col] - temp_df_wide[col].mean()) / temp_df_wide[col].std()


# In[ ]:


# std_cols = ["std_diff_" + store for store in store_list]


# In[ ]:


# endog = temp_df_wide.loc[:, std_cols]

# # Create the model
# mod = sm.tsa.DynamicFactor(endog, k_factors=1, factor_order=3, error_order=3)
# initial_res = mod.fit(method='powell', disp=False)
# res = mod.fit(initial_res.params, disp=False)


# In[ ]:


# print(res.summary(separate_params=False))


# In[ ]:


# from pandas_datareader.data import DataReader

# fig, ax = plt.subplots(figsize=(13,3))

# # Plot the factor
# dates = endog.index._mpl_repr()
# ax.plot(dates, res.factors.filtered[0], label='Factor')
# ax.legend()

# # Retrieve and also plot the NBER recession indicators
# rec = DataReader('USREC', 'fred', start=temp_df_wide.index.min(), end=temp_df_wide.index.max())
# ylim = ax.get_ylim()
# ax.fill_between(dates[:-3], ylim[0], ylim[1], rec.values[:-4,0], facecolor='k', alpha=0.1);


# In[ ]:


# This doesn't seem make sense.
# res.plot_coefficients_of_determination(figsize=(8,2));


# # Store Analysis

# In[ ]:


temp_series = df.groupby(["store_id", "cat_id"])["value"].sum()


# In[ ]:


store_id_list_by_state = [["CA_1", "CA_2", "CA_3", "CA_4"], ["TX_1", "TX_2", "TX_3"], ["WI_1", "WI_2", "WI_3"]] 


# In[ ]:


fig, axs = plt.subplots(3, 4, figsize=(16, 12), sharey=True) 

for row in range(len(store_id_list_by_state)):
    for col in range(len(store_id_list_by_state[row])):
        axs[row, col].bar(x=temp_series[temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col]].index.get_level_values("cat_id"),
                          height=temp_series[temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col]].values,
                         color=["orange", "green", "blue"], label=["FOODS", "HOBBIES", "HOUSEHOLD"])
        axs[row, col].set_title(store_id_list_by_state[row][col])
        axs[row, col].set_ylabel("# of items")

fig.suptitle("Each category item sold in each store")


# In[ ]:


fig, axs = plt.subplots(3, 4, figsize=(16, 12), sharey=True) 

for row in range(len(store_id_list_by_state)):
    for col in range(len(store_id_list_by_state[row])):
        axs[row, col].bar(x=temp_series[temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col]].index.get_level_values("cat_id"),
                          height=temp_series[temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col]].values / temp_series[temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col]].sum(),
                         color=["orange", "green", "blue"], label=["FOODS", "HOBBIES", "HOUSEHOLD"])
        axs[row, col].set_title(store_id_list_by_state[row][col])
        axs[row, col].set_ylabel("% of each category")

fig.suptitle("Each category item sold percentage in each store")


# In[ ]:


cat_id = "FOODS"

temp_series = df.groupby(["store_id", "cat_id", "wday"])["value"].sum()
temp_series = temp_series[temp_series.index.get_level_values("cat_id") == cat_id]
temp_series


# In[ ]:


weekday = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]


# In[ ]:


# Combine all these three figures.
cat_list = ["FOODS", "HOBBIES", "HOUSEHOLD"]
color_list = ["orange", "green", "blue"]
temp_series = df.groupby(["store_id", "cat_id", "wday"])["value"].sum()
width = 0.25

fig, axs = plt.subplots(3, 4, figsize=(20, 12), sharey=True) 

for row in range(len(store_id_list_by_state)):
    for col in range(len(store_id_list_by_state[row])):
        for i, cat in enumerate(cat_list):
            height_numerator = temp_series[(temp_series.index.get_level_values("cat_id") == cat) & (temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col])].values
            height_denominater = height_numerator.sum()

            axs[row, col].bar(x=temp_series[(temp_series.index.get_level_values("cat_id") == cat) & (temp_series.index.get_level_values("store_id") == store_id_list_by_state[row][col])].index.get_level_values("wday") + width * (i-1),
                              height=height_numerator / height_denominater,
                             tick_label=weekday, color=color_list[i], width=width, label=cat)
            axs[row, col].set_title(store_id_list_by_state[row][col])
            axs[row, col].legend()
            
fig.suptitle("HOBBIES item sold in each store in each day")


# # Snap Purchase Analysis
# Let's see how snap purchase allowed day is distributed.

# In[ ]:


fig, axs = plt.subplots(1, 3, sharey=True)
fig.suptitle("Snap Purchase Enable Day Count of each store")

sns.countplot(x="snap_CA", data =calendar_df, ax=axs[0])
sns.countplot(x="snap_TX", data =calendar_df, ax=axs[1])
sns.countplot(x="snap_WI", data =calendar_df, ax=axs[2])


# 1. OK, the total count of snap purchase enable day looks similar in these three stores.
# 
# 2. The total count of snap purchase enable day is about one half of that of non-enable day.
# 
# Next Let's see whether snap purchase is how-distributed in one year.

# In[ ]:


temp_df = calendar_df.groupby(["year"])[["snap_CA", "snap_TX", "snap_WI"]].sum()
temp_df


# 1. OK. Total snap purchase allowed day of each state is the same in all years.
# 
# 2. From 2011 to 2015, there are about 120 days when snap purchase is allowed.  
#    (As for 2016, we only have the first half of whole year.)

# In[ ]:


# This cell is just visuallizing the above dataframe.
plt.bar(temp_df.index, temp_df.snap_CA)
plt.ylabel("# of snap purchase allowed day")
plt.xlabel("Year")
plt.title("Snap Purchase allowed day yearly transition")


# OK, total count in one year is almost the same in all years and all states.
# How about monthly distribution?

# In[ ]:


temp_df = calendar_df[calendar_df["year"] == 2015].groupby(["month"])[["snap_CA", "snap_TX", "snap_WI"]].sum()
temp_df


# Through the year, we have 10 snap purchase allowed days in one month.  
# This tendency is the same from 2012 to 2015.  
# (In 2011, no snap days in January)

# In[ ]:


# Just visualizing the above dataframe
plt.bar(temp_df.index, temp_df.snap_CA)
plt.ylabel("# of snap purchase allowed day")
plt.xlabel("Month")
plt.title("Snap Purchase allowed day monthly trend")


# OK, total count in one month is the same through the whole year.
# How about weekly distribution?

# In[ ]:


temp_df = calendar_df[calendar_df["year"] == 2015].groupby(["weekday"])[["snap_CA", "snap_TX", "snap_WI"]].sum()
temp_df


# Regarding weekly trend, we can find no biased distribution.  
# This is also almost uniformly distributed like year total and month total.

# In[ ]:


plt.bar(temp_df.index, temp_df.snap_CA)
plt.xticks(rotation=60)
plt.ylabel("# of snap purchase allowed day")
plt.xlabel("Day type")
plt.title("Snap Purchase allowed day weekly trend")


# From above things, we may think "oh, snap_enable day is distributed uniformly, like 1 day in 3 consecutive days." because all of these barplot show it's not so different in any month, any day.  
# However, it is **completely different** as I'll show you below.  
# (i.e. snap purchase enable day is distributed biasedly.)

# In[ ]:


# Make temp dataframe with necessary information
temp_df = df.groupby(["date", "state_id"])[["value"]].sum()
temp_df = temp_df.reset_index()
temp_df = temp_df.merge(calendar_df[["date", "snap_CA", "snap_TX", "snap_WI"]], on="date")
temp_df


# Find the most item sold day for example and take a look at the relationship between snap purchase allowed flag and values.

# In[ ]:


np.argmax(temp_df.groupby(["date", "state_id"])["value"].sum())


# In[ ]:


temp_df = temp_df[(temp_df.date >= "2016-02-15") & (temp_df.date <= "2016-03-25") & (temp_df.state_id == "CA")]
temp_df


# In[ ]:


fig, ax1 = plt.subplots()
plt.xticks(rotation=60)
ax1.plot("date", "value", data=temp_df[temp_df.state_id == "CA"])
ax2 = ax1.twinx()  
ax2.scatter("date", "snap_CA", data=temp_df[temp_df.state_id == "CA"])


# In above figure, each plot means whether the day allows snap purchase or not in CA.  
# As you can see, snap purchase enable day is not regularly distributed like one day in three consective days.  
# (ex. 2016-03-01 > 2016-03-04 > 2016-03-07 > ...)  
# It is actually biasedly distributed like the figure above.  
# (i.e. Snap purchase Enable Day continues from 2016-03-01 to 2016-03-10)  
# And on these days, sales are also increased.  

# # Event Pattern Analysis
# Let's check event pattern in event_name_1 column.  
# (As for event_name_2 column, there are much less non-null values compeared to event_name_1 column.

# In[ ]:


plt.figure(figsize=(8, 6))
sns.countplot(x="event_type_1", data=calendar_df[calendar_df["event_name_1"] != "unknown"])
plt.xticks(rotation=90)
plt.title("Event Type Count in event name 1 column")


# OK, event tyoe distributes like the graph above.   
# (Most of the values are actually "unknown", but for visualization, I omitted unknown value)

# In[ ]:


# Let's check the distribution of snap purchase day and event day
# Accirding to the graph, Snap CA is allowed especially when sport event occurs.

plt.figure(figsize=(8, 6))
sns.countplot(x="event_type_1", data=calendar_df[calendar_df["event_name_1"] != "unknown"], hue="snap_CA")
plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.01, 1.01))
plt.title("Snap Purchse allowed day Count in each event category")


# Let's check the sales of event day!

# In[ ]:


temp_series = df.groupby(["cat_id", "event_type_1"])["value"].mean()
temp_series


# In[ ]:


plt.bar(x=temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("event_type_1"), 
        height=temp_series[temp_series.index.get_level_values("cat_id") == "HOBBIES"].values)
plt.title("HOBBIES Item Sold mean in each event type")
plt.ylabel("Item sold mean")
plt.xlabel("Event Type")


# I thought when some cultual or sporting event occurs, HOBBIES item are more likely to be sold.  
# However, this plot doesn't mean this hypothesis clearly.

# ## One Item Features Analysis

# In[ ]:


# find out most sold item for example
df[df["value"] == df["value"].max()]


# The most sold out item in this dataseet is FOODS_3_090_CA_3_validation

# In[ ]:


target_id = "FOODS_3_090_CA_3_validation"
temp_df = df[df["id"] == target_id]
temp_df


# In[ ]:


weekday = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]

# Create one hot weekday column from wday column to calculate correlation later. 
for idx, val in enumerate(weekday):
    temp_df.loc[:, val] = (temp_df["wday"] == idx + 1).astype("int8")

temp_df
# sns.heatmap(temp_df[["value", "snap_CA", ]].corr(), annot=True)


# In[ ]:


# Create Event Flag (Any events occur: 1, otherwise: 0)
# Create Each Event Type Flag
temp_df.loc[:, "is_event_day"] = (temp_df["event_name_1"] != "unknown").astype("int8")
temp_df.loc[:, "is_sport_event"] = (temp_df["event_type_1"] == "Sporting").astype("int8")
temp_df.loc[:, "is_cultural_event"] = (temp_df["event_type_1"] == "Cultural").astype("int8")
temp_df.loc[:, "is_national_event"] = (temp_df["event_type_1"] == "National").astype("int8")
temp_df.loc[:, "is_religious_event"] = (temp_df["event_type_1"] == "Religious").astype("int8")

temp_df.head()


# In[ ]:


# Plot Heatmap with these columns made in previous cells
plt.figure(figsize=(14, 10))
sns.heatmap(temp_df[["value", "sell_price", "snap_CA", "is_event_day", "is_sport_event", "is_cultural_event", "is_national_event", "is_religious_event"] + weekday].corr(), annot=True)
plt.title("Heatmap with values, snap_CA,  event_flag and weekday columns")


# We can find the following things.
# 1. Regarding value and other columns correlation:
#    - snap purchase and other events flag has little correltion.
#    - Saturday has the most positive effect on values, and Tuesday has the most negative effect.  
#      (We've previously seen Saturday is the most item sold day in one week [here](#Item-Sold-in-each-day-type).)
#      
# 2. Regarding snap_CA and weekdays columns correlation:
#    - As we've previously seen, snap_CA is uniformly distributed in each day type.  
#      Thus, the correlation between snap_CA and weekdays columns (ex. Monday, Tuesday, ...) are almost 0.
# 
# 3. Others:
#    - Looking at event and sunday correlation,it is just 0.089.  
#      I thought most part of events oocur on Sunday, but it wasn't so much as I had exoected.
#    - Regarding sell price, we look below.

# ## Sell Price Analysis

# In[ ]:


df.groupby("cat_id")["sell_price"].mean()


# In[ ]:


df.groupby("cat_id")["sell_price"].describe()


# I don't understand why mean = NaN when using *.describe* method, however, *.mean()* method accurately calculate category's sell price mean.  
# Let's plot it with some ways!

# In[ ]:


sns.boxplot(data=df, x="cat_id", y='sell_price')
plt.title("Boxplot of sell prices in each category")


# The price of some Household category is super expensive like over 100 \$.  
# On the other hand, foods are mostly around 5 to 10 \$ and don't have a large deviation.

# In[ ]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="cat_id", y='sell_price', hue="store_id")
plt.title("Boxplot of sell prices in each store")


# We've already seen HOUSEHOLD has some exceptionally expensive sell price.  
# Now we found these items aren't sold anywhere, but only in some stores.

# In[ ]:


# One Item Sell Price Transition
sns.lineplot(data=df[df["item_id"] == "FOODS_3_090"], x='date', y='sell_price', hue="store_id")
plt.legend(bbox_to_anchor=(1.01, 1.01))
plt.title("Sell price change of 'FOODS_3_090' in each store")


# Though the item is same, sell price is slightly different in each store and each season. -> Some promotion season?

# In[ ]:


df["is_event_day"] = (df["event_name_1"] != "unknown").astype("int8")
df.head()


# In[ ]:


sns.heatmap(df[df["item_id"] == "FOODS_3_090"][["value", "sell_price", "is_event_day"]].corr(), annot=True)
plt.title("Heatmap of value, sell_price and event flag")


# Event Day flag and Sell Price don't have so strong relationship.  
# However, when we buy items for some events, we perhaps buy items 1 week ~ 1 day before the event, not on the same day.  
# So we have to take this into consideration.  (I'll tackle with this analysis later.)

# In[ ]:


temp_df = df.groupby(["date", "cat_id"])["sell_price"].mean()
temp_df


# In[ ]:


plt.figure(figsize=(8,4))
sns.lineplot(x=temp_df[temp_df.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), y=temp_df[temp_df.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")
sns.lineplot(x=temp_df[temp_df.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), y=temp_df[temp_df.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")
sns.lineplot(x=temp_df[temp_df.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), y=temp_df[temp_df.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Mean price")
plt.title("Mean price transition of each category")


# The mean of item price in each category seems to have some change in these dataset periods.  
# Is this simply because the expensive item increased in later periods?

# In[ ]:


temp_df = df.groupby(["date", "cat_id"])["item_id"].count()


# In[ ]:


plt.figure(figsize=(8,4))
sns.lineplot(x=temp_df[temp_df.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), y=temp_df[temp_df.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")
sns.lineplot(x=temp_df[temp_df.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), y=temp_df[temp_df.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")
sns.lineplot(x=temp_df[temp_df.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), y=temp_df[temp_df.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Registered Item Counts")
plt.title("Registered Item Counts Transition in each category")


# All categories have similar trends though the volume of these values are different.  
# From 2015, the number of registered items is seemed to be relatively constant.

# ## Sell price and value relationship

# In[ ]:


sns.jointplot(df["value"], df["sell_price"])


# This result is also interesting.  
# Of course, when the price of item is expensive, less items are sold. (left top field)  
# And when the price gets lower, more items are sold.  (Right bottom field)  
# However, the relationship is not likely linear but likely to be inverse propotion.

# ## Discount Season Presumption
# Is there a period when all items have lower price than usual in Walmart?

# In[ ]:


df["sell_price_diff"] = df.groupby("id")["sell_price"].transform(lambda x: x - x.mean()).astype("float32")


# In[ ]:


sns.lineplot(df[df["item_id"] == "FOODS_3_090"]["date"],df[df["item_id"] == "FOODS_3_090"]["sell_price_diff"], hue=df["store_id"]) 
plt.legend(bbox_to_anchor=(1.01, 1.01))


# All store has lower price than mean value in the latter half of 2013.  

# In[ ]:





# ## Relationship of Lag Variables

# In[ ]:


df["lag_1"] = df.groupby("id")["value"].transform(lambda x: x.shift(1)).astype("float32")
df["lag_7"] = df.groupby("id")["value"].transform(lambda x: x.shift(7)).astype("float32")


# In[ ]:


# plt.figure(figsize=(8, 8))
# sns.pairplot(df[["cat_id", "value", "lag_1"]], hue="cat_id")


# In[ ]:


sns.pairplot(df[["cat_id", "value", "lag_1", "lag_7"]], hue="cat_id")


# Maybe 1 day or 1 week lag variables are important in this case.  
# I'll find the strength of correlation between the sold of the day and past few days in my next version.

# ## PCA Trial

# In[ ]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


# In[ ]:


pca = PCA()


# In[ ]:


temp_df = df.loc[df["date"] >= "2015-01-01", ["sell_price", "dept_id", "value", "state_id", "cat_id"]]


# In[ ]:


# For memory usage reduction
del df
gc.collect()


# In[ ]:


temp_df.loc[temp_df["cat_id"] == "HOBBIES", "cat_color"]  = "orange"
temp_df.loc[temp_df["cat_id"] == "FOODS", "cat_color"]  = "blue"
temp_df.loc[temp_df["cat_id"] == "HOUSEHOLD", "cat_color"]  = "green"
color = temp_df["cat_color"]


# In[ ]:


le = LabelEncoder()
temp_df["enc_state_id"] = le.fit_transform(temp_df["state_id"])
temp_df.drop(columns=["state_id", "cat_id", "cat_color"], inplace=True)


# In[ ]:


# temp_df = temp_df.apply(lambda x: (x-x.mean()/ x.std(), axis=0))


# In[ ]:


pca.fit(temp_df)
feature = pca.transform(temp_df)


# In[ ]:


plt.figure(figsize=(6, 6))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=color)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()


# In[ ]:


del temp_df
gc.collect()


# # Summary
#    In this notebook, through some easy data visualization, we found some points regarding this dataset like below.
#    1. The transition of all items sold in each category 
#       - Some periodical effect. (Weekly and monthly)
#       - On christmas day, there are almost no sales
#    2. Which category is the most sold one?  
#       - FOODS is the most sold item category of these three categories.
#       - HOUSEHOLD category is the 2nd one, and the HOBBIES are the least sold one.
#    3. Which day type is the most sold day? 
#       - Saturday and Sunday is the most item sold day types.
#       - In contrast, on weekdays like Tuesday, there are less item sold.
#    4. The transition in all stores by each state
#       - In CA, CA_3 store is the most sold store.  
#       - In other states, not so much difference appeared.  
#       - These sales trasition often corresponds to the registered item entries.
#    5. Snap purchase allowed day visuaizaiton
#       - The total count of Snap purchase allowed day in whole year is almost the same from 2011.
#       - The total count of Snap purchase allowed day in one month is 10 in every month.
#       - The total count of Snap purchase allowed day in each day type is almost uniformly distributed.
#       - However, there are some biased patterns regarding snap purchase allowed flag.
#         (i.e. it is not like one day in three consective days regularly, but all days in one week and none in next week)
#     
#    And finally we visualize some points by using heatmap.

# # Future Work
# If I have time, I'd like to tackle with the following things.
# 
# 1. Apply Dynamic Analysis and find out the relationship among state and stores.
# 2. Make One item or one store prediction model for beginners like me to learn how to use lightgbm as a regressor.
# 3. Check out the pre-processing effect. Is that effective considering the noise samples like Christmas or other irregularly days.
# 4. More detailed analysis and find out some useful information for making prediction.

# # References
# Following notebooks are the great notebooks in this competition. 
# For whom hasn't check these notebooks, I strongly recommend you to take a look at these notebooks.
# (I'm sorry if I missed some other great kernels, I'll take a lookt at other notebooks if I have enough time.)
# 
# Data Visualization:
# 
# - **M5 Forecasting - Starter Data Exploration**  
#   https://www.kaggle.com/robikscube/m5-forecasting-starter-data-exploration  
# 
# -  **Back to (predict) the future - Interactive M5 EDA**  
#    https://www.kaggle.com/headsortails/back-to-predict-the-future-interactive-m5-eda
# 
# Making Prediction:
# 
# - **M5 - Three shades of Dark: Darker magic**  
#   https://www.kaggle.com/kyakovlev/m5-three-shades-of-dark-darker-magic
# 
# -  **Very fst Model**  
#    https://www.kaggle.com/ragnar123/very-fst-model

# # Acknowledgment
# 
# I apologize that my english are somewhat wrong and my codes are not so beautiful one like others' codes.  
# However, I tried hard to make simle codes as much as I can especially for beginners like me to learn how to use matplotlib and seaborn to do data visualization.  
# Any comments or upvotes can be my very strong motivation towards much harder work! Thank you!

# In[ ]:




