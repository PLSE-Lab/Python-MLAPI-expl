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


calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")
sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")
sales_train = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")
sample_sub = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")


# In[ ]:


def memory_reduction(dataset):
    column_types = dataset.dtypes
    temp = None
    for x in range(len(column_types)):
        column_types[x] = str(column_types[x])
    for x in range(len(column_types)):
        temp = dataset.columns[x]
        if dataset.columns[x] == "date":
            dataset[temp] = dataset[temp].astype("datetime64")
        if column_types[x] == "int64" and dataset.columns[x] != "date":
            dataset[temp] = dataset[temp].astype("int16")
        if column_types[x] == "object" and dataset.columns[x] != "date":
            dataset[temp] = dataset[temp].astype("category")
        if column_types[x] == "float64" and dataset.columns[x] != "date":
            dataset[temp] = dataset[temp].astype("float16")
    return dataset


# In[ ]:


calendar_df = memory_reduction(calendar)


# In[ ]:


sell_prices["id"] = sell_prices["item_id"] + "_" + sell_prices["store_id"] + "_validation" 
sell_prices = pd.merge(sell_prices, sales_train[["cat_id", "id", "state_id"]], on = "id")
sell_prices_df = memory_reduction(sell_prices)


# In[ ]:


sales_train_df = memory_reduction(sales_train)


# In[ ]:


calendar_df = calendar_df[:1913]
calendar_df["day"] = pd.DatetimeIndex(calendar_df["date"]).day
calendar_df["day"] = calendar_df["day"].astype("int8")
calendar_df["week_num"] = (calendar_df["day"] - 1) // 7 + 1
calendar_df["week_num"] = calendar_df["week_num"].astype("int8")


# In[ ]:


import gc


# In[ ]:


def make_dataframe():
    # Wide format dataset 
    df_wide_train = sales_train_df.drop(columns=["item_id", "dept_id", "cat_id", "state_id","store_id", "id"]).T
    df_wide_train.index = calendar_df["date"]
    df_wide_train.columns = sales_train_df["id"]
    
   
    # Convert wide format to long format
    df_long = df_wide_train.stack().reset_index(1)
    df_long.columns = ["id", "value"]

    del df_wide_train
    gc.collect()
    
    df = pd.merge(pd.merge(df_long.reset_index(), calendar_df, on="date"), sell_prices_df, on=["id", "wm_yr_wk"])
    df = df.drop(columns=["d"])
    #df[["cat_id", "store_id", "item_id", "id", "dept_id"]] = df[["cat_id"", store_id", "item_id", "id", "dept_id"]].astype("category")
    df["sell_price"] = df["sell_price"].astype("float16")   
    df["value"] = df["value"].astype("int32")
    df["state_id"] = df["store_id"].str[:2].astype("category")


    del df_long
    gc.collect()

    return df

df = make_dataframe()


# In[ ]:


del calendar, sales_train, sell_prices
gc.collect()


# In[ ]:





# # EDA STARTS

# In[ ]:


df.head()


# In[ ]:


#importing all necessary libraries
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# ## BASIC ANALYSIS

# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


plt.figure(figsize = (12,12))
sn.heatmap(df.corr(), annot=True)


# The correlation before feature engineering can be seen here -
# * snap has good correlation with day and week num
# * value has a little correlation with sell price at the moment

# In[ ]:


df.dtypes


# In[ ]:


df.isnull().sum()


# we can leave all the null values of the events as it isnt possible for each day of the year to have a event

# In[ ]:


sales_train_df.describe()


# In[ ]:


calendar_df.describe()


# In[ ]:


sell_prices_df.describe()


# ## UNIVARIATE ANALYSIS

# In[ ]:


#value
temp = df.groupby(["cat_id", "date"])["value"].sum()


# In[ ]:


x = temp[temp.index.get_level_values("cat_id") == 'FOODS'].values


# In[ ]:


plt.hist(x)


# In[ ]:


x = temp[temp.index.get_level_values("cat_id") == 'HOUSEHOLD'].values
plt.hist(x)


# In[ ]:


gc.collect()


# In[ ]:


x = temp[temp.index.get_level_values("cat_id") == 'HOBBIES'].values
plt.hist(x)


# In[ ]:


del x, temp
gc.collect()


# ### so  as we can note few points in this section
# * Food has the highest frequency among the others 
# * Food and Hobbies are highly left skewed 
# * Household is little left skewed

# In[ ]:


# ALTERNATE PROCEDURE Of DOING THIS WILL BE SHOWN HERE after some time


# #### Next is EVENTS UNIVARIATE ANALYSIS

# In[ ]:


calendar_df.head()


# In[ ]:


calendar_df['event_name_1'].value_counts().plot.bar()


# In[ ]:


calendar_df['event_type_1'].value_counts().plot.bar()


# In[ ]:


calendar_df['event_name_2'].value_counts().plot.bar()


# In[ ]:


calendar_df['event_type_2'].value_counts().plot.bar()


# #### So as we can see in event type one in event type one -
# * The highest frequncy is of 7 events - lentweek2 , superbowl, StPatricksDay, Purim End, President start, LentStart, Valentines day
# * The highest frequency is of Relegious events in whole
# 
# #### And in event type 2 - 
# 
# * All events have same frequency
# * although as whole cultral events can be seen more

# ### Lets move on to SNAP analyis 

# In[ ]:


calendar_df["snap_CA"].value_counts()


# In[ ]:


calendar_df["snap_TX"].value_counts()


# In[ ]:


calendar_df["snap_WI"].value_counts()


# #### There are equal no. of times when Snap has happened in each of the states

# Lets check this in df 

# In[ ]:


df["snap_CA"].value_counts()


# In[ ]:


df["snap_TX"].value_counts()


# In[ ]:


df["snap_WI"].value_counts()


# #### Therefore having same ratio in snap indicate that our data has been correctly been fitted

# ### CAT_ID

# In[ ]:


df["cat_id"].value_counts()


# In[ ]:


sell_prices_df["cat_id"].value_counts()


# In[ ]:


sell_prices_df["cat_id"].value_counts().plot.bar()


# #### We have 
# * 3181789 products in foods category
# * 2375427 products in HouseHold category
# * 1283905 products in Hobbies category
# 
# to conclude this we can say highest no. of products have been registered in Foods category while lowest in Hobbies category

# ### STATE_ID

# In[ ]:


sell_prices_df["state_id"].value_counts()


# In[ ]:


sell_prices_df["state_id"].value_counts().plot.bar()


# * Canada has highest no. of products registered
# * Texas and West indies have apporximately the same no. of products registered if round of by 10,000

# ### STORE_ID

# In[ ]:


sn.countplot(sell_prices_df.store_id)


# In[ ]:


sell_prices_df.store_id.value_counts()


# In this we got to know 
# * TX_2 has highest no. of products followed by TX_1
# * At third comes CA_1
# * last position is attained by CA_2

# # BIVARIATE ANALYSIS

# In[ ]:


df.head()


# 1. We have to find relations of these with respect to the column value.
# 2. Inter category relations

# ### TIME SERIES ANALYSIS W.R.T cat_id

# #### first as whole 

# In[ ]:


temp  = df.groupby(["cat_id", "date"])["value"].sum()


# In[ ]:


plt.figure(figsize = (8,6))
plt.plot(temp[temp.index.get_level_values('cat_id') == "FOODS"].index.get_level_values("date"), temp[temp.index.get_level_values('cat_id') == "FOODS"].values, label ="FOODS")
plt.plot(temp[temp.index.get_level_values('cat_id') == "HOUSEHOLD"].index.get_level_values("date"), temp[temp.index.get_level_values('cat_id') == "HOUSEHOLD"].values, label ="HOUSEHOLD")
plt.plot(temp[temp.index.get_level_values('cat_id') == "HOBBIES"].index.get_level_values("date"), temp[temp.index.get_level_values('cat_id') == "HOBBIES"].values, label ="HOBBIES")
plt.legend()
plt.show()


# In[ ]:





# #### EVENT CLEANING UP SEQUENCE

# In[ ]:


df.head()


# In[ ]:


#So lets analyze events first like how do they affect the data 
#first we will take event_type_1 in considerations
event_call = df.groupby(["event_name_1", "date"])["value"].sum()


# In[ ]:


plt.figure(figsize = (8,6))
plt.plot(temp[temp.index.get_level_values('cat_id') == "FOODS"].index.get_level_values("date"), temp[temp.index.get_level_values('cat_id') == "FOODS"].values, label ="FOODS")
plt.plot(temp[temp.index.get_level_values('cat_id') == "HOUSEHOLD"].index.get_level_values("date"), temp[temp.index.get_level_values('cat_id') == "HOUSEHOLD"].values, label ="HOUSEHOLD")
plt.plot(temp[temp.index.get_level_values('cat_id') == "HOBBIES"].index.get_level_values("date"), temp[temp.index.get_level_values('cat_id') == "HOBBIES"].values, label ="HOBBIES")
plt.legend()
plt.show()


# In[ ]:


pd.get_dummies(calendar_df, columns=["event_name_1"]).head()


# In[ ]:




