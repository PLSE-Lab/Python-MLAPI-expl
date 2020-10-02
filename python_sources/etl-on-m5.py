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





# ## Loading The data

# In[ ]:


calendar = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")
sell_prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")
sales_train = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")
sample_sub = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")


# ## importing ploting library

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sn
get_ipython().run_line_magic('matplotlib', 'inline')


# ## importing mathematical library

# In[ ]:


import numpy as np
import scipy as sc


# memory reduction function
# 

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

           


# ## Reading The Data

# In[ ]:


calendar.tail(56)


# ## calendar

# In[ ]:


calendar_df = memory_reduction(calendar)


# In[ ]:


calendar_df.dtypes


# In[ ]:


calendar_df.shape


# ## Sell prices

# In[ ]:


sell_prices.dtypes


# In[ ]:


sell_prices.head()


# In[ ]:


sell_prices["id"] = sell_prices["item_id"] + "_" + sell_prices["store_id"] + "_validation" 


# In[ ]:


sell_prices = pd.merge(sell_prices, sales_train[["cat_id", "id", "state_id"]], on = "id")


# In[ ]:


sell_prices.columns


# In[ ]:


sell_prices_df = memory_reduction(sell_prices)


# In[ ]:


sell_prices_df.dtypes


# In[ ]:


sell_prices_df.shape


# ## Sales_train

# In[ ]:


sales_train.head()


# In[ ]:


sales_train_df = memory_reduction(sales_train)


# In[ ]:


sales_train_df.dtypes


# In[ ]:


sales_train_df.shape


# #### as we  can see the data set in calendar has 1969 rows although sales_train has 1914 days
#      thus we need to reduce calendar data to 1919  

# In[ ]:


calendar_df = calendar_df[:1913]


# In[ ]:


calendar_df.shape


# In[ ]:


calendar_df.head()


# In[ ]:


calendar_df["day"] = pd.DatetimeIndex(calendar_df["date"]).day


# In[ ]:


calendar_df["day"] = calendar_df["day"].astype("int8")


# In[ ]:


calendar_df["week_num"] = (calendar_df["day"] - 1) // 7 + 1


# In[ ]:


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


#calendar_df.to_csv("calendar_normalized.csv", index = False)
#sales_train_df.to_csv("sales_train_normalized.csv", index = False)
#sell_prices_df.to_csv("sell_prices_normalized.csv", index = False)


# In[ ]:


del calendar, sales_train, sell_prices
gc.collect()


# # DATA ANALYSIS

# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sn 
import numpy as np
import scipy as sc
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


temp = df.groupby(["cat_id", "date"])["value"].sum()


# In[ ]:


temp


# In[ ]:


plt.figure(figsize = (12,4))
plt.plot(temp[temp.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), temp[temp.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")
plt.plot(temp[temp.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), temp[temp.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")
plt.plot(temp[temp.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), temp[temp.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")


# In[ ]:


temp = temp.loc[temp.index.get_level_values("date") >= "2015-01-01"]


# In[ ]:


plt.figure(figsize = (12,4))
plt.plot(temp[temp.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), temp[temp.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")
plt.plot(temp[temp.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), temp[temp.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")
plt.plot(temp[temp.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), temp[temp.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")
plt.legend()


# In[ ]:


temp = df.groupby(["cat_id", "date"])["value"].sum()


# In[ ]:


temp = temp.loc[temp.index.get_level_values("date") >= "2015-01-01"]


# In[ ]:


temp = temp.loc[temp.index.get_level_values("date") <= "2015-02-01"]


# In[ ]:


plt.figure(figsize = (12,4))
plt.plot(temp[temp.index.get_level_values("cat_id") == "FOODS"].index.get_level_values("date"), temp[temp.index.get_level_values("cat_id") == "FOODS"].values, label="FOODS")
plt.plot(temp[temp.index.get_level_values("cat_id") == "HOBBIES"].index.get_level_values("date"), temp[temp.index.get_level_values("cat_id") == "HOBBIES"].values, label="HOBBIES")
plt.plot(temp[temp.index.get_level_values("cat_id") == "HOUSEHOLD"].index.get_level_values("date"), temp[temp.index.get_level_values("cat_id") == "HOUSEHOLD"].values, label="HOUSEHOLD")
plt.legend()


# In[ ]:


plt.figure(figsize = (12,12))
sn.heatmap(df.corr(), annot=True)


# In[ ]:


gc.collect()


# In[ ]:


temp = df.groupby(["cat_id", "wday"])["value"].sum()


# In[ ]:


plt.figure(figsize=(6, 4))
left = np.arange(1,8) 
width = 0.3
weeklabel = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]    # Please Confirm df


plt.bar(left, temp[temp.index.get_level_values("cat_id") == "FOODS"].values, width=width, label="FOODS")
plt.bar(left + width, temp[temp.index.get_level_values("cat_id") == "HOUSEHOLD"].values, width=width, label="HOUSEHOLD")
plt.bar(left + width + width, temp[temp.index.get_level_values("cat_id") == "HOBBIES"].values, width=width, label="HOBBIES")
plt.legend(bbox_to_anchor=(1.01, 1.01))
plt.xticks(left, weeklabel, rotation=60)
plt.xlabel("day of week")
plt.ylabel("# of sold items")
plt.title("Total sold item in each daytype")


# In[ ]:


temp = df.groupby(["cat_id", "day"])["value"].sum()


# In[ ]:


plt.figure(figsize=(12, 8))
left = np.arange(1,32) 
width = 0.3
weeklabel = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]    # Please Confirm df


plt.bar(left, temp[temp.index.get_level_values("cat_id") == "FOODS"].values, width=width, label="FOODS")
plt.bar(left + width, temp[temp.index.get_level_values("cat_id") == "HOUSEHOLD"].values, width=width, label="HOUSEHOLD")
plt.bar(left + width + width, temp[temp.index.get_level_values("cat_id") == "HOBBIES"].values, width=width, label="HOBBIES")
plt.legend(bbox_to_anchor=(1.01, 1.01))
plt.xlabel("day of month")
plt.ylabel("# of sold items")
plt.title("Total sold item in each day")


# ## State analysis

# In[ ]:


gc.collect()


# In[ ]:


temp = df.groupby(["state_id", "wday"])["value"].sum()


# In[ ]:


plt.figure(figsize=(6, 4))
left = np.arange(1,8) 
width = 0.3
weeklabel = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]    # Please Confirm df


plt.bar(left, temp[temp.index.get_level_values("state_id") == "CA"].values, width=width, label="CA")
plt.bar(left + width, temp[temp.index.get_level_values("state_id") == "TX"].values, width=width, label="TX")
plt.bar(left + width + width, temp[temp.index.get_level_values("state_id") == "WI"].values, width=width, label="WI")
plt.legend(bbox_to_anchor=(1.01, 1.01))
plt.xticks(left, weeklabel, rotation=60)
plt.xlabel("day of week")
plt.ylabel("# of sold items")
plt.title("Total sold item in each daytype")


# In[ ]:


temp = df.groupby(["state_id", "day"])["value"].sum()


# In[ ]:


plt.figure(figsize=(12, 8))
left = np.arange(1,32) 
width = 0.3
weeklabel = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]    # Please Confirm df


plt.bar(left, temp[temp.index.get_level_values("state_id") == "CA"].values, width=width, label="CA")
plt.bar(left + width, temp[temp.index.get_level_values("state_id") == "TX"].values, width=width, label="TX")
plt.bar(left + width + width, temp[temp.index.get_level_values("state_id") == "WI"].values, width=width, label="WI")
plt.legend(bbox_to_anchor=(1.01, 1.01))
plt.xlabel("day of month")
plt.ylabel("# of sold items")
plt.title("Total sold item in each day")


# In[ ]:


gc.collect()


# In[ ]:


temp = df.groupby(["state_id", "date"])["value"].sum()


# In[ ]:


plt.figure(figsize = (12,4))
plt.plot(temp[temp.index.get_level_values("state_id") == "CA"].index.get_level_values("date"), temp[temp.index.get_level_values("state_id") == "CA"].values, label="CA")
plt.plot(temp[temp.index.get_level_values("state_id") == "TX"].index.get_level_values("date"), temp[temp.index.get_level_values("state_id") == "TX"].values, label="TX")
plt.plot(temp[temp.index.get_level_values("state_id") == "WI"].index.get_level_values("date"), temp[temp.index.get_level_values("state_id") == "WI"].values, label="WI")
plt.legend()


# In[ ]:


temp = temp.loc[temp.index.get_level_values("date") >= "2015-12-01"]  
temp = temp.loc[temp.index.get_level_values("date") <= "2016-01-01"]


# In[ ]:


plt.figure(figsize = (16,4))
plt.plot(temp[temp.index.get_level_values("state_id") == "CA"].index.get_level_values("date"), temp[temp.index.get_level_values("state_id") == "CA"].values, label="CA")
plt.plot(temp[temp.index.get_level_values("state_id") == "TX"].index.get_level_values("date"), temp[temp.index.get_level_values("state_id") == "TX"].values, label="TX")
plt.plot(temp[temp.index.get_level_values("state_id") == "WI"].index.get_level_values("date"), temp[temp.index.get_level_values("state_id") == "WI"].values, label="WI")
plt.legend()


# ## SNAP ANALYSIS
# 

# In[ ]:


temp = df.groupby(["year"])[["snap_CA", "snap_TX", "snap_WI"]].sum()
temp


# In[ ]:


fig, axs = plt.subplots(3, figsize = (8,6)) 

axs[0].plot(temp.index, temp.snap_CA, label = "SNAP_CA")
axs[1].plot(temp.index, temp.snap_TX, label = "SNAP_TX")
axs[2].plot(temp.index, temp.snap_WI, label = "SNAP_WI")
axs[0].legend()
axs[1].legend()
axs[2].legend()


# In[ ]:


fig, axs = plt.subplots(3, figsize = (8,6)) 

axs[0].bar(temp.index, temp.snap_CA, label = "SNAP_CA")
axs[1].bar(temp.index, temp.snap_TX, label = "SNAP_TX")
axs[2].bar(temp.index, temp.snap_WI, label = "SNAP_WI")
axs[0].legend()
axs[1].legend()
axs[2].legend()


# In[ ]:


temp = calendar_df.groupby(["year"])[["snap_CA", "snap_TX", "snap_WI"]].sum()
temp


# In[ ]:


fig, axs = plt.subplots(3, figsize = (8,6)) 

axs[0].bar(temp.index, temp.snap_CA, label = "SNAP_CA")
axs[1].bar(temp.index, temp.snap_TX, label = "SNAP_TX")
axs[2].bar(temp.index, temp.snap_WI, label = "SNAP_WI")
axs[0].legend()
axs[1].legend()
axs[2].legend()


# In[ ]:


temp = df.groupby(["month"])[["snap_CA", "snap_TX", "snap_WI"]].sum()
temp


# In[ ]:


fig, axs = plt.subplots(3, figsize = (8,6)) 

axs[0].plot(temp.index, temp.snap_CA, label = "SNAP_CA")
axs[1].plot(temp.index, temp.snap_TX, label = "SNAP_TX")
axs[2].plot(temp.index, temp.snap_WI, label = "SNAP_WI")
axs[0].legend()
axs[1].legend()
axs[2].legend()


# In[ ]:


fig, axs = plt.subplots(3, figsize = (8,6)) 

axs[0].bar(temp.index, temp.snap_CA, label = "SNAP_CA")
axs[1].bar(temp.index, temp.snap_TX, label = "SNAP_TX")
axs[2].bar(temp.index, temp.snap_WI, label = "SNAP_WI")
axs[0].legend()
axs[1].legend()
axs[2].legend()


# In[ ]:


temp = df.groupby(["week_num"])[["snap_CA", "snap_TX", "snap_WI"]].sum()
temp


# In[ ]:


fig, axs = plt.subplots(3, figsize = (8,6)) 

axs[0].plot(temp.index, temp.snap_CA, label = "SNAP_CA")
axs[1].plot(temp.index, temp.snap_TX, label = "SNAP_TX")
axs[2].plot(temp.index, temp.snap_WI, label = "SNAP_WI")
axs[0].legend()
axs[1].legend()
axs[2].legend()


# In[ ]:


fig, axs = plt.subplots(3, figsize = (8,6)) 

axs[0].bar(temp.index, temp.snap_CA, label = "SNAP_CA")
axs[1].bar(temp.index, temp.snap_TX, label = "SNAP_TX")
axs[2].bar(temp.index, temp.snap_WI, label = "SNAP_WI")
axs[0].legend()
axs[1].legend()
axs[2].legend()


# In[ ]:




