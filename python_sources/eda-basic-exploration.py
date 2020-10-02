#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, subprocess
import seaborn as sns
import matplotlib.pyplot as plt
print(os.listdir("../input"))


# ## Get the number of lines in each file

# In[2]:


print('# Line count:')
for file in ['sales_train.csv.gz', 'sample_submission.csv.gz', 'test.csv.gz']:
    cmd = "xargs zcat ../input/{} | wc -l".format(file)
    counts = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].decode('utf-8')
    print("{}\t{}".format(counts.rstrip(), file))


# In[3]:


print('# Line count:')
for file in ['items.csv', 'shops.csv', 'item_categories.csv']:
    lines = subprocess.run(['wc', '-l', '../input/{}'.format(file)], stdout=subprocess.PIPE).stdout.decode('utf-8')
    print(lines, end='', flush=True)


# ## Load Training Data

# In[4]:


fpath = "../input/sales_train.csv.gz"
dtype = {"date_block_num": "int8",
         "item_id": "uint16",
         "shop_id": "int8",
         "item_price": "float64",
         "item_cnt_day": "int16"}
df_Train = pd.read_csv(fpath, compression="gzip", parse_dates=["date"], dtype=dtype,
                       date_parser=lambda x: pd.to_datetime(x, format="%d.%m.%Y"))


# In[5]:


df_Train.head()


# ## Basic Statics of Training Data

# In[6]:


df_Train.describe()


# ## The trend of number of records
# ---
# It looks like the transactions double around new year.

# In[7]:


df_Train.groupby("date").agg({"date": "count"}).plot(figsize=(10, 6));


# We can clearly see two peaks happened at 2014/01 and 2015/01.

# In[8]:


df_Train.groupby("date_block_num").agg({"date_block_num": "count"}).plot(figsize=(10, 6));


# * We can see that the transactions go up very quickly from Dec. to Jan.
# * November surprisingly has lowest number of transactions.

# In[9]:


df_Train["month"] = df_Train["date"].dt.month
df_Train.groupby("month").agg({"month": "count"}).plot(figsize=(10, 6));


# In[10]:


df_Train["day"] = df_Train["date"].dt.day
df_Train.groupby("day").agg({"day": "count"}).plot(figsize=(10, 6));


# ## Does every item exist in all shops for training data
# ---
# No, for the 21807 unique items, there's no single item exits in all 60 shops

# In[11]:


df_Train["shop_id"].nunique(), df_Train["item_id"].nunique()


# In[12]:


df_Train.groupby("item_id").agg({"shop_id": "nunique"}).reset_index().plot.scatter("item_id", "shop_id", figsize=(10, 6), s=10);


# ## Load Testing Data

# In[13]:


fpath = "../input/test.csv.gz"
dtype = {"item_id": "uint16",
         "shop_id": "int8",}
df_Test = pd.read_csv(fpath, dtype=dtype, index_col="ID")


# ## Does every item exist in all shops for testing data
# ---
# Yes, for the 5100 unique items, each item has been sold in all 42 shops

# In[14]:


set(df_Test["shop_id"])-set(df_Train["shop_id"])


# In[15]:


df_Test["shop_id"].nunique(), df_Test["item_id"].nunique()


# In[16]:


df_Test.groupby("item_id").agg({"shop_id": "nunique"}).reset_index().plot.scatter("item_id", "shop_id", figsize=(10, 6), s=10)
plt.ylim(40, 45);


# ## Does every shop in testing data have history record for every item.
# ---
# Wow, we can see that some shops do have historical data almost across all items.

# In[17]:


cols = ["shop_id", "item_id"]
df_ShopItem = df_Train.groupby(cols).agg({"item_id": "nunique"}).rename(columns={"item_id": "Exist"}).reset_index()
df_ShopItem = df_Test.merge(df_ShopItem, on=cols, how="left").fillna(0)


# In[18]:


_, ax = plt.subplots(1, 1, figsize=(20, 6))
sns.heatmap(df_ShopItem.pivot("shop_id", "item_id", "Exist"), ax=ax);


# In[19]:


df_ShopItem.groupby(["shop_id"]).agg({"Exist": "sum"}).sort_values("Exist")


# ## Does every item in testing data exist in training data
# ---
# No. There are only 4737 items exist in training data but there are 363 items missing. Also, some items are rarely selling in shops.

# In[20]:


items_testing = set(df_Test["item_id"])
items_training = set(df_Train["item_id"])
len(items_testing - items_training)


# In[ ]:




