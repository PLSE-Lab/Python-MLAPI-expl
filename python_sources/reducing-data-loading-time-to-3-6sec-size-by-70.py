#!/usr/bin/env python
# coding: utf-8

# Loading this data from csv is pain when you have slow machine with limited memory (like I do).
# 
# So, by several steps we gonna reduce both memory size and loading time:
# 
# 1. Label encode Card ID and Merchant ID labels in transaction histories and train/test sets
# 2. Merge historical and "new" transaction logs into one (also add distinguishing column)
# 3. Encode date field
# 4. Encode text categorical fields as numeric
# 5. Compress all numeric fields if possible with reduce_mem_usage*
# 6. At last: save all the data in parquete format 
# 
# \* reduce_mem_usage is a slightly simplified version of [this kernel](https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65) by [arjanso](https://www.kaggle.com/arjanso)
# 
# \** Saving data in parquete format (by using [Pyarrow](https://arrow.apache.org/docs/python/parquet.html)) gets us great loading speed due to compression, muilthreaded loading, etc.)
# 
# On my home machine (cpu: 4 cores, mem: 16GB) loading time is reduced down to **2.7 sec.**
# 
# In this notebook it's around **6 sec.**
# 

# In[ ]:


import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder


# First load all the data:

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
merchants = pd.read_csv("../input/merchants.csv")
historical_transactions = pd.read_csv("../input/historical_transactions.csv")
new_merchant_transactions = pd.read_csv("../input/new_merchant_transactions.csv")


# Let's see how much memory it occupies in the beginning:

# In[ ]:


def mem_usage_mb(dataframes):
    return int(sum([df.memory_usage().sum() for df in dataframes]) / 1024**2)
 
mem_usg_0 = mem_usage_mb([train, test, merchants, historical_transactions, new_merchant_transactions])
print("Memory usage: {} MB".format(mem_usg_0))


# Label encoding for Card ID and Merchant ID:

# In[ ]:


card_ids = np.hstack((train["card_id"].values, test["card_id"].values))
encoder = LabelEncoder().fit(card_ids)

for df in (test, train, historical_transactions, new_merchant_transactions):
    df["card_id"] = encoder.transform(df["card_id"])
    
encoder = LabelEncoder().fit(merchants["merchant_id"])
for df in (merchants, historical_transactions, new_merchant_transactions):
    # rows with non-null merchant_id
    df.loc[~df["merchant_id"].isnull(), "merchant_id"]         = encoder.transform(df.loc[~df["merchant_id"].isnull(), "merchant_id"])
    # rows with null merchant_id
    df.loc[df["merchant_id"].isnull(), "merchant_id"] = -1


# Merge old and new transactions in one table:

# In[ ]:


historical_transactions["historical"] = True
new_merchant_transactions["historical"] = False
transactions = pd.concat((historical_transactions, new_merchant_transactions), axis=0).reset_index(drop=True)


# Encode rest of the text columns:

# In[ ]:


transactions["authorized_flag"] = transactions["authorized_flag"] == "Y"
  
transactions["category_1"] = transactions["category_1"] == "Y"

transactions["category_3"] = transactions["category_3"].fillna("")
transactions["category_3"] = LabelEncoder().fit(transactions["category_3"]).transform(transactions["category_3"])

transactions["purchase_date"] = pd.to_datetime(transactions["purchase_date"])


# Simplified version of reduce_mem_usage:

# In[ ]:


def reduce_mem_usage(dataframe, skip=[]):
    
    for col in dataframe.columns:
        
        if col in skip:
            continue
        
        col_type = str(dataframe[col].dtype)
        
        if col_type.startswith("int"):
            
            mx = dataframe[col].max()
            mn = dataframe[col].min()
            
            if mn >= 0:
                if mx < 255:
                    dataframe[col] = dataframe[col].astype(np.uint8)
                elif mx < 65535:
                    dataframe[col] = dataframe[col].astype(np.uint16)
                elif mx < 4294967295:
                    dataframe[col] = dataframe[col].astype(np.uint32)
                else:
                    dataframe[col] = dataframe[col].astype(np.uint64)
            else:
                if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    dataframe[col] = dataframe[col].astype(np.int8)
                elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    dataframe[col] = dataframe[col].astype(np.int16)
                elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    dataframe[col] = dataframe[col].astype(np.int32)
                elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                    dataframe[col] = dataframe[col].astype(np.int64)
                    
        elif col_type.startswith("float"):
            
            mx = dataframe[col].max()
            mn = dataframe[col].min()
    
            if mn > np.finfo(np.float32).min and mx < np.finfo(np.float32).max:
                dataframe[col] = dataframe[col].astype(np.float32)


# Compress the old and new numeric fields is possible:

# In[ ]:


reduce_mem_usage(transactions)
reduce_mem_usage(train)
reduce_mem_usage(test)

reduce_mem_usage(merchants, skip=["avg_purchases_lag3", "avg_purchases_lag6", "avg_purchases_lag12"])


# Let's see how much memory we've saved:

# In[ ]:


mem_usg_1 = mem_usage_mb([train, test, merchants, transactions])
print("Memory usage: {} MB".format(mem_usg_1))
print(mem_usg_1 / mem_usg_0 * 100, "% of initial size")


# Save to parquet format:

# In[ ]:


def save_as_parquet(df, path):
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, use_dictionary=True, compression='snappy')

save_as_parquet(train, "train.parquet")
save_as_parquet(test, "test.parquet")
save_as_parquet(merchants, "merchants.parquet")
save_as_parquet(historical_transactions, "historical_transactions.parquet")
save_as_parquet(new_merchant_transactions, "new_merchant_transactions.parquet")
save_as_parquet(transactions, "transactions.parquet")


# How fast is the loading now:

# In[ ]:


def load_parquet(path):
    table = pq.read_table(path, nthreads=4)
    return table.to_pandas()

p0 = time.time()
load_parquet("train.parquet")
load_parquet("test.parquet")
load_parquet("merchants.parquet")
load_parquet("transactions.parquet")
p1 = time.time()
p1 - p0
    

