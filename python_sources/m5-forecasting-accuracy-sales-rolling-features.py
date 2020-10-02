#!/usr/bin/env python
# coding: utf-8

# # About this notebook
# - To continue the creation of lag features, we definitely want to have more information from the past sales,
# - but it is just not possible and not efficiently to create features by shifting "sales" column until day 1941,
# - so we must have a better way to get the information we want with reasonable calculation and memory usage.
# - Basic features: https://www.kaggle.com/kaiweihuang/m5-forecasting-accuracy-sales-basic-features
# - Lag features: https://www.kaggle.com/kaiweihuang/m5-forecasting-accuracy-sales-lag-features

# In[ ]:


# Set environment variables
import os
import time
import warnings
import numpy as np
import pandas as pd

VERSION = 1
INPUT_PATH = f"/kaggle/input/m5-forecasting-accuracy-sales-basic-features"
BASE_PATH = f"/kaggle/working/m5-forecasting-accuracy-ver{VERSION}"


# In[ ]:


# Turn off warnings

warnings.filterwarnings("ignore")


# In[ ]:


# Change directory

os.chdir(INPUT_PATH)
print(f"Change to directory: {os.getcwd()}")


# In[ ]:


# Memory usage function

def format_memory_usage(total_bytes):
    unit_list = ["", "Ki", "Mi", "Gi"]
    for unit in unit_list:
        if total_bytes < 1024:
            return f"{total_bytes:.2f}{unit}B"
        total_bytes /= 1024
    return f"{total_bytes:.2f}{unit}B"


# In[ ]:


# Set global variables

days_to_predict = 28
rolling_days = [60, 90, 180, 365]


# In[ ]:


# Load dataset from our previous work

df_rolling_features = pd.read_pickle("m5-forecasting-accuracy-ver1/sales_basic_features.pkl")
df_rolling_features.head(10)


# # Feature Engineering - Sales - Rolling Lag Features
# - From day 1 to 28, we have created lag features to contain raw information.
# - Because of memory constraint, it is not possible to keep creating those features until day 1941,
# - and even if we have enough spaces, it is still not the most efficient way to utilize the memory.

# - For each longer period, we calculate some descriptive statistics of its distribution to capture the characteristics.
# - For example, for sales information in past 60 days, instead of shifting "sales" column 60 times,
# - we can create features like mean or median by a moving window with 1 step each time,
# - which stores the crucial information efficiently, and condense 60 columns into one.
# - And remember that 28 days shift is to ensure that every prediction row contains those features.

# In[ ]:


# Get necessary columns only

df_rolling_features = df_rolling_features[["id", "d", "sales"]]
df_rolling_features.head(10)


# In[ ]:


# Create features
# Generate rolling lag features and control the memory usage

df_rolling_grouped = df_rolling_features.groupby(["id"])["sales"]

for day in rolling_days:

    start_time = time.time()
    print(f"Rolling {str(day)} Start.")

    df_rolling_features[f"rolling_{str(day)}_max"] = df_rolling_grouped.transform(lambda x: x.shift(days_to_predict).rolling(day).max()).astype(np.float16)
    df_rolling_features[f"rolling_{str(day)}_min"] = df_rolling_grouped.transform(lambda x: x.shift(days_to_predict).rolling(day).min()).astype(np.float16)
    df_rolling_features[f"rolling_{str(day)}_median"] = df_rolling_grouped.transform(lambda x: x.shift(days_to_predict).rolling(day).median()).astype(np.float16)
    df_rolling_features[f"rolling_{str(day)}_mean"] = df_rolling_grouped.transform(lambda x: x.shift(days_to_predict).rolling(day).mean()).astype(np.float16)
    df_rolling_features[f"rolling_{str(day)}_std"] = df_rolling_grouped.transform(lambda x: x.shift(days_to_predict).rolling(day).std()).astype(np.float16)

    end_time = time.time()
    print(f"Calculation time: {round(end_time - start_time)} seconds")


# In[ ]:


# Check dataset

df_rolling_features.head(120)


# In[ ]:


# Check data type

df_rolling_features.info()


# In[ ]:


# Check current memory usage

memory_usage_string = format_memory_usage(df_rolling_features.memory_usage().sum())
print(f"Current memory usage: {memory_usage_string}")


# # Note
# - Some more ways to make features better:
# - Now, for sales within past 28 days, we collect all raw data as features in previous notebook,
# - but we don't know if raw data will perform better than descriptive statistics.
# - Also, keep shifting 28 days is making the training data have a lot of NaN, especially in longer periods.
# - So, 1) Shifting within 28 days, which will increase the effective training data, but let features in prediction row fewer.
# - 2) Calculating descriptive statistics in shorter periods, such as 7, 14, 21, 30.
# - 3) We have also canceled the calculation of skewness and kurtosis due to memory limit, and they are worth trying actually.

# In[ ]:


# Change to output path

try:
    os.chdir(BASE_PATH)
    print(f"Change to directory: {os.getcwd()}")
except:
    os.mkdir(BASE_PATH)
    os.chdir(BASE_PATH)
    print(f"Create and change to directory: {os.getcwd()}")


# In[ ]:


# Save pickle file

df_rolling_features.to_pickle("sales_rolling_features.pkl")

