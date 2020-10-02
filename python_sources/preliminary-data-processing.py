#!/usr/bin/env python
# coding: utf-8

# # Preliminary data processing

# In[ ]:


import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize


# In[ ]:


def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    print(f"read from file '{csv_path}'...")
    df = pd.read_csv(csv_path,
                     converters={
                         column: json.loads for column in JSON_COLUMNS},
                     dtype={'fullVisitorId': 'str'},  # Important!!
                     nrows=nrows)
    print("convert columns from json format to plain text...")
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [
            f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(
            column_as_df, right_index=True, left_index=True)
    print(f"Loaded data from '{os.path.basename(csv_path)}'. Shape: {df.shape}")
    return df


# ## Load data

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = load_df()\ntest_df = load_df("../input/test.csv")')


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# Columns not present in test data:

# In[ ]:


set(train_df.columns).difference(set(test_df.columns))


# In[ ]:


train_df.drop('trafficSource.campaignCode', axis=1, inplace=True)


# ## Features with constant values: 

# In[ ]:


nconst_cols_train = [c for c in train_df.columns if train_df[c].nunique(dropna=False) == 1]
print(f"Number of features with constant value in train.csv = {len(nconst_cols_train)}")
print(nconst_cols_train)


# In[ ]:


nconst_cols_test = [c for c in test_df.columns if test_df[c].nunique(dropna=False) == 1]
print(
    f"Number of features with constant value in test.csv = {len(nconst_cols_test)}")
print(nconst_cols_test)


# In[ ]:


print(
    f"Is two features lists match? {set(nconst_cols_train).intersection(set(nconst_cols_test)) == set(nconst_cols_train)}"
)


# Remove features with constant values:

# In[ ]:


train_df.drop(nconst_cols_train, axis=1, inplace=True)
test_df.drop(nconst_cols_test, axis=1, inplace=True)
print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")


# In[ ]:


train_df.head()


# ## Unique values

# In[ ]:


def unique_table(df: pd.DataFrame, show_result=False) -> pd.DataFrame:
    ret_val = pd.DataFrame(columns=["Column", "N", "Values"])
    for column in df.columns:
        uniq = df[column].unique()
        ret_val = ret_val.append({
            "Column": column,
            "N": len(uniq),
            "Values": uniq
        },
            ignore_index=True)
    if show_result:
        for row in ret_val.values:
            print("=" * 80)
            print(f"Column - '{row[0]}' has {row[1]} unique values.")
            print(f"Unique values in '{row[0]}':")
            print("-" * 80)
            print(row[2])
    return ret_val


# In[ ]:


u_val_train = unique_table(train_df)


# In[ ]:


u_val_test = unique_table(test_df)


# In[ ]:


sv_u_val = u_val_train.merge(u_val_test, how="left", on="Column", suffixes=("_train","_test"))


# In[ ]:


def diff_values(x):
    try:
        return len(set(x[0]).difference(set(x[1])))
    except:
        return ""
    
def match_values(x):
    try:
        return len(set(x[0]).intersection(set(x[1])))
    except:
        return ""
                                      


# In[ ]:


sv_u_val["Diff_train"] = sv_u_val[["Values_train", "Values_test"]].apply(diff_values, axis=1)
sv_u_val["Diff_test"] = sv_u_val[["Values_test", "Values_train"]].apply(diff_values, axis=1)
sv_u_val["Matching"] = sv_u_val[["Values_train", "Values_test"]].apply(match_values, axis=1)


# In[ ]:


sv_u_val


# Replace for device.isMobile:

# In[ ]:


train_df["device.isMobile"] = train_df["device.isMobile"].astype("int")
test_df["device.isMobile"] = test_df["device.isMobile"].astype("int")


# ## Save data

# In[ ]:


train_df.to_csv("GA_train.csv", index=False)
test_df.to_csv("GA_test.csv", index=False)


# **Commands for loading data:**
# ```python
# df_train = pd.read_csv("GA_train.csv", dtype={'fullVisitorId': str})
# df_test = pd.read_csv("GA_test.csv", dtype={'fullVisitorId': str})
# ```
# *If you want to replace dots in column names with underscores:*
# ```python
# df_train.columns = df_train.columns.str.replace(".", "_")
# df_test.columns = df_test.columns.str.replace(".", "_")
# ```
