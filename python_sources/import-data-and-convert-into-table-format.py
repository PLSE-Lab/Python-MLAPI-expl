#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas.io.json import json_normalize

import os
print(os.listdir("../input"))


# 'device', 'geoNetwork', 'totals' and 'trafficSource' are in JSON format in training and test datasets. We'll use the function from Julian's great [kernel](https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook) to flatten the format.

# In[ ]:


def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_train = load_df()\ndf_test = load_df("../input/test.csv")')


# 19 columns have constant values. Removing them from the dataset.

# In[ ]:


constant_columns = [col for col in df_train.columns if df_train[col].nunique(dropna=False)==1]

df_train.drop(columns=constant_columns,inplace=True)
df_test.drop(columns=constant_columns,inplace=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_train.to_csv("df_train.csv", index=False)\ndf_test.to_csv("df_test.csv", index=False)')

