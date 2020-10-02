#!/usr/bin/env python
# coding: utf-8

# While working on my first EDA, I found that `visitId` seems to be **broken**.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import json
from pandas.io.json import json_normalize


# In[ ]:


# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields

JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

def load_df(csv_path):
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'})

    json_part = df[JSON_COLUMNS]
    df = df.drop(JSON_COLUMNS, axis=1)
    normed_json_part = []
    for col in JSON_COLUMNS:
        col_as_df = json_normalize(json_part[col])
        col_as_df.rename(columns=lambda x: f'{col}.{x}', inplace=True)
        normed_json_part.append(col_as_df)
    df = pd.concat([df] + normed_json_part, axis=1)

    return df


# In[ ]:


train = load_df('../input/train.csv')


# 'visitId' seems to be created from 'visitStartTime'

# In[ ]:


train[['visitId', 'visitStartTime']].head()


# However, **4709 (0.52%) rows have different values.**

# In[ ]:


mismatch = train['visitId'].astype(str) != train['visitStartTime'].astype(str)
mismatch.sum(), mismatch.mean()


# It cause 898 duplicate sessionId's discussed in [#66100](https://www.kaggle.com/c/google-analytics-customer-revenue-prediction/discussion/66100).

# In[ ]:


# confirm `sessionId` == `fullVisitorId` + '_' + `visitId`
((train['fullVisitorId'] + '_' + train['visitId'].astype(str)) == train['sessionId']).all()


# In[ ]:


train['sessionId'].value_counts().max(), (train['sessionId'].value_counts() > 1).sum()


# In[ ]:


duplicates = train[train['sessionId'].map(train['sessionId'].value_counts()) > 1].sort_values(by='sessionId')
duplicates.head()


# For all rows where `visitId` != `visitStartTime`, `visitId` is smaller than `visitStartTime`, and the differences tend to be small values.

# In[ ]:


(train.loc[mismatch, 'visitId'] < train.loc[mismatch, 'visitStartTime']).all()


# In[ ]:


diff = train.loc[mismatch, 'visitStartTime'] - train.loc[mismatch, 'visitId']
diff.hist()


# In[ ]:


diff.value_counts().iloc[:20]


# In[ ]:


diff.max()


# I found that some of mismatches are caused by setting previous `visitStartTime` as `visitId`.
# 895 of 898 duplicates are caused by the mismatches.

# In[ ]:


train = train.sort_values(by='visitStartTime')
# convert to `str` so that the series will not be converted to `float` automatically.
train['previous_visitStartTime'] = train['visitStartTime'].astype(str).groupby(train['fullVisitorId']).shift(1).fillna('-1').astype(np.int64)
train = train.sort_index()


# In[ ]:


((train['previous_visitStartTime'] == train['visitId']) & mismatch).sum()


# In[ ]:


duplicates = train[train['sessionId'].map(train['sessionId'].value_counts()) > 1].sort_values(by=['sessionId', 'date'])
(duplicates['visitId'] == duplicates['previous_visitStartTime']).sum()


# `fullVisitorId` + '_' + `visitId` will work well as true session ids.

# In[ ]:


(train['fullVisitorId'] + '_' + train['visitStartTime'].astype(str)).nunique() == train.shape[0]


# In[ ]:




