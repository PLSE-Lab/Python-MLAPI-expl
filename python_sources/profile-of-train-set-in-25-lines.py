#!/usr/bin/env python
# coding: utf-8

# This is a profile of the training data - i.e., an overview of variables with counts, distributions and so on. It's a good start to a more thorough EDA. It includes unpacking the JSON columns and looking at the contents.

# In[ ]:


import json
import pandas as pd
import hvplot.pandas


# Here's our train data - notice the lovely JSON!

# In[ ]:


train = pd.read_csv('../input/train.csv', dtype={'fullVisitorId': str, 'date': str}, parse_dates=['date'],
        index_col=('fullVisitorId', 'sessionId'))
train.head()


# We'll start with the non-json columns first. Note that the bar plots only include the top 50 categories.

# In[ ]:


json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']
train_jsons = train[json_cols]
train_nonjsons = train.drop(json_cols, axis=1)

def profile(df):
    numerics = df.select_dtypes(exclude=['bool', 'object']).columns.tolist()
    try:
        display(df.hvplot.hist(numerics, height=300, width=350, subplots=True, shared_axes=False).cols(2))
    except: print("no numerics")
    for n in df.select_dtypes(['bool', 'object']).columns:
        dcounts = df[n].value_counts(normalize=True)
        dcounts_df = pd.DataFrame({'label': dcounts.index.tolist(), 'percent of total': dcounts})
        dcounts_df.reset_index(drop=True, inplace=True)
        display(dcounts_df[0:50].hvplot.bar(x='label', y='percent of total', invert=True, flip_yaxis=True, 
               height=450, width=450, ylim=(0,1), title=n))
profile(train_nonjsons)


# Now we can do the json columns. You'll see the target to predict, transactionRevenue, as part of the 'totals' display.

# In[ ]:


for jc in json_cols: 
    print(jc)
    flat_df = pd.io.json.json_normalize(train_jsons[jc].apply(json.loads)) 
    flat_df = flat_df.apply(pd.to_numeric, errors='ignore')
    profile(flat_df)


# All for now, good luck!
