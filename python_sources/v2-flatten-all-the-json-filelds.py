#!/usr/bin/env python
# coding: utf-8

# This notebook loads train_v2.csv/test_v2.csv file and flatten the json fields. 

# In[ ]:


import pandas as pd 
import numpy as np 
import json 
from ast import literal_eval


# In[ ]:


df = pd.read_csv('../input/train_v2.csv', nrows=10000)


# In[ ]:


df.head()


# We have 6 special columns which should be flattened. `device`, `geoNetwork`, `totals`,`trafficSource` is standard json fields and can't be easily flattened by json module; `customDimensions` and `hit` columns can not be processed by json module, bu can be processed by ast.literal_eval. 

# ## device, geoNetwork, totals and trafficSource

# In[ ]:


json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']

def parse_json_col(raw_str):
    return pd.Series(json.loads(raw_str))

for col in json_cols:
    parsed_df = df[col].apply(parse_json_col)
    parsed_df.columns = [f'{col}_{x}' for x in parsed_df.columns]
    df = pd.concat([df, parsed_df], axis=1)
    df.drop(col, axis=1, inplace=True)


# In[ ]:


df.shape


# Let's check the new columns. We found that `trafficSource_adwordsClickInfo` is also a json column which should be flattened.

# In[ ]:


df.filter(regex='.*_.*', axis=1).head()


# In[ ]:


trafficSource_adwordsClickInfo_df = df.trafficSource_adwordsClickInfo.apply(pd.Series)
trafficSource_adwordsClickInfo_df.columns = [f'trafficSource_adwordsClickInfo_{x}' for x in trafficSource_adwordsClickInfo_df.columns]
df = pd.concat([df, trafficSource_adwordsClickInfo_df], axis=1)
df.drop('trafficSource_adwordsClickInfo', axis=1, inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.filter(regex='trafficSource_adwordsClickInfo_.*', axis=1).head()


# ## customDimensions

# In[ ]:


# for customDimensions and hits columns
def parse_special_col(raw_str):
    lst = literal_eval(raw_str)
    if isinstance(lst, list) and lst:
        return pd.Series(lst[0])
    else:
        return pd.Series({})


# In[ ]:


customDimensions_df = df.customDimensions.apply(parse_special_col)
customDimensions_df.columns = [f'customDimensions_{x}' for x in customDimensions_df.columns]
df = pd.concat([df, customDimensions_df], axis=1)
df.drop('customDimensions', axis=1, inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.filter(regex='customDimensions_.*', axis=1).head()


# ## hits

# In[ ]:


hits_df = df.hits.apply(parse_special_col)
hits_df.columns = [f'hits_{x}' for x in hits_df.columns]
df = pd.concat([df, hits_df], axis=1)
df.drop('hits', axis=1, inplace=True)


# In[ ]:


df.shape


# In[ ]:


df.filter(regex='hits_.*', axis=1).head()


# `hits_experiment`, `hits_customVariables`, `hits_customMetrics`, `hits_publisher_infos`, `hits_customDimensions` are empty, we can drop it.

# In[ ]:


df.drop(['hits_experiment', 'hits_customVariables', 'hits_customMetrics', 'hits_publisher_infos', 'hits_customDimensions'], axis=1, inplace=True)


# `hits_page`, `hits_transaction`, `hits_item`, `hits_appInfo`, `hits_exceptionInfo`, `hits_eCommerceAction`, `hits_social`, `hits_contentGroup`, `hits_promotionActionInfo` are python dict, we can should flatten it.

# In[ ]:


dict_cols = ['hits_page', 'hits_transaction', 'hits_item', 'hits_appInfo', 
        'hits_exceptionInfo', 'hits_eCommerceAction', 'hits_social', 'hits_contentGroup', 'hits_promotionActionInfo']
for col in dict_cols:
    parsed_df = hits_df[col].apply(pd.Series)
    parsed_df.columns = [f'{col}_{x}' for x in parsed_df.columns]
    df = pd.concat([df, parsed_df], axis=1)
    df.drop(col, axis=1, inplace=True)


# `hits_product`, `hits_promotion` are python list, we should flatten it.

# In[ ]:


def parse_list(x):
    if isinstance(x, list) and x:
        return pd.Series(x[0])
    else:
        return pd.Series({})
    
for col in ['hits_product', 'hits_promotion']:
    parsed_df = hits_df[col].apply(parse_list)
    parsed_df.columns = [f'{col}_{x}' for x in parsed_df.columns]
    df = pd.concat([df, parsed_df], axis=1)
    df.drop(col, axis=1, inplace=True)


# In[ ]:


df.shape


# ## Pack it to a function
# I have put the code in one function so you can copy it easily.

# In[ ]:


def flatten(in_csv, out_csv, nrows=None):
    df = pd.read_csv(in_csv, dtype=np.object, nrows=nrows)
    # json columns
    json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']

    def parse_json_col(raw_str):
        return pd.Series(json.loads(raw_str))
    
    for col in json_cols:
        parsed_df = df[col].apply(parse_json_col)
        parsed_df.columns = [f'{col}_{x}' for x in parsed_df.columns]
        df = pd.concat([df, parsed_df], axis=1)
        df.drop(col, axis=1, inplace=True)
    
    # trafficSource_adwordsClickInfo
    trafficSource_adwordsClickInfo_df = df.trafficSource_adwordsClickInfo.apply(pd.Series)
    trafficSource_adwordsClickInfo_df.columns = [f'trafficSource_adwordsClickInfo_{x}' for x in trafficSource_adwordsClickInfo_df.columns]
    df = pd.concat([df, trafficSource_adwordsClickInfo_df], axis=1)
    df.drop('trafficSource_adwordsClickInfo', axis=1, inplace=True)

    # customDimensions
    def parse_customDimensions(raw_str):
        lst = literal_eval(raw_str)
        if isinstance(lst, list) and lst:
            return pd.Series(lst[0])
        else:
            return pd.Series({})
    
    customDimensions_df = df.customDimensions.apply(parse_customDimensions)
    customDimensions_df.columns = [f'customDimensions_{x}' for x in customDimensions_df.columns]
    df = pd.concat([df, customDimensions_df], axis=1)
    df.drop('customDimensions', axis=1, inplace=True)

    # hits
    def parse_hits(raw_str):
        lst = literal_eval(raw_str)
        if isinstance(lst, list) and lst:
            return pd.Series(lst[0])
        else:
            return pd.Series({})
    
    hits_df = df.hits.apply(parse_hits)
    hits_df.columns = [f'hits_{x}' for x in hits_df.columns]
    df = pd.concat([df, hits_df], axis=1)
    df.drop('hits', axis=1, inplace=True)

    # 'hits_page', 'hits_transaction', 'hits_item', 'hits_appInfo',
    # 'hits_exceptionInfo', 'hits_eCommerceAction', 'hits_social', 'hits_contentGroup', 'hits_promotionActionInfo'
    dict_cols = ['hits_page', 'hits_transaction', 'hits_item', 'hits_appInfo', 
        'hits_exceptionInfo', 'hits_eCommerceAction', 'hits_social', 'hits_contentGroup', 'hits_promotionActionInfo']
    for col in dict_cols:
        parsed_df = hits_df[col].apply(pd.Series)
        parsed_df.columns = [f'{col}_{x}' for x in parsed_df.columns]
        df = pd.concat([df, parsed_df], axis=1)
        df.drop(col, axis=1, inplace=True)
    
    # 'hits_experiment', 'hits_customVariables', 'hits_customMetrics', 'hits_publisher_infos', 'hits_customDimensions' are empty
    df.drop(['hits_experiment', 'hits_customVariables', 'hits_customMetrics', 'hits_publisher_infos', 'hits_customDimensions'], axis=1, inplace=True)

    # 'hits_product', 'hits_promotion'
    def parse_list(x):
        if isinstance(x, list) and x:
            return pd.Series(x[0])
        else:
            return pd.Series({})
    
    for col in ['hits_product', 'hits_promotion']:
        parsed_df = hits_df[col].apply(parse_list)
        parsed_df.columns = [f'{col}_{x}' for x in parsed_df.columns]
        df = pd.concat([df, parsed_df], axis=1)
        df.drop(col, axis=1, inplace=True)

    df.to_csv(out_csv, index=False)

    return df.shape


# It takes about 1~2h to flatten the train_v2.csv and test_v2.csv. Have fun!
