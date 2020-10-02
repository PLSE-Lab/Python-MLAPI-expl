#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd

import os


# The idea of this notebook is to create a baseline submission based simply on the last 28 days of sales. Note that we could probably get a better score if we could identify, say, the start of a month and select by calendar cycle. But as it's just a baseline, lets keep it simple.
# 
# First lets load the data and ensure it is compressed into appropriate data types

# In[ ]:


input_path = "../input/m5-forecasting-accuracy"

def get_salesval_coltypes():
    keys = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'] +         [f"d_{i}" for i in range(1, 1914)]
    values = ['object', 'category', 'category', 'category', 'category', 'category'] +        ["uint16" for i in range(1, 1914)]
    return dict(zip(keys, values))

submission = pd.read_csv(os.path.join(input_path, 'sample_submission.csv'))
sales_train_val = pd.read_csv(os.path.join(input_path, 'sales_train_validation.csv'), 
                              dtype=get_salesval_coltypes())


# So lets grab the last 28 days of sales to use as our predictions.

# In[ ]:


preds = sales_train_val.iloc[:, -28:]


# Now we have 28 days of predictions, lets duplicate and stack them on top of each other so we have 56 days of predictions as required.

# In[ ]:


all_preds = pd.concat([preds, preds])


# Now we just need to match the required format of the submission file. The easiest way to do this would be simply, to take the id column from the existing submission file and add it to our prediction dataframe. For this to match up correctly, we need to reindex our current prediction dataframe so that it lines up with the index in the loaded example submission file.
# 
# Once we have a matching format, we can then save it to a CSV file ready to be submitted.

# In[ ]:


all_preds.reset_index(inplace=True, drop=True)
all_preds['id'] = submission.id
all_preds = all_preds.reindex(
        columns=['id'] + [c for c in all_preds.columns if c != 'id'], copy=False)
all_preds.columns = ['id'] + [f"F{i}" for i in range(1, 29)]

all_preds.to_csv('submission.csv', index=False)


# In[ ]:




