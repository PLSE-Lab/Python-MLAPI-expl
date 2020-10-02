#!/usr/bin/env python
# coding: utf-8

# ## Walkthrough 4-1: Clean Data

# In[ ]:


import numpy as np
import pandas as pd

pd.options.display.max_rows = 10


# In[ ]:


# Read in Excel data
input_df = pd.read_excel('../input/transactions.xlsx','data')


# In[ ]:


## Check how many rows data set contains
input_df.shape


# In[ ]:


## Drop any duplicates ad check how many rows were preserved
input_df = input_df.drop_duplicates()
input_df.shape


# In[ ]:


## Get data types for all columns
input_df.dtypes

# Another way is to use input_df.info()
#input_df.info()


# In[ ]:


## Confirm Total Net Amount is a String type
type(input_df['Total Net Amount'].iat[0])


# In[ ]:


# Rename columns by replacing spaces with underscores and lowercasing everything for easier column references
input_df.columns = [ col.replace(' ','_').lower() for col in input_df.columns]


# In[ ]:


## Use value_counts and see what this data looks like
input_df.total_net_amount.value_counts()


# In[ ]:


## Confirm whether Total Amount and Item Net Amount is affected at the same occurrences as Total Net Amount for '--error--' values
input_df[input_df.total_net_amount == '--error--'].head(15)


# In[ ]:


## Get rid of the data with --error-- and start a new DataFrame that will be a clean data set for analysis
clean_df = input_df[input_df.total_net_amount != '--error--'].copy()
clean_df.head(15)


# In[ ]:


## Convert item_net_amount,total_net_amount, and total_amount to float
clean_df.item_net_amount = clean_df.item_net_amount.astype(float)
clean_df.total_net_amount = clean_df.total_net_amount.astype(float)
clean_df.total_amount = clean_df.total_amount.astype(float)


# In[ ]:


## Use Series apply function to convert values from cents to dollars

def toDollars(val):
    return val/100

clean_df.item_net_amount = clean_df.item_net_amount.apply(toDollars)
clean_df.total_net_amount = clean_df.total_net_amount.apply(toDollars)
clean_df.total_amount = clean_df.total_amount.apply(toDollars)

clean_df.head(5)


# In[ ]:


## Use DataFrame apply to create a new column dependent the values in each row of dat

# Define create key function
def createKey(series):
    return '{store}_{itemname}_{price:.1f}_{qty}'.format(store=series.store,
                                                     itemname=series.item_name,
                                                     price=series.item_net_amount,
                                                     qty=series.quantity_sold)

## Do row-wise apply and assign result of createKey to a new "key" column
key_col = clean_df.apply(createKey,axis=1)
clean_df["key"] = key_col
clean_df.head(5)

# Alternative method: Use insert command to insert the new column and a column position we want
# clean_df.insert(0, "key", key_col)


# In[ ]:


## Check if item_names all match up to each other
clean_df.item_name.value_counts()


# In[ ]:


## Answer this question: Knowing the context of who these transactions are from, is the data totally accurate?

