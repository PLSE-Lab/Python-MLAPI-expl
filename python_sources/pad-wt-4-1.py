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


# In[ ]:


## Drop any duplicates ad check how many rows were preserved


# In[ ]:


## Get data types for all columns

# Another method is to use info()
#input_df.info()


# In[ ]:


## Confirm Total Net Amount is a String type


# In[ ]:


# Rename columns by replacing spaces with underscores and lowercasing everything for easier column references
input_df.columns = [ col.replace(' ','_').lower() for col in input_df.columns]


# In[ ]:


## Use value_counts and see what this data looks like


# In[ ]:


## Confirm whether Total Amount and Item Net Amount is affected at the same occurrences as Total Net Amount for '--error--' values


# In[ ]:


## Get rid of the data with --error-- and start a new DataFrame that will be a clean data set for analysis


# In[ ]:


## Convert item_net_amount,total_net_amount, and total_amount to float
clean_df.item_net_amount = 
clean_df.total_net_amount = 
clean_df.total_amount = 


# In[ ]:


## Use Series apply function to convert values from cents to dollars

def toDollars(val):
    return val/100

clean_df.item_net_amount = 
clean_df.total_net_amount = 
clean_df.total_amount = 

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

clean_df.head(5)

# Alternative method: Use insert command to insert the new column and a column position we want
# clean_df.insert(0, "key", key_col)


# In[ ]:


## Check if item_names all match up to each other


# In[ ]:


## Answer this question: Knowing the context of who these transactions are from, is the data totally accurate?

