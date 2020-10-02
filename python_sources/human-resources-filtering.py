#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# Reading data and deleting empty cells.

# In[ ]:


data = pd.read_csv("../input/human-resources-data-set/HRDataset_v13.csv").dropna()


# Removing dublacited rows and columns.

# In[ ]:


cols = list( pd.Series(data.columns).unique() )
colsToDrop = list( set(data.columns)-set(cols) )

data = data.drop( labels=colsToDrop, axis=1 )


# So we have data with next parameters:

# In[ ]:


print( list(data.columns) )


# So if you need add some data enter parameters using this sequence.
