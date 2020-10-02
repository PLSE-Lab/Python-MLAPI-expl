#!/usr/bin/env python
# coding: utf-8

# This kernel will show you how to load complete parquet files into Pandas and how to read just a subset of the data.

# In[ ]:


import pandas as pd
import pyarrow.parquet as pq
import os


# In[ ]:


os.listdir('../input')


# Reading the entire parquet file is a one liner. Parquet will handle the parallelization and recover the original int8 datatype.

# In[ ]:


train = pq.read_pandas('../input/train.parquet').to_pandas()


# Note that each column contains a single 800,000 measurement signal.

# In[ ]:


train.info()


# If we wanted to instead load a subset of the data, we could load the metadata file to get the names of a few signals of interest. However, since the column names are just the column enumerations, we'll skip that step for now and just load the first five.

# In[ ]:


train.columns[:5]


# In[ ]:


subset_train = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(5)]).to_pandas()


# In[ ]:


subset_train.info()


# In[ ]:


subset_train.head()


# In[ ]:




