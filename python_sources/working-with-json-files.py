#!/usr/bin/env python
# coding: utf-8

# # Working with JSON files in Python
# Working with JSON files isn't the most fun.  While pandas has the read_json method that is useful for reading the .json file into a dataframe, we are often left with lists or dictionaries inside of columns.  Since nested column values aren't really helpful for analzying data, we'll explore some methods for unpacking the json and creating clean and orderly dataframes.

# In[ ]:


import numpy as np
import pandas as pd
import ijson
from pandas.io.json import json_normalize


# In[ ]:


get_ipython().run_cell_magic('bash', '', '# we can use %%bash magic to print a preview of our file\n\nhead ../input/roam_prescription_based_prediction.jsonl')


# In[ ]:


# read in data
raw_data = pd.read_json("../input/roam_prescription_based_prediction.jsonl",
                        lines=True,
                        orient='columns')
print(raw_data.shape)
raw_data.head()


# We can see from above that we have nested values inside our cells.  There are several options for extracting these values.  In this kernel we will explore using list comprehensions and json_normalize.
# 
# ## Extract Prescriber Data
# ### List Comprehension

# In[ ]:


get_ipython().run_line_magic('time', 'provider = pd.DataFrame([md for md in raw_data.provider_variables])')
provider.head()


# In[ ]:


# add npi as index
provider['npi'] = raw_data.npi
provider.set_index('npi', inplace=True)
provider.head()


# ### JSON Normalize

# In[ ]:


get_ipython().run_line_magic('time', 'provider = json_normalize(data=raw_data.provider_variables)')
provider.head()


# ## Extract Rx Data
# ### List Comprehension

# In[ ]:


get_ipython().run_line_magic('time', 'rx_counts = pd.DataFrame([rx for rx in raw_data.cms_prescription_counts])')


# In[ ]:


print(rx_counts.shape)
rx_counts.head()

