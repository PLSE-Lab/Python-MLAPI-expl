#!/usr/bin/env python
# coding: utf-8

# # How to read a .csv file if you get a UnicodeDecodeError

# Step 1: Try to load the .csv file using [pd.read_csv()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

# In[ ]:


import pandas as pd
file = '/kaggle/input/demographics-of-academy-awards-oscars-winners/Oscars-demographics-DFE.csv'        
oscar_demographics = pd.read_csv(file)


# Step 2: If you get a unicode decode error then next you need to [try to determine what the character encoding is](https://chardet.readthedocs.io/en/latest/)

# In[ ]:


import chardet
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result


# Step 3: Now you can manually define the encoding in [pd.read_csv()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

# In[ ]:


oscar_demographics = pd.read_csv(file,encoding='ISO-8859-1')
oscar_demographics.head()


# In[ ]:




