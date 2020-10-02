#!/usr/bin/env python
# coding: utf-8

# Hi people !
# 
# The aim of this kernel is to show a very simple method **to reduce our dataframe size by 55%**. The principle is to convert numeric data from float64 ou int64 to more memory efficient types.
# 
# I'll use a sample of 59,633,310 rows as an example (every clicks from 2017-11-07 00:00:00 to 2017-11-07 23:59:59).

# In[3]:


import pandas as pd
import numpy as np
import gc


# **Importation of a entire day data :**

# In[6]:


# Rows importation
df = pd.read_csv('../input/train.csv', skiprows = 9308568, nrows = 59633310)

# Header importation
header = pd.read_csv('../input/train.csv', nrows = 0) 
df.columns = header.columns
df

# Cleaning
del header
gc.collect()

# And check his size        
print("The created dataframe contains", df.shape[0], "rows.")   


# **The magical function :**

# In[7]:


total_before_opti = sum(df.memory_usage())

# Type's conversions
def conversion (var):
    if df[var].dtype != object:
        maxi = df[var].max()
        if maxi < 255:
            df[var] = df[var].astype(np.uint8)
            print(var,"converted to uint8")
        elif maxi < 65535:
            df[var] = df[var].astype(np.uint16)
            print(var,"converted to uint16")
        elif maxi < 4294967295:
            df[var] = df[var].astype(np.uint32)
            print(var,"converted to uint32")
        else:
            df[var] = df[var].astype(np.uint64)
            print(var,"converted to uint64")


# **Function's launch :**

# In[8]:


for v in ['ip', 'app', 'device','os', 'channel', 'is_attributed'] :
    conversion(v)


# **We can now print the results :**

# In[13]:


print("Memory usage before optimization :", str(round(total_before_opti/1000000000,2))+'GB')
print("Memory usage after optimization :", str(round(sum(df.memory_usage())/1000000000,2))+'GB')
print("We reduced the dataframe size by",str(round(((total_before_opti - sum(df.memory_usage())) /total_before_opti)*100,2))+'%')


# **You can find my EDA here **(which really need so visibility ! :p ) : https://www.kaggle.com/valentinw/eda-of-the-whole-dataset
# 
# Thanks to this kernel :  https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65/notebook
# 
# Thanks for your time !
