#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
get_ipython().run_line_magic('matplotlib', 'inline')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


PATH = '/kaggle/input/m5-forecasting-accuracy/'
stv = pd.read_csv(f'{PATH}sales_train_validation.csv')
ss = pd.read_csv(f'{PATH}sample_submission.csv')
cal = pd.read_csv(f'{PATH}calendar.csv')
sell_prices = pd.read_csv(f'{PATH}sales_train_validation.csv')


# # Function: Adds column to sales_train_validation which shows the first day with a positive value for each series.

# In[ ]:


def fsd(stv): 
    """Get the first non zero entry of a row in the stv DataFrame, 
    and add it as the 'fsd' column, giving first sale day for M5 series"""
    stv['fsd'] = 0 # initialize at zero, awaiting update from loop
    for i in range(1,1914):
        # Does fsd still need first day value? If yes, show True
        fsd_as_reverse_bool = ~(stv.fsd.astype(bool))
        # Convert to int to multiply
        fsd_int = fsd_as_reverse_bool.astype('int')
        # Does day i have a non-zero sale? Convert to int
        d_i = stv['d_' + str(i)].astype('bool').astype('int')
        # Update fsd column 
        stv['fsd'] += i * (fsd_int) * d_i


# In[ ]:


fsd(stv)


# # Adding on the effective time series length as a column

# In[ ]:


# Set the time series length column
stv['ts_length'] = 1914 - stv.fsd


# In[ ]:


stv.ts_length.head()


# # Histogram of the effective time series lengths of all items at all stores.

# In[ ]:


tsl = stv.ts_length
tsl.hist(figsize=(10,7))


# # Csv file with id, fsd, and ts_length in output

# In[ ]:


stv[['id', 'fsd', 'ts_length']].to_csv('id_fsd_ts_length.csv', index=False)


# In[ ]:




