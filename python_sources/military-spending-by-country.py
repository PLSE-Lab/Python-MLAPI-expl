#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.display.float_format = '{:,}'.format
df_military = pd.read_csv('../input/military-expenditure-of-countries-19602019/Military Expenditure.csv',index_col=[0])


# # Biggest military spenders in 2018

# In[ ]:


df_military[df_military['Type'] == 'Country'].sort_values(by='2018',ascending=False)[['2018']][:10]


# # Largest $$$ Increase in military spending in 2018

# In[ ]:


(df_military[df_military['Type'] == 'Country']['2018'] - df_military[df_military['Type'] == 'Country']['2017']).sort_values(ascending=False)[:10]


# # Largest % increase in military spending in 2018

# In[ ]:


((df_military[df_military['Type'] == 'Country']['2018'] - df_military[df_military['Type'] == 'Country']['2017'])/df_military[df_military['Type'] == 'Country']['2018']).sort_values(ascending=False)[:10].round(2)

