#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

import seaborn as sns
sns.set(style="whitegrid")


# In[ ]:


df = pd.read_csv("../input/Indicators.csv")


# ## Forest area(in sq. km) of India over the period of time
# - Indicator used is AG.LND.FRST.K2(Forest area (sq. km))

# In[ ]:


df_india_fc = df[(df.CountryName=='India')&(df.IndicatorCode=='AG.LND.FRST.K2')]
fig = plt.figure()
plt.plot(df_india_fc.Year,df_india_fc.Value,'o-',color='g')
plt.xlabel('Years')
plt.ylabel('forest area in sq. km')
plt.title('India forest cover area over time')
fig.savefig('forestarea.png')


# ## Forest area (% of land area) of India over the period of time
# - Indicator used is AG.LND.FRST.ZS(Forest area (% of land area))

# In[ ]:


df_india_fc_landperc = df[(df.CountryName=='India')&(df.IndicatorCode=='AG.LND.FRST.ZS')]
fig = plt.figure()
plt.plot(df_india_fc_landperc.Year,df_india_fc_landperc.Value,'o-',color='g')
plt.xlabel('Years')
plt.ylabel('forest area in sq. km')
plt.title('India forest cover area as percentage of land over time')
fig.savefig('forestcover_percetange_of_land.png')


# ## Some notes:
# Though it seems quite opposite to what I was expecting and even any of us would be expecting since I have 
# myself witnessed all this urbanization by myself which is responsible for the deforestation.
# 
# I think there is some issue with the data and suspect whether it's even accurate or not.

# In[ ]:




