#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from warnings import filterwarnings
filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpp


# In[ ]:


ibm_df = pd.read_csv('../input/useu-stocks/IBM_max.csv')


# In[ ]:


ibm_df.isnull().sum()


# ## Note:
# SMA10 = Simple Moving Average for 10 days. <br/>
# SMA50 = Simple Moving Average for 50 days. <br/>
# Return = The Return on Stock caused changing in price. <br/>
# Close-1 = yesterday Close price. <br/>
# Trend = the Trend 1 for Up or 0 for Down. <br/>
# ChngVolume = Changing in Volume. <br/>
# TrendVolume = Trend Changing in Volume 1 for Up 0 for Down. <br/>

# In[ ]:


ibm_df['SMA10'] = ibm_df['Close'].rolling(10).mean()
ibm_df['SMA50'] = ibm_df['Close'].rolling(50).mean()
Close_1 = ibm_df['Close'].shift(1)
ibm_df['Return'] = ibm_df['Close'] - Close_1
ibm_df['ReturnPerc'] = (ibm_df['Return'] / ibm_df['Close']) * 100
ibm_df['Trend'] = [1 if Rtn > 0 else 0 for Rtn in ibm_df['Return']]
Volume_1 = ibm_df['Volume'].shift(1)
ibm_df['ChngVolume'] = ibm_df['Volume'] - Volume_1
ibm_df['TrendVolume'] = [1 if Vol > 0 else 0 for Vol in ibm_df['ChngVolume']]


# In[ ]:


ibm_df.sort_index(ascending=False, inplace=True)


# In[ ]:


ibm_df.head()


# In[ ]:


mpp.figure(figsize=(26, 14))
grid = mpp.GridSpec(4, 1, hspace=0)
mpp.subplot(grid[:3])
ibm_df.iloc[:100, 4].plot()
mpp.subplot(grid[3])
ibm_df.iloc[:100, 6].plot(kind='bar')


# In[ ]:


dt1 = ibm_df[['Close', 'SMA10', 'SMA50']].iloc[:100, :]
mpp.figure(figsize=(24, 12))
mpp.plot(dt1)
mpp.legend(['Close', 'SMA10', 'SMA50'])
mpp.show()

