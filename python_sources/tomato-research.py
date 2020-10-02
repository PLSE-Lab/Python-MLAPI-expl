#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from statistics import mode, mean
import copy


# In[ ]:


data = pd.read_csv("/kaggle/input/vegetablepricetomato/Price of Tomato Karnataka(2016-2018).csv")


# ### Data preprocessing###

# In[ ]:


data.isnull().sum()


# In[ ]:


data.dropna(subset=['Variety'], inplace=True)
data.fillna(method='ffill', inplace=True)


# In[ ]:


data.isnull().sum()


# ### Financial analysis ###

# In[ ]:


data['Arrival Date'] = pd.to_datetime(data['Arrival Date'])
df_gb = data.groupby(['Arrival Date']).sum()
df_gb['periods']=df_gb.index.to_period("M")
df_p = df_gb.groupby(['periods']).sum()
print(df_p.head(8))


# ![image.png](attachment:image.png)
# 
# Production volume of tomatoes across India from FY 2015 to FY 2018(in million metric tons)

# In[ ]:


fig, ax = plt.subplots(figsize=(16,5))
ax.bar(df_gb.index, df_gb['Arrivals (Tonnes)'])
ax.set_ylabel('Arrivals (Tonnes)')
plt.gcf().autofmt_xdate()


# The supply in the markets has declined steadily from 2016 to early 2019. While tomato production in India was increasing. Perhaps the deliveries were to other stores or for export.

# In[ ]:


data['Modal Price(Rs./Quintal)'] = data['Modal Price(Rs./Quintal)'].astype('int')


# In[ ]:


fig, ax = plt.subplots(figsize=(9,9))
ax.scatter(data['Arrivals (Tonnes)'], data['Market'], marker='*');
ax.set(xlabel='Arrivals (Tonnes) in market', ylabel='Market', title='The volume of supply of tomatoes to the market');


# The largest number of tomatoes were in markets: Kolar, Chikkamagalore, Chickkaballapura, Chintamani.

# In[ ]:


data['revenue'] = data['Arrivals (Tonnes)']*data['Modal Price(Rs./Quintal)']*10


# In[ ]:


dt = pd.pivot_table(data, values='revenue', index=['Market'], aggfunc=np.sum)


# In[ ]:


fig, ax = plt.subplots(figsize=(18, 4))
ax.scatter(dt.index, dt);
plt.gcf().autofmt_xdate()


# The largest revenue were in markets: Kolar, Chintamani, Mysore (Bandipalya).
