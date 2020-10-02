#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt 
import os
import plotly.graph_objects as go
import plotly.express as px


# In[ ]:


#Loading Data
lvl = pd.read_csv("../input/chennai_reservoir_levels.csv")
rain = pd.read_csv("../input/chennai_reservoir_rainfall.csv")


# In[ ]:


fig = px.line(lvl, x='Date', y='POONDI')
fig.show();


# In[ ]:


fig = px.line(lvl, x='Date', y='CHOLAVARAM')
fig.show();


# In[ ]:


fig = px.line(lvl, x='Date', y='REDHILLS')
fig.show();


# In[ ]:


fig = px.line(lvl, x='Date', y='CHEMBARAMBAKKAM')
fig.show();


# ****Water level data has a seasonality that can be seen in these curves. Although this Trend is keep on decreasing and currently, Only Poondi Reservoir has some water left. All other three reservoirs are running empty.
# On 2014 December approx, there's a sudden growth in water level that is very unnatural. Maybe supplied from some other region. Downfall in water level is significant in 2019 Feb-July. Evidence of water shortage can be noticed in all four reservoir between a period of 2017-2018 year transition.
# Also the dry period generally ends at the starting of December just after the dry period which generally occurs in october.****

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15,5))
ax1.plot(rain.REDHILLS, 'r')
ax2.plot(rain.CHOLAVARAM, 'b')
plt.show();
#Water reservoir level in million cubic feet (mcf), y axis


# In[ ]:


fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))
ax1.plot(rain.CHOLAVARAM)
ax2.plot(rain.POONDI,'g')
plt.show();
#Water reservoir level in million cubic feet (mcf), y axis


# In[ ]:


fig ,(ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
ax1.plot(rain.REDHILLS,'r')
ax2.plot(rain.CHEMBARAMBAKKAM);
#Water reservoir level in million cubic feet (mcf), y axis


# In[ ]:


fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(15,5))
ax1.plot(rain.CHEMBARAMBAKKAM ,'b') 
ax2.plot(rain.POONDI )
plt.show;
#Water reservoir level in million cubic feet (mcf), y axis


# In[ ]:


corr = lvl.corr()
corr.style.background_gradient(cmap='coolwarm')


# In[ ]:


corr = rain.corr()
corr.style.background_gradient(cmap='coolwarm')


# **Seasonality in graphs can be explained using this evidence. Signs of heavy coorelation is observed in this coorelation matrix.**

# In[ ]:


rain.Date = pd.to_datetime(rain.Date)
rain.set_index('Date', inplace=True)


# In[ ]:


rain.plot(figsize=(20,10), linewidth=3, fontsize=15)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Rain Level', fontsize=15);


# **Above bar chart describes the water availity during the last 15 years in 4 different reservoirs located in the cities of chennai. **

# In[ ]:


rain.total = rain.POONDI + rain.CHOLAVARAM + rain.REDHILLS + rain.CHEMBARAMBAKKAM
rain.total.plot(figsize=(20,10), linewidth=3, fontsize=15)
plt.xlabel('Year', fontsize=15)
plt.ylabel('Rain Level', fontsize=15);


# **Total water availibility is very low in 2019. Perodic cycle is disturbed in current phase. Significant downfall in total water level has been observed since 2015 and it keeps on decreasing till 2019.**
