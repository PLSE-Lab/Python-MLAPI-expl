#!/usr/bin/env python
# coding: utf-8

# # **Hey there! This is my notebook where I will learn data visualization. One thing at a time. Hehe! Let's start. [6/17/20]**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Using geojason for plotting India. 

# The jason file is sourced from GitHub

# In[ ]:


df_india = gpd.read_file('https://raw.githubusercontent.com/geohacker/india/master/taluk/india_taluk.geojson')
df_maharashtra = df_india[df_india['NAME_1']== 'Maharashtra']
df_jalgaon = df_maharashtra[df_maharashtra['NAME_2']=='Jalgaon']
df_jamner = df_jalgaon[df_jalgaon['NAME_3']=='Jamner']


# In[ ]:


plt.rcParams['figure.figsize'] = (30, 10)
ax = df_india.plot(color='blue')


# In[ ]:


plt.rcParams['figure.figsize'] = (30, 10)
ax = df_maharashtra.plot(color='blue')


# In[ ]:


plt.rcParams['figure.figsize'] = (30, 10)
ax = df_jalgaon.plot(color='blue')


# In[ ]:


plt.rcParams['figure.figsize'] = (30, 10)
ax = df_jamner.plot(color='blue')


# In[ ]:


plt.rcParams['figure.figsize'] = (30, 10)
ax = df_maharashtra[df_maharashtra['NAME_2']=='Latur'].plot(color='blue')


# In[ ]:




