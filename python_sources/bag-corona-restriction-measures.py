#!/usr/bin/env python
# coding: utf-8

# # Overview
# A look at the public transportation data from Open Data Zurich (https://data.stadt-zuerich.ch/) to evaluate how much has changed since the measures went into place

# In[ ]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})
plt.rcParams['image.cmap'] = 'gray' # grayscale looks better
from itertools import cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


# In[ ]:


hardbruecke_df = pd.read_csv('../input/zurich-mobility-data-2020/frequenzen_hardbruecke_2020.csv')
hardbruecke_df['Date'] = pd.to_datetime(hardbruecke_df['Timestamp'])
hardbruecke_df['WeekNo'] = hardbruecke_df['Date'].dt.weekofyear
hardbruecke_df['DayOfWeek'] = hardbruecke_df['Date'].dt.dayofweek
hardbruecke_df['DayNo'] = hardbruecke_df['Date'].dt.dayofyear
hardbruecke_df['Month'] = hardbruecke_df['Date'].dt.month
hardbruecke_df['Day'] = hardbruecke_df['Date'].dt.day

hardbruecke_df.sample(3)


# In[ ]:


sns.heatmap(hardbruecke_df.pivot_table(columns='WeekNo', index='DayOfWeek', values='Out', aggfunc='sum'), 
            annot=True, fmt='2.0f',
           )


# In[ ]:


daily_summary_df = hardbruecke_df.    groupby(['WeekNo', 'DayOfWeek', 'DayNo', 'Name']).    apply(lambda x: pd.Series(
    {'Total_Out': sum(x['Out']),
     'Total_in': sum(x['In'])
    }
)).reset_index()
daily_summary_df.head(3)


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(20, 5))
sns.lineplot(data=daily_summary_df, x='DayNo', y='Total_Out', hue='Name', ax=ax1)


# In[ ]:


sns.heatmap(
    daily_summary_df.pivot_table(index='DayNo', columns='Name', values='Total_Out', aggfunc='sum')
)


# # VBZ Guests
# - To be continued

# In[ ]:


passangers_df = pd.read_csv('../input/zurich-mobility-data-2020/reisende.csv', sep=";")


# In[ ]:


passangers_df.shape


# In[ ]:


passangers_df.sample(3).T


# In[ ]:


passangers_df.query('Linien_Id==80').plot.scatter(x='Sequenz', y='Besetzung')


# In[ ]:


sns.lineplot(data=passangers_df.query('Linien_Id==80'), x='Sequenz', y='Besetzung')


# In[ ]:




