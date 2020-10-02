#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.express as px
from mpl_toolkits.basemap import Basemap

sb.set_style('white')
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = (10,6)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Read the data
covid = pd.read_csv('../input/coronavirus-cases-in-india-june2020/Covid19-India.csv')


# In[ ]:


display(covid)


# In[ ]:


covid.info()


# ## Active Cases in Top 3 State

# In[ ]:


order_active = pd.DataFrame()
order_active[['State', 'Active Cases']] = covid[['State', 'Active Cases']].groupby(['State'], as_index=False).mean().sort_values(by='Active Cases', ascending=False)

top_3 = order_active.iloc[:3]
plt.subplots(figsize=(9, 7))
sb.barplot(top_3['State'], top_3['Active Cases'], palette='spring')
plt.xticks(rotation=45)


# ## Active Cases in Each State

# In[ ]:


order_active


# In[ ]:


plt.figure(1)
f, ax = plt.subplots(figsize=(13, 12))
sb.barplot(y=order_active['State'], x=order_active['Active Cases'], palette='spring', saturation=0.85)
ax.set(ylabel='')

plt.figure(2)
f, ax = plt.subplots(figsize=(13, 11))
sb.pointplot(y=order_active['State'], x=order_active['Active Cases'], palette='spring', markers='+')
ax.set(ylabel='')


# ## Cured/Migrated Cases in Each State

# In[ ]:


order_cured = pd.DataFrame()
order_cured[['State', 'Cured/Migrated']] = covid[['State', 'Cured/Migrated']].groupby(['State'], as_index=False).mean().sort_values(by='Cured/Migrated', ascending=False)
order_cured


# In[ ]:


plt.figure(1)
f, ax = plt.subplots(figsize=(13, 12))
sb.barplot(y=order_cured['State'], x=order_cured['Cured/Migrated'], palette='spring', saturation=0.85)
ax.set(ylabel='')

plt.figure(2)
f, ax = plt.subplots(figsize=(13, 11))
sb.pointplot(y=order_cured['State'], x=order_cured['Cured/Migrated'], palette='spring', markers='+')
ax.set(ylabel='')


# ## Deaths in Each State

# In[ ]:


order_died = pd.DataFrame()
order_died[['State', 'Deaths']] = covid[['State', 'Deaths']].groupby(['State'], as_index=False).mean().sort_values(by='Deaths', ascending=False)
order_died


# In[ ]:


plt.figure(1)
f, ax = plt.subplots(figsize=(13, 12))
sb.barplot(y=order_died['State'], x=order_died['Deaths'], palette='magma', saturation=0.85)
ax.set(ylabel='')

plt.figure(2)
f, ax = plt.subplots(figsize=(13, 11))
sb.pointplot(y=order_died['State'], x=order_died['Deaths'], palette='magma', markers='+')
ax.set(ylabel='')


# ## Total Confirmed Cases in Each State

# In[ ]:


order_total = pd.DataFrame()
order_total[['State', 'Total Confirmed Cases']] = covid[['State', 'Total Confirmed Cases']].groupby(['State'], as_index=False).mean().sort_values(by='Total Confirmed Cases', ascending=False)
order_total


# In[ ]:


plt.figure(1)
f, ax = plt.subplots(figsize=(13, 12))
sb.barplot(y=order_total['State'], x=order_total['Total Confirmed Cases'], palette='spring', saturation=0.85)
ax.set(ylabel='')

plt.figure(2)
f, ax = plt.subplots(figsize=(13, 11))
sb.pointplot(y=order_total['State'], x=order_total['Total Confirmed Cases'], palette='magma', markers='+')
ax.set(ylabel='')


# We can see that Maharashtra is top in each plot and this shows how much it is affected by Coronavirus.

# In[ ]:


order_cured_total = covid[['State', 'Cured/Migrated', 'Total Confirmed Cases']]
order_cured_total.sort_values(by='Total Confirmed Cases', ascending=False, inplace=True)


# In[ ]:


f, ax = plt.subplots(figsize=(13, 12))

sb.set_color_codes("pastel")
sb.barplot('Total Confirmed Cases', 'State', data=order_cured_total, label='Total', color='r')

sb.set_color_codes("muted")
sb.barplot('Cured/Migrated', 'State', data=order_cured_total, label='Cured', color='g')

ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 110000), ylabel="",
       xlabel="Cases")
sb.despine(left=True, bottom=True)

