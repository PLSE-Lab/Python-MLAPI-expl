#!/usr/bin/env python
# coding: utf-8

# # Greenland Surface Melt Extent
# The [NSIDC has a very interesting melt-chart](https://nsidc.org/greenland-today/greenland-surface-melt-extent-interactive-chart/) but unfortunately doesn't quite show all the steps taken to make it. So here we attempt to reproduce the chart from the raw data by pulling it all from their API.
# ## Acknowledgements
# - Greenland Ice Sheet Today is produced at the National Snow and Ice Data Center by Ted Scambos, Julienne Stroeve, and Lora Koenig with support from NASA. NSIDC thanks Jason Box, Xavier Fettweis, and Thomas Mote for data and collaboration.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["figure.dpi"] = 125
plt.rcParams["font.size"] = 14
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.style.use('ggplot')
sns.set_style("whitegrid", {'axes.grid': False})
from itertools import cycle
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


# In[ ]:


import pandas as pd
import plotly as py
import plotly_express as px


# # Load Data
# Here we pull down all the years

# In[ ]:


melt_urls = ['https://nsidc.org/api/greenland/melt_area/{:04d}'.format(x) for x in range(1979, 2020)]
def read_melt_df(c_url):
    c_melt = pd.read_json(c_url, typ="series")
    return pd.DataFrame({'date': c_melt.index, 
                         'MeltAreaSqKm': c_melt.values})
all_melt_df = pd.concat([read_melt_df(c_url) for c_url in melt_urls]).reset_index(drop=True)
all_melt_df.sample(3)


# In[ ]:


all_melt_df['Month'] = all_melt_df['date'].dt.month*1.0
all_melt_df['MonthDecimal'] = all_melt_df['Month']*1.0+all_melt_df['date'].dt.day/all_melt_df['date'].dt.daysinmonth
all_melt_df['Month Name'] = all_melt_df['date'].dt.month_name()
all_melt_df['Year'] = all_melt_df['date'].dt.year
all_melt_df['YearDecimal'] = all_melt_df['Year']*1.0+all_melt_df['Month']/12.0 # approximate
all_melt_df.to_csv('melt_area.csv', index=False)
all_melt_df.sample(5)


# # Plots
# Here we put together a few plots to show the data and try to see some of the trends

# ## Show all years

# In[ ]:


all_melt_df.plot(x='YearDecimal', y='MeltAreaSqKm')


# ## Monthly Summary

# In[ ]:


month_summary_df = all_melt_df.groupby(['Month', 'Year']).agg({'MeltAreaSqKm': 'mean'}).reset_index()
sns.lineplot(x='Month', y='MeltAreaSqKm', hue='Year', data=month_summary_df)


# # All vs 2019
# We can try and recreate the decile range by using the 90% confidence interval (that is similar enough, right?). We can then show 2019 on top of that to see how far away the curve lies from the normal behavior.

# In[ ]:


fig, ax1 = plt.subplots(1, 1)
sns.lineplot(x='Month', y='MeltAreaSqKm', data=month_summary_df.query('Year<2019'), ax=ax1, ci=95)
df_2019 = all_melt_df.query('Year==2019')
ax1.plot(df_2019['MonthDecimal'], df_2019['MeltAreaSqKm'])
ax1.legend(['1979-2018', '2019'])


# In[ ]:


fig, ax1 = plt.subplots(1, 1)
sns.boxplot(x='Month', y='MeltAreaSqKm', data=month_summary_df.query('Year<2019'), ax=ax1)
df_2019 = all_melt_df.query('Year==2019')
ax1.plot(df_2019['MonthDecimal']-3, df_2019['MeltAreaSqKm'])
ax1.legend(['1979-2018', '2019'])


# # Heatmap over the years

# In[ ]:


all_melt_df['Month_Cat'] = (all_melt_df['MonthDecimal']*4).astype('int')/4.0
year_month_pivot = all_melt_df.pivot_table(columns='Year', index='Month_Cat', values='MeltAreaSqKm')
sns.heatmap(year_month_pivot, vmin=0, vmax=600000)


# # Random Other Plots
# Here are some random other possibly interesting plots

# In[ ]:


px.line(all_melt_df, x='MonthDecimal', y='MeltAreaSqKm', color='Year')


# ### Monthly Extreme Values
# Here we calculate the extreme values 99% for each month and compare it to 2019

# In[ ]:


import numpy as np
# who says you can't make pandas a bit tidier
monthly_95_melt_df = pd.concat([
    all_melt_df.\
        query('Year<2019').\
        dropna().\
        groupby(['Month']).\
        agg({'MeltAreaSqKm': lambda x: np.percentile(x, 95)}).\
        reset_index().\
        rename(columns={'Month': 'MonthDecimal'}).\
        assign(Source = '95 Percentile of Melting (1979-2018)'),
    all_melt_df.query('Year==2019')[['MonthDecimal', 'MeltAreaSqKm']].assign(Source='2019')
]).reset_index(drop=True)
px.line(monthly_95_melt_df, x='MonthDecimal', y='MeltAreaSqKm', color='Source')


# In[ ]:


# slightly more fine grained
all_melt_df['Month_Cat'] = (all_melt_df['MonthDecimal']*8).astype('int')/8.0
monthly_95_melt_df = pd.concat([
    all_melt_df.\
        query('Year<2019').\
        dropna().\
        groupby(['Month_Cat']).\
        agg({'MeltAreaSqKm': lambda x: np.percentile(x, 95)}).\
        reset_index().\
        rename(columns={'Month_Cat': 'MonthDecimal'}).\
        assign(Source = '95 Percentile of Melting (1979-2018)'),
    all_melt_df.query('Year==2019')[['MonthDecimal', 'MeltAreaSqKm']].assign(Source='2019')
]).reset_index(drop=True)
px.line(monthly_95_melt_df, x='MonthDecimal', y='MeltAreaSqKm', color='Source')


# In[ ]:


sns.catplot(x='Month', 
            y='MeltAreaSqKm', 
            hue='Year', 
            kind='swarm', 
            data=month_summary_df)

