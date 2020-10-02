#!/usr/bin/env python
# coding: utf-8

# # Overview
# - Basically trying to reproduce this graph: https://twitter.com/MarkJHandley/status/1237119688578138112/photo/1 showing a linear (on a semilog scale) growth and a particularly slow/quick peaking growth in Japan.
# - I try to also see how number of people infected relates to the average spread in a country
# - Finally I drop the country pretense and just look at grids of latitute and longitude

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
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


corona_df = pd.read_csv("../input/corona-virus-report/covid_19_clean_complete.csv")
corona_df['DateCode'] = pd.to_datetime(corona_df['Date'])
corona_df.head(5)                      


# In[ ]:


date_country_df = corona_df.    groupby(['DateCode', 'Country/Region']).    agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'}).    reset_index()
date_country_df.head(3)


# In[ ]:


px.line(date_country_df.query('Confirmed>0'), 
        x='DateCode', 
        y='Confirmed', 
        color='Country/Region', 
        log_y=True)


# ## Big Summary Table
# This should show all the countries, not sure why they are missing

# In[ ]:


fig, ax1 = plt.subplots(1,1, figsize=(15, 15))
sns.heatmap(
    corona_df.\
        pivot_table(index='Country/Region', columns='Date', values='Confirmed', aggfunc='sum').\
        sort_values('2020-04-05').\
        applymap(lambda x: np.log(x) if x>0 else -1),
    ax=ax1
)


# In[ ]:


corona_df.pivot_table(index='Country/Region', columns='Date', values='Confirmed', aggfunc='sum').applymap(lambda x: np.clip(x, 0, 1000)) 


# In[ ]:


px.density_heatmap(date_country_df, y='Country/Region', x='DateCode', z='Confirmed')


# # Twitter Post Line
# Here I try to reproduce the twitter post

# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
twitter_countries = [('Italy', 0), ('Germany', -9), ('France', -9), ('Spain', -10.5), 
                     ('US', -11.5) ,('UK', -13.5), ('Japan', 0), ('Korea, South', +2.5), 
                    ('Taiwan*', +2.5)]
graph_day_zero = pd.to_datetime('2020-03-09')
twitter_rows = []

for c_country, offset in twitter_countries:
    c_rows = date_country_df[date_country_df['Country/Region'].isin([c_country])].query('Confirmed>0').sort_values(['DateCode'])
    days_since_start = (c_rows['DateCode']-graph_day_zero).dt.total_seconds()/(24*3600)+offset
    group_label = f'{c_country} (Offset:{offset:5.1f})'
    ax1.plot(days_since_start, c_rows['Confirmed'],'.-', label=group_label)
    twitter_rows += [c_rows.assign(days_since_start=days_since_start, label=group_label)]

# 33% growth
fake_days = np.arange(-22, 3)
ax1.plot(fake_days, 30*np.power(1.33, np.arange(len(fake_days))), 'b-', label='33% Daily Increase', lw=8, alpha=0.25)

ax1.legend()
ax1.set_yscale('log')
ax1.set_xlabel('Days before March 9\nWith Offset to Italy')
ax1.set_ylabel('Confirmed Cases')
ax1.set_ylim(10, 1e6)


# In[ ]:


px.line(pd.concat(twitter_rows), x='days_since_start', y='Confirmed', color='label', log_y=True)


# ## Top Country Overview

# In[ ]:


countries_by_cases_df = date_country_df.    groupby('Country/Region').    agg({'Confirmed': 'sum'}).    reset_index().    sort_values('Confirmed', ascending=False)


# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
top_countries = countries_by_cases_df.head(8)['Country/Region'].values.tolist()
top_df = date_country_df[
        date_country_df['Country/Region'].isin(top_countries)
    ].query('Confirmed>0').sort_values('DateCode')

for c_country, raw_rows in top_df.groupby(['Country/Region']):
    c_rows = raw_rows.sort_values(['DateCode'])
    # define day zero as when there are 100 cases
    day_zero = c_rows.query('Confirmed>100')['DateCode'].min()
    days_since_start = (c_rows['DateCode']-day_zero).dt.total_seconds()/(24*3600)
    ax1.semilogy(c_rows['DateCode']-day_zero, c_rows['Confirmed'], '-', label=f'{c_country} (First Case:{day_zero:%m-%d})')
ax1.legend()
ax1.set_xlabel('Days Since Outbreak')
ax1.set_ylabel('Confirmed Cases')


# In[ ]:


from sklearn.linear_model import LinearRegression
out_rows = []
lr_rows = []
for c_country, raw_rows in date_country_df.groupby(['Country/Region']):
    c_rows = raw_rows.sort_values(['DateCode'])
    # define day zero as when there are 100 cases
    if c_rows.query('Confirmed>100').shape[0]>1:
        day_zero = c_rows.query('Confirmed>100')['DateCode'].min()
        days_since_start = (c_rows['DateCode']-day_zero).dt.total_seconds()/(24*3600)
        group_label = f'{c_country} (First Case:{day_zero:%m-%d})'
        
        out_rows += [c_rows.assign(days_since_start=days_since_start, 
                                   day_zero=day_zero, 
                                   label=group_label)]
norm_rows_df = pd.concat(out_rows)


# In[ ]:


px.line(norm_rows_df, 
        x='days_since_start', 
        y='Confirmed', 
        color='Country/Region', 
        log_y=True)


# # Slope for Each Country

# In[ ]:


from sklearn.linear_model import LinearRegression
lr_rows = []
for c_country, raw_rows in date_country_df.groupby(['Country/Region']):
    c_rows = raw_rows.sort_values(['DateCode'])
    # define day zero as when there are 100 cases
    if c_rows.query('Confirmed>100').shape[0]>1:
        day_zero = c_rows.query('Confirmed>100')['DateCode'].min()
        days_since_start = (c_rows['DateCode']-day_zero).dt.total_seconds()/(24*3600)
        group_label = f'{c_country} (Outbreak Day: {day_zero:%m-%d})'
        
        lr = LinearRegression()
        lr.fit(days_since_start.values.reshape((-1, 1)), c_rows['Confirmed'])
        lr_rows.append(dict(New_Cases_Per_Day=lr.coef_[0], day_zero=day_zero, label=group_label, Total_Confirmed_Cases=c_rows['Confirmed'].max()))
lr_rows_df = pd.DataFrame(lr_rows)
lr_rows_df.sample(3)


# In[ ]:


import plotly.graph_objects as go
fig = px.scatter(lr_rows_df, 
        y='New_Cases_Per_Day', 
        x='Total_Confirmed_Cases', 
        color='label', 
        log_x=True,
        log_y=True)

# 33% growth
total_cases = 50*np.power(1.33, np.arange(20))
new_cases_per_day = np.diff(total_cases, axis=0)
fig.add_trace(go.Scatter(x=total_cases[1:], y=new_cases_per_day,
                    mode='lines',
                    name='33% Daily Increase'))


# ## Capture Intermediate Points

# In[ ]:


all_diff_rows = []
for c_country, raw_rows in date_country_df.groupby(['Country/Region']):
    c_rows = raw_rows.sort_values(['DateCode'])
    # define day zero as when there are 100 cases
    if c_rows.query('Confirmed>100').shape[0]>1:
        day_zero = c_rows.query('Confirmed>100')['DateCode'].min()
        days_since_start = (c_rows['DateCode']-day_zero).dt.total_seconds()/(24*3600)
        group_label = f'{c_country} (Outbreak Day: {day_zero:%m-%d})'
        
        all_diff_rows.append(c_rows.assign(
            New_Cases_Per_Day=c_rows['Confirmed'].diff(), 
            day_zero=day_zero, 
            label=group_label, 
            Total_Confirmed_Cases=c_rows['Confirmed'].max(),
            days_since_start=days_since_start
        ))
all_diff_rows_df = pd.concat(all_diff_rows).dropna()
all_diff_rows_df.sample(3)


# In[ ]:


fig = px.line(all_diff_rows_df.query('New_Cases_Per_Day>0'), 
        y='New_Cases_Per_Day', 
        x='Confirmed', 
        color='label', 
        log_x=True,
        log_y=True)

# 33% growth
total_cases = 10*np.power(1.33, np.arange(35))
new_cases_per_day = np.diff(total_cases, axis=0)
fig.add_trace(go.Scatter(x=total_cases[1:], y=new_cases_per_day,
                    mode='lines',
                         line=dict(width=12),
                         opacity=0.5,
                    name='33% Daily Increase'))


# In[ ]:


fig = px.scatter(all_diff_rows_df, 
        y='New_Cases_Per_Day', 
        x='Confirmed', 
        color='label', 
        log_x=True,
        log_y=True)

# 33% growth
total_cases = 50*np.power(1.33, np.arange(20))
new_cases_per_day = np.diff(total_cases, axis=0)
fig.add_trace(go.Scatter(x=total_cases[1:], y=new_cases_per_day,
                    mode='lines',
                    name='33% Daily Increase'))


# # Deaths
# Since confirmed infections are a fairly difficult to measure number (and highly depenedent on testing). Using deaths as an end-point is probably more reliable

# In[ ]:


fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
top_countries = countries_by_cases_df.head(8)['Country/Region'].values.tolist()
top_df = date_country_df[
        date_country_df['Country/Region'].isin(top_countries)
    ].query('Deaths>0').sort_values('DateCode')

for c_country, raw_rows in top_df.groupby(['Country/Region']):
    c_rows = raw_rows.sort_values(['DateCode'])
    # define day zero as when there are 100 cases
    day_zero = c_rows.query('Deaths>1')['DateCode'].min()
    days_since_start = (c_rows['DateCode']-day_zero).dt.total_seconds()/(24*3600)
    ax1.semilogy(c_rows['DateCode']-day_zero, c_rows['Deaths'], '-', label=f'{c_country} (First Case:{day_zero:%m-%d})')
ax1.legend()
ax1.set_xlabel('Days Since Outbreak')
ax1.set_ylabel('Confirmed Cases')


# In[ ]:


all_death_diff_rows = []
for c_country, raw_rows in date_country_df.groupby(['Country/Region']):
    c_rows = raw_rows.sort_values(['DateCode'])
    # define day zero as when there are 10 deaths
    if c_rows.query('Deaths>10').shape[0]>1:
        day_zero = c_rows.query('Deaths>10')['DateCode'].min()
        days_since_start = (c_rows['DateCode']-day_zero).dt.total_seconds()/(24*3600)
        group_label = f'{c_country} (Outbreak Day: {day_zero:%m-%d})'
        
        all_death_diff_rows.append(c_rows.assign(
            New_Deaths_Per_Day=c_rows['Deaths'].diff(), 
            New_Cases_Per_Day=c_rows['Confirmed'].diff(), 
            New_Recovery_Per_Day=c_rows['Recovered'].diff(), 
            day_zero=day_zero, 
            label=group_label, 
            Total_Deaths_Cases=c_rows['Deaths'].max(),
            days_since_start=days_since_start
        ))
all_death_diff_rows_df = pd.concat(all_death_diff_rows).dropna()
all_death_diff_rows_df.sample(3)


# In[ ]:


fig = px.scatter(all_death_diff_rows_df, 
        y='New_Deaths_Per_Day', 
        x='Deaths', 
        color='label', 
        log_x=True,
        log_y=True)

# 33% growth
total_cases = 50*np.power(1.33, np.arange(10))
new_cases_per_day = np.diff(total_cases, axis=0)
fig.add_trace(go.Scatter(x=total_cases[1:], y=new_cases_per_day,
                    mode='lines',
                    name='33% Daily Increase'))


# # Lat/Long Grids
# Rather than country we can focus on latitude longitude grids

# In[ ]:


corona_df['QLat'] = pd.cut(corona_df['Lat'], 100)
corona_df['QLong'] = pd.cut(corona_df['Long'], 100)


# In[ ]:


def summarize_grid(in_rows):
    return in_rows.        groupby('DateCode').        agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum', 
         'Country/Region': 'first', 'Province/State': 'first'}).\
        reset_index()
def cut_to_num(in_str: str) -> float:
    """Takes the middle of qcut range"""
    clean_str = str(in_str).replace('(', '').replace('[', '').replace(']', '').replace(')', '')
    return np.mean([float(x) for x in clean_str.split(',')])    
date_grid_df = corona_df.    groupby(['QLat', 'QLong']).    apply(summarize_grid).    reset_index().    dropna().    assign(Lat=lambda x: x['QLat'].astype(str).map(cut_to_num), 
           Long=lambda x: x['QLong'].astype(str).map(cut_to_num))
date_grid_df.head(3)


# In[ ]:


sum_grid_df = date_grid_df.groupby(['Lat', 'Long']).agg({'Confirmed': 'sum'}).reset_index().query('Confirmed>0')


# In[ ]:


plt.scatter(sum_grid_df['Long'], sum_grid_df['Lat'], s=10*np.log10(sum_grid_df['Confirmed']))


# In[ ]:


from mpl_toolkits.basemap import Basemap
world_map = Basemap(projection='ortho',lat_0=45,lon_0=100,resolution='l')
world_map.drawcoastlines(linewidth=0.25)
world_map.drawcountries(linewidth=0.25)
world_map.fillcontinents(color='lightgreen',lake_color='aqua', alpha=0.25)

world_map.scatter(sum_grid_df['Long'].values, 
                  sum_grid_df['Lat'].values, 
                  s=10*np.log10(sum_grid_df['Confirmed']),
                  c='r',
                  latlon=True)


# In[ ]:


out_rows = []
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
for c_bucket, raw_rows in date_grid_df.groupby(['QLat', 'QLong']):
    if raw_rows.query('Confirmed>10').shape[0]>0:
        c_rows = raw_rows.sort_values(['DateCode'])
        # define day zero as when there are 100 cases
        day_zero = c_rows.query('Confirmed>10')['DateCode'].min()
        days_since_start = (c_rows['DateCode']-day_zero).dt.total_seconds()/(24*3600)
        ax1.semilogy(days_since_start, c_rows['Confirmed'], '-', label=f'{c_bucket} (First Case:{day_zero:%m-%d})')
        ax2.semilogy(days_since_start, c_rows['Deaths'], '-', label=f'{c_bucket} (First Case:{day_zero:%m-%d})')
        lr = LinearRegression()
        lr.fit(days_since_start.values.reshape((-1, 1)), c_rows['Confirmed'])
    else:
        lr = LinearRegression()
        day_zero = raw_rows['DateCode'].min()
        days_since_start = (raw_rows['DateCode']-day_zero).dt.total_seconds()/(24*3600)
        lr.fit(days_since_start.values.reshape((-1, 1)), raw_rows['Confirmed'])
    out_rows.append(dict({
        'Lat': raw_rows['Lat'].iloc[0],
        'Long': raw_rows['Long'].iloc[0],
        'Region': raw_rows['Country/Region'].iloc[0],
        'State': raw_rows['Province/State'].iloc[0],
        'day_zero': day_zero,
        'spread_rate': lr.coef_[0],
        'max_confirmed': raw_rows['Confirmed'].max(),
        'max_deaths': raw_rows['Deaths'].max()
                         }))
ax1.set_xlabel('Days Since Outbreak')
ax1.set_ylabel('Confirmed Cases')
ax2.set_ylabel('Confirmed Deaths')


# In[ ]:


grid_growth_df = pd.DataFrame(out_rows).sort_values('spread_rate', ascending=False)
print(grid_growth_df.shape[0])
grid_growth_df.head(10)


# In[ ]:


plt.scatter(grid_growth_df['Long'], grid_growth_df['Lat'], s=np.clip(grid_growth_df['spread_rate'], 0, 25))


# In[ ]:


world_map = Basemap(projection='ortho',lat_0=30,lon_0=50,resolution='l')
world_map.drawcoastlines(linewidth=0.25)
world_map.drawcountries(linewidth=0.25)
world_map.fillcontinents(color='lightgreen',lake_color='aqua', alpha=0.25)

world_map.scatter(grid_growth_df['Long'].values, 
                  grid_growth_df['Lat'].values, 
                  s=np.clip(grid_growth_df['spread_rate'], 0, 25),
                  c='r',
                  latlon=True)


# In[ ]:


px.scatter_geo(grid_growth_df, lat='Lat', lon='Long', color='spread_rate', size='max_deaths')


# In[ ]:




