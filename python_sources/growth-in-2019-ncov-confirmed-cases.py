#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)

import ipywidgets as widgets
from IPython.display import clear_output
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data_df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')


# # Dataset
# 
# Source:
# The dataset is scraped from https://docs.google.com/spreadsheets/d/1yZv9w9zRKwrGTaR-YzmAqMefw4wMlaXocejdxZaTs6w/htmlview?usp=sharing&sle=true# by @sudalairajkumar to the Kaggle platform.
# 
# 

# # Features Summary
# * Sno: Serial number (Location Id)
# * Province/State: Province or State of observation
# * Country: Country of observation
# * Last Update: Date of observation
# * Confirmed: Number of confirmed cases
# * Deaths: Number of deaths
# * Recovered: Number of recovered cases

# In[ ]:


# convert to datetime
data_df['Last Update'] = pd.to_datetime(data_df['Last Update'])
# map Mainland China to China
data_df['Country'] = data_df['Country'].replace('Mainland China', 'China')


# In[ ]:


for column in data_df.columns:
    print(f'{column}: {data_df[column].dtype}')
    print(f'\tUnique Values:{data_df[column].nunique()}')
    print(f'\tNormalized Unique Values:{data_df[column].nunique()/data_df.shape[0]: .4f}')
    print(f'\tMissing:{data_df[column].isna().sum()}')


# In[ ]:


# fill missing values
data_df['Province/State'] = data_df['Province/State'].fillna('NA')
data_df['Full Location'] = data_df['Country'] + '_' + data_df['Province/State']


# # Visualization by Country

# In[ ]:


# Vis
def plot_trend(df, y_feature, loc_feature='Country', loc='All', freq='Daily'):
    df = df.copy()
    # filter by Location
    if loc == 'All':
        df = df.sort_values('Last Update').reset_index(drop=True)
    else:
        df = df[df[loc_feature] == loc].sort_values('Last Update').reset_index(drop=True)
    # group by freq
    df.index = df['Last Update']
    if freq == 'Daily':
        df = df.groupby(pd.Grouper(freq='D'))[y_feature].sum().reset_index()
    elif freq == 'Hourly':
        df = df.groupby(pd.Grouper(freq='H'))[y_feature].sum().reset_index()
        
    trace1 = go.Scatter(
                        x = df['Last Update'],
                        y = df[y_feature],
                        mode = "lines",
                        marker = dict(color = 'rgba(16, 112, 2, 0.8)'))

    traces = [trace1]
    layout = dict(title = f'[{y_feature}] {freq}',
                  xaxis= dict(title='Date',ticklen= 5,zeroline= False)
                 )
    fig = dict(data=traces, layout = layout)
    iplot(fig)

# UI
country_dropdown_1 = widgets.Dropdown(
    options=data_df['Country'].unique().tolist() + ['All'],
    value='All',
    description='Country:',
    disabled=False,
)

y_dropdown_1 = widgets.Dropdown(
    options=['Confirmed', 'Deaths', 'Recovered'],
    value='Confirmed',
    description='Stats:',
    disabled=False,
)

def on_value_change(change):
    clear_output()
    display(country_dropdown_1, y_dropdown_1)
    plot_trend(data_df, y_feature=y_dropdown_1.value, loc=country_dropdown_1.value, freq='Daily')

country_dropdown_1.observe(on_value_change, names='value')
y_dropdown_1.observe(on_value_change, names='value')
# trigger init
on_value_change(None)


# In[ ]:


# UI
location_dropdown_2 = widgets.Dropdown(
    options=data_df['Full Location'].unique().tolist() + ['All'],
    value='All',
    description='Location:',
    disabled=False,
)

y_dropdown_2 = widgets.Dropdown(
    options=['Confirmed', 'Deaths', 'Recovered'],
    value='Confirmed',
    description='Stats:',
    disabled=False,
)

def on_value_change(change):
    clear_output()
    display(location_dropdown_2, y_dropdown_2)
    plot_trend(data_df, y_feature=y_dropdown_2.value, loc_feature='Full Location', loc=location_dropdown_2.value, freq='Daily')

location_dropdown_2.observe(on_value_change, names='value')
y_dropdown_2.observe(on_value_change, names='value')
# trigger init
on_value_change(None)


# # Growth Visualization By Country

# In[ ]:


# calculate % change
def get_daily_change(df, y_feature, loc, loc_feature='Country'):
    df = df.copy()
    # filter by Location
    if loc == 'All':
        df = df.sort_values('Last Update').reset_index(drop=True)
    else:
        df = df[df[loc_feature] == loc].reset_index(drop=True).sort_values('Last Update')
        
    df.index = df['Last Update']
    df = df.groupby(pd.Grouper(freq='D'))[y_feature].sum().reset_index()
    df[f'{y_feature} Change'] = df[y_feature].diff() 
    df[f'{y_feature} % Change'] = df[y_feature].pct_change() * 100
    return df

country_dropdown_3 = widgets.Dropdown(
    options=data_df['Country'].unique().tolist() + ['All'],
    value='All',
    description='Country:',
    disabled=False,
)

y_dropdown_3 = widgets.Dropdown(
    options=['Confirmed', 'Deaths', 'Recovered'],
    value='Confirmed',
    description='Stats:',
    disabled=False,
)

abs_dropdown_3 = widgets.Dropdown(
    options=['Abs. Change', '% Change'],
    value='% Change',
    description='Abs Change/% Change:',
    disabled=False,
)


def on_value_change(change):
    clear_output()
    display(country_dropdown_3, y_dropdown_3, abs_dropdown_3)
    df = get_daily_change(data_df, y_dropdown_3.value, country_dropdown_3.value)
    if abs_dropdown_3.value == 'Abs. Change':
        plot_trend(df, y_feature=f'{y_dropdown_3.value} Change', loc='All', freq='Daily')
    else:
        plot_trend(df, y_feature=f'{y_dropdown_3.value} % Change', loc='All', freq='Daily')

country_dropdown_3.observe(on_value_change, names='value')
y_dropdown_3.observe(on_value_change, names='value')
abs_dropdown_3.observe(on_value_change, names='value')
# trigger init
on_value_change(None)


# # Growth Visualization By Location

# In[ ]:


location_dropdown_4 = widgets.Dropdown(
    options=data_df['Full Location'].unique().tolist() + ['All'],
    value='China_Hubei',
    description='Location:',
    disabled=False,
)

y_dropdown_4 = widgets.Dropdown(
    options=['Confirmed', 'Deaths', 'Recovered'],
    value='Confirmed',
    description='Stats:',
    disabled=False,
)

abs_dropdown_4 = widgets.Dropdown(
    options=['Abs. Change', '% Change'],
    value='% Change',
    description='Abs Change/% Change:',
    disabled=False,
)


def on_value_change(change):
    clear_output()
    display(location_dropdown_4, y_dropdown_4, abs_dropdown_4)
    df = get_daily_change(data_df, y_dropdown_4.value, loc=location_dropdown_4.value, loc_feature='Full Location')
    if abs_dropdown_4.value == 'Abs. Change':
        plot_trend(df, y_feature=f'{y_dropdown_4.value} Change', loc_feature='Full Location', loc='All', freq='Daily')
    else:
        plot_trend(df, y_feature=f'{y_dropdown_4.value} % Change', loc_feature='Full Location', loc='All', freq='Daily')

location_dropdown_4.observe(on_value_change, names='value')
y_dropdown_4.observe(on_value_change, names='value')
abs_dropdown_4.observe(on_value_change, names='value')
# trigger init
on_value_change(None)


# # Last N Day Confirmed Cases Change By Country

# In[ ]:


def get_moving_stat(df, y_feature, loc_feature='Country'):
    df = df.copy()
    df.index = df['Last Update']
    df = df.groupby([pd.Grouper(freq='D'), loc_feature])[y_feature].sum().reset_index()
    day_count_df = df.groupby(loc_feature)[y_feature].count().reset_index()
    filted_locations = day_count_df[day_count_df['Confirmed']>=3][loc_feature].tolist()
    df = df[df[loc_feature].isin(filted_locations)]    
    df[f'{y_feature} Change'] = (df.groupby(loc_feature)[y_feature].apply(pd.Series.diff))
    
    for window_size in [1, 2, 3, 4]:
        df[f'Lag {window_size} {y_feature}'] =  df.groupby(loc_feature)[y_feature].shift(window_size)
        df[f'Last {window_size}Dy {y_feature} Change'] = df.groupby(loc_feature)[f'{y_feature} Change']            .transform(lambda x: x.rolling(window_size, min_periods=window_size).sum()).round(2)
    df = df.groupby(loc_feature).last()
    
    for window_size in [1, 2, 3, 4]:
        df[f'Last {window_size}Dy {y_feature} % Change'] = df[f'Last {window_size}Dy {y_feature} Change'] /  df[f'Lag {window_size} {y_feature}']
        
#     display(df)
    return df


# 1. Sorted by: 3 Day Moving Sum of Confirmed Case Change. i.e Abs. Growth

# In[ ]:


country_growth_df = get_moving_stat(data_df, y_feature='Confirmed', loc_feature='Country')
country_growth_df = country_growth_df.sort_values('Last 4Dy Confirmed Change', ascending=False)
display(country_growth_df[[f for f in country_growth_df.columns if 'Confirmed Change' in f or 'Update' in f or f == 'Confirmed']])


# # Moving Sum of Confirmed Cases Change By Locations with Country == 'China'

# 1. Sorted by: 3 Day Moving Sum of Confirmed Case Change. i.e Abs. Growth

# In[ ]:


china_df = data_df[data_df['Country']=='China']
country_growth_df = get_moving_stat(data_df, y_feature='Confirmed', loc_feature='Full Location')
country_growth_df = country_growth_df.sort_values('Last 4Dy Confirmed Change', ascending=False)
display(country_growth_df[[f for f in country_growth_df.columns if 'Confirmed Change' in f or 'Update' in f or f == 'Confirmed']])


# # Last N Day Confirmed Cases % Change By Locations with Country == 'China'

# In[ ]:


china_df = data_df[data_df['Country']=='China']
country_growth_df = get_moving_stat(china_df, y_feature='Confirmed', loc_feature='Full Location')
country_growth_df = country_growth_df.sort_values('Last 4Dy Confirmed % Change', ascending=False)
display(country_growth_df[[f for f in country_growth_df.columns if 'Confirmed % Change' in f or 'Update' in f or f == 'Confirmed']])


# In[ ]:




