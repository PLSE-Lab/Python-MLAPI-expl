#!/usr/bin/env python
# coding: utf-8

# #### To play around with interactive widgets, *Fork* the notebook!
# 
# *>* Charts in increasing order of difficulty to create *<*

# In[ ]:


import numpy as np 
import pandas as pd 
from path import Path

import os

from ipywidgets import interact, interact_manual
from IPython.display import display

import cufflinks as cf


# In[ ]:


pd.set_option('display.max_rows', 500)
pd.set_option('use_inf_as_na', True)
cf.set_config_file(offline=True, theme='solar');


# In[ ]:


input_path = Path("../input/novel-corona-virus-2019-dataset/")


# In[ ]:


master_df = pd.read_csv(input_path/'covid_19_data.csv', index_col='SNo', 
                        parse_dates=['Last Update', 'ObservationDate'])


# In[ ]:


master_df = master_df.sort_values(['ObservationDate', 'Last Update'])
master_df['Province/State'] = master_df['Province/State'].fillna(master_df['Country/Region'])


# # Cumulative Change

# In[ ]:


@interact
def plot_cumulative(countries=list(master_df['Country/Region'].sort_values().unique())):
    df = master_df[master_df['Country/Region'] == countries].drop_duplicates(subset=['Last Update'])
    df = df.groupby('ObservationDate', as_index=False).sum() 
    day_num_df = pd.DataFrame(data={'ObservationDate': pd.date_range(df['ObservationDate'].min(), df['ObservationDate'].max()), 
                                    'day_num': 1})
    day_num_df['day_num'] = day_num_df['day_num'].cumsum() - 1
    df = df.merge(day_num_df, on='ObservationDate', how='left')
#     display(df)
    df.iplot(kind='line', x='day_num', y='Confirmed', secondary_y=['Deaths', 'Recovered'], 
             title=countries, xTitle='Day Number', yTitle='Number of People')


# # Daily Absolute Change

# In[ ]:


desc_countries_by_confirmed = master_df.groupby(['Country/Region']).last().sort_values('Confirmed', ascending=False).index.to_list()


# In[ ]:


@interact
def plot_daily(countries=desc_countries_by_confirmed):
    df = master_df[master_df['Country/Region'] == countries].drop_duplicates(subset=['Last Update'])
    df = df.groupby('ObservationDate', as_index=False).sum() 
    day_num_df = pd.DataFrame(data={'ObservationDate': pd.date_range(df['ObservationDate'].min(), df['ObservationDate'].max()), 
                                    'day_num': 1})
    day_num_df['day_num'] = day_num_df['day_num'].cumsum() - 1
    df = df.merge(day_num_df, on='ObservationDate', how='left').sort_values('day_num', ascending=False)
    df = df.set_index('day_num')

    df['Confirmed'] = df['Confirmed'] -  df['Confirmed'].shift(-1)
    df['Deaths'] = df['Deaths'] -  df['Deaths'].shift(-1)
    df['Recovered'] = df['Recovered'] -  df['Recovered'].shift(-1)
#     display(df)
    df.iplot(kind='scatter', mode='markers', y='Confirmed', secondary_y=['Recovered', 'Deaths'], 
             title=countries, xTitle='Day Number', yTitle='Number of People', size=7)


# # Daily Percentage Change

# In[ ]:


@interact
def pct_change(countries=list(master_df['Country/Region'].unique())):
    df = master_df[master_df['Country/Region'] == countries].drop_duplicates(subset=['Last Update'])
    df = df.groupby('ObservationDate', as_index=False).sum() 
    day_num_df = pd.DataFrame(data={'ObservationDate': pd.date_range(df['ObservationDate'].min(), df['ObservationDate'].max()), 
                                    'day_num': 1})
    day_num_df['day_num'] = day_num_df['day_num'].cumsum() - 1
    df = df.merge(day_num_df, on='ObservationDate', how='left') 
    df = df.set_index('day_num')

    df[['Confirmed', 'Deaths', 'Recovered']] = df[['Confirmed', 'Deaths', 'Recovered']].pct_change()*100
    df = df.dropna(how='any')
#     display(df)
    df.iplot(kind='scatter', mode='markers+lines', x='ObservationDate', y=['Confirmed', 'Deaths', 'Recovered'], 
             title=countries, xTitle='Date', yTitle='Percentage Change', size=7)


# In[ ]:




