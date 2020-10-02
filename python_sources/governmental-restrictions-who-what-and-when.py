#!/usr/bin/env python
# coding: utf-8

# ## Governmental Restrictions - Who, What and When
# 
# 
# <img src="https://specials-images.forbesimg.com/imageserve/5e700122aa542800075a14cd/960x0.jpg" width="700">
# 
# 
# ### Overview
# Everything disperses quickly in a connected world - physical goods, information, and unfortunately, disease. Governments and their decisions can have a huge impact on the spread of disease. This notebook shows the timing of a government's restrictions relative to the first confirmed cases in China and the first cases in that country.
# 
# (TODO: Enter summary opinion and/or insights)
# 
# ![](http://)

# ### Historical Comparison
# 
# Speed and connectivity are relative. Here is a fascinating map showing the spread of Black Plague through Europe in the mid 1300's. The timeline for the spread was influenced by several factors:
# 
#   - Trade routes, including a terribly active market for slaves
#   - Modes of travel; i.e., flea-ridden caravans and sailing ships
#   - Boundaries between adversarial empires
#   
#   
# <img src="https://www.historytoday.com/sites/default/files/blackdeathmap.jpg" width="700">
# 
# 

# ### Present Day
# 
# (TODO: Show a map of COVID-19 spread and timing)
# 
# 
# 

# ### Analysis
# 
# Data comes from the [OXFORD COVID-19 GOVERNMENT RESPONSE TRACKER](https://www.bsg.ox.ac.uk/research/research-projects/oxford-covid-19-government-response-tracker). Paul Mooney is refreshing this dataset for Kaggle on a regular basis. The dataset contains data for all countries from 2020-01-01 to present day. I restricted the analysis to countries with economies ranked in the top 15 by GDP.

# In[ ]:


import numpy as np
import pandas as pd
pd.set_option("display.max_columns", 1000)

import holoviews as hv
hv.extension('bokeh')
from bokeh.models import HoverTool

from IPython.display import HTML


# In[ ]:


# 15 largest national economies in order of reported first infection
# https://worldpopulationreview.com/countries/countries-by-gdp/
top15 = ['China',
         'Japan',
         'South Korea',
         'United States',
         'France',
         'Australia',
         'Canada',
         'Germany',
         'India',
         'United Kingdom',
         'Italy',
         'Spain',
         'Russia',
         'Brazil',
         'Mexico']

cols = ['CountryName',
         'Date',
         'ConfirmedCases',
         'S1_School closing',  #Internal
         'S2_Workplace closing',  #Internal
         'S3_Cancel public events',  #Internal
         'S4_Close public transport',  #Internal
         'S6_Restrictions on internal movement',  #Internal
         'S7_International travel controls'  #Global travel
         ]


# In[ ]:


df = pd.read_excel('/kaggle/input/oxford-covid19-government-response-tracker'
                   '/OxCGRT_Download_latest_data.xlsx',
                   sheet_name=0, usecols=cols, parse_dates=['Date'])
df = df.loc[df.CountryName.isin(top15)]        .set_index(['CountryName', 'Date'])        .sort_index()
display(df)


# The charts below show the timing of actions taken by countries across the world. I restricted the analysis to countries with economies ranked in the top 15 by GDP.
# 

# In[ ]:



df = df.reset_index()        .fillna(method='ffill')        .melt(id_vars=['CountryName', 'Date'], var_name='Restriction', value_name='Level')        .sort_values(['CountryName', 'Date', 'Restriction'])        .loc[lambda x: x.Level>0]        .drop(columns='Level')        .drop_duplicates(['CountryName', 'Restriction'])        .reset_index(drop=True)

df = df.assign(ConfirmedDate=df.Date.where(df.Restriction.eq('ConfirmedCases'))                                     .groupby(df.CountryName).transform('first'),
               ConfirmedOrder=df.CountryName.map(dict(zip(top15, range(15)))),
               FirstActionDate=df.Date.where(df.Restriction.ne('ConfirmedCases')) \
                                  .groupby(df.CountryName).transform('min'),
               TravelBanDate=df.Date.where(df.Restriction.eq('S7_International travel controls')) \
                                    .groupby(df.CountryName).transform('first'),
               CV_Day=lambda x: (x.Date-x.ConfirmedDate).dt.days,
               )

# Simplify to internal and external limits
df.loc[df.Restriction.str.startswith("S7"), 'Restriction'] = "Global travel limits"
df.loc[df.Restriction.str.startswith("S"), 'Restriction'] = "Internal limits"
df = df.sort_values('Date')        .drop_duplicates(['CountryName', 'Restriction'])        .sort_values(['CountryName', 'Restriction'])

df


# In[ ]:





# TODO: Add commentary

# In[ ]:


tooltips_pts = [('Restriction', '@Restriction'),
                ('Days Since Confirmed', '@CV_Day')
                ]

hover_pts = HoverTool(tooltips=tooltips_pts)

y_ticks = list(tuple(zip(range(15), top15)))

font_size={'legend': 7,
           'labels': 14,
           'xticks': 14,
           'yticks': 12}

plot_opts = dict(width=800, 
                 height=600, 
                 color_index='Restriction',
                 show_legend=True,
                 yticks=y_ticks,
                 invert_yaxis=True,
                 ylabel="Country (in order of reporting cases)",
                )

grid_style = {'ygrid_line_dash': [1, 4],
              'ygrid_line_color': 'gray',
              'grid_line_width': 1}

style_opts = dict(cmap=['red', 'green', 'blue'],
                  size=10,
                  fill_alpha=0.6,
                  marker='square',
                  show_grid=True,
                  fontsize=font_size,
                  gridstyle=grid_style,
                  default_tools=[]
                 )

p = hv.Points(df, kdims=['Date', 'ConfirmedOrder'], vdims=['Restriction','CV_Day'])     .opts(**plot_opts, tools=[hover_pts], title="Timeline of Government Restrictions",
          **style_opts, legend_position='bottom_left')\
display(HTML("Hover over squares for more information."), p)


# TODO: Add commentary

# In[ ]:


xticks = list(range(-100, 101, 10))

tooltips_pts = [('Restriction', '@Restriction'),
                ('Date', '@{Date}{%F}')
                ]
formatters={'@{Date}': 'datetime'}

hover_pts = HoverTool(tooltips=tooltips_pts, formatters=formatters)

p2 = hv.Points(df, kdims=['CV_Day', 'ConfirmedOrder'], vdims=['Restriction', 'Date'])             .opts(**plot_opts, xticks=xticks, xlabel="Days since Cases were Confirmed",
                  tools=[hover_pts], title="Timeline of Restrictions relative to Confirmed Cases",
                  **style_opts, legend_position='top_left')
display(HTML("Hover over squares for more information."), p2)


# TODO: Add commentary

# In[ ]:


# Bar charts

grid_style = {'xgrid_line_dash': [1, 4],
              'xgrid_line_color': 'gray',
              'ygrid_line_width': 0}
x_ticks = list(range(-60, 51, 10))

plot_opts = dict(width=800, 
                 height=600,
                 xticks=x_ticks,
                 invert_axes=True,
                 invert_yaxis=True,
                 )

style_opts = dict(fontsize=font_size,
                  xrotation=0,
                  default_tools=[],
                  show_grid=True,
                  gridstyle=grid_style,
                  color='Tan',
                  line_alpha=0
                  )

df['ActionDelay'] = (df.FirstActionDate-df.ConfirmedDate).dt.days

b1 = hv.Bars(df.sort_values('ConfirmedOrder'), 'CountryName', 'ActionDelay').opts(**plot_opts, 
        title='Days between first case and first restriction',
        **style_opts)
b1


# In[ ]:


df['TravelBanDelay'] = (df.TravelBanDate-df.ConfirmedDate).dt.days


b1 = hv.Bars(df.sort_values('ConfirmedOrder'), 'CountryName', 'TravelBanDelay').opts(**plot_opts, 
        title='Days between first case and global travel restrictions',
        **style_opts)
b1


# ### Wrap-up
# 
# TODO: Add closing.

# In[ ]:


# plot_opts = dict(width=1000, 
#                  height=600, 
#                 )

# df_travel = df.sort_values('EconomyRank').loc[df.Restriction == "International travel controls"]
# hv.Bars(df_travel, 'CountryName', 'CV_Day').opts(**plot_opts)

