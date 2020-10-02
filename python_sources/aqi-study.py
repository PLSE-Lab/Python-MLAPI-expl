#!/usr/bin/env python
# coding: utf-8

# Goal of this notebook is to look at PM2.5 pollutant levels across the USA over time.  Later the plan is to query from the global data set to view other country's PM2.5 pollutant levels.
# 
# * PM2.5 is particulate matter (PM) that has a diameter less than 2.5 micrometers.  
# * PM2.5 sources include power plants, motor vehicles, airplanes, residential wood burning, forest fires, agricultural burning, volcanic eruptions.
# * PM2.5 can be emitted directly into the air, others are formed when gases and particles interact with one another in the atmosphere.
# * The small size allows the particles to bypass the noise and throat and directly deep into the lungs and can even enter the circulatory system.
# * Long term exposure may lead to plaque deposits in the arteries, causing vascular inflammation and hardening of the arteries.
# * Potential damage from PM2.5 exposure depends on the concentration and the duration of exposure.
# 
# Why the interest, after moving from a low pollutant area, to Shanghai, a rather high concentration location, I have an interest in how the concentrations are changing over time, and how the concentrations change during the year.
# 
# Ref (https://blissair.com/what-is-pm-2-5.htm)
# 
# Thanks to Mhamed Jabri and Niyamat Ullah for there notebook that provided a great starting point.

# In[ ]:


# Load packages

import pandas as pd
import numpy as np
from google.cloud import bigquery
from bq_helper import BigQueryHelper

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# import plotly
import chart_studio as plotly
plotly.tools.set_config_file(world_readable=True, sharing='public')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.tools as tls
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as fig_fact
from plotly.subplots import make_subplots
from plotnine import *


get_ipython().run_line_magic('matplotlib', 'inline')

bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
pollutants = ['pm25_frm']


# Query the USA from the 2.5 daily data as a starting point.
# 
# 

# In[ ]:


QUERY = """
    SELECT
        pm25_daily.state_name,
        avg(pm25_daily.aqi) as pm25_avg_aqi,
        EXTRACT(YEAR FROM pm25_daily.date_local) AS year
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary` AS pm25_daily
    WHERE
      pm25_daily.poc = 1
      AND EXTRACT(YEAR FROM pm25_daily.date_local) > 1990
    GROUP BY pm25_daily.state_name,
       EXTRACT(YEAR FROM pm25_daily.date_local)
      
    
        """
df_states_pm25 = bq_assistant.query_to_pandas(QUERY)


# In[ ]:


states = {'AL': 'Alabama',
'AK': 'Alaska',
'AZ':'Arizona',
'AR':'Arkansas',
'CA':'California',
'CO':'Colorado',
'CT':'Connecticut',
'DE':'Delaware',
'FL':'Florida',
'GA':'Georgia',
'HI':'Hawaii',
'ID':'Idaho',
'IL':'Illinois',
'IN':'Indiana',
'IA':'Iowa',
'KS':'Kansas',
'KY':'Kentucky',
'LA':'Louisiana',
'ME':'Maine',
'MD':'Maryland',
'MA':'Massachusetts',
'MI':'Michigan',
'MN':'Minnesota',
'MS':'Mississippi',
'MO':'Missouri',
'MT':'Montana',
'NE':'Nebraska',
'NV':'Nevada',
'NH':'New Hampshire',
'NJ':'New Jersey',
'NM':'New Mexico',
'NY':'New York',
'NC':'North Carolina',
'ND':'North Dakota',
'OH':'Ohio',
'OK':'Oklahoma',
'OR':'Oregon',
'PA':'Pennsylvania',
'RI':'Rhode Island',
'SC':'South Carolina',
'SD':'South Dakota',
'TN':'Tennessee',
'TX':'Texas',
'UT':'Utah',
'VT':'Vermont',
'VA':'Virginia',
'WA':'Washington',
'WV':'West Virginia',
'WI':'Wisconsin',
'WY':'Wyoming'}


# In[ ]:


df_states = pd.DataFrame.from_dict(states,orient='index').reset_index()
df_states.columns = ['code', 'code_name']
df_states_pm25['state_code'] = df_states_pm25['state_name'].map(df_states.set_index('code_name')['code'])


# In[ ]:


df_states_pm25['pm25_avg_aqi'] = df_states_pm25['pm25_avg_aqi'].round(1)


# In[ ]:


df_states_pm25 = df_states_pm25.sort_values(by=['year','state_code'])


# In[ ]:


df_states_pm25.head(10)


# In[ ]:


scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],[0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

data = []

years = sorted(df_states_pm25['year'].unique())
COLS = len(years)

layout = dict(
        title = 'The average air quality index of pm25 By State<br>(Hover for breakdown)',
)

for i in range(len(years)):
    geo_key = 'geo'+str(i+1) if i != 0 else 'geo'
    pm25data = list(df_states_pm25[df_states_pm25['year'] == years[i]]['pm25_avg_aqi'].astype(float))
    # 
    data.append(
        dict(
        type = 'choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = df_states_pm25['state_code'],
        z = pm25data,
        zmin = 0,
        zmax = 70,
        showscale=True,
        locationmode = 'USA-states',
        text =  'Average AQI: ' + df_states_pm25['pm25_avg_aqi'].astype(str), 
        geo = geo_key,
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title =  "AQI of pm25",
            )
        )
    )
    
    layout[geo_key] = dict(
        scope = 'usa',
        projection=dict( type='albers usa' ),
        showlakes = True,
        domain = dict( x = [], y = [] ),
        lakecolor = 'rgb(255, 255, 255)',
    )
    
    # Year markers
    data.append(
        dict(
            type = 'scattergeo',
            showlegend = False,
            lon = [-78],
            lat = [47],
            geo = geo_key,
            text = [years[i]],
            mode = 'text',
        )
    )
    
COLS = 7
ROWS = 3
z=0     
for j in reversed(range(ROWS)):
    for k in range(COLS):
        geo_key = 'geo'+str(z+1) if z != 0 else 'geo'
        layout[geo_key]['domain']['x'] = [k/COLS,(k+1)/COLS]
        layout[geo_key]['domain']['y'] = [j/ROWS,(j+1)/ROWS]
        z = z + 1
         
fig = go.Figure(data=data, layout=layout)

fig.show()


# AQI of average AQI PM 2.5 time based by state.
# 
# Queary daily PM2.5

# In[ ]:


QUERY = """
    SELECT
        pm25_daily.state_name,
        avg(pm25_daily.aqi) as pm25_avg_aqi,
        EXTRACT(YEAR FROM pm25_daily.date_local) AS year,
        EXTRACT(MONTH FROM pm25_daily.date_local) AS month
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary` AS pm25_daily
    WHERE
      pm25_daily.poc = 1
      AND EXTRACT(YEAR FROM pm25_daily.date_local) > 1990
    GROUP BY pm25_daily.state_name,
       EXTRACT(YEAR FROM pm25_daily.date_local),
       EXTRACT(MONTH FROM pm25_daily.date_local)
      
    
        """
df_states_pm25_daily = bq_assistant.query_to_pandas(QUERY)


# In[ ]:


df_states_pm25_daily.dtypes


# In[ ]:


df_states_pm25_daily['state_code'] = df_states_pm25_daily['state_name'].map(df_states.set_index('code_name')['code'])
df_states_pm25_daily['date'] = pd.to_datetime(df_states_pm25_daily['year'].astype(str)  + df_states_pm25_daily['month'].astype(str), format='%Y%m')


# In[ ]:


df_states_pm25_daily = df_states_pm25_daily.sort_values(by=['year','month','state_code'])


# In[ ]:


df_states_pm25_daily.head(3)


# In[ ]:


plt.subplots(figsize=(25,15))
sns.scatterplot(x='date',y='pm25_avg_aqi',data=df_states_pm25_daily,palette='inferno',edgecolor=sns.color_palette('dark',7),hue = 'state_code')
plt.ylabel('Air Quality Index PM2.5', fontsize=20)
plt.xlim(df_states_pm25_daily.date.min(), df_states_pm25_daily.date.max())
plt.xticks(rotation=90)
plt.xlabel('Date', fontsize=20)
plt.title('Average AQI of PM2.5 By Month', fontsize=24)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# Plot appears to show a trend of PM2.5 being reduced over time.
# 
# **Does PM2.5 change month to month?**
# 
# Assumption is that the hotter and colder months would have higer PM2.5 levels due to increased heating and cooling.

# In[ ]:


g = sns.FacetGrid(df_states_pm25_daily, col="year", col_wrap=6, hue="state_code",palette='inferno')
g.map(plt.scatter, "month", "pm25_avg_aqi", alpha=.7)
g.add_legend();


# Look at a few mid west states (IL,MI,OH,PA,NY,VT,IA)

# In[ ]:


g = sns.FacetGrid(df_states_pm25_daily[(df_states_pm25_daily.state_code == 'PA') | (df_states_pm25_daily.state_code ==  'OH')|(df_states_pm25_daily.state_code ==  'IL')|
                                      (df_states_pm25_daily.state_code ==  'NY')|(df_states_pm25_daily.state_code ==  'MI')|(df_states_pm25_daily.state_code ==  'VT')|
                                       (df_states_pm25_daily.state_code ==  'IA')], col="year", col_wrap=6, hue="state_code",palette='inferno')
g.map(plt.scatter, "month", "pm25_avg_aqi", alpha=.7)
g.add_legend();

                                       
                                      


# Looking at a few mid west and eastern states (IL,MI,OH,PA,NY,VT,IA) there seems to be a pattern of higher vaules in the summer and winter.
# 
# **Next, Queary from the global database to look at another Country**

# In[ ]:


bq_assistant_global = BigQueryHelper(active_project="bigquery-public-data", dataset_name="openaq")

QUERY_year = """
    SELECT
       city,
       country,
       pollutant ,
       value AS pm25,
       timestamp,
       unit,
       latitude,
       longitude,
       EXTRACT(YEAR FROM timestamp) AS year,
       EXTRACT(MONTH FROM timestamp) AS month
    FROM
      `bigquery-public-data.openaq.global_air_quality` as globalAQ
    WHERE
       country = 'CN' AND pollutant = 'pm25' AND value > 0
    ORDER BY city,
       EXTRACT(MONTH FROM timestamp),
       EXTRACT(YEAR FROM timestamp)
    #LIMIT 1000;
"""

df_global = bq_assistant_global.query_to_pandas(QUERY_year)
df_global.head()


# In[ ]:


df_global['date'] = pd.to_datetime(df_global['year'].astype(str)  + df_global['month'].astype(str), format='%Y%m')

df_global_avg = df_global.groupby(["year","month","city"]).agg({"pm25":"mean"})
df_global_avg.reset_index(inplace=True)
df_global_avg


# In[ ]:


g = sns.FacetGrid(df_global_avg[(df_global_avg.city == 'Beijing') | (df_global_avg.city ==  'Anqing')|
                         (df_global_avg.city == 'Chizhou') | (df_global_avg.city ==  'Shanghai')|      
                         (df_global_avg.city == 'Shenyang') | (df_global_avg.city ==  'Wuhu')|      
                         (df_global_avg.city == 'Xuancheng') | (df_global_avg.city ==  'Bozhou')     
                               ], col="year", col_wrap=6, hue="city",palette='inferno')
g.map(plt.scatter, "month", "pm25", alpha=.7)
g.add_legend();
g


# Interesting, not a lot of data.

# In[ ]:


df_global.groupby(["year","latitude","longitude"]).agg({"pm25":"mean"})

