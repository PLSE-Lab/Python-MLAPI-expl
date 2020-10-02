#!/usr/bin/env python
# coding: utf-8

# <img align="left" src="https://gcn.com/-/media/GIG/GCN/Redesign/Articles/2020/February/covid19.jpg"></img>

# 
# ######Created by AGBaysa
# 
# ###The following graphs shows the update as of ** March 24, 2020** on the number of COVID-19 confirmed cases in the Philippines and the rest of Southeast Asia (ex China and South Korea). This report is updated every 24 hours based on the availability of new data.
#  

# ### References
# 
# 
# * [Novel Coronavirus (COVID-19) Cases, provided by JHU CSSE](https://github.com/CSSEGISandData/COVID-19)
# 
# * [COVID19 Global Forecasting (Week 1)](https://www.kaggle.com/c/covid19-global-forecasting-week-1)
# 
# * [Novel Corona Virus 2019 Dataset](https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset)
# 
# * [COVID-19 Complete Dataset (Updated every 24hrs)](https://www.kaggle.com/imdevskp/corona-virus-report)
# 

# In[ ]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
from plotly.subplots import make_subplots

from IPython.display import display, HTML
js = "<script>$('.output_scroll').removeClass('output_scroll')</script>"
display(HTML(js))


df = pd.read_csv('../input/covid19-coronavirus/2019_nCoV_data.csv')


# In[ ]:


# Hide
df.rename(columns={'Date': 'date', 
                     'Id': 'id',
                     'Province/State':'state',
                     'Country':'country',
                     'Lat':'lat',
                     'Long': 'long',
                     'ConfirmedCases': 'confirmed',
                     'Fatalities':'deaths',
                     'ObservationDate': 'obsdate',
                     'Last Update': 'last_upd'
                    }, inplace=True)


# df["obsdate"] = pd.to_datetime(df["obsdate"]).dt.strftime('%m-%d')
# df["last_upd"] = pd.to_datetime(df["last_upd"]).dt.strftime('%m-%d')
# df


# In[ ]:


# Hide
sea = ['Bangladesh','Cambodia','India','Indonesia','Japan','Malaysia','Philippines','Singapore','Taiwan*','Vietnam']
df.columns
df_sea = df[df.country.isin(sea)]
df_sea.sort_values('date', ascending = False)


# ##**Total Confirmed Cases in Southeast Asia**
# ###* The number of total confirmed cases in Southeast Asia surpassed the 4,000 level and is now at `4,938`, up from `4,361`, a 13% increase. Cases started to surge dramatically on March 4 which was driven by the increase in confirmed cases from Japan and Malaysia
# ###* Last March 4, the total confirmed cases in Southeast Asia was at 541 and has dramatically increased thereafter.

# In[ ]:


# SEA Cases
grouped = df_sea.groupby('date')['date', 'Confirmed', 'Deaths'].sum().reset_index()


fig = px.line(grouped, x="date", y="Confirmed", 
              title="Total Confirmed Cases in Southeast Asia Over Time",
              labels={'obsdate': 'Date', 'Confirmed': 'Confirmed Cases'})

fig.show()

fig = px.line(grouped, x="date", y="Confirmed", 
              title="Total Confirmed Cases in Southease Asia (Logarithmic Scale) Over Time",
              labels={'date': 'Date', 'Confirmed': 'Confirmed Cases (Log)'},
              log_y=True)
fig.show()


# ##**Confirmed Cases by Southeast Asian Countries**
# ###* `Malaysia` has reported `1,624` cases, up by 6.9% from yesterday's `1,518`.
# ###* `Japan` has `1,193` confirmed cases, up by 5.7% from `1,128`. Japan's trajectory seems to be slowing down.
# ###* `Indonesia` has `686` cases, up by 18% from `579`.
# ###* `Singapore` reported `558` cases, up by 9.6% from `509`. Singapore's trend seems to be also slowing down.
# ###* `India` continues to increase but at a slower pace. Confirmed cases increased by 9.8%, from `488` to `536`.
# ###* Confirmed cases in the `Philippines` is  up by 19%, slightly slower than the previous 22% increase, but still comparatively high. The latest count is at `552` versus `462`, an increase of 19% (As of March 27,the country reported `707` cases).

# In[ ]:


fig = px.line(df_sea, x='obsdate', y='Confirmed', color='country',
             title='Confirmed Cases in Southeast Asian Countries',
             labels={'obsdate': 'Date', 'Confirmed': 'Confirmed Cases'})
fig.show()


# ##**Cumulative Daily Percent Increase**
# ###* The trajectory for all Southeast Asian countries remains the same as the previous days. `Singapore` and `Vietnam` seems to follow the same trajectory in terms of the cumulative percentage increase per day.
# ###* For the `Philippines`, there seem to be two (2) inflection points: Jan 30 and March 8 as the trend in the daily cumulative percentage increased. The trajectory continues to be steep in terms of daily cumulative percentage.
# ###* `Indonesia` initially reported on March 2 (late reporting only?) and the trajectory continues to be steep.
# ###* India's inflection point was last March 3.

# In[ ]:


# Percentage change
import matplotlib.pyplot as plt
import numpy as np

foo = df_sea.sort_values(['country', 'obsdate'], ascending = (True, True))
foo = foo.loc[:,['country','obsdate','Confirmed']]

country = ['Singapore','Indonesia','India','Philippines','Vietnam']
foo = foo.loc[foo['country'].isin(country)]

foo = foo.groupby(['country', 'obsdate']).sum().groupby(level=[0]).cumsum().pct_change().replace(np.inf, 0).cumsum().fillna(0)
foo = foo.reset_index()
foo.sort_values(['obsdate'], ascending=True, inplace=True)

# foo[foo['country'] == 'Philippines']
# foo.to_csv('foo.csv', index=False)

fig = px.line(foo, x='obsdate', y='Confirmed', color='country',
             title='Confirmed Cases in SEA - Cum. Daily Percentage',
             labels={'obsdate': 'Date', 'Confirmed': 'Confirmed Cases'})
fig.show()

