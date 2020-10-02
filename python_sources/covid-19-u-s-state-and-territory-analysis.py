#!/usr/bin/env python
# coding: utf-8

# My original analysis was started in this other [notebook](https://www.kaggle.com/ryanglasnapp/covid-19-eda-and-country-comparisons); however, after noticing a large jump in the number of deaths I decided to create a separate notebook for a state by state analysis.
# 
# Nomenclature note: I'm calling everything here a state, even though this encompasses territories and the Distric of Columbia, it's just easier.
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt 
from matplotlib import dates as mdates
from seaborn import heatmap
from matplotlib.colors import LogNorm

get_ipython().run_line_magic('matplotlib', 'inline')

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

all_covid = pd.read_csv("../input/coronavirus-2019ncov/covid-19-all.csv")

# Additional county level data set

us_counties_data = pd.read_csv("../input/us-counties-covid-19-dataset/us-counties.csv", dtype={'fips': object})

# Any results you write to the current directory are saved as output.


# # Data Cleaning

# In[ ]:


# Data cleaning and sorting

us_all = all_covid.loc[all_covid['Country/Region'].isin(['US'])]

state_abbrv = {"AL": "Alabama", "AK":"Alaska", "AZ":"Arizona", "AR":"Arkansas", "CA":"California", "CO":"Colorado", "CT":"Connecticut", "DC":"Washington, D.C.", "DE":"Delaware", "FL":"Florida", 
               "GA":"Georgia", "HI":"Hawaii", "ID":"Idaho", "IL":"Illinois", "IN":"Indiana", "IA":"Iowa", "KS":"Kansas", "KY":"Kentucky", "LA":"Louisiana", "ME":"Maine", "MD":"Maryland", 
          "MA":"Massachusetts", "MI":"Michigan", "MN":"Minnesota", "MS":"Mississippi", "MO":"Missouri", "MT":"Montana", "NE":"Nebraska", "NV":"Nevada", "NH":"New Hampshire", "NJ":"New Jersey", 
          "NM":"New Mexico", "NY":"New York", "NC":"North Carolina", "ND":"North Dakota", "OH":"Ohio", "OK":"Oklahoma", "OR":"Oregon", "PA":"Pennsylvania", "RI":"Rhode Island", 
          "SC":"South Carolina", "SD":"South Dakota", "TN":"Tennessee", "TX":"Texas", "UT":"Utah", "VT":"Vermont", "VA":"Virginia", "WA":"Washington", "WV":"West Virginia", "WI":"Wisconsin", 
          "WY":"Wyoming", "Virgin Islands, U.S.": "Virgin Islands", "District of Columbia":"Washington, D.C.", "Chicago":"Illinois", "United States Virgin Islands":"Virgin Islands"  }

# Dropping odd data
us_all.drop(us_all.loc[us_all['Province/State'] == 'US'].index,inplace=True)
us_all.drop(us_all.loc[us_all['Province/State'] == 'Recovered'].index,inplace=True)
us_all.drop(us_all.loc[us_all['Province/State'].str.contains("Princess")].index,inplace=True)

for state in state_abbrv:
    us_all.loc[us_all.iloc[:,1].str.contains(state), 'Province/State'] = state_abbrv[state]

state_list = np.sort(us_all['Province/State'].unique())

date_list = pd.to_datetime(us_all.Date.unique())

us_confirmed = pd.DataFrame(index=date_list)
us_deaths = pd.DataFrame(index=date_list)
us_recovered = pd.DataFrame(index=date_list)


# Note: This is my old way of doing things, I'm leaving it in because I'm reasonable confident it works and it makes for a good comparison against my newer method.
for state in state_list:
    cur_state = us_all.loc[us_all['Province/State']==state,['Confirmed', 'Recovered', 'Deaths', 'Date']].groupby('Date').sum()
    us_confirmed = us_confirmed.join(cur_state['Confirmed'])
    us_confirmed.rename(columns={'Confirmed':state}, inplace=True)
    us_deaths = us_deaths.join(cur_state['Deaths'])
    us_deaths.rename(columns={'Deaths':state}, inplace=True)
    us_recovered = us_recovered.join(cur_state['Recovered'])
    us_recovered.rename(columns={'Recovered':state}, inplace=True)
    
#us_all.loc[us_all['Province/State']=='Washington',['Confirmed', 'Recovered', 'Deaths', 'Date']].groupby('Date').sum()


# This dataset is based on the NYT data. Ultimately I would like to have data pulled from the same sources just with different granularity, as well as datasets that have data up to the same dates; however, this is what I've got to work with currently.

# In[ ]:


# Counties level data set cleaning

us_counties_data = us_counties_data.rename(columns={'date':'Date',
                                                    'county':'County',
                                                    'state':'Province/State',
                                                    'fips': 'FIPS',
                                                    'cases':'Confirmed',
                                                    'deaths':'Deaths'})


#us_counties_data.drop(us_counties_data['FIPS'].isna().index)
us_counties_data['County_and_State'] = us_counties_data['County'] + ', ' + us_counties_data['Province/State']

us_counties_data.drop(us_counties_data.loc[us_counties_data['FIPS'].isna()].index,inplace=True)

#us_counties_data['FIPS'] = us_counties_data['FIPS'].apply(lambda x: f'{x:.0f}')


# # Data Exploration and Analysis
# Note as of 7/19/2020: Slight change to the code below because the original dataset I was using hasn't been updated in 14 days, so I'm using the dataset I was using for the county level data instead (for now).

# ## Calculations

# In[ ]:


# Calculations 
us_death_dt = us_deaths.diff()
us_death_dt2 = us_death_dt.diff()

us_confirmed_dt = us_confirmed.diff()
us_confirmed_dt2 = us_confirmed_dt.diff()


# In[ ]:


state_grp_orig = us_all.groupby(['Province/State','Date'], as_index=False).sum()
state_grp = us_counties_data.groupby(['Province/State', 'Date'], as_index=False).sum()
#state_grp2.loc[state_grp2['Province/State'] == 'New York'].groupby('Date').first()

state_grp['Confirmed_dt'] = state_grp.groupby(['Province/State'])['Confirmed'].diff().rolling(6).mean()
#state_grp['Recovered_dt'] = state_grp.groupby(['Province/State'])['Recovered'].diff().rolling(6).mean()
state_grp['Deaths_dt'] = state_grp.groupby(['Province/State'])['Deaths'].diff().rolling(6).mean()

us_counties_data['Confirmed_dt'] = us_counties_data.groupby(['County_and_State'])['Confirmed'].diff().rolling(6).mean()
us_counties_data['Deaths_dt'] = us_counties_data.groupby(['County_and_State'])['Deaths'].diff().rolling(6).mean()



state_grp['Confirmed_dt2'] = state_grp.groupby(['Province/State'])['Confirmed_dt'].diff()
#state_grp['Recovered_dt2'] = state_grp.groupby(['Province/State'])['Recovered_dt'].diff()
state_grp['Deaths_dt2'] = state_grp.groupby(['Province/State'])['Deaths_dt'].diff()

us_counties_data['Confirmed_dt2'] = us_counties_data.groupby(['County_and_State'])['Confirmed_dt'].diff()
us_counties_data['Deaths_dt2'] = us_counties_data.groupby(['County_and_State'])['Deaths_dt'].diff()

state_grp['Confirmed_pct'] = state_grp.groupby(['Province/State'])['Confirmed'].pct_change() + 1
state_grp['Neg_Confirmed_dt2'] = state_grp.groupby(['Province/State'])['Confirmed_dt2'].apply(lambda x: (x < 0).rolling(14).sum())
state_grp['Mean_Confirmed_dt2'] = state_grp.groupby(['Province/State'])['Confirmed_dt2'].transform(lambda x: x.rolling(14).mean())

state_grp['Deaths_pct'] = state_grp.groupby(['Province/State'])['Deaths'].pct_change() + 1
state_grp['Neg_Deaths_dt2'] = state_grp.groupby(['Province/State'])['Deaths_dt2'].apply(lambda x: (x < 0).rolling(14).sum())
state_grp['Mean_Deaths_dt2'] = state_grp.groupby(['Province/State'])['Deaths_dt2'].transform(lambda x: x.rolling(14).mean())


us_counties_data['Confirmed_pct'] = us_counties_data.groupby(['County_and_State'])['Confirmed'].pct_change() + 1
us_counties_data['Neg_Confirmed_dt2'] = us_counties_data.groupby(['County_and_State'])['Confirmed_dt2'].apply(lambda x: (x < 0).rolling(14).sum())
us_counties_data['Mean_Confirmed_dt2'] = us_counties_data.groupby(['County_and_State'])['Confirmed_dt2'].transform(lambda x: x.rolling(14).mean())


#state_grp['Mean_Confirmed_dt2'] = state_grp['Mean_Confirmed_dt2'].div(state_grp['Confirmed'])*10000


# ## Recent trends (14 days)
# This next set of data shows the trend for the last 14 days. The case growth is straightforward, it's just the median case growth for the last 14 days. The case deceleration is the percentage of the last 14 days that had days where the case growth was decelerating. 
# 
# States with a low case growth percentage and a high deceleration are doing well. 

# In[ ]:


def calculate_recent_trends(dataset, columns, case_type='Confirmed'):
    last_pct = pd.pivot_table(dataset,values=case_type+'_pct',index='Date',columns=columns).rolling(14).median().tail(1).T
    last_pct.columns = ['Case_growth']

    neg_dt2 = pd.pivot_table(dataset,values='Neg_'+case_type+'_dt2',index='Date',columns=columns).tail(1).T/14*100
    neg_dt2.columns = ['Case_deceleration']

    avg_accel = pd.pivot_table(dataset,values='Mean_'+case_type+'_dt2',index='Date',columns=columns).tail(1).T
    avg_accel.columns = ['Avg_Case_acceleration']

    latest_cases = pd.pivot_table(dataset,values=case_type,index='Date',columns=columns).tail(1).T
    latest_cases.columns = ['Latest_Case_totals'] 
    latest_cases['Latest_Case_totals'] = latest_cases['Latest_Case_totals'].map('{:,.1f}'.format)

    new_cases = pd.pivot_table(dataset,values=case_type+'_dt',index='Date',columns=columns).tail(1).T
    new_cases.columns = ['New_Daily_Cases']
    new_cases['New_Daily_Cases'] = new_cases['New_Daily_Cases'].map('{:,.0f}'.format)

    recent_trends = last_pct.join(neg_dt2).join(avg_accel).join(latest_cases).join(new_cases)

    recent_trends = recent_trends.round({'Case_growth':4, 'Case_deceleration':2, 'Avg_Case_acceleration':2})

    for col in recent_trends:
        recent_trends[col] = recent_trends[col].astype(str)

    return recent_trends


# In[ ]:


def test_callback(trace, points, selector):
    print(trace)
    print(points)
    print(selector)


# In[ ]:


recent_trends = calculate_recent_trends(state_grp, ['Province/State'])

reverse_abrv = dict((v, k) for k, v in state_abbrv.items())
# Fix for some items since the original dictionary isn't one to one.
reverse_abrv['Illinois'] = 'IL'

[key for key in state_abbrv.keys() if state_abbrv[key] in recent_trends.index]

recent_trends['state_abrvs'] = [reverse_abrv[state] if state in reverse_abrv else state for state in recent_trends.index]
recent_trends['text'] = 'Total Confirmed Cases: ' + recent_trends['Latest_Case_totals'] + '</br>' +                         'New Daily Cases: ' + recent_trends['New_Daily_Cases'] + '</br>' +                         'Case Growth Rate: ' + recent_trends['Case_growth'] + '</br>' +                         'Average Case Acceleration: ' + recent_trends['Avg_Case_acceleration']

fig = go.Figure(data=go.Choropleth(
    locations=recent_trends['state_abrvs'], # Spatial coordinates
    text = recent_trends['text'],
    z = recent_trends['Case_deceleration'], # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'RdYlGn',
    colorbar_title = "% Deceleration Rate",
    zmin = 0,
    zmax = 100,
))

fig.update_layout(
    title_text = 'Confirmed Case Recent Trends for the last 14 days, as of ' + state_grp['Date'].max()+'. <br>States are colored by the percentage of the last 14 days that case growth has slowed.',
    geo_scope='usa', # limit map scope to USA
)

fig.data[0].on_click(test_callback)

fig.show()


# In[ ]:


recent_trends_deaths = calculate_recent_trends(state_grp, ['Province/State'], case_type='Deaths')

reverse_abrv = dict((v, k) for k, v in state_abbrv.items())
# Fix for some items since the original dictionary isn't one to one.
reverse_abrv['Illinois'] = 'IL'

[key for key in state_abbrv.keys() if state_abbrv[key] in recent_trends_deaths.index]

recent_trends_deaths['state_abrvs'] = [reverse_abrv[state] if state in reverse_abrv else state for state in recent_trends_deaths.index]
recent_trends_deaths['text'] = 'Total Confirmed Fatalities: ' + recent_trends_deaths['Latest_Case_totals'] + '</br>' +                         'New Daily Fatalities: ' + recent_trends_deaths['New_Daily_Cases'] + '</br>' +                         'Fatality Growth Rate: ' + recent_trends_deaths['Case_growth'] + '</br>' +                         'Average Fatality Acceleration: ' + recent_trends_deaths['Avg_Case_acceleration']

fig_deaths = go.Figure(data=go.Choropleth(
    locations=recent_trends_deaths['state_abrvs'], # Spatial coordinates
    text = recent_trends_deaths['text'],
    z = recent_trends_deaths['Case_deceleration'], # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'RdYlGn',
    colorbar_title = "% Deceleration Rate",
    zmin = 0,
    zmax = 100,
))

fig_deaths.update_layout(
    title_text = 'Fatality Recent Trends for the last 14 days, as of ' + state_grp['Date'].max()+'. <br>States are colored by the percentage of the last 14 days that fatality growth has slowed.',
    geo_scope='usa', # limit map scope to USA
)

fig_deaths.data[0].on_click(test_callback)

fig_deaths.show()


# ### County Recent Trends
# Same analysis as I did with the states, but with the county level data.
# 

# In[ ]:


county_recent_trends = calculate_recent_trends(us_counties_data, ['County_and_State'])

county_recent_trends = county_recent_trends.join(us_counties_data[['FIPS','County_and_State']].set_index('County_and_State').drop_duplicates())


# In[ ]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)


# In[ ]:


#import plotly.express as px

#fig = px.choropleth_mapbox(county_recent_trends, geojson=counties, locations='FIPS', color='Case_deceleration',
#                           color_continuous_scale=px.colors.diverging.RdYlGn,
#                           #colorscale = 'RdYlGn',
#                           range_color=(0, 100),
#                           mapbox_style="carto-positron",
#                           zoom=3, center = {"lat": 37.0902, "lon": -95.7129},
#                           opacity=0.5,
#                           #labels={'unemp':'unemployment rate'}
#                          )
    
#fig = go.Figure(data=go.Choroplethmapbox(
#    geojson=counties, # Spatial coordinates
#    locations = county_recent_trends['FIPS'],
#    #text = recent_trends['text'],
#    z = county_recent_trends['Case_deceleration'], # Data to be color-coded
#    colorscale = 'RdYlGn',
#    colorbar_title = "% Deceleration Rate",
#    zmin = 0,
#    zmax = 100,
#))

#fig.update_layout(
#    title_text = 'Recent Trends for the last 14 days, as of ' + us_counties_data['Date'].max()+'. <br>Counties are colored by the percentage of the last 14 days that case growth has slowed.',
#    #geo_scope='usa', # limit map scope to USA
#)
#fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#
#fig.show()


# In[ ]:


county_recent_trends['Location']=county_recent_trends.index


county_recent_trends['text'] = county_recent_trends['Location'] + '</br>' +                         'Total Confirmed Cases: ' + county_recent_trends['Latest_Case_totals'] + '</br>' +                         'New Daily Cases: ' + county_recent_trends['New_Daily_Cases'] + '</br>' +                         'Case Growth Rate: ' + county_recent_trends['Case_growth'] + '</br>' +                         'Average Case Acceleration: ' + county_recent_trends['Avg_Case_acceleration']

fig_test = go.Figure(data=go.Choroplethmapbox(
    geojson=counties, # Spatial coordinate
    locations = county_recent_trends['FIPS'],
    text = county_recent_trends['text'],
    z = county_recent_trends['Case_deceleration'], # Data to be color-coded
    colorscale = 'RdYlGn',
    colorbar_title = "% Deceleration Rate",
    zmin = 0,
    zmax = 100,
))

fig_test.update_layout(
    title_text = 'Recent Trends for the last 14 days, as of ' + us_counties_data['Date'].max()+'. <br>Counties are colored by the percentage of the last 14 days that case growth has slowed.',
    #geo_scope='usa', # limit map scope to USA
)
fig_test.update_layout(mapbox_style="carto-positron")
fig_test.update_layout(margin={"r":10,"t":30,"l":10,"b":10})

fig_test.show()


# To keep an eye on the larger picture of the U.S. as a whole I added the graph below. It's based on the same analysis as the individual states comparison further below, where I compare the number of cases versus the new daily cases. It's based on a common technique in physics called a phase space diagram. A phase space diagram is used to analyze the dynamics of a system by plotting the position $x(t)$ versus velocity $v(t)^*$. 
# 
# In this particular case it's useful because it gives a very clear indication of when things across the country as a whole are spiking, or getting better. (As of 6/25/2020 things are spiking again, hence the addition of this section).
# 
# $^*$ Note: In physics the position and velocity are usually changed to the generalized coordinates $q$ and $\dot{q}$. This allows for a broader use of coordinates as $q$ can represent any standard cartesian cordinate ($x,y,z$), radial coordinates ($r, \theta$), or in this particular situation, total cases and new daily cases.

# In[ ]:


us_totals = state_grp.groupby('Date', as_index=False)['Confirmed', 'Confirmed_dt', 'Confirmed_dt2'].sum()

plt_data_us = dict(data=[], layout=dict(
    title='US Only, Cases vs Daily Differential (1st derivative)',
    #width=1500,
    #height=1500,
    xaxis=dict(title='Confirmed Cases',type='log'),
    yaxis=dict(title='Daily Difference of Confirmed Cases',type='log')
))

plt_data_us['data'].append(go.Scatter(
    x = us_totals.Confirmed,
    y = us_totals.Confirmed_dt,
    mode = 'lines+markers',
        name = 'US',
        text = us_totals.Date) )

iplot(plt_data_us)


# ## Heatmaps

# In[ ]:


us_confirmed_subset = us_confirmed.tail(60)[us_confirmed.tail(60) > 200].dropna(axis=1).T


# In[ ]:


fig, ax = plt.subplots(figsize=(20,20))

heatmap(us_confirmed_subset.fillna(1),ax=ax,norm=LogNorm(),cmap='seismic')

ax.set_xticklabels(us_confirmed.iloc[-60:].index.strftime('%m-%d-%Y'))
ax.set_title('Heatmap of Confirmed Cases for the last 60 days')


# In[ ]:


fig, ax = plt.subplots(figsize=(20,20))
heatmap(us_deaths.fillna(0.5).iloc[-60:].T,ax=ax,norm=LogNorm(),cmap='seismic')

ax.set_xticklabels(us_deaths.iloc[-60:].index.strftime('%m-%d-%Y'))
ax.set_title('Heatmap of Fatalities for the last 60 days')


# ## Differential Comparision Graphs
# The graphs below are based upon the ideas presented in MinutePhysics's video here: https://www.youtube.com/watch?v=54XLXg4fYsc. Once I saw that video I recalled phase space diagrams from classical mechanics courses. The generalized coordinate $q$ in this case is Confirmed cases (or Fatalities), and $\dot{q} = \frac{dq}{dt}$ is the daily new cases. In hindsight this is a very common approach in physics, which had just slipped my mind.
# 

# In[ ]:


def create_dropdown_button(name, new_title, tf_position_list):
    return dict(label=name,
               method='update',
               args=[{'visible': tf_position_list},
                    {'title': new_title,
                    'showlegend': True}])


# In[ ]:


#fig, ax = plt.subplots(figsize=(20,20))
#for state, data in state_grp.groupby('Province/State'):
#    plt.plot(data['Confirmed'],data['Confirmed_dt'],label=state)

#ax.set_xlabel('Confirmed Cases')
#ax.set_ylabel('Daily change in Confirmed Cases')
#ax.set_xscale('log')
#ax.set_yscale('log')

#chartBox = ax.get_position()
#ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.6, chartBox.height])
#ax.legend(loc='upper center', bbox_to_anchor=(1.45, 0.8), shadow=True, ncol=1)
#plt.show()


# In[ ]:


state_grps = state_grp.groupby('Province/State')

plt_data = dict(data=[], layout=dict(
    title='State Comparison, Cases vs Daily Differential (1st derivative)',
    #width=1500,
    #height=1500,
    xaxis=dict(title='Confirmed Cases',type='log'),
    yaxis=dict(title='Daily Difference of Confirmed Cases', type='log')
))

all_buttons = list([ create_dropdown_button('All', 'State Comparison, Cases vs Daily Differential (1st derivative)', [True for x in state_grps.groups.keys()]) ])

for state, state_data in state_grps:
    plt_data['data'].append(go.Scatter(
        x = state_data.Confirmed,
        y = state_data.Confirmed_dt,
        mode = 'lines+markers',
        name = state,
        text = state_data.Date) )
    all_buttons.append(create_dropdown_button(state, state+' Only, Cases vs Daily Differential (1st derivative)', [x == state for x in state_grps.groups.keys()] ) )
    
plt_data['layout']['updatemenus'] = [go.layout.Updatemenu(
    active=0,
    buttons= all_buttons
    )]
    
iplot(plt_data)


# In[ ]:


state_grps = state_grp.groupby('Province/State')

plt_data = dict(data=[], layout=dict(
    title='State Comparison, Fatalities vs Daily Differential (1st derivative)',
    #width=1500,
    #height=1500,
    xaxis=dict(title='Fatalities',type='log'),
    yaxis=dict(title='Daily Difference of Fatalities', type='log')
))

all_buttons = list([ create_dropdown_button('All', 'State Comparison, Fatalities vs Daily Differential (1st derivative)', [True for x in state_grps.groups.keys()]) ])

for state, state_data in state_grp.groupby('Province/State'):
    plt_data['data'].append(go.Scatter(
        x = state_data.Deaths,
        y = state_data.Deaths_dt,
        mode = 'lines+markers',
        name = state,
        text = state_data.Date) )
    all_buttons.append(create_dropdown_button(state, state+' Only, Fatalities vs Daily Differential (1st derivative)', [x == state for x in state_grps.groups.keys()] ) )

plt_data['layout']['updatemenus'] = [go.layout.Updatemenu(
    active=0,
    buttons= all_buttons
    )]
    
iplot(plt_data)


# # Changelog
# * 6/25/2020: Added a total US comparison for a higher level view. Moved this section to the bottom.
# * ~6/20/2020: Added a US Map with recent trends data.
# * 6/5/2020: Added a recent trends section. 
# * 6/1/2020: Added drop downs for the state comparisons 
# * 5/28/2020: Fixed the date display on the heatmaps and changed it to only display the last 60 days. 
# * 5/21/2020: Added a 6 day rolling mean to difference computations. I decided on the 6 day mean based on my fourier calculations over in https://www.kaggle.com/ryanglasnapp/covid-19-eda-and-country-comparisons . Also changed the comparison graph to an interactive one using plotly. 
# 
# * 5/19/2020: Finally got the graphs of confirmed vs confirmed.diff() working. It's ugly currently, but in the near future I plan on cleaning it up so that it's readable (and thus useable).

# ## Sanity Checks
# 
# This is sort of a playground of ideas for me to try things that may or may not be useful.

# In[ ]:


state_grp.head(20)


# In[ ]:


state_grps = state_grp.groupby('Province/State')

states = ('New York', 'Alaska', 'New Mexico', 'Washington', 'Texas')
pops = (19.45e6, 731545, 2.097e6, 7.615e6, 29e6)

plt_data_tst = dict(data=[], layout=dict(
    title='State Comparison, Cases vs Daily Differential (1st derivative)',
    #width=1500,
    #height=1500,
    xaxis=dict(title='Confirmed Cases',type='log'),
    yaxis=dict(title='Daily Difference of Confirmed Cases (6 day running avg)',type='log')
))

all_buttons = list([ create_dropdown_button('All', 'State Comparison, Cases vs Daily Differential (1st derivative)', [True for x in states]) ])



for state, pop  in zip(states,pops):
    curstate = state_grps.get_group(state)
    plt_data_tst['data'].append(go.Scatter(
        x = curstate.Confirmed/pop,
        y = curstate.Confirmed_dt/pop,
        mode = 'lines+markers',
        name = state,
        text = curstate.Date) )
    all_buttons.append(create_dropdown_button(state, state+' Only, Cases vs Daily Differential (1st derivative)', [x == state for x in states] ) )

plt_data_tst['layout']['updatemenus'] = [go.layout.Updatemenu(
    active=0,
    buttons= all_buttons
    )]

iplot(plt_data_tst)


# In[ ]:


plt_data_2nd = dict(data=[], layout=dict(
    title='State Comparison, Daily Differential vs 2nd Daily Differential',
    #width=1500,
    #height=1500,
    xaxis=dict(title='Confirmed Case per capita',type='log'),
    yaxis=dict(title='2nd Derivative/1st Derivative')
))

for state, pop  in zip(states,pops):
    curstate = state_grps.get_group(state)
    plt_data_2nd['data'].append(go.Scatter(
        x = curstate.Confirmed,
        y = curstate.Confirmed_dt2/curstate.Confirmed_dt,
        mode = 'lines+markers',
        name = state,
        text = curstate.Date) )
    
iplot(plt_data_2nd)


# In[ ]:


state_grps = state_grp.groupby('Province/State')

plt_data = dict(data=[], layout=dict(
    title='State Comparison, Daily Differential vs 2nd Daily Differential',
    #width=1500,
    #height=1500,
    xaxis=dict(title='Confirmed Case Daily Diff'),
    yaxis=dict(title='Confirmed Cases 2nd Daily Differential')
))

all_buttons = list([ create_dropdown_button('All', 'State Comparison, Daily Differential vs 2nd Daily Differential', [True for x in state_grps.groups.keys()]) ])

for state, state_data in state_grp.groupby('Province/State'):
    plt_data['data'].append(go.Scatter(
        x = state_data.Confirmed_dt,
        y = state_data.Confirmed_dt2,
        mode = 'lines+markers',
        name = state,
        text = state_data.Date) )
    all_buttons.append(create_dropdown_button(state, state+' Only, Daily Differential vs 2nd Daily Differential', [x == state for x in state_grps.groups.keys()] ) )

plt_data['layout']['updatemenus'] = [go.layout.Updatemenu(
    active=0,
    buttons= all_buttons
    )]
    
iplot(plt_data)


# In[ ]:


# state_grp.groupby('Province/State').plot('Confirmed', 'Confirmed_dt')

