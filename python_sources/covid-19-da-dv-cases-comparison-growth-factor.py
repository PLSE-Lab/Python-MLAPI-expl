#!/usr/bin/env python
# coding: utf-8

# ## About CoronaVirus
# 
# Coronavirus Disease (COVID-19) is an infectious disease caused by the newly discovered coronavirus virus that first appeared in Wuhan, China, in December 2019.
# 
# Read more about coronavirus from [World Health Organization](https://www.who.int/emergencies/diseases/novel-coronavirus-2019), [Centers for Disease Control and Prevention](https://www.cdc.gov/coronavirus/2019-ncov/)
# 
# 
# ## About this NoteBook
# 
# This notebook provides visualization (Linear & Logarithmic scale) on the spread of Coronavirus around the world, as well as for countries named China, Italy, US, India, Spain, Germany.
# 
# Thanks to John Hopkins University for providing the [Covid-19 data](https://github.com/CSSEGISandData/COVID-19) to the DS community.
# 
# ## <font color='red'>I'm going to update my notebook (every 24 hours) with a new data interpretation and visualization chart. Please up-vote the kernel if you like the work. In addition, your feedback is very much needed.</font>

# # Data Analysis

# ### Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_dark"
sns.set(style="darkgrid")
import matplotlib.dates as mdates
import datetime as dt


# ### Import Datasets

# In[ ]:


f = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv')
f_recovered = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')
f_deaths = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')
f_confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
f['ObservationDate'] = pd.to_datetime(f['ObservationDate'], infer_datetime_format=True)
f.drop(columns='SNo')


# ### Information about the Covid_19_data

# In[ ]:


f.info()


# ### Add Active case column to "f" DataFrame and convert data type of Confirmed, Deaths and Recovered columns to integer

# In[ ]:


f[["Confirmed","Deaths","Recovered"]] = f[["Confirmed","Deaths","Recovered"]].astype(int) 
f['Active'] = f['Confirmed'] - f['Deaths'] - f['Recovered']
f.head()


# ### Total Number of countries affected by Covid-19

# In[ ]:


f['Country/Region'].nunique()


# ### List of Countries affected by Covid-19

# In[ ]:


f['Country/Region'].unique()


# ### Total Number of cases (Confirmed, Deaths, Recovered and Active Cases)

# In[ ]:


Total = f.groupby('ObservationDate').sum()
Total.drop(columns = 'SNo').max()


# # World Data Analysis and Visualizations

# ### Data Analysis

# In[ ]:


# World data
world = f.groupby(['ObservationDate'])[['SNo','Confirmed', 'Deaths','Recovered','Active']].sum().reset_index()
world = world.sort_values(by ='ObservationDate',ascending=0)
world[['Confirmed', 'Deaths','Recovered','Active']] = world[['Confirmed', 'Deaths','Recovered','Active']].astype(float)
world = world.sort_values(by = 'ObservationDate', ascending = 0)

# Countrywise data
df = f[f['ObservationDate'] == max(f['ObservationDate'])]
country = df.groupby(["Country/Region"])[['SNo','Confirmed', 'Deaths','Recovered','Active']].sum().reset_index().sort_values("Confirmed",ascending=False).reset_index(drop=True)
country.head()

# Worldwide - Confirmed, Deaths, Recovered and Active Covid-19 Cases
confirmed = f.groupby('ObservationDate').sum()['Confirmed'].reset_index()
deaths = f.groupby('ObservationDate').sum()['Deaths'].reset_index()
recovered = f.groupby('ObservationDate').sum()['Recovered'].reset_index()
active = f.groupby('ObservationDate').sum()['Active'].reset_index()

# Add 'Conf_change', 'Death_change', 'Recov_change', 'Active_change' to world dataframe
world['Conf_change'] = world.apply(lambda _: '', axis=1)
world['Death_change'] = world.apply(lambda _: '', axis=1)
world['Recov_change'] = world.apply(lambda _: '', axis=1)
world['Active_change'] = world.apply(lambda _: '', axis=1)
world = world.append(pd.Series(name='Start'))
world = world.fillna(0)
world['Conf_change'] = world['Confirmed'].diff(periods=-1)
world['Death_change'] = world['Deaths'].diff(periods=-1)
world['Recov_change'] = world['Recovered'].diff(periods=-1)
world['Active_change'] = world['Active'].diff(periods=-1)

# Add 'grow_fact_conf', 'grow_fact_death', 'grow_fact_recov', 'grow_fact_active' to world dataframe
world['grow_fact_conf'] = world.apply(lambda _: '', axis=1)
world['grow_fact_death'] = world.apply(lambda _: '', axis=1)
world['grow_fact_recov'] = world.apply(lambda _: '', axis=1)
world['grow_fact_active'] = world.apply(lambda _: '', axis=1)
world['grow_fact_conf'] = world['Conf_change'].pct_change(periods = -1) + 1
world['grow_fact_death'] = world['Death_change'].pct_change(periods = -1) + 1 
world['grow_fact_recov'] = world['Recov_change'].pct_change(periods = -1) + 1 
world['grow_fact_active'] = world['Active_change'].pct_change(periods = -1) + 1 
world = world.sort_values(by = 'Confirmed', ascending = 1)

# weekly dataframe
week_df = world[1:]
week_df = week_df[::7]
week_df = week_df.sort_values(by = 'Confirmed', ascending = 0)
week_df
# Add 'Conf_change', 'Death_change', 'Recov_change', 'Active_change' to world dataframe
week_df['Conf_change'] = week_df.apply(lambda _: '', axis=1)
week_df['Death_change'] = week_df.apply(lambda _: '', axis=1)
week_df['Recov_change'] = week_df.apply(lambda _: '', axis=1)
week_df['Active_change'] = week_df.apply(lambda _: '', axis=1)
week_df = week_df.append(pd.Series(name='Start'))
week_df = week_df.fillna(0)
week_df['Conf_change'] = week_df['Confirmed'].diff(periods=-1)
week_df['Death_change'] = week_df['Deaths'].diff(periods=-1)
week_df['Recov_change'] = week_df['Recovered'].diff(periods=-1)
week_df['Active_change'] = week_df['Active'].diff(periods=-1)

# Add 'grow_fact_conf', 'grow_fact_death', 'grow_fact_recov', 'grow_fact_active' to world dataframe
week_df['grow_fact_conf'] = week_df.apply(lambda _: '', axis=1)
week_df['grow_fact_death'] = week_df.apply(lambda _: '', axis=1)
week_df['grow_fact_recov'] = week_df.apply(lambda _: '', axis=1)
week_df['grow_fact_active'] = week_df.apply(lambda _: '', axis=1)
week_df['grow_fact_conf'] = week_df['Conf_change'].pct_change(periods = -1) + 1
week_df['grow_fact_death'] = week_df['Death_change'].pct_change(periods = -1) + 1 
week_df['grow_fact_recov'] = week_df['Recov_change'].pct_change(periods = -1) + 1 
week_df['grow_fact_active'] = week_df['Active_change'].pct_change(periods = -1) + 1 
week_df = week_df.sort_values(by = 'Confirmed', ascending = 1)
week_df = week_df[1:]


# ### Data Visualizations

# ### Worldwide Coronavirus - Confirmed, Deaths, Recovered and Active Cases

# #### Worldwide Coronavirus - Confirmed, Deaths, Recovered and Active Cases (Linear Scale)

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=confirmed['ObservationDate'],
                y=confirmed['Confirmed'],
                name='Confirmed',
                marker_color='white'
                ))
fig.add_trace(go.Scatter(x=deaths['ObservationDate'],
                y=deaths['Deaths'],
                name='Deaths',
                marker_color='Red'
                ))
fig.add_trace(go.Scatter(x=recovered['ObservationDate'],
                y=recovered['Recovered'],
                name='Recovered',
                marker_color='Green'
                ))
fig.add_trace(go.Scatter(x=active['ObservationDate'],
                y=active['Active'],
                name='Active',
                marker_color='Blue'
                ))
fig.update_layout(title='Worldwide Coronavirus - Confirmed, Deaths, Recovered and Active Cases (Linear Scale)',
                  xaxis = dict(title = 'Observation Dates'),
                  yaxis = dict(title = 'Number of Cases')
                  )
fig.show()


# #### Worldwide Corona Virus - Confirmed, Deaths, Recovered and Active Cases (Logarithmic Scale)

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=confirmed['ObservationDate'],
                y=confirmed['Confirmed'],
                name='Confirmed',
                marker_color='white'
                ))
fig.add_trace(go.Scatter(x=deaths['ObservationDate'],
                y=deaths['Deaths'],
                name='Deaths',
                marker_color='Red'
                ))
fig.add_trace(go.Scatter(x=recovered['ObservationDate'],
                y=recovered['Recovered'],
                name='Recovered',
                marker_color='Green'
                ))
fig.add_trace(go.Scatter(x=active['ObservationDate'],
                y=active['Active'],
                name='Active',
                marker_color='Blue'
                ))
fig.update_layout(title='Worldwide Coronavirus - Confirmed, Deaths, Recovered and Active Cases (Logarithmic)',
                  xaxis = dict(title = 'Observation Dates'),
                  yaxis = dict(title = 'Number of Cases'),
                  yaxis_type="log"
                  )
fig.show()


# ### Confirmed vs Recovered Cases - Worldwide - Bubble Chart

# In[ ]:


fig = px.scatter(data_frame = country, x=country['Confirmed'], y=country['Recovered'], size = country['Confirmed'], 
                 color = country['Country/Region'], hover_name=country['Country/Region'],log_x=1, size_max=80)
fig.update_layout(title='Confirmed vs Recovered Cases - Worldwide - Bubble Chart')
fig.show()


# ### Deaths vs Recovered Cases - Worldwide - Bubble Chart

# In[ ]:


fig = px.scatter(data_frame = country, x=country['Deaths'], y=country['Recovered'], size = country['Deaths'], 
                 color = country['Country/Region'], hover_name=country['Country/Region'],log_x=1, size_max=80)
fig.update_layout(title='Deaths vs Recovered Cases - Worldwide - Bubble Chart')
fig.show()


# ### Growth Factor for Confirmed, Deaths and Recovered Cases (Worldwide)

# #### Daily Growth Factor - Confirmed, Deaths and Recovered Cases (Worldwide)

# In[ ]:


world = world.sort_values(by = 'Confirmed', ascending = 1)

fig = go.Figure()

fig.add_trace(go.Scatter(x = world['ObservationDate'][1:],
                         y = world['grow_fact_conf'][1:],
                         mode='lines+markers',
                         name = 'Confirmed'))

fig.add_trace(go.Scatter(x = world['ObservationDate'][1:],
                         y = world['grow_fact_death'][1:],
                         mode='lines+markers',
                         name = 'Deaths'))

fig.add_trace(go.Scatter(x = world['ObservationDate'][1:],
                         y = world['grow_fact_recov'][1:],
                         mode='lines+markers',
                         name = 'Recovered'))

fig.update_layout(title='Daily Growth Factor - Confirmed, Deaths and Recovered Cases (Worldwide)',
                  xaxis_title="Observation Dates",
                  yaxis_title="Growth Factor",
                  yaxis_type = "log")

fig.show()


# #### Weekly Growth Factor - Confirmed, Deaths and Recovered Cases (Worldwide)

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(x = week_df['ObservationDate'],
                         y = week_df['grow_fact_conf'],
                         mode='lines+markers',
                         name = 'Confirmed'))

fig.add_trace(go.Scatter(x = week_df['ObservationDate'],
                         y = week_df['grow_fact_death'],
                         mode='lines+markers',
                         name = 'Deaths'))

fig.add_trace(go.Scatter(x = week_df['ObservationDate'],
                         y = week_df['grow_fact_recov'],
                         mode='lines+markers',
                         name = 'Recovered'))

fig.update_layout(title='Weekly Growth Factor - Confirmed, Deaths and Recovered Cases (Worldwide)',
                  xaxis_title="Observation Dates",
                  yaxis_title="Growth Factor",
                  yaxis_type = "log")

fig.show()


# ### New Cases for Confirmed, Deaths and Recovered Cases (Worldwide)

# #### Weekly New Confirmed, Death and Recovered Cases - Worldwide

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(x = week_df['ObservationDate'],
                     y = week_df['Conf_change'],
                     marker_color='white',                  
                     name = 'New Confirmed Cases'
                     ))

fig.add_trace(go.Scatter(x = week_df['ObservationDate'],
                     y = week_df['Death_change'],
                     marker_color='red',                  
                     name = 'New Death Cases'
                     ))

fig.add_trace(go.Scatter(x = week_df['ObservationDate'],
                     y = week_df['Recov_change'],
                     marker_color='Green',                  
                     name = 'New Recovered Cases'
                     ))
fig.update_layout(title='Weekly New Confirmed, Death and Recovered Cases - Worldwide',
                  xaxis_title="Observation Date",
                  yaxis_title="New Cases",
                  yaxis_type = "log"
                  )


# #### Daily New Confirmed, Death and Recovered Cases - Worldwide

# In[ ]:



fig = go.Figure()

fig.add_trace(go.Scatter(x = world['ObservationDate'][1:],
                     y = world['Conf_change'][1:],
                     marker_color='white',                  
                     name = 'New Confirmed Cases'
                     ))

fig.add_trace(go.Scatter(x = world['ObservationDate'][1:],
                     y = world['Death_change'][1:],
                     marker_color='red',                  
                     name = 'New Death Cases'
                     ))

fig.add_trace(go.Scatter(x = world['ObservationDate'][1:],
                     y = world['Recov_change'][1:],
                     marker_color='Green',                  
                     name = 'New Recovered Cases'
                     ))
fig.update_layout(title='Daily New Confirmed, Death and Recovered Cases - Worldwide',
                  xaxis_title="Observation Date",
                  yaxis_title="New Cases"
                  )


# #### World Mortality Rate

# In[ ]:


world['mortality_rate'] = np.round(100*world['Deaths']/world['Confirmed'],2)

fig = go.Figure()

fig.add_trace(go.Scatter(x = world['ObservationDate'][1:],
                         y = world['mortality_rate'][1:],
                         name = 'World Mortality Rate',
                         marker_color = '#00a1c1'))

fig.update_layout(title = 'Covid-19 Mortality Rate (Worldwide)')

fig.show()


# # USA Data Analysis and Visualization

# ### Data Analysis

# In[ ]:


# Import USA's States and Counties data
f_USA = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv')
df_USA = f_USA[f_USA['date'] == max(f_USA['date'])]
states = df_USA.groupby(['state'])[['cases','deaths']].sum().reset_index().sort_values("cases",ascending=0).reset_index(drop=True)
counties = df_USA.groupby(['county'])[['cases','deaths']].sum().reset_index().sort_values("cases",ascending=0).reset_index(drop=True)
states['mortality_rate'] = np.round(100*(states['deaths']/states['cases']),2)
counties['mortality_rate'] = np.round(100*(counties['deaths']/counties['cases']),2)


# In[ ]:


US = f[f['Country/Region'] == 'US'].reset_index()
US = US.groupby('ObservationDate')[['SNo','Confirmed', 'Deaths','Recovered','Active']].sum().reset_index()
US[['Confirmed', 'Deaths','Recovered','Active']] = US[['Confirmed', 'Deaths','Recovered','Active']].astype(float)
US = US.sort_values(by = 'ObservationDate', ascending = 0)

# Add 'Conf_change', 'Death_change', 'Recov_change', 'Active_change' to world dataframe
US['Conf_change'] = US.apply(lambda _: '', axis=1)
US['Death_change'] = US.apply(lambda _: '', axis=1)
US['Recov_change'] = US.apply(lambda _: '', axis=1)
US['Active_change'] = US.apply(lambda _: '', axis=1)
US = US.append(pd.Series(name='Start'))
US = US.fillna(0)
US['Conf_change'] = US['Confirmed'].diff(periods=-1)
US['Death_change'] = US['Deaths'].diff(periods=-1)
US['Recov_change'] = US['Recovered'].diff(periods=-1)
US['Active_change'] = US['Active'].diff(periods=-1)

# Add 'grow_fact_conf', 'grow_fact_death', 'grow_fact_recov', 'grow_fact_active' to world dataframe
US['grow_fact_conf'] = US.apply(lambda _: '', axis=1)
US['grow_fact_death'] = US.apply(lambda _: '', axis=1)
US['grow_fact_recov'] = US.apply(lambda _: '', axis=1)
US['grow_fact_active'] = US.apply(lambda _: '', axis=1)
US['grow_fact_conf'] = US['Conf_change'].pct_change(periods = -1) + 1
US['grow_fact_death'] = US['Death_change'].pct_change(periods = -1) + 1 
US['grow_fact_recov'] = US['Recov_change'].pct_change(periods = -1) + 1 
US['grow_fact_active'] = US['Active_change'].pct_change(periods = -1) + 1
US = US.replace([np.inf])
US = US.replace([-np.inf])
US = US.fillna(0)
US = US.sort_values('Confirmed',ascending = 1)
US_temp = US.sort_values('SNo',ascending = 1)
# weekly dataframe
US_week_df = US_temp[1:]
US_week_df = US_week_df[::7]
US_week_df = US_week_df.sort_values(by = 'SNo', ascending = 0)
US_week_df
# Add 'Conf_change', 'Death_change', 'Recov_change', 'Active_change' to world dataframe
US_week_df['Conf_change'] = US_week_df.apply(lambda _: '', axis=1)
US_week_df['Death_change'] = US_week_df.apply(lambda _: '', axis=1)
US_week_df['Recov_change'] = US_week_df.apply(lambda _: '', axis=1)
US_week_df['Active_change'] = US_week_df.apply(lambda _: '', axis=1)
US_week_df = US_week_df.append(pd.Series(name='Start'))
US_week_df = US_week_df.fillna(0)
US_week_df['Conf_change'] = US_week_df['Confirmed'].diff(periods=-1)
US_week_df['Death_change'] = US_week_df['Deaths'].diff(periods=-1)
US_week_df['Recov_change'] = US_week_df['Recovered'].diff(periods=-1)
US_week_df['Active_change'] = US_week_df['Active'].diff(periods=-1)

# Add 'grow_fact_conf', 'grow_fact_death', 'grow_fact_recov', 'grow_fact_active' to world dataframe
US_week_df['grow_fact_conf'] = US_week_df.apply(lambda _: '', axis=1)
US_week_df['grow_fact_death'] = US_week_df.apply(lambda _: '', axis=1)
US_week_df['grow_fact_recov'] = US_week_df.apply(lambda _: '', axis=1)
US_week_df['grow_fact_active'] = US_week_df.apply(lambda _: '', axis=1)
US_week_df['grow_fact_conf'] = US_week_df['Conf_change'].pct_change(periods = -1) + 1
US_week_df['grow_fact_death'] = US_week_df['Death_change'].pct_change(periods = -1) + 1 
US_week_df['grow_fact_recov'] = US_week_df['Recov_change'].pct_change(periods = -1) + 1 
US_week_df['grow_fact_active'] = US_week_df['Active_change'].pct_change(periods = -1) + 1 
US_week_df = US_week_df.sort_values(by = 'SNo', ascending = 1)
US_week_df = US_week_df[1:]


# ## Data Visualization

# ### US Corona Virus - Confirmed, Deaths, Recovered and Active Case

# #### US Corona Virus - Confirmed, Deaths, Recovered and Active Cases (Linear Scale)

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x = US['ObservationDate'][1:],
                         y = US['Confirmed'][1:],
                         name='Confirmed',
                         marker_color='white'
                         ))
fig.add_trace(go.Scatter(x = US['ObservationDate'][1:],
                         y = US['Deaths'][1:],
                         name='Deaths',
                         marker_color='red'
                         ))
fig.add_trace(go.Scatter(x = US['ObservationDate'][1:],
                         y = US['Recovered'][1:],
                         name='Recovered',
                         marker_color='green'
                         ))
fig.add_trace(go.Scatter(x = US['ObservationDate'][1:],
                         y = US['Active'][1:],
                         name='Active',
                         marker_color='blue'
                         ))

fig.update_layout(title='US Corona Virus - Confirmed, Deaths, Recovered and Active Cases (Linear Scale)',
                  xaxis = dict(title = 'Observation Dates'),
                  yaxis = dict(title = 'Number of Cases')
                  )
fig.show()


# #### US Corona Virus - Confirmed, Deaths, Recovered and Active Cases (Logarithmic Scale)

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x = US['ObservationDate'][1:],
                         y = US['Confirmed'][1:],
                         name='Confirmed',
                         marker_color='white'
                         ))
fig.add_trace(go.Scatter(x = US['ObservationDate'][1:],
                         y = US['Deaths'][1:],
                         name='Deaths',
                         marker_color='red'
                         ))
fig.add_trace(go.Scatter(x = US['ObservationDate'][1:],
                         y = US['Recovered'][1:],
                         name='Recovered',
                         marker_color='green'
                         ))
fig.add_trace(go.Scatter(x = US['ObservationDate'][1:],
                         y = US['Active'][1:],
                         name='Active',
                         marker_color='blue'
                         ))

fig.update_layout(title='US Corona Virus - Confirmed, Deaths, Recovered and Active Cases (Logarithmic Scale)',
                  xaxis = dict(title = 'Observation Dates'),
                  yaxis = dict(title = 'Number of Cases'),
                  yaxis_type = "log"
                  )
fig.show()


# ### New Cases for Confirmed, Deaths and Recovered Cases (US

# #### Daily New Confirmed, Death and Recovered Cases - US

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(x = US['ObservationDate'][1:],
                     y = US['Conf_change'][1:],
                     marker_color='white',                  
                     name = 'New Confirmed Cases'
                     ))

fig.add_trace(go.Scatter(x = US['ObservationDate'][1:],
                     y = US['Death_change'][1:],
                     marker_color='red',                  
                     name = 'New Death Cases'
                     ))

fig.add_trace(go.Scatter(x = US['ObservationDate'][1:],
                     y = US['Recov_change'][1:],
                     marker_color='Green',                  
                     name = 'New Recovered Cases'
                     ))
fig.update_layout(title='Daily New Confirmed, Death and Recovered Cases - US',
                  xaxis_title="Observation Date",
                  yaxis_title="New Cases"
                  )
fig.show()


# #### Weekly New Confirmed, Death and Recovered Cases - US
# 

# In[ ]:



fig = go.Figure()

fig.add_trace(go.Scatter(x = US_week_df['ObservationDate'],
                     y = US_week_df['Conf_change'],
                     marker_color='white',                  
                     name = 'New Confirmed Cases'
                     ))

fig.add_trace(go.Scatter(x = US_week_df['ObservationDate'],
                     y = US_week_df['Death_change'],
                     marker_color='red',                  
                     name = 'New Death Cases'
                     ))

fig.add_trace(go.Scatter(x = US_week_df['ObservationDate'],
                     y = US_week_df['Recov_change'],
                     marker_color='Green',                  
                     name = 'New Recovered Cases'
                     ))
fig.update_layout(title='Weekly New Confirmed, Death and Recovered Cases - US',
                  xaxis_title="Observation Date",
                  yaxis_title="New Cases"
                  )
fig.show()


# #### Confirmed cases - Top 15 States in USA

# In[ ]:


states = states.sort_values(by = 'cases',ascending=0)

fig = go.Figure()

fig.add_trace(go.Bar(x = states['state'][:15],
                     y = states['cases'][:15],
                     text = states['cases'][:15],
                     textposition = "outside",
                     name = 'Confirmed',
                     marker_color = '#00a1c1'
                     ))

fig.update_layout(title='Confirmed - Top 15 States in USA',
                  xaxis_title="States",
                  yaxis_title="Confirmed Cases"
                  )
fig.show()


# #### Deaths cases - Top 15 States in USA

# In[ ]:


states = states.sort_values(by = 'deaths',ascending=0)

fig = go.Figure()

fig.add_trace(go.Bar(x = states['state'][:15],
                     y = states['deaths'][:15],
                     text = states['deaths'][:15],
                     textposition = "outside",
                     name = 'Deaths',
                     marker_color = '#00a1c1'
                     ))

fig.update_layout(title='Death cases - Top 15 States in USA',
                  xaxis_title="States",
                  yaxis_title="Deaths Cases"
                  )
fig.show()


# #### Mortality Rate - Top 15 States in USA

# In[ ]:


states = states.sort_values(by = 'mortality_rate',ascending=0)

fig = go.Figure()

fig.add_trace(go.Bar(x = states['state'][:15],
                     y = states['mortality_rate'][:15],
                     text = states['mortality_rate'][:15],
                     textposition = "outside",
                     name = 'Deaths',
                     marker_color = '#00a1c1'
                     ))

fig.update_layout(title='Mortality Rate - Top 15 States in USA',
                  xaxis_title="States",
                  yaxis_title="Mortality Rate"
                  )
fig.show()


# #### Covid-19 Mortality Rate (USA)

# In[ ]:


US['mortality_rate'] = np.round(100*US['Deaths']/US['Confirmed'],2)

fig = go.Figure()

fig.add_trace(go.Scatter(x = US['ObservationDate'][1:],
                         y = US['mortality_rate'][1:],
                         name = 'USA Mortality Rate',
                         marker_color = '#00a1c1'))

fig.update_layout(title = 'Covid-19 Mortality Rate (USA)')

fig.show()


# # Italy Data Analysis and Visualization

# ## Data Analysis

# In[ ]:


Italy = f[f['Country/Region'] == 'Italy'].reset_index()
Italy = Italy.groupby('ObservationDate')[['SNo','Confirmed', 'Deaths','Recovered','Active']].sum().reset_index()
Italy[['Confirmed', 'Deaths','Recovered','Active']] = Italy[['Confirmed', 'Deaths','Recovered','Active']].astype(float)
Italy = Italy.sort_values(by = 'ObservationDate', ascending = 0)

# Add 'Conf_change', 'Death_change', 'Recov_change', 'Active_change' to world dataframe
Italy['Conf_change'] = Italy.apply(lambda _: '', axis=1)
Italy['Death_change'] = Italy.apply(lambda _: '', axis=1)
Italy['Recov_change'] = Italy.apply(lambda _: '', axis=1)
Italy['Active_change'] = Italy.apply(lambda _: '', axis=1)
Italy = Italy.append(pd.Series(name='Start'))
Italy = Italy.fillna(0)
Italy['Conf_change'] = Italy['Confirmed'].diff(periods=-1)
Italy['Death_change'] = Italy['Deaths'].diff(periods=-1)
Italy['Recov_change'] = Italy['Recovered'].diff(periods=-1)
Italy['Active_change'] = Italy['Active'].diff(periods=-1)

# Add 'grow_fact_conf', 'grow_fact_death', 'grow_fact_recov', 'grow_fact_active' to world dataframe
Italy['grow_fact_conf'] = Italy.apply(lambda _: '', axis=1)
Italy['grow_fact_death'] = Italy.apply(lambda _: '', axis=1)
Italy['grow_fact_recov'] = Italy.apply(lambda _: '', axis=1)
Italy['grow_fact_active'] = Italy.apply(lambda _: '', axis=1)
Italy['grow_fact_conf'] = Italy['Conf_change'].pct_change(periods = -1) + 1
Italy['grow_fact_death'] = Italy['Death_change'].pct_change(periods = -1) + 1 
Italy['grow_fact_recov'] = Italy['Recov_change'].pct_change(periods = -1) + 1 
Italy['grow_fact_active'] = Italy['Active_change'].pct_change(periods = -1) + 1
Italy = Italy.replace([np.inf])
Italy = Italy.replace([-np.inf])
Italy = Italy.fillna(0)
Italy = Italy.sort_values('SNo',ascending = 1)
Italy_temp = Italy.sort_values('SNo',ascending = 1)
# weekly dataframe
Italy_week_df = Italy_temp[1:]
Italy_week_df = Italy_week_df[::7]
Italy_week_df = Italy_week_df.sort_values(by = 'SNo', ascending = 0)
Italy_week_df
# Add 'Conf_change', 'Death_change', 'Recov_change', 'Active_change' to world dataframe
Italy_week_df['Conf_change'] = Italy_week_df.apply(lambda _: '', axis=1)
Italy_week_df['Death_change'] = Italy_week_df.apply(lambda _: '', axis=1)
Italy_week_df['Recov_change'] = Italy_week_df.apply(lambda _: '', axis=1)
Italy_week_df['Active_change'] = Italy_week_df.apply(lambda _: '', axis=1)
Italy_week_df = Italy_week_df.append(pd.Series(name='Start'))
Italy_week_df = Italy_week_df.fillna(0)
Italy_week_df['Conf_change'] = Italy_week_df['Confirmed'].diff(periods=-1)
Italy_week_df['Death_change'] = Italy_week_df['Deaths'].diff(periods=-1)
Italy_week_df['Recov_change'] = Italy_week_df['Recovered'].diff(periods=-1)
Italy_week_df['Active_change'] = Italy_week_df['Active'].diff(periods=-1)

# Add 'grow_fact_conf', 'grow_fact_death', 'grow_fact_recov', 'grow_fact_active' to world dataframe
Italy_week_df['grow_fact_conf'] = Italy_week_df.apply(lambda _: '', axis=1)
Italy_week_df['grow_fact_death'] = Italy_week_df.apply(lambda _: '', axis=1)
Italy_week_df['grow_fact_recov'] = Italy_week_df.apply(lambda _: '', axis=1)
Italy_week_df['grow_fact_active'] = Italy_week_df.apply(lambda _: '', axis=1)
Italy_week_df['grow_fact_conf'] = Italy_week_df['Conf_change'].pct_change(periods = -1) + 1
Italy_week_df['grow_fact_death'] = Italy_week_df['Death_change'].pct_change(periods = -1) + 1 
Italy_week_df['grow_fact_recov'] = Italy_week_df['Recov_change'].pct_change(periods = -1) + 1 
Italy_week_df['grow_fact_active'] = Italy_week_df['Active_change'].pct_change(periods = -1) + 1 
Italy_week_df = Italy_week_df.sort_values(by = 'SNo', ascending = 1)
Italy_week_df = Italy_week_df[1:]


# ## Data Visualization

# ### Italy Corona Virus - Confirmed, Deaths, Recovered and Active Cases

# #### Italy Corona Virus - Confirmed, Deaths, Recovered and Active Cases (Linear Scale)

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x = Italy['ObservationDate'][1:],
                         y = Italy['Confirmed'][1:],
                         name='Confirmed',
                         marker_color='white'
                         ))
fig.add_trace(go.Scatter(x = Italy['ObservationDate'][1:],
                         y = Italy['Deaths'][1:],
                         name='Deaths',
                         marker_color='red'
                         ))
fig.add_trace(go.Scatter(x = Italy['ObservationDate'][1:],
                         y = Italy['Recovered'][1:],
                         name='Recovered',
                         marker_color='green'
                         ))
fig.add_trace(go.Scatter(x = Italy['ObservationDate'][1:],
                         y = Italy['Active'][1:],
                         name='Active',
                         marker_color='blue'
                         ))

fig.update_layout(title='Italy Corona Virus - Confirmed, Deaths, Recovered and Active Cases (Linear Scale)',
                  xaxis = dict(title = 'Observation Dates'),
                  yaxis = dict(title = 'Number of Cases')
                  )
fig.show()


# #### Italy Corona Virus - Confirmed, Deaths, Recovered and Active Cases (Logarithmic Scale)

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x = Italy['ObservationDate'][1:],
                         y = Italy['Confirmed'][1:],
                         name='Confirmed',
                         marker_color='white'
                         ))
fig.add_trace(go.Scatter(x = Italy['ObservationDate'][1:],
                         y = Italy['Deaths'][1:],
                         name='Deaths',
                         marker_color='red'
                         ))
fig.add_trace(go.Scatter(x = Italy['ObservationDate'][1:],
                         y = Italy['Recovered'][1:],
                         name='Recovered',
                         marker_color='green'
                         ))
fig.add_trace(go.Scatter(x = Italy['ObservationDate'][1:],
                         y = Italy['Active'][1:],
                         name='Active',
                         marker_color='blue'
                         ))

fig.update_layout(title='Italy Corona Virus - Confirmed, Deaths, Recovered and Active Cases (Logarithmic Scale)',
                  xaxis = dict(title = 'Observation Dates'),
                  yaxis = dict(title = 'Number of Cases'),
                  yaxis_type = "log"
                  )
fig.show()


# ### New Cases for Confirmed, Deaths and Recovered Cases (Italy)

# #### Daily New Confirmed, Death and Recovered Cases - Italy

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(x = Italy['ObservationDate'][1:],
                     y = Italy['Conf_change'][1:],
                     marker_color='white',                  
                     name = 'New Confirmed Cases'
                     ))

fig.add_trace(go.Scatter(x = Italy['ObservationDate'][1:],
                     y = Italy['Death_change'][1:],
                     marker_color='red',                  
                     name = 'New Death Cases'
                     ))

fig.add_trace(go.Scatter(x = Italy['ObservationDate'][1:],
                     y = Italy['Recov_change'][1:],
                     marker_color='Green',                  
                     name = 'New Recovered Cases'
                     ))
fig.update_layout(title='Daily New Confirmed, Death and Recovered Cases - Italy',
                  xaxis_title="Observation Date",
                  yaxis_title="New Cases"
                  )
fig.show()


# #### Weekly New Confirmed, Death and Recovered Cases - Italy

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(x = Italy_week_df['ObservationDate'],
                     y = Italy_week_df['Conf_change'],
                     marker_color='white',                  
                     name = 'New Confirmed Cases'
                     ))

fig.add_trace(go.Scatter(x = Italy_week_df['ObservationDate'],
                     y = Italy_week_df['Death_change'],
                     marker_color='red',                  
                     name = 'New Death Cases'
                     ))

fig.add_trace(go.Scatter(x = Italy_week_df['ObservationDate'],
                     y = Italy_week_df['Recov_change'],
                     marker_color='Green',                  
                     name = 'New Recovered Cases'
                     ))
fig.update_layout(title='Weekly New Confirmed, Death and Recovered Cases - Italy',
                  xaxis_title="Observation Date",
                  yaxis_title="New Cases"
                  )
fig.show()


# #### Covid-19 Mortality Rate (Italy)

# In[ ]:


Italy['mortality_rate'] = np.round(100*Italy['Deaths']/Italy['Confirmed'],2)

fig = go.Figure()

fig.add_trace(go.Scatter(x = Italy['ObservationDate'][1:],
                         y = Italy['mortality_rate'][1:],
                         name = 'Italy Mortality Rate',
                         marker_color = '#00a1c1'))

fig.update_layout(title = 'Covid-19 Mortality Rate (Italy)')

fig.show()


# # India Data Analysis and Visualization

# ## Data Analysis

# In[ ]:


India = f[f['Country/Region'] == 'India'].reset_index()
India = India.groupby('ObservationDate')[['SNo','Confirmed', 'Deaths','Recovered','Active']].sum().reset_index()
India[['Confirmed', 'Deaths','Recovered','Active']] = India[['Confirmed', 'Deaths','Recovered','Active']].astype(float)
India = India.sort_values(by = 'ObservationDate', ascending = 0)

# Add 'Conf_change', 'Death_change', 'Recov_change', 'Active_change' to world dataframe
India['Conf_change'] = India.apply(lambda _: '', axis=1)
India['Death_change'] = India.apply(lambda _: '', axis=1)
India['Recov_change'] = India.apply(lambda _: '', axis=1)
India['Active_change'] = India.apply(lambda _: '', axis=1)
India = India.append(pd.Series(name='Start'))
India = India.fillna(0)
India['Conf_change'] = India['Confirmed'].diff(periods=-1)
India['Death_change'] = India['Deaths'].diff(periods=-1)
India['Recov_change'] = India['Recovered'].diff(periods=-1)
India['Active_change'] = India['Active'].diff(periods=-1)

# Add 'grow_fact_conf', 'grow_fact_death', 'grow_fact_recov', 'grow_fact_active' to world dataframe
India['grow_fact_conf'] = India.apply(lambda _: '', axis=1)
India['grow_fact_death'] = India.apply(lambda _: '', axis=1)
India['grow_fact_recov'] = India.apply(lambda _: '', axis=1)
India['grow_fact_active'] = India.apply(lambda _: '', axis=1)
India['grow_fact_conf'] = India['Conf_change'].pct_change(periods = -1) + 1
India['grow_fact_death'] = India['Death_change'].pct_change(periods = -1) + 1 
India['grow_fact_recov'] = India['Recov_change'].pct_change(periods = -1) + 1 
India['grow_fact_active'] = India['Active_change'].pct_change(periods = -1) + 1
India = India.replace([np.inf])
India = India.replace([-np.inf])
India = India.fillna(0)
India = India.sort_values('SNo',ascending = 1)
India_temp = India.sort_values('SNo',ascending = 1)
# weekly dataframe
India_week_df = India_temp[1:]
India_week_df = India_week_df[::7]
India_week_df = India_week_df.sort_values(by = 'SNo', ascending = 0)
India_week_df
# Add 'Conf_change', 'Death_change', 'Recov_change', 'Active_change' to world dataframe
India_week_df['Conf_change'] = India_week_df.apply(lambda _: '', axis=1)
India_week_df['Death_change'] = India_week_df.apply(lambda _: '', axis=1)
India_week_df['Recov_change'] = India_week_df.apply(lambda _: '', axis=1)
India_week_df['Active_change'] = India_week_df.apply(lambda _: '', axis=1)
India_week_df = India_week_df.append(pd.Series(name='Start'))
India_week_df = India_week_df.fillna(0)
India_week_df['Conf_change'] = India_week_df['Confirmed'].diff(periods=-1)
India_week_df['Death_change'] = India_week_df['Deaths'].diff(periods=-1)
India_week_df['Recov_change'] = India_week_df['Recovered'].diff(periods=-1)
India_week_df['Active_change'] = India_week_df['Active'].diff(periods=-1)

# Add 'grow_fact_conf', 'grow_fact_death', 'grow_fact_recov', 'grow_fact_active' to world dataframe
India_week_df['grow_fact_conf'] = India_week_df.apply(lambda _: '', axis=1)
India_week_df['grow_fact_death'] = India_week_df.apply(lambda _: '', axis=1)
India_week_df['grow_fact_recov'] = India_week_df.apply(lambda _: '', axis=1)
India_week_df['grow_fact_active'] = India_week_df.apply(lambda _: '', axis=1)
India_week_df['grow_fact_conf'] = India_week_df['Conf_change'].pct_change(periods = -1) + 1
India_week_df['grow_fact_death'] = India_week_df['Death_change'].pct_change(periods = -1) + 1 
India_week_df['grow_fact_recov'] = India_week_df['Recov_change'].pct_change(periods = -1) + 1 
India_week_df['grow_fact_active'] = India_week_df['Active_change'].pct_change(periods = -1) + 1 
India_week_df = India_week_df.sort_values(by = 'SNo', ascending = 1)
India_week_df = India_week_df[1:]


# ## Data Visualization

# ### India Corona Virus - Confirmed, Deaths, Recovered and Active Cases
# 

# #### India Corona Virus - Confirmed, Deaths, Recovered and Active Cases (Linear Scale)

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x = India['ObservationDate'][1:],
                         y = India['Confirmed'][1:],
                         name='Confirmed',
                         marker_color='white'
                         ))
fig.add_trace(go.Scatter(x = India['ObservationDate'][1:],
                         y = India['Deaths'][1:],
                         name='Deaths',
                         marker_color='red'
                         ))
fig.add_trace(go.Scatter(x = India['ObservationDate'][1:],
                         y = India['Recovered'][1:],
                         name='Recovered',
                         marker_color='green'
                         ))
fig.add_trace(go.Scatter(x = India['ObservationDate'][1:],
                         y = India['Active'][1:],
                         name='Active',
                         marker_color='blue'
                         ))

fig.update_layout(title='India Corona Virus - Confirmed, Deaths, Recovered and Active Cases (Linear Scale)',
                  xaxis = dict(title = 'Observation Dates'),
                  yaxis = dict(title = 'Number of Cases')
                  )
fig.show()


# #### India Corona Virus - Confirmed, Deaths, Recovered and Active Cases (Logarithmic Scale)

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x = India['ObservationDate'][1:],
                         y = India['Confirmed'][1:],
                         name='Confirmed',
                         marker_color='white'
                         ))
fig.add_trace(go.Scatter(x = India['ObservationDate'][1:],
                         y = India['Deaths'][1:],
                         name='Deaths',
                         marker_color='red'
                         ))
fig.add_trace(go.Scatter(x = India['ObservationDate'][1:],
                         y = India['Recovered'][1:],
                         name='Recovered',
                         marker_color='green'
                         ))
fig.add_trace(go.Scatter(x = India['ObservationDate'][1:],
                         y = India['Active'][1:],
                         name='Active',
                         marker_color='blue'
                         ))

fig.update_layout(title='India Corona Virus - Confirmed, Deaths, Recovered and Active Cases (Logarithmic Scale)',
                  xaxis = dict(title = 'Observation Dates'),
                  yaxis = dict(title = 'Number of Cases'),
                  yaxis_type = "log"
                  )
fig.show()


# ## New Cases for Confirmed, Deaths and Recovered Cases (India)

# #### Daily New Confirmed, Death and Recovered Cases - India

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(x = India['ObservationDate'][1:],
                     y = India['Conf_change'][1:],
                     marker_color='white',                  
                     name = 'New Confirmed Cases'
                     ))

fig.add_trace(go.Scatter(x = India['ObservationDate'][1:],
                     y = India['Death_change'][1:],
                     marker_color='red',                  
                     name = 'New Death Cases'
                     ))

fig.add_trace(go.Scatter(x = India['ObservationDate'][1:],
                     y = India['Recov_change'][1:],
                     marker_color='Green',                  
                     name = 'New Recovered Cases'
                     ))
fig.update_layout(title='Daily New Confirmed, Death and Recovered Cases - India',
                  xaxis_title="Observation Date",
                  yaxis_title="New Cases"
                  )
fig.show()


# #### Weekly New Confirmed, Death and Recovered Cases - India

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Scatter(x = India_week_df['ObservationDate'],
                     y = India_week_df['Conf_change'],
                     marker_color='white',                  
                     name = 'New Confirmed Cases'
                     ))

fig.add_trace(go.Scatter(x = India_week_df['ObservationDate'],
                     y = India_week_df['Death_change'],
                     marker_color='red',                  
                     name = 'New Death Cases'
                     ))

fig.add_trace(go.Scatter(x = India_week_df['ObservationDate'],
                     y = India_week_df['Recov_change'],
                     marker_color='Green',                  
                     name = 'New Recovered Cases'
                     ))
fig.update_layout(title='Weekly New Confirmed, Death and Recovered Cases - India',
                  xaxis_title="Observation Date",
                  yaxis_title="New Cases"
                  )
fig.show()


# #### Covid-19 Mortality Rate (India)

# In[ ]:


India['mortality_rate'] = np.round(100*India['Deaths']/India['Confirmed'],2)

fig = go.Figure()

fig.add_trace(go.Scatter(x = India['ObservationDate'][1:],
                         y = India['mortality_rate'][1:],
                         name = 'Italy Mortality Rate',
                         marker_color = '#00a1c1'))

fig.update_layout(title = 'Covid-19 Mortality Rate (India)')

fig.show()


# # Confirmed, Deaths, Recovered and Active Cases Comparison

# ### Comparison between Confirmed Cases (China vs Italy vs United States vs India vs Spain vs Germany)

# In[ ]:


# china
china = f[f['Country/Region'] == 'Mainland China'].reset_index()
china_obs_date = china.groupby('ObservationDate')[['Confirmed', 'Deaths','Recovered','Active']].sum().reset_index()
china_obs_date.sort_values(by = 'ObservationDate', ascending = 0)

# Italy
Italy = f[f['Country/Region'] == 'Italy'].reset_index()
Italy_obs_date = Italy.groupby('ObservationDate')[['Confirmed', 'Deaths','Recovered','Active']].sum().reset_index()
Italy_obs_date.sort_values(by = 'ObservationDate', ascending = 0)

# US
US = f[f['Country/Region'] == 'US'].reset_index()
US_obs_date = US.groupby('ObservationDate')[['Confirmed', 'Deaths','Recovered','Active']].sum().reset_index()
US_obs_date.sort_values(by = 'ObservationDate', ascending = 0)

# India
India = f[f['Country/Region'] == 'India'].reset_index()
India_obs_date = India.groupby('ObservationDate')[['Confirmed', 'Deaths','Recovered','Active']].sum().reset_index()
India_obs_date.sort_values(by = 'ObservationDate', ascending = 0)

# Spain
Spain = f[f['Country/Region'] == 'Spain'].reset_index()
Spain_obs_date = Spain.groupby('ObservationDate')[['Confirmed', 'Deaths','Recovered','Active']].sum().reset_index()
Spain_obs_date.sort_values(by = 'ObservationDate', ascending = 0)

# Germany
Germany = f[f['Country/Region'] == 'Germany'].reset_index()
Germany_obs_date = Germany.groupby('ObservationDate')[['Confirmed', 'Deaths','Recovered','Active']].sum().reset_index()
Germany_obs_date.sort_values(by = 'ObservationDate', ascending = 0)

fig = go.Figure()
fig.add_trace(go.Scatter(x = china_obs_date['ObservationDate'],
                         y = china_obs_date['Confirmed'],
                         name='China',
                         marker_color='red'
                         ))
fig.add_trace(go.Scatter(x = Italy_obs_date['ObservationDate'],
                         y = Italy_obs_date['Confirmed'],
                         name='Italy',
                         marker_color='cyan'
                         ))
fig.add_trace(go.Scatter(x = US_obs_date['ObservationDate'],
                         y = US_obs_date['Confirmed'],
                         name='US',
                         marker_color='magenta'
                         ))
fig.add_trace(go.Scatter(x = India_obs_date['ObservationDate'],
                         y = India_obs_date['Confirmed'],
                         name='India',
                         marker_color='Green'
                         ))
fig.add_trace(go.Scatter(x = Spain_obs_date['ObservationDate'],
                         y = Spain_obs_date['Confirmed'],
                         name='Spain',
                         marker_color='white'
                         ))
fig.add_trace(go.Scatter(x = Germany_obs_date['ObservationDate'],
                         y = Germany_obs_date['Confirmed'],
                         name='Germany',
                         marker_color='Yellow'
                         ))
fig.update_layout(title='Comparison between Confirmed Cases (China vs Italy vs United States vs India vs Spain vs Germany)',
                  xaxis = dict(title = 'Observation Dates'),
                  yaxis = dict(title = 'Number of Cases')
                  )
fig.show()


# #### Comparison between Death Cases (China vs Italy vs United States vs India vs Spain vs Germany)

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x = china_obs_date['ObservationDate'],
                         y = china_obs_date['Deaths'],
                         name='China',
                         marker_color='red'
                         ))
fig.add_trace(go.Scatter(x = Italy_obs_date['ObservationDate'],
                         y = Italy_obs_date['Deaths'],
                         name='Italy',
                         marker_color='cyan'
                         ))
fig.add_trace(go.Scatter(x = US_obs_date['ObservationDate'],
                         y = US_obs_date['Deaths'],
                         name='US',
                         marker_color='magenta'
                         ))
fig.add_trace(go.Scatter(x = India_obs_date['ObservationDate'],
                         y = India_obs_date['Deaths'],
                         name='India',
                         marker_color='green'
                         ))
fig.add_trace(go.Scatter(x = Spain_obs_date['ObservationDate'],
                         y = Spain_obs_date['Deaths'],
                         name='Spain',
                         marker_color='white'
                         ))
fig.add_trace(go.Scatter(x = Germany_obs_date['ObservationDate'],
                         y = Germany_obs_date['Deaths'],
                         name='Germany',
                         marker_color='yellow'
                         ))
fig.update_layout(title='Comparison between Death Cases (China vs Italy vs United States vs India vs Spain vs Germany)',
                  xaxis = dict(title = 'Observation Dates'),
                  yaxis = dict(title = 'Number of Cases')
                  )
fig.show()


# #### Comparison between Recovered Cases (China vs Italy vs United States vs India vs Spain vs Germany)

# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x = china_obs_date['ObservationDate'],
                         y = china_obs_date['Recovered'],
                         name='China',
                         marker_color='red'
                         ))
fig.add_trace(go.Scatter(x = Italy_obs_date['ObservationDate'],
                         y = Italy_obs_date['Recovered'],
                         name='Italy',
                         marker_color='cyan'
                         ))
fig.add_trace(go.Scatter(x = US_obs_date['ObservationDate'],
                         y = US_obs_date['Recovered'],
                         name='US',
                         marker_color='magenta'
                         ))
fig.add_trace(go.Scatter(x = India_obs_date['ObservationDate'],
                         y = India_obs_date['Recovered'],
                         name='India',
                         marker_color='green'
                         ))
fig.add_trace(go.Scatter(x = Spain_obs_date['ObservationDate'],
                         y = Spain_obs_date['Recovered'],
                         name='Spain',
                         marker_color='white'
                         ))
fig.add_trace(go.Scatter(x = Germany_obs_date['ObservationDate'],
                         y = Germany_obs_date['Recovered'],
                         name='Germany',
                         marker_color='yellow'
                         ))
fig.update_layout(title='Comparison between Recovered Cases (China vs Italy vs United States vs India vs Spain vs Germany)',
                  xaxis = dict(title = 'Observation Dates'),
                  yaxis = dict(title = 'Number of Cases')
                  )
fig.show()


# #### Coronavirus Confirmed cases - Top 5 Countries vs India

# In[ ]:


fig = go.Figure()

fig.add_trace(go.Bar(x = country['Country/Region'][0:5],
                     y = country['Confirmed'][0:5],
                     marker_color='white',
                     text = country['Confirmed'][0:5],
                     textposition = 'outside',
                     name = 'Confirmed'
                     ))

fig.add_trace(go.Bar(x = country['Country/Region'][country[country['Country/Region'] == 'India'].index],
                     y = country['Confirmed'][country[country['Country/Region'] == 'India'].index],
                     marker_color='white',
                     text = country['Confirmed'][country[country['Country/Region'] == 'India'].index],
                     textposition = 'outside',
                     showlegend=False
                     ))

fig.update_layout(title='Confirmed cases - Top 5 Countries vs India',
                  xaxis_title="Countries",
                  yaxis_title="Confirmed Cases",
                 )
fig.show()


# #### Coronavirus Death cases - Top 5 Countries vs India

# In[ ]:


country_death = df.groupby(["Country/Region"])[['Confirmed', 'Deaths','Recovered','Active']].sum().reset_index().sort_values("Deaths",ascending=False).reset_index(drop=True)

fig = go.Figure()

fig.add_trace(go.Bar(x = country_death['Country/Region'][0:5],
                     y = country_death['Deaths'][0:5],
                     marker_color='red',
                     text = country_death['Deaths'][0:5],
                     textposition = 'outside',
                     name = 'Deaths'
                     ))

fig.add_trace(go.Bar(x = country_death['Country/Region'][country_death[country_death['Country/Region'] == 'India'].index],
                     y = country_death['Deaths'][country_death[country_death['Country/Region'] == 'India'].index],
                     marker_color='red',
                     text = country_death['Deaths'][country_death[country_death['Country/Region'] == 'India'].index],
                     textposition = 'outside',
                     showlegend=False
                     ))

fig.update_layout(title='Death cases - Top 5 Countries vs India',
                  xaxis_title="Countries",
                  yaxis_title="Deaths Cases",
                 )
fig.show()


# #### Coronavirus Recovered cases - Top 5 Countries vs India

# In[ ]:


country_recovered = df.groupby(["Country/Region"])[['Confirmed', 'Deaths','Recovered','Active']].sum().reset_index().sort_values('Recovered',ascending=False).reset_index(drop=True)

fig = go.Figure()

fig.add_trace(go.Bar(x = country_recovered['Country/Region'][0:5],
                     y = country_recovered['Recovered'][0:5],
                     marker_color='green',
                     text = country_recovered['Recovered'][0:5],
                     textposition = 'outside',
                     name = 'Recovered'
                     ))

fig.add_trace(go.Bar(x = country_recovered['Country/Region'][country_recovered[country_recovered['Country/Region'] == 'India'].index],
                     y = country_recovered['Recovered'][country_recovered[country_recovered['Country/Region'] == 'India'].index],
                     marker_color='green',
                     text = country_recovered['Recovered'][country_recovered[country_recovered['Country/Region'] == 'India'].index],
                     textposition = 'outside',
                     showlegend=False
                     ))

fig.update_layout(title='Recovered cases - Top 5 Countries vs India',
                  xaxis_title="Countries",
                  yaxis_title="Recovered Cases"
                  )
fig.show()


# #### Coronavirus Active cases - Top 5 Countries vs India

# In[ ]:


country_active = df.groupby(["Country/Region"])[['Confirmed', 'Deaths','Recovered','Active']].sum().reset_index().sort_values('Active',ascending=False).reset_index(drop=True)

fig = go.Figure()

fig.add_trace(go.Bar(x = country_active['Country/Region'][0:5],
                     y = country_active['Active'][0:5],
                     marker_color='blue',
                     text = country_active['Active'][0:5],
                     textposition = 'outside',
                     name = 'Active'
                     ))

fig.add_trace(go.Bar(x = country_active['Country/Region'][country_active[country_active['Country/Region'] == 'India'].index],
                     y = country_active['Active'][country_active[country_active['Country/Region'] == 'India'].index],
                     marker_color='blue',
                     text = country_active['Active'][country_active[country_active['Country/Region'] == 'India'].index],
                     textposition = 'outside',
                     showlegend=False
                     ))

fig.update_layout(title='Active cases - Top 5 Countries vs India',
                  xaxis_title="Countries",
                  yaxis_title="Active Cases"
                  )
fig.show()


# # Number of days to cross '10,000' confirmed cases for the top 10 countries with the most reported cases

# In[ ]:


w = f.groupby(['ObservationDate','Country/Region'])[['SNo','Confirmed', 'Deaths','Recovered','Active']].sum().reset_index()
lst = country['Country/Region'][:10]
num_df = pd.DataFrame()

for i in lst:
    fd = w[w['Country/Region'] == i].sort_values(by = 'SNo',ascending  = 1)
    days = len(fd[fd['Confirmed'] < 10000]) + 1
    data = {'Country':[i],'Number of Days':[days]}
    num_df = num_df.append(pd.DataFrame(data),ignore_index = True)
    
num_df = num_df.sort_values(by = 'Number of Days', ascending = 1)

fig = go.Figure()

fig.add_trace(go.Bar(x=num_df['Country'], y=num_df['Number of Days'],text=num_df['Number of Days'],
                       textposition='outside', marker_color = '#00a1c1'
                    )
             )

fig.update_layout(title='Number of Days to cross 10,000 Confirmed Cases',
                  xaxis_title="Country",
                  yaxis_title="Number of Days"
                 )

fig.show()


# ## Thank you and Be Strong, Be Safe! Now, because things will get better. It might be stormy now but it can't rain forever.
