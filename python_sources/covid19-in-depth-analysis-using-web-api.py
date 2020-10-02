#!/usr/bin/env python
# coding: utf-8

# # Covid19 Detailed Analysis in Web API

# **The whole analytics is deployed into an Web API format**
# [Advanced Covid19 Tracker](http://covidtracker-teslacoil.herokuapp.com/)

# The Web App is based on **Covid19 analysis** , developed by us for making more awareness among people and to provide in depth analytics of Covid19 to people. It includes future prediction , prevention and cure methodologies and a symptoms analyzer for common people to ease the understandibility of the Covid19 pandemic.
# The prophet model and the analysis is shown in the below notebook.
# 
# * Country wise In depth analysis of Covid19
# * Prevention and Cure techniques
# * Symptoms Anlysis for social awareness
# * Prediction of Covid19 Spread Countrywise based on prophet Model
# 
# Suggestions would be appreciated.

# **Please if you like our effort upvote this notebook**

# ![Coronavirus](http://cdn.kalingatv.com/wp-content/uploads/2020/02/coronavirus-Youtube.jpg)

# Coronaviruses are a group of related viruses that cause diseases in mammals and birds. In humans, coronaviruses cause respiratory tract infections that can be mild, such as some cases of the common cold (among other possible causes, predominantly rhinoviruses), and others that can be lethal, such as SARS, MERS, and COVID-19. Symptoms in other species vary: in chickens, they cause an upper respiratory tract disease, while in cows and pigs they cause diarrhea. There are yet to be vaccines or antiviral drugs to prevent or treat human coronavirus infections.

#  [Advanced Covid19 Tracker](http://covidtracker-teslacoil.herokuapp.com)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Importing Necessary Libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from fbprophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
from datetime import date, timedelta
import plotly.offline as py


# In[ ]:


#Complete Data
complete_data = pd.read_csv("../input/covid19cleancompletedata/covid_19_clean_complete.csv", parse_dates=['Date'])
complete_data.head()


# Defining the Active Cases and Mainland China is replaced by China

# In[ ]:


# cases 
cases = ['Confirmed', 'Deaths', 'Recovered', 'Active']

# Active Case = (confirmed - deaths - recovered)
complete_data['Active'] = complete_data['Confirmed'] - complete_data['Deaths'] - complete_data['Recovered']

# replacing Mainland china with just China
complete_data['Country/Region'] = complete_data['Country/Region'].replace('Mainland China', 'China')


# In[ ]:


complete_data.head()


# In[ ]:


complete_data.info()


# In[ ]:


sns.heatmap(complete_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# **Data Preprocessing**

# As from the above graph it is visible that there are maximum missing values in the Province/Stste Column and there are no missing values in the other features. 

# In[ ]:


complete_data[['Province/State']] = complete_data[['Province/State']].fillna('')
complete_data.rename(columns={'Date':'date'}, inplace=True)

data = complete_data


# Now let's have a look on the time series,earliest entry and the time span of Covid19 spread.

# In[ ]:


print(f"Earliest Entry: {data['date'].min()}")
print(f"Last Entry:     {data['date'].max()}")
print(f"Total Days:     {data['date'].max() - data['date'].min()}")


# Grouping the data based on Dates to analyze the growth rate of the Coronavirus

# In[ ]:


grouped = data.groupby('date')['date', 'Confirmed', 'Deaths','Active'].sum().reset_index()


# In[ ]:


grouped.head()


# In[ ]:


fig = px.line(grouped, x="date", y="Confirmed", title="Worldwide Confirmed Cases Over Time")
fig.show()


# In[ ]:


fig = px.line(grouped, x="date", y="Deaths", title="Worldwide Death Cases Over Time")
fig.show()


# In[ ]:


data['Province/State'] = data['Province/State'].fillna('')
temp = data[[col for col in data.columns if col != 'Province/State']]

latest = temp[temp['date'] == max(temp['date'])].reset_index()
latest_grouped = latest.groupby('Country/Region')['Confirmed', 'Deaths'].sum().reset_index()
fig = px.bar(latest_grouped.sort_values('Confirmed', ascending=False)[:30][::-1], 
             x='Confirmed', y='Country/Region',
             title='Confirmed Cases Worldwide', text='Confirmed', height=1000, orientation='h')
fig.show()


# In[ ]:


#latest_grouped = latest.groupby('Country/Region')['Confirmed', 'Deaths'].sum().reset_index()
fig = px.bar(latest_grouped.sort_values('Deaths', ascending=False)[:20][::-1], 
             x='Deaths', y='Country/Region',color_discrete_sequence=['#84DCC6'],
             title='Deaths Cases Worldwide', text='Deaths', height=1000, orientation='h')
fig.show()


# In[ ]:


formated_gdf = data.groupby(['date', 'Country/Region'])['Confirmed', 'Deaths'].max()
formated_gdf = formated_gdf.reset_index()
formated_gdf['date'] = pd.to_datetime(formated_gdf['date'])
formated_gdf['date'] = formated_gdf['date'].dt.strftime('%m/%d/%Y')
formated_gdf['size'] = formated_gdf['Confirmed'].pow(0.3)

fig = px.scatter_geo(formated_gdf, locations="Country/Region", locationmode='country names', 
                     color="Confirmed", size='size', hover_name="Country/Region", 
                     range_color= [0, 1500], 
                     projection="natural earth", animation_frame="date", 
                     title='COVID-19: Spread Over Time', color_continuous_scale="portland")
# fig.update(layout_coloraxis_showscale=False)
fig.show()


# Covid19 Spread Analysis on Italy

# In[ ]:


#Cleaned Data
data = pd.read_csv("../input/covid19cleancompletedata/covid_19_clean_complete.csv", parse_dates=['Date'])
data.head()
data_grouped=data.groupby("Country/Region")[['Confirmed', 'Deaths', 'Recovered']].sum().reset_index()
data_grouped.head()


# Facebook Prophet Model

# In[ ]:


country_data = data[data['Country/Region']=='Italy']
idata = country_data.tail(30)
idata.head()


# In[ ]:


prophet=country_data.iloc[: , [4,5 ]].copy() 
prophet.head()
prophet.columns = ['ds','y']
prophet.head()


# In[ ]:


m=Prophet()
m.fit(prophet)
future=m.make_future_dataframe(periods=100)
forecast=m.predict(future)
forecast


# In[ ]:


confirm = forecast.loc[:,['ds','trend']]
confirm = confirm[confirm['trend']>0]
confirm=confirm.tail(40)
confirm.columns = ['Date','Confirm']
confirm.tail()


# Increaing Confirmed Cases Prediction over time in Italy

# In[ ]:


pio.templates.default = "plotly_white"
figure = plot_plotly(m, forecast)
py.iplot(figure) 


# In[ ]:


prophet_dth=country_data.iloc[: , [4,6 ]].copy() 
prophet_dth.head()
prophet_dth.columns = ['ds','y']
prophet_dth.head()


# In[ ]:


m2=Prophet()
m2.fit(prophet_dth)
future_dth=m2.make_future_dataframe(periods=100)
forecast_dth=m2.predict(future_dth)
forecast_dth


# In[ ]:


death = forecast_dth.loc[:,['ds','trend']]
death = death[death['trend']>0]
death=death.tail(15)
death.columns = ['Date','Death']
death.head()


# Increaing Death Cases Prediction over time in Italy

# In[ ]:


figure_death = plot_plotly(m2, forecast_dth)
py.iplot(figure_death)


# **Covid19 Statewise Analysis of India**

# In[ ]:


import numpy as np
import pandas as pd
# visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium


# In[ ]:


# color pallette
cnf = '#393e46' # confirmed - grey
dth = '#ff2e63' # death - red
rec = '#21bf73' # recovered - cyan
act = '#fe9801' # active case - yellow


# In[ ]:


df = pd.read_csv('../input/covid19-india/complete.csv', parse_dates=['Date'])
df['Name of State / UT'] = df['Name of State / UT'].str.replace('Union Territory of ', '')
df.head()


# In[ ]:


df = df[['Date', 'Name of State / UT', 'Latitude', 'Longitude', 'Total Confirmed cases', 'Death', 'Cured/Discharged/Migrated']]
df.columns = ['Date', 'State/UT', 'Latitude', 'Longitude', 'Confirmed', 'Deaths', 'Cured']

for i in ['Confirmed', 'Deaths', 'Cured']:
    df[i] = df[i].astype('int')
    
df['Active'] = df['Confirmed'] - df['Deaths'] - df['Cured']
df['Mortality rate'] = df['Deaths']/df['Confirmed']
df['Recovery rate'] = df['Cured']/df['Confirmed']

df = df[['Date', 'State/UT', 'Latitude', 'Longitude', 'Confirmed', 'Active', 'Deaths', 'Mortality rate', 'Cured', 'Recovery rate']]

df.head()


# In[ ]:


latest = df[df['Date']==max(df['Date'])]

# days
latest_day = max(df['Date'])
day_before = latest_day - timedelta(days = 1)

# state and total cases 
latest_day_df = df[df['Date']==latest_day].set_index('State/UT')
day_before_df = df[df['Date']==day_before].set_index('State/UT')

temp = pd.merge(left = latest_day_df, right = day_before_df, on='State/UT', suffixes=('_lat', '_bfr'), how='outer')
latest_day_df['New cases'] = temp['Confirmed_lat'] - temp['Confirmed_bfr']
latest = latest_day_df.reset_index()
latest.fillna(1, inplace=True)


# In[ ]:


temp = latest[['State/UT', 'Confirmed', 'Active', 'New cases', 'Deaths', 'Mortality rate', 'Cured', 'Recovery rate']]
temp = temp.sort_values('Confirmed', ascending=False).reset_index(drop=True)

temp.style    .background_gradient(cmap="Blues", subset=['Confirmed', 'Active', 'New cases'])    .background_gradient(cmap="Greens", subset=['Cured', 'Recovery rate'])    .background_gradient(cmap="Reds", subset=['Deaths', 'Mortality rate'])


# In[ ]:


import folium


# In[ ]:


india_map = folium.Map(location=[20.5937, 78.9629], tiles='cartodbpositron',
               min_zoom=4, max_zoom=6, zoom_start=4)

for i in range(0, len(latest)):
    if latest.iloc[i]['Confirmed']>0:
        folium.Circle(
            location=[latest.iloc[i]['Latitude'], latest.iloc[i]['Longitude']],
            color='#e84545', 
            fill='#e84545',
            tooltip =   '<li><bold>Name of State / UT : '+str(latest.iloc[i]['State/UT'])+
                        '<li><bold>Confirmed cases  : '+str(latest.iloc[i]['Confirmed'])+
                        '<li><bold>Cured cases  : '+str(latest.iloc[i]['Cured'])+
                        '<li><bold>Death cases  : '+str(latest.iloc[i]['Deaths']),
            radius=int(latest.iloc[i]['Confirmed'])*300).add_to(india_map)


# In[ ]:


india_map


# State Wise Analysis of Covid19 Spread in India

# In[ ]:


temp = latest.sort_values('Confirmed', ascending=False)
import plotly.graph_objects as go
states=temp['State/UT']

fig = go.Figure(data=[
    go.Bar(name='Confirmed', x=states, y=temp['Confirmed']),
    go.Bar(name='Recovered', x=states, y=temp['Cured']),
    go.Bar(name='Deaths', x=states, y=temp['Deaths'])
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# In[ ]:


px.scatter(latest[latest['Confirmed']>10], x='Confirmed', y='Deaths', color='State/UT', size='Confirmed', 
           text='State/UT', log_x =True, title='Confirmed vs Death')

