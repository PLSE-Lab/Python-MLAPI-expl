#!/usr/bin/env python
# coding: utf-8

# # This is exercise 6
# 
# In this exercise, I chose Mexico to analyze the COVID-19 data sets. The reason I chose this country is because I barely see it on the news or other SNS platforms. Yet they are in top 20 countries with the most cases of novel corona virus.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='Mexico'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)


# Note: This 'df' is actually a cumulative information. 
# 
# June 12, 2020 shows the total cases since initial date.
# 
# Below code is the day-to-day increase in cases.

# In[ ]:


df['daily_confirmed'] = df['Confirmed'].diff()
df['daily_deaths'] = df['Deaths'].diff()
df['daily_recovery'] = df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()
df['daily_deaths'].plot()
plt.show()


# Below I made an interactive chart

# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')
daily_recovered_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily recovered')

layout_object = go.Layout(title='Mexico daily cases 20M52067',xaxis=dict(title='Date'),yaxis=dict(title='Number of People'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object],layout=layout_object)
iplot(fig)
fig.write_html('Mexico_daily_case_20M52067.html')


# # How can we make an informative table
# 
# Large number values as bright color
# Low number values as some dark color

# In[ ]:


df1 = df#[['daily_confirmed']]
df1 = df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('table_20M52067.html','w')
f.write(styled_object.render())


# # How can we calculate global ranking?

# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df.index=df['ObservationDate']
df = df.drop(['SNo','ObservationDate'],axis=1)
df.head()
df_Mexico = df[df['Country/Region']=='Mexico']
df_Mexico.tail()
latest = df[df.index=='06/12/2020']
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 

print('Ranking of Mexico: ', latest[latest['Country/Region']=='Mexico'].index.values[0]+1)


# # Discussion
# # How Mexico is fighting with Corona Virus
# 
# First confirmed case; 28th of February
# Number of cases; 139196
# Rank of the country; 14th 
# 
# peak days; 06/02-06/12
# 
# From the graph, we can see that the number of daily cases are increasing and the pandemic in Mexico rapidly increasing. As of June 12th, country is ranked 14th by the number of confirmed cases. 
# 
# Government response; 
# 
# February 29th; First local confirmed case
# March 14th; Closed high schools
# March 18th; Closed higher education institutions 
# March 22nd; Bars, nightclubs and other entertainment places are closed. 
# March 23rd; Limited the number of people who are entering grocery stores (1 person per family) and checking temperature before entering
# March 25th; Announced phase two of pandemic
# March 27th; Bought 5000 ventilators from China
# March 28th; Government urging people to stay home
# March 30th; Declared health emergency
# April 12th; Established National Contigency Center to fight COVID-19. It was lead by scientist and health experts
# April 16th; Suspended transportation between affected and non-affected regions
# April 21st; Entered phase three pandemic
# May 4th; Started using military forces to fight COVID-19.
# May 13th; Announced plan to reopen Mexico's economy. Announced to lift restriction in areas without confirmend cases.
# June 10th; Announced that Mexico will increase the testing numbers.
# June 12th; Announced that non-essential business will be opened from June 14th
# 
# My opinion about Mexico;
# 
# I am shocked how mexican government is acting slow to prevent further infection of COVID-19. The video, urging people to stay home, was only published after 1 month from the first local case. And more shockingly, government is lifting restriction when daily cases are increasing day-by-day. After seeing the data and graph, I think government actions are not enough to contain the COVID-19 infection. In conclusion, I believe lifting the restrictions will make the situation worse. 
# 
# Reference;
# US embassy in Mexico
# https://mx.usembassy.gov/health-alert-mexico-covid-19-update-06-11-2020/
# World Health Organization
# https://www.who.int/emergencies/diseases/novel-coronavirus-2019/events-as-they-happen
