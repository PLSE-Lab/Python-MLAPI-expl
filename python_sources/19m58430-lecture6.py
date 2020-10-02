#!/usr/bin/env python
# coding: utf-8

# # Importing important libraries

# In[ ]:


#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#for advanced ploting
import seaborn as sns
#for interactive visualizations
import plotly.express as px
import plotly.graph_objs as go


# # 1. Reading the data, choosing Germany as a case country

# In[ ]:


#read data 
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df_germany = (df[df['Country/Region']=='Germany'])
df = df.groupby('ObservationDate').sum()
df.tail(10)


# In[ ]:


df_germany = df_germany.groupby('ObservationDate').sum()
df_germany.tail(10)


# # 2. Constructing daily cases in Germany and creating interactive table

# In[ ]:


df_germany['daily_confirmed'] = df_germany['Confirmed'].diff()
df_germany['daily_deaths'] = df_germany['Deaths'].diff()
df_germany['daily_recovered'] = df_germany['Recovered'].diff()
df_germany['daily_confirmed'].plot(color=['b'], label='daily confirmed')
df_germany['daily_recovered'].plot(color=['g'], label='daily recovered')
df_germany['daily_deaths'].plot(color='r', label='daily deaths')
plt.ylabel('Number of people')
plt.xticks(rotation=45)
plt.title('Coronavirus cases in Germany 19M58430')
plt.legend()


# In[ ]:


from plotly.offline import iplot
import plotly.graph_objs as go


# In[ ]:


df_germany


# In[ ]:


daily_confirmed_object = go.Scatter(x=df.index, y=df_germany['daily_confirmed'].values, name='Daily Confirmed',mode='markers', marker_line_width=1, marker_size=8)
daily_deaths_object = go.Scatter(x=df.index, y=df_germany['daily_deaths'].values, name='Daily Deaths', mode='markers',marker_line_width=1, marker_size=9)
daily_recovered_object = go.Scatter(x=df.index, y=df_germany['daily_recovered'].values, name='Daily Recovered', mode='markers',marker_line_width=1, marker_size=8)
layout_object = go.Layout(title='Germany Daily Cases 19M58430', xaxis=dict(title='Date'), yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object], layout=layout_object)
fig.update_xaxes(tickangle=60,tickfont=dict(family='Arial', color='black',size=12))
iplot(fig)


# # 3.Analyzing the trend & Global Ranking of Germany by confirmed cases 
# 

# In[ ]:


df_germany


# In[ ]:


df_germany['Active'] = df_germany['Confirmed']- df_germany['Deaths'] - df_germany['Recovered']
df_germany['NewConfirmed'] = df_germany['Confirmed'].shift(-1) - df_germany['Confirmed']
df_germany['NewRecovered'] = df_germany['Recovered'] - df_germany['Recovered'].shift(periods=1)
df_germany['NewDeaths'] = df_germany['Deaths'] - df_germany['Deaths'].shift(periods=1)


# In[ ]:



print(df_germany)


# In[ ]:


active = df_germany['Active'].plot()
plt.xticks(rotation=45)
plt.ylabel('Number of people')
plt.title('Active cases in Germany 19M58430')


# In[ ]:


new = df_germany['NewConfirmed'].plot()
plt.xticks(rotation=45)
plt.ylabel('Number of people')
plt.title('New Confirmed Cases 19M58430')


# In[ ]:


newrecov = df_germany['NewRecovered'].plot()
plt.xticks(rotation=45)
plt.ylabel('Number of people')
plt.title('New Recovered Cases 19M58430')


# In[ ]:


newdeaths = df_germany['NewDeaths'].plot()
plt.xticks(rotation=45)
plt.ylabel('Number of people')
plt.title('New Deaths 19M58430')


# ## From the graphs above we see that Active cases, New Confirmed, New Death cases are drastically decreasing. Then, if all the three variables are decreasing there would not be new patients, thus new recovered cases will decrease. 

# In[ ]:


df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df.index=df['ObservationDate']
df = df.drop(['SNo','ObservationDate'],axis=1)
df_Germany = df[df['Country/Region']=='Germany']

latest = df[df.index=='06/16/2020']
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index()

print('Germany Rank: ', latest[latest['Country/Region']=='Germany'].index.values[0]+1)


# # Success story of German government policies:

# * 1. There are many number of hospital intensive care beds in Germany as compared to other countries like Italy. For instance, for 100 thousand people, there are 43 thousand beds in Germany. Versus, in Italy there are 8.5 thousand beds per 100 thousand people
# 
# * 2. When Angela Merkel adressed on national television that 70% population of Germany may become infected with Covid, citizens started to take precautionary approaches. Many hospotals started to test early on extensively. [https://www.youtube.com/watch?v=TC8wxecpcfk](http://) As of April, 1.5 million Germans were tested for Covid 19. [https://www.youtube.com/watch?v=229Fi8fOtho](http://)
# 
# * 3. Germany has good healthcare system, good general practicitioners. They have broad network of laboratories which helped them from the start. First test for Corona in the world was actually developed in Germany. 
# 
# * 4. From the latest news, Germany just launched an app for tracking the coronavirus cases. [https://www.investing.com/news/coronavirus/germany-says-coronavirus-tracing-app-ready-to-go-2201436](http://)

# [](http://https://www.youtube.com/watch?v=TC8wxecpcfk)
