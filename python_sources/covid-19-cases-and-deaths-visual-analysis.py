#!/usr/bin/env python
# coding: utf-8

# # COVID-19 Cases and Deaths Visual Analysis
# 
# This notebook purposes to give some useful information about COVID-19 cases in United States.
# 
# 1. [COVID-19 World Statistics](#5)
# 1. [COVID-19 Cases State by State (US)](#2)
# 1. [COVID-19 California Detailed Statistics](#3)
# 1. [COVID-19 Turkey Statistics](#6)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# <a id="5"> </a> <br/> 
# ## COVID-19 World Statistics

# In[ ]:


df_novel_covid = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df_novel_covid = df_novel_covid[df_novel_covid.ObservationDate == max(df_novel_covid['ObservationDate'])].sort_values(by=['Confirmed'], ascending=False)
df_novel_covid = df_novel_covid.groupby(['Country/Region']).sum()[['Confirmed','Deaths','Recovered']].sort_values(by=['Confirmed'], ascending=False).reset_index()
df_novel_covid['Closed Cases'] = df_novel_covid['Deaths'] + df_novel_covid['Recovered']
df_novel_covid['Death Ratio in Closed Cases(%)'] = (df_novel_covid['Deaths'] * 100) / (df_novel_covid['Deaths'] + df_novel_covid['Recovered'])
df_novel_covid['Death Ratio in All Cases(%)'] = (df_novel_covid['Deaths'] * 100) / df_novel_covid['Confirmed']
df_novel_covid.index = np.arange(1, len(df_novel_covid) + 1)


df_novel_covid.style.background_gradient(cmap='Blues',subset=["Confirmed"])                        .background_gradient(cmap='YlGn',subset=["Death Ratio in Closed Cases(%)"])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Recovered"])                        .background_gradient(cmap='PuBu',subset=["Closed Cases"])                        .background_gradient(cmap='Purples',subset=["Death Ratio in All Cases(%)"])


# In[ ]:


df_novel_covid2 = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')
df_novel_covid2['ActiveCases']= df_novel_covid2['Confirmed'] - (df_novel_covid2['Deaths'] + df_novel_covid2['Recovered'])
df_germany = df_novel_covid2[df_novel_covid2['Country/Region']=='Germany'].groupby(['ObservationDate']).sum().reset_index()
df_italy = df_novel_covid2[df_novel_covid2['Country/Region']=='Italy'].groupby(['ObservationDate']).sum().reset_index()
df_turkey = df_novel_covid2[df_novel_covid2['Country/Region']=='Turkey'].groupby(['ObservationDate']).sum().reset_index()
df_spain =  df_novel_covid2[df_novel_covid2['Country/Region']=='Spain'].groupby(['ObservationDate']).sum().reset_index()
df_us = df_novel_covid2[df_novel_covid2['Country/Region']=='US'].groupby(['ObservationDate']).sum().reset_index()
df_france = df_novel_covid2[df_novel_covid2['Country/Region']=='France'].groupby(['ObservationDate']).sum().reset_index()
df_germany['ObservationDate2'] = pd.to_datetime(df_germany['ObservationDate'])
df_germany = df_germany.sort_values(by=['ObservationDate2'],ascending=True)
df_germany['DeathRateClosed']= df_germany['Deaths'] * 100 / (df_germany['Deaths']+df_germany['Recovered'])
df_germany['RecoveryRateClosed']= df_germany['Recovered'] * 100 / (df_germany['Deaths']+df_germany['Recovered'])
df_germany.dropna(inplace=True)
df_italy['ObservationDate2'] = pd.to_datetime(df_italy['ObservationDate'])
df_italy = df_italy.sort_values(by=['ObservationDate2'],ascending=True)
df_italy['DeathRateClosed']= df_italy['Deaths'] * 100 / (df_italy['Deaths']+df_italy['Recovered'])
df_italy['RecoveryRateClosed']= df_italy['Recovered'] * 100 / (df_italy['Deaths']+df_italy['Recovered'])
df_italy.dropna(inplace=True)
df_turkey['ObservationDate2'] = pd.to_datetime(df_turkey['ObservationDate'])
df_turkey = df_turkey.sort_values(by=['ObservationDate2'],ascending=True)
df_turkey['DeathRateClosed']= df_turkey['Deaths'] * 100 / (df_turkey['Deaths']+df_turkey['Recovered'])
df_turkey['RecoveryRateClosed']= df_turkey['Recovered'] * 100 / (df_turkey['Deaths']+df_turkey['Recovered'])
df_turkey.dropna(inplace=True)
df_spain['ObservationDate2'] = pd.to_datetime(df_spain['ObservationDate'])
df_spain = df_spain.sort_values(by=['ObservationDate2'],ascending=True)
df_spain['DeathRateClosed']= df_spain['Deaths'] * 100 / (df_spain['Deaths']+df_spain['Recovered'])
df_spain['RecoveryRateClosed']= df_spain['Recovered'] * 100 / (df_spain['Deaths']+df_spain['Recovered'])
df_spain.dropna(inplace=True)
df_us['ObservationDate2'] = pd.to_datetime(df_us['ObservationDate'])
df_us = df_us.sort_values(by=['ObservationDate2'],ascending=True)
df_us['DeathRateClosed']= df_us['Deaths'] * 100 / (df_us['Deaths']+df_us['Recovered'])
df_us['RecoveryRateClosed']= df_us['Recovered'] * 100 / (df_us['Deaths']+df_us['Recovered'])
df_us.dropna(inplace=True)
df_france['ObservationDate2'] = pd.to_datetime(df_france['ObservationDate'])
df_france = df_france.sort_values(by=['ObservationDate2'],ascending=True)
df_france['DeathRateClosed']= df_france['Deaths'] * 100 / (df_france['Deaths']+df_france['Recovered'])
df_france['RecoveryRateClosed']= df_france['Recovered'] * 100 / (df_france['Deaths']+df_france['Recovered'])
df_france.dropna(inplace=True)


# In[ ]:


# Creating trace1
trace1 = go.Scatter(
                    x = df_italy.ObservationDate2,
                    y = df_italy.DeathRateClosed,
                    mode = "lines+markers",
                    name = 'Death Rate',
                    marker = dict(color = 'rgba(128, 0, 0, 0.8)'))
# Creating trace2
trace2 = go.Scatter(
                    x = df_italy.ObservationDate2,
                    y = df_italy.RecoveryRateClosed,
                    mode = "lines+markers",
                    name = 'Recovery Rate',
                    marker = dict(color = 'rgba(0, 128, 0, 0.8)'))
data = [trace1, trace2]
layout = dict(title = 'Italy COVID-19 Recovery Rate vs Death Rate in Closed Cases(%)',
              xaxis= dict(title= 'Days',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

# Creating trace1
trace1 = go.Scatter(
                    x = df_germany.ObservationDate2,
                    y = df_germany.DeathRateClosed,
                    mode = "lines+markers",
                    name = 'Death Rate',
                    marker = dict(color = 'rgba(128, 0, 0, 0.7)'))
# Creating trace2
trace2 = go.Scatter(
                    x = df_germany.ObservationDate2,
                    y = df_germany.RecoveryRateClosed,
                    mode = "lines+markers",
                    name = 'Recovery Rate',
                    marker = dict(color = 'rgba(0, 128, 0, 0.8)'))
data = [trace1, trace2]
layout = dict(title = 'Germany COVID-19 Recovery Rate vs Death Rate in Closed Cases(%)',
              xaxis= dict(title= 'Days',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


# Creating trace1
trace1 = go.Scatter(
                    x = df_turkey.ObservationDate2,
                    y = df_turkey.DeathRateClosed,
                    mode = "lines+markers",
                    name = 'Death Rate',
                    marker = dict(color = 'rgba(128, 0, 0, 0.8)'))
# Creating trace2
trace2 = go.Scatter(
                    x = df_turkey.ObservationDate2,
                    y = df_turkey.RecoveryRateClosed,
                    mode = "lines+markers",
                    name = 'Recovery Rate',
                    marker = dict(color = 'rgba(0, 128, 0, 0.8)'))
data = [trace1, trace2]
layout = dict(title = 'Turkey COVID-19 Recovery Rate vs Death Rate in Closed Cases(%)',
              xaxis= dict(title= 'Days',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

# Creating trace1
trace1 = go.Scatter(
                    x = df_spain.ObservationDate2,
                    y = df_spain.DeathRateClosed,
                    mode = "lines+markers",
                    name = 'Death Rate',
                    marker = dict(color = 'rgba(128, 0, 0, 0.8)'))
# Creating trace2
trace2 = go.Scatter(
                    x = df_spain.ObservationDate2,
                    y = df_spain.RecoveryRateClosed,
                    mode = "lines+markers",
                    name = 'Recovery Rate',
                    marker = dict(color = 'rgba(0, 128, 0, 0.8)'))
data = [trace1, trace2]
layout = dict(title = 'Spain COVID-19 Recovery Rate vs Death Rate in Closed Cases(%)',
              xaxis= dict(title= 'Days',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


# Creating trace1
trace1 = go.Scatter(
                    x = df_us.ObservationDate2,
                    y = df_us.DeathRateClosed,
                    mode = "lines+markers",
                    name = 'Death Rate',
                    marker = dict(color = 'rgba(128, 0, 0, 0.8)'))
# Creating trace2
trace2 = go.Scatter(
                    x = df_us.ObservationDate2,
                    y = df_us.RecoveryRateClosed,
                    mode = "lines+markers",
                    name = 'Recovery Rate',
                    marker = dict(color = 'rgba(0, 128, 0, 0.8)'))
data = [trace1, trace2]
layout = dict(title = 'United States COVID-19 Recovery Rate vs Death Rate in Closed Cases(%)',
              xaxis= dict(title= 'Days',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

trace1 = go.Scatter(
                    x = df_france.ObservationDate2,
                    y = df_france.DeathRateClosed,
                    mode = "lines+markers",
                    name = 'Death Rate',
                    marker = dict(color = 'rgba(128, 0, 0, 0.8)'))
# Creating trace2
trace2 = go.Scatter(
                    x = df_france.ObservationDate2,
                    y = df_france.RecoveryRateClosed,
                    mode = "lines+markers",
                    name = 'Recovery Rate',
                    marker = dict(color = 'rgba(0, 128, 0, 0.8)'))
data = [trace1, trace2]
layout = dict(title = 'France COVID-19 Recovery Rate vs Death Rate in Closed Cases (%)',
              xaxis= dict(title= 'Days',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# ## Active Cases Day by Day
# 
# This is an important graphic because it can represent the current burden of the health system of countries.

# In[ ]:


trace1 = go.Scatter(
    x=df_italy.ObservationDate2,
    y=df_italy.ActiveCases,
    name = "Italy"
)
trace2 = go.Scatter(
    x=df_germany.ObservationDate2,
    y=df_germany.ActiveCases,
    xaxis='x2',
    yaxis='y2',
    name = "Germany"
)
trace3 = go.Scatter(
    x=df_turkey.ObservationDate2,
    y=df_turkey.ActiveCases,
    xaxis='x3',
    yaxis='y3',
    name = "Turkey"
)
trace4 = go.Scatter(
    x=df_spain.ObservationDate2,
    y=df_spain.ActiveCases,
    xaxis='x4',
    yaxis='y4',
    name = "Spain"

)
data = [trace1, trace2, trace3, trace4]
layout = go.Layout(
    xaxis=dict(
        domain=[0, 0.45]
    ),
    yaxis=dict(
        domain=[0, 0.45]
    ),
    xaxis2=dict(
        domain=[0.55, 1]
    ),
    xaxis3=dict(
        domain=[0, 0.45],
        anchor='y3'
    ),
    xaxis4=dict(
        domain=[0.55, 1],
        anchor='y4'
    ),
    yaxis2=dict(
        domain=[0, 0.45],
        anchor='x2'
    ),
    yaxis3=dict(
        domain=[0.55, 1]
    ),
    yaxis4=dict(
        domain=[0.55, 1],
        anchor='x4'
    ),
     
    title = 'Daily COVID-19 Active Cases in Italy, Germany, Turkey and Spain'
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)


# <a id="7"></a><br/>
# ## Wordcloud

# In[ ]:


df_dene = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df_dene.head()


# In[ ]:


#convert string to date
#df_world['dateRep2'] = pd.to_datetime(df_world['dateRep'])
#max(df_world.dateRep2)
df_world = pd.read_csv('http://opendata.ecdc.europa.eu/covid19/casedistribution/csv')
# Top 20 countries where COVID-19 has most seen
top_5 = df_world.groupby("countriesAndTerritories").sum().sort_values(by=['cases'], ascending=False).head(5).reset_index()['countriesAndTerritories']
type(top_5)
#df_world_top_5 = df_world[[each in top_5.values for each in df_world.countriesAndTerritories]]
df_world_top_5 = df_world[[each in top_5.values for each in df_world['countriesAndTerritories']]]
df_world_top_5 = df_world_top_5.reindex(index=df_world_top_5.index[::-1])

df_world3 = df_world.groupby(['countriesAndTerritories']).sum()[['cases','deaths']]
df_world3 = df_world3.sort_values(by=['cases'],ascending=False).reset_index()


# In[ ]:


#Covid-19 Cases and Deaths wordcloud

#countries = df_world3.countriesAndTerritories
countries=df_world3['countriesAndTerritories']
plt.subplots(figsize=(12,12))
wordcloud = WordCloud(
                          background_color='white',
                          width=512,
                          height=512
                         ).generate(" ".join(countries))
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('graph.png')

plt.show()


# In[ ]:





# In[ ]:


df_world3.countriesAndTerritories = df_world3.countriesAndTerritories.replace('United_States_of_America','USA')


# In[ ]:


# I modified the map code from this link:https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-kiva 
data = [ dict(
        type = 'choropleth',
        locations = df_novel_covid['Country/Region'],
        locationmode = 'country names',
        z = df_novel_covid['Confirmed'],
        text = df_novel_covid['Country/Region'],
        #colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(56, 142, 60)']],
        #colorscale = [[0,'rgb(255, 255, 255)'],[1,'rgb(220, 83, 67)']],
        colorscale = [[0,"rgb(5, 10, 172)"],[0.85,"rgb(40, 60, 190)"],[0.9,"rgb(70, 100, 245)"],\
            [0.94,"rgb(90, 120, 245)"],[0.97,"rgb(106, 137, 247)"],[1,"rgb(220, 220, 220)"]],
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(128,0,0)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            tickprefix = '',
            title = 'Number of Cases'),
      ) ]

layout = dict(
    title = 'Number of cases by Country',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        projection = dict(
            type = 'equirectangular'
        )
    )
)

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False, filename='covid19-cases-world-map')


# <a id="8"> </a> <br/>
# ## Daily COVID-19 Cases and Deaths

# In[ ]:


from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter(
                    x = df_world_top_5.dateRep,
                    y = df_world_top_5[df_world_top_5.countriesAndTerritories==top_5.values[0]].cases,
                    mode = "lines+markers",
                    name = top_5.values[0],
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'))
# Creating trace2
trace2 = go.Scatter(
                    x = df_world_top_5.dateRep,
                    y = df_world_top_5[df_world_top_5.countriesAndTerritories==top_5.values[1]].cases,
                    mode = "lines+markers",
                    name = top_5.values[1],
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'))
# Creating trace3
trace3 = go.Scatter(
                    x = df_world_top_5.dateRep,
                    y = df_world_top_5[df_world_top_5.countriesAndTerritories==top_5.values[2]].cases,
                    mode = "lines+markers",
                    name = top_5.values[2],
                    marker = dict(color = 'rgba(255, 255, 0, 0.8)'))
# Creating trace4
trace4 = go.Scatter(
                    x = df_world_top_5.dateRep,
                    y = df_world_top_5[df_world_top_5.countriesAndTerritories==top_5.values[3]].cases,
                    mode = "lines+markers",
                    name = top_5.values[3],
                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'))
# Creating trace5
trace5 = go.Scatter(
                    x = df_world_top_5.dateRep,
                    y = df_world_top_5[df_world_top_5.countriesAndTerritories==top_5.values[4]].cases,
                    mode = "lines+markers",
                    name = top_5.values[4],
                    marker = dict(color = 'rgba(0, 255, 0, 0.8)'))

data = [trace1, trace2,trace3, trace4,trace5]
layout = dict(title = 'World COVID-19 Daily Cases',
              xaxis= dict(title= 'Days',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# In[ ]:


from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go

# Creating trace1
trace1 = go.Scatter(
                    x = df_world_top_5.dateRep,
                    y = df_world_top_5[df_world_top_5.countriesAndTerritories==top_5.values[0]].deaths,
                    mode = "lines+markers",
                    name = top_5.values[0],
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'))
# Creating trace2
trace2 = go.Scatter(
                    x = df_world_top_5.dateRep,
                    y = df_world_top_5[df_world_top_5.countriesAndTerritories==top_5.values[1]].deaths,
                    mode = "lines+markers",
                    name = top_5.values[1],
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'))
# Creating trace3
trace3 = go.Scatter(
                    x = df_world_top_5.dateRep,
                    y = df_world_top_5[df_world_top_5.countriesAndTerritories==top_5.values[2]].deaths,
                    mode = "lines+markers",
                    name = top_5.values[2],
                    marker = dict(color = 'rgba(255, 255, 0, 0.8)'))
# Creating trace4
trace4 = go.Scatter(
                    x = df_world_top_5.dateRep,
                    y = df_world_top_5[df_world_top_5.countriesAndTerritories==top_5.values[3]].deaths,
                    mode = "lines+markers",
                    name = top_5.values[3],
                    marker = dict(color = 'rgba(255, 0, 0, 0.8)'))
# Creating trace5
trace5 = go.Scatter(
                    x = df_world_top_5.dateRep,
                    y = df_world_top_5[df_world_top_5.countriesAndTerritories==top_5.values[4]].deaths,
                    mode = "lines+markers",
                    name = top_5.values[4],
                    marker = dict(color = 'rgba(0, 255, 0, 0.8)'))

data = [trace1, trace2,trace3, trace4,trace5]
layout = dict(title = 'World COVID-19 Daily Deaths',
              xaxis= dict(title= 'Days',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# <a id = "1"></a><br>
# # COVID-19 US Data

# In[ ]:



df = pd.read_csv("/kaggle/input/us-counties-covid-19-dataset/us-counties.csv")


# With using info method of dataframe we can get useful information about the dataset we work on.

# In[ ]:


df.info()


# corr method gives correlation coefficients of numeric columns.

# In[ ]:


df.corr().index


# In[ ]:



#correlation map
f,ax = plt.subplots(figsize=(6, 6))
sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


#  **To get what columns dataframe have and inspect some examples of the dataset we can use "*columns*" and "*head*" methods respectively.**

# In[ ]:


df.columns


# In[ ]:


df.tail()


# 

# In[ ]:


#For time series data, it is convenient to use line diagrams
# For this graph I've created a derived dataset from original dataset
df2 = df.groupby(['date']).sum()
'''df2.columns
df2.cases.plot(kind = 'line', color = 'b',label = 'Cases',linewidth=2,alpha = 0.5,grid = True,linestyle = '-',figsize=(18,6))
df2.deaths.plot(kind = 'line', color = 'r',label = 'Deaths',linewidth=2,alpha = 0.5,grid = True,linestyle = '-',figsize=(18,6))
plt.legend(loc='upper left')     # legend = puts label into plot
plt.xlabel('Days')              # label = name of label
plt.ylabel('# of Cases')
plt.title('Corona Virus US Statistics')            # title = title of plot
plt.show()'''


# In[ ]:



# Creating trace1
trace1 = go.Scatter(
                    x = df2.index,
                    y = df2.cases,
                    mode = "lines+markers",
                    name = "Cases",
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'))
# Creating trace2
trace2 = go.Scatter(
                    x = df2.index,
                    y = df2.deaths,
                    mode = "lines+markers",
                    name = "Deaths",
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'))
data = [trace1, trace2]
layout = dict(title = 'US Corona Cases and Deaths',
              xaxis= dict(title= 'Days',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)


# <a id="2"></a> <br/>
# ## COVID-19 Cases State by State (US)

# In[ ]:


# Here is the top 10 states where covid-19 has most seen
df_hist = df[df.date == max(df.date)].groupby(['state']).sum().sort_values(by=['cases'],ascending=False).head(10)
states = df_hist.index.values
#df_hist.cases.plot(kind = 'hist',figsize = (12,12))
'''f, ax = plt.subplots(figsize=(18,8)) 
plt.bar(df_hist.index, df_hist.cases)
plt.show()'''


# In[ ]:


#Plotly bar graphic

# import graph objects as "go"
import plotly.graph_objs as go
# create trace1 
trace1 = go.Bar(
                x = df_hist.index,
                y = df_hist.cases,
                name = "Cases",
                marker = dict(color = 'rgba(0, 0, 255, 0.5)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))
# create trace2 
trace2 = go.Bar(
                x = df_hist.index,
                y = df_hist.deaths,
                name = "Deaths",
                marker = dict(color = 'rgba(255, 0, 0, 0.5)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
data = [trace1, trace2]
layout = go.Layout(barmode = "group")
fig = go.Figure(data = data, layout = layout)
iplot(fig)


# In[ ]:


df_for_states = df.groupby(["date","state"]).sum()
df_for_states.reset_index(inplace=True)

'''df_for_states.loc['2020-04-10']
df_for_states.index = range(1,len(df_for_states)+1,1)
df_for_states.head()'''
df_for_states = df_for_states[['date','state','cases','deaths']]
#Last 10 days and top 10 states
df_us_deaths =  df_for_states.pivot(index='date',columns='state',values='deaths').tail(10)
df_us_cases =  df_for_states.pivot(index='date',columns='state',values='cases').tail(10)


# In[ ]:





# <a id="4"> </a> <br/>
# ### Heatmaps of Cases and Deaths by State (Last 10 days and top 10 states)

# In[ ]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df_us_deaths[states], annot=True, linewidths=1, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(df_us_cases[states], annot=True, linewidths=1, fmt= '.1f',ax=ax)
plt.show()


# <a id="3"></a> <br/>
# ## COVID-19 California Detailed Statistics

# In[ ]:


df_california = df[df['state']=='California']
df_california.groupby(["date","county"]).sum()
ratio =[df_california['deaths']/df_california['cases']]
df_california['death_ratio'] = df_california['deaths']/df_california['cases']


# In[ ]:


df_california_last_day = df_california[df_california['date'] == max(df_california['date'])].sort_values(by=['cases'],ascending=False).head(10)


# In[ ]:


fig, ax = plt.subplots(figsize=(18,6))
x = np.arange(len(df_california_last_day['cases']))  # the label locations
width = 0.35
rects1 = ax.bar(x - width/2, df_california_last_day['cases'], width, label='Cases')
rects2 = ax.bar(x + width/2, df_california_last_day['deaths'], width, label='Deaths')
ax.set_ylabel('Numbers')
ax.set_title('Cases and Deaths For Each County in California')
ax.set_xticks(x)
ax.set_xticklabels(df_california_last_day['county'])
ax.legend()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(10,10))
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
county = df_california_last_day.sort_values(by=['death_ratio'],ascending=False)['county']
death_ratio = df_california_last_day.sort_values(by=['death_ratio'],ascending=False)['death_ratio']
ax.pie(death_ratio, labels = county,autopct='%1.2f%%')
ax.set_title('Death Ratios')
plt.show()


# <a id="6"> </a> <br/> 
# ## COVID-19 Turkey Statistics

# In[ ]:


df_turkey = df_world[df_world['countriesAndTerritories']=='Turkey']
df_turkey = df_turkey.reindex(index=df_turkey.index[::-1])
'''
plt.figure(figsize=(18,6))
plt.plot(df_turkey.dateRep,df_turkey.cases,'bo-')
plt.xticks(rotation=90)
plt.title("Turkey Daily COVID-19 Cases")
plt.show()'''


# In[ ]:


# Creating trace1
trace1 = go.Scatter(
                    x = df_turkey.dateRep,
                    y = df_turkey.cases,
                    mode = "lines+markers",
                    name = 'Cases',
                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'))
# Creating trace2
trace2 = go.Scatter(
                    x = df_turkey.dateRep,
                    y = df_turkey.deaths,
                    mode = "lines+markers",
                    name = 'Deaths',
                    marker = dict(color = 'rgba(80, 26, 80, 0.8)'))
data = [trace1, trace2]
layout = dict(title = 'Turkey COVID-19 Daily Cases and Deaths',
              xaxis= dict(title= 'Days',ticklen= 5,zeroline= False)
             )
fig = dict(data = data, layout = layout)
iplot(fig)

