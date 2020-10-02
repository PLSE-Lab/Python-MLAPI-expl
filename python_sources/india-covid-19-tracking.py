#!/usr/bin/env python
# coding: utf-8

# # **COVID INTRODUCTION:**

# ![](https://img.medscape.com/thumbnail_library/dt_200221_covid_19_coronavirus_800x450.jpg)
# 
# **Coronavirus disease (COVID-19)** is an infectious disease caused by a newly discovered coronavirus.
# 
# Most people infected with the COVID-19 virus will experience mild to moderate respiratory illness and recover without requiring special treatment.  Older people, and those with underlying medical problems like cardiovascular disease, diabetes, chronic respiratory disease, and cancer are more likely to develop serious illness.
# 
# **Symptoms to be aware of:** The COVID-19 virus affects different people in different ways. Common symptoms include fever, tiredness and dry cough. Other symptoms that are related to this are sore throat, shortness of breath, bodyaches and with a very few people reporting diarrhoea, nausea or a runny nose. 
# 
# **To prevent infection and to slow transmission of COVID-19**, the following is recommended. 
# - Wash your hands regularly with soap and water, avoiding touching your face and covering your mouth and nose when coughing or sneezing.
# - Staying at home if you feel unwell is highly recommended to aviod the spread. 
# - Practice social distancing by avoiding unnecessary travel and staying away from large groups of people.
# 
# Another important way to prevent and slow down the spread is to be well informed about COVID-19 itself. This Exercise will try to serve a dual purpose to:
# 1. Understanding and building awareness about the disease, its spread and its effect so far on the world.  
# 2. Easily understand and being able to replicate the step by step process involved in data analysis, dataframe operations and visualisations. 
# 
# 
# Sources:
# 
# Information: https://www.who.int/health-topics/coronavirus#tab=tab_1
# 
# Image: https://www.medscape.com/viewarticle/925588*

# # **OBJECTIVE:**

# ***Notebook Objective:** *
# 
# The aim of this notebook is to understand, shed knowledge on the current global pandemic and try to add insightful information based on current data. Here we will analyse all relevent datasets and develop visuals that are easy to understand and replicate. The work is intended to be done in 3 parts with Part-A focusing on the data analysis of Covid-19 cases in India and Part-B will aim to analyse data for different countries such as Italy, Iran, UK, US, China and South Korea. The work in later part (Part:C) will focus on comparing trends using different visual methods to better understand the situation.  
# 
# ***Personal Objective:***
# 
# New to this, eager to learn and explore data analysis and representations.
# 
# Get the feel of Kaggle community. 
# 
# Kill time and boredom due to lockdown. 
# 
# ***Sources of knowledge:***
# 
# I have gone through a couple of really insightful notebooks that has given me the drive to build one myself. A few mentions below: 
# 
# Tracking India's Coronavirus Spread[WIP] - by Parul Pandey
# 
# COVID-19 | SARS-CoV-2 - A Statistical Analysis - by Saurav Mishra. 
# 
# Apart from this, its the mighty internet itself that served to be a great source of knowledge (Plotly and Pandas docs). 
# 

# # **PART:A - Indian Data Analysis**

# Initial Setup and configuration:

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import folium 
from folium.plugins import HeatMap, HeatMapWithTime
# plt.style.use("fivethirtyeight")# for pretty graphs

# # Increase the default plot size and set the color scheme
# plt.rcParams['figure.figsize'] = 8, 5
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# > Import Datasets

# In[ ]:


# Read and convert the csv dataset into pandas dataframe 
# 
#
India_data = pd.read_csv('/kaggle/input/covid19-in-india/covid_19_india.csv',parse_dates=['Date'], dayfirst=True)
India_bed = pd.read_csv('/kaggle/input/covid19-in-india/HospitalBedsIndia.csv')
India_population = pd.read_csv('/kaggle/input/covid19-in-india/population_india_census2011.csv')
India_agegrp = pd.read_csv('/kaggle/input/covid19-in-india/AgeGroupDetails.csv')
India_nodays = pd.read_excel('/kaggle/input/coronavirus-cases-in-india/per_day_cases.xlsx',sheet_name='India')
India_coordi = pd.read_csv('/kaggle/input/coronavirus-cases-in-india/Indian Coordinates.csv')


# # **Data Cleanup**

# In[ ]:


India_data.drop(['Sno'], axis=1, inplace=True)
India_data.rename(columns={"State/UnionTerritory": "States"}, inplace=True)

India_data['Active Cases'] = India_data['Confirmed'] - India_data['Cured'] - India_data['Deaths']

# India_data.head(10)


# In[ ]:


India_coordi.drop(['Unnamed: 3'], axis=1, inplace=True)
India_coordi.rename(columns={"Name of State / UT": "States"}, inplace=True)
India_coordi.sort_values('States',ascending = True) 
India_coordi=India_coordi.set_index('States')
# India_coordi.head(50)


# In[ ]:


State_data=India_data.copy()
State_data=State_data.groupby("States")['Confirmed','Deaths','Cured'].max().reset_index()

State_data['Active Cases'] = State_data['Confirmed'] - State_data['Cured'] - State_data['Deaths']
State_data['CD_ratio'] = State_data['Deaths']/State_data["Confirmed"]
State_data['CR_ratio'] = State_data['Cured']/State_data["Confirmed"]
State_data=State_data.set_index('States')
# State_data.info()


# In[ ]:


# This section cleans dataframe and drops non essential coloums 
# Also forms coloums to determine current active cases and ratios 
# 
India_population.drop(['Sno'], axis=1, inplace=True)
India_population.drop(['Density'], axis=1, inplace=True)
India_population.drop(['Gender Ratio'], axis=1, inplace=True)
India_population.drop(['Rural population'], axis=1, inplace=True)
India_population.drop(['Urban population'], axis=1, inplace=True)
India_population.rename(columns={"State / Union Territory": "States"}, inplace=True)

India_population = India_population.set_index('States')
# India_population.info()


# In[ ]:


# India_bed
India_bed.drop(['Sno'], axis=1, inplace=True)
India_bed.drop(['NumPrimaryHealthCenters_HMIS'], axis=1, inplace=True)
India_bed.drop(['NumCommunityHealthCenters_HMIS'], axis=1, inplace=True)
India_bed.drop(['NumSubDistrictHospitals_HMIS'], axis=1, inplace=True)
India_bed.drop(['NumDistrictHospitals_HMIS'], axis=1, inplace=True)



India_bed.rename(columns={"State/UT": "States"}, inplace=True)

India_bed = India_bed.set_index('States')
# India_bed.head(10)


# In[ ]:


India_info = India_population.merge(India_coordi, left_index=True, right_index=True)
# India_info.info()


# In[ ]:


India_agegrp.drop(['Sno'],axis=1,inplace=True)
# India_agegrp.head(10)


# In[ ]:


India_perday = India_data.groupby(['Date'])['Confirmed'].sum().reset_index().sort_values('Confirmed',ascending = True)
India_perday['New Daily Cases'] = India_perday['Confirmed'].sub(India_perday['Confirmed'].shift())
India_perday['New Daily Cases'].iloc[0] = India_perday['Confirmed'].iloc[0]
India_perday['New Daily Cases'] = India_perday['New Daily Cases'].astype(int)
# India_perday.head(100)


# > # **INDIA DATA SUMMARY:**

# In[ ]:


x = State_data['Confirmed'].sum()
Tot_CR = State_data['Cured'].sum()/State_data['Confirmed'].sum()
Tot_CD = State_data['Deaths'].sum()/State_data['Confirmed'].sum()
PopM = (x/1380004385)*1000000

fig = go.Figure()
fig.add_trace(go.Indicator(
    mode = "number",
    value = State_data['Confirmed'].sum(),
    title = {'text': "Total Confirmed cases"},
    domain = {'row': 0, 'column': 0}))
fig.add_trace(go.Indicator(
    mode = "number",
    value = State_data['Cured'].sum(),
    title = {'text': "Total Recovered"},
    domain = {'row': 0, 'column': 1}))
fig.add_trace(go.Indicator(
    mode = "number",
    value = State_data['Deaths'].sum(),
    title = {'text': "Total Deaths"},
    domain = {'row': 0, 'column': 2}))
fig.add_trace(go.Indicator(
    mode = "number",
    value = Tot_CD,
    title = {'text': "Death Ratio"},
    domain = {'row': 1, 'column': 2}))
fig.add_trace(go.Indicator(
    mode = "number",
    value = Tot_CR,
    title = {'text': "Recovery Ratio"},
    domain = {'row': 1, 'column': 1}))
fig.add_trace(go.Indicator(
    mode = "number",
    value = PopM,
    title = {'text': "Cases per 1M Population"},
    domain = {'row': 1, 'column': 0}))


fig.update_layout(
    grid = {'rows': 2, 'columns': 3, 'pattern': "independent"},
    width=1000,
    height=600)


# # **DATA VISUALS**

# In[ ]:


# -------------------- Plotting Daily increase in cases country-wise

India_data['Date']=pd.to_datetime(India_data.Date,dayfirst=True)
India_daily= India_data.groupby(['Date'])['Confirmed'].sum().reset_index().sort_values('Confirmed',ascending=True)

fig=go.Figure()
fig.add_trace(go.Bar(x=India_daily['Date'],y=India_daily['Confirmed'],marker_color='black'))
fig.update_layout(
    title="Overall cases",
    xaxis_title="Date",
    yaxis_title="Total Confirmed cases",
    font=dict(
#     family="Airel, monospace",
    size=18,
    color="black"),
    annotations=[             #----------- Annotation section to add specific details on plots
        dict(
            x='2020-03-22',
            y=0,
            xref="x",
            yref="y",
            text="India Lockdown",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-300
        )
    ])


# In[ ]:


India_CDR = India_data.groupby(['Date'])['Confirmed','Active Cases','Cured','Deaths'].sum().reset_index().sort_values('Date',ascending=False)
India_CDR.head()

fig=go.Figure()
fig.add_trace(go.Scatter(x=India_CDR['Date'], y=India_CDR['Confirmed'],
                    mode='lines', marker_color='Gray',name='Total Confirmed',fill='tozeroy'
                    ))
fig.add_trace(go.Scatter(x=India_CDR['Date'], y=India_CDR['Cured'],
                    mode= 'lines',marker_color='Green',name='Total Cured',fill='tozeroy'
                    ))
fig.add_trace(go.Scatter(x=India_CDR['Date'], y=India_CDR['Deaths'],
                    mode= 'lines',marker_color='Orange',name='Total Deaths',fill='tozeroy'
                    ))

fig.update_layout(
    title="Overall Cumulative cases",
    xaxis_title="Date",
    yaxis_title="Cumulative Cases",
    font=dict(
#     family="Airel, monospace",
    size=18,
    color="black"))


# In[ ]:


fig=go.Figure()
fig.add_trace(go.Bar(x=India_perday['Date'],y=India_perday['New Daily Cases'],marker_color='black'))

fig.add_annotation(             #----------- Annotation section to add specific details on plots
            x='2020-03-22',
            y=0,
            text="India Lockdown")
fig.add_annotation(             #----------- Annotation section to add specific details on plots
            x='2020-01-30',
            y=0,
            text="First case")
fig.update_annotations(dict(
            xref="x",
            yref="y",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-300
))
fig.update_layout(
    title="Daily New Cases",
    xaxis_title="Date",
    yaxis_title="Daily New Cases",
    font=dict(
#     family="Airel, monospace",
    size=18,
    color="black"))


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=India_agegrp['AgeGroup'], y=India_agegrp['TotalCases'], fill='tozeroy', line_shape='spline',
                        hovertext=["Percentage= 3.18%", "Percentage= 3.90%", "Percentage= 24.86%", "Percentage= 21.10%", "Percentage= 16.18%", "Percentage= 11.13%", "Percentage= 12.86%", "Percentage= 16.18%"],
                        hoverinfo='text',)) 
fig.update_layout(
    title="Affected Age group",
    yaxis_title="Total cases",
    xaxis_title="Age",
    font=dict(
#     family="Airel, monospace",
    size=18,
    color="black"))


## Adding a subplot with percentage graph 


# fig = make_subplots(rows=1, cols=2,specs=[[{'type':'xy'}, {'type':'domain'}]])
# fig.add_trace(go.Scatter(x=India_agegrp['AgeGroup'], y=India_agegrp['TotalCases'], fill='tozeroy', line_shape='spline',
#                         hovertext=["Percentage= 3.18%", "Percentage= 3.90%", "Percentage= 24.86%", "Percentage= 21.10%", "Percentage= 16.18%", "Percentage= 11.13%", "Percentage= 12.86%", "Percentage= 16.18%"],
#                         hoverinfo='text',), 1, 1) 
# fig.add_trace(go.Pie(labels=India_agegrp['AgeGroup'], values=India_agegrp['TotalCases'], hole=.3),1,2)
# fig.update_layout(
#     barmode='group',
#     width=1100,
#     height=500,
#     title="Affected Age group",
#     font=dict(
# #     family="Airel, monospace",
#     size=14,
#     color="black"))
# fig.show()


# # **State-wise Case overview table**

# In[ ]:



State_data.sort_values('Confirmed', ascending= False).fillna(0).style.background_gradient(cmap='Oranges',subset=["Confirmed"])                        .background_gradient(cmap='Reds',subset=["Deaths"])                        .background_gradient(cmap='Greens',subset=["Cured"])                        .background_gradient(cmap='Blues',subset=["Active Cases"])                        .background_gradient(cmap='Purples',subset=["CD_ratio"])                        .background_gradient(cmap='Greens',subset=["CR_ratio"])


# In[ ]:


# Creating merged Dataset for Demographic visuals

India_info = India_info.merge(State_data, left_index=True, right_index=True)
India_info.sort_values('States',ascending = False) 
# India_info.head(20)


# # **Demograpic representation**

# ** Needs fix, data table and coordinates mismatch

# In[ ]:


mapped_region = folium.Map(location=[20.5937, 78.9629],tiles='Stamen Toner',zoom_start=5,max_zoom=5,min_zoom=5,height = 800,width = '70%')
# Adding a heatmap feature
HeatMap(data=India_info[['Latitude','Longitude','Confirmed']].groupby(['Latitude','Longitude']).sum().reset_index().values.tolist(),
         radius=20, max_zoom=10).add_to(mapped_region)
mapped_region


# In order to keep statistics more meaningful, we remove the outlier data for further Statewise analysis by selecting states that have only had over a **100 confirmed cases**. 

# In[ ]:


State_data=State_data[(State_data['Confirmed'] > 100)]
State_data.sort_values('Confirmed', ascending = False)
# State_data.head(20)


# In[ ]:


State_data_Totcase=pd.DataFrame(State_data.sort_values(by=['Confirmed'],ascending=True).reset_index())
fig=go.Figure()
fig.add_trace(go.Bar(x=State_data_Totcase['Confirmed'],y=State_data_Totcase['States'],orientation='h',marker_color='teal',text=State_data_Totcase['Confirmed']))
fig.update_traces(textposition='inside')
fig.update_layout(
    width=1000,
    height=900,
    title="Statewise Total Cases",
    xaxis_title="Total Confirmed cases",
#     yaxis_title="States",
    font=dict(
#     family="Airel, monospace",
    size=18,
    color="black"))


# In[ ]:


State=pd.DataFrame(State_data.sort_values(by=['Confirmed'],ascending=True).reset_index())
fig=go.Figure()
fig.add_trace(go.Bar(y=State['States'],x=State['Cured'],base=0,name='Cured',orientation='h',marker_color='green'))
fig.add_trace(go.Bar(y=State['States'],x=State['Deaths'],base=0,name='Deaths',marker_color='red',orientation='h'))

fig.add_trace(go.Bar(y=State['States'],x=State['Active Cases'],base=0,name='Active',marker_color='blue',orientation='h'))

fig.update_layout(
    barmode='group',
    width=900,
    height=900,
    title="Statewise Current Case Status",
    xaxis_title="Cured/Deceased/Active cases",
#     yaxis_title="States",
    font=dict(
#     family="Airel, monospace",
    size=18,
    color="black"))


# In[ ]:


India_CD=pd.DataFrame(State_data.sort_values(by=['CD_ratio'],ascending=False).reset_index())
India_CR=pd.DataFrame(State_data.sort_values(by=['CR_ratio'],ascending=False).reset_index())

fig = make_subplots(rows=1, cols=2,shared_yaxes=True)
fig.add_trace(go.Bar(x=India_CD['States'],y=India_CD['CD_ratio'],base=0,name='Death Ratio',marker_color='Orange'),row=1,col=1)

fig.add_trace(go.Bar(x=India_CR['States'],y=India_CR['CR_ratio'],base=0,name='Recovery Ratio',marker_color='Green'),row=1, col=2)

fig.update_yaxes(title_text="Death Ratio", row=1, col=1)
fig.update_yaxes(title_text="Recovery Ratio", row=1, col=2)

fig.update_layout(
    barmode='group',
    width=1100,
    height=500,
    title="Death and recovery ratios",
    font=dict(
#     family="Airel, monospace",
    size=16,
    color="black"))
fig.show()


# *The ratios above are a result of comparing Deaths/Recovered cases with its respective Total confirmed cases for each state. 
# 
# An important thing to notice is that how **Kerala** has managed to have a high recovery ratio while keeping their death ratio to the minimum with respect to its total confirmed cases. 

# An Update of current Health facility vs cases will be added soon. 

# # **PART:B - Global Data Analysis** - Work in progress

# # **PART:C - Visual insights and Trend Indicators ** - Work in progress

# Initial write up and current work in progress. 
# 
# **Kindly upvote and feel free to suggest any changes via comments. Thank you! **
