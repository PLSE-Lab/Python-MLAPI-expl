#!/usr/bin/env python
# coding: utf-8

# # COVID-19 - NSW Health Data Analysis

# This analysis focuses on data specifically relating to COVID-19 cases in New South Wales (Australia) - as made avaialble by Data NSW.
# 
# It analyses the data in a few different ways, including the prevalence of cases in the Sydney metropolitan region vs non-metro cases. The mapping done was achieved with the assistance of an open source index of Australian postcodes and associated latitude and longditude values.
# 
# The size of NSW - and indeed all Australian states - is more indicative of the geographical layout of many small countries. As such it could be useful to understand the problem from a Sydney and non-Sydney perspective.
# 
# I've also conducted a short analysis using the ongoing John Hopkins data set - https://www.kaggle.com/adrianagius/australian-covid-19-data-analysis

# # The Data

# DATA NSW has recently made available three separata COVID-19 data sources relating to cases in New South Wales. They can be found in the following locations.
# 
# 1. https://data.nsw.gov.au/data/dataset/covid-19-cases-by-location
# 2. https://data.nsw.gov.au/data/dataset/nsw-covid-19-cases-by-likely-source-of-infection
# 3. https://data.nsw.gov.au/data/dataset/nsw-covid-19-cases-by-age-range
# 
# As things currently stand, the datasets are not linked.

# In[ ]:


get_ipython().system('pip install pyvis')

#package imports
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import os
import datetime
import cufflinks as cf
import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px
import geopandas as gpd
import pyvis
from pyvis.network import Network
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[ ]:


#load in data from Data NSW
nswdata = pd.read_csv('https://data.nsw.gov.au/data/dataset/aefcde60-3b0c-4bc0-9af1-6fe652944ec2/resource/21304414-1ff1-4243-a5d2-f52778048b29/download/covid-19-cases-by-notification-date-and-postcode-local-health-district-and-local-government-area.csv')
nswages = pd.read_csv('https://data.nsw.gov.au/data/dataset/3dc5dc39-40b4-4ee9-8ec6-2d862a916dcf/resource/24b34cb5-8b01-4008-9d93-d14cf5518aec/download/covid-19-cases-by-notification-date-and-age-range.csv')
nswfactors = pd.read_csv('https://data.nsw.gov.au/data/dataset/c647a815-5eb7-4df6-8c88-f9c537a4f21e/resource/2f1ba0f3-8c21-4a86-acaf-444be4401a6d/download/covid-19-cases-by-notification-date-and-likely-source-of-infection.csv')
ausmapinfo = pd.read_csv('https://raw.githubusercontent.com/matthewproctor/australianpostcodes/master/australian_postcodes.csv')

#amend duplicate postcodes and combine with data
ausmapinfo = ausmapinfo.groupby('postcode').agg({
                             'locality': ' | '.join, 
                             'long':'first',
                             'lat':'first'}).reset_index()
nswmaster = pd.merge(nswdata, ausmapinfo, on='postcode', how='left')


# # NSW Case Analysis

# Most recent count of new cases in NSW

# In[ ]:


#summarise by cases over time
summarised_cases = nswmaster.groupby('notification_date').agg({'notification_date':'count'}).rename(columns={'notification_date':'Count'}).reset_index()
summarised_cases = summarised_cases.sort_values(by=['notification_date'], ascending=False)
summarised_cases[['notification_date','Count']].head(1).style


# In[ ]:


#New cases by day in NSW over time
fig = px.bar(summarised_cases, x="notification_date", y="Count", title='New cases by day in NSW over time')
fig.update_layout(xaxis_title="Date", yaxis_title="Count")
fig.show()


# Most recent number of cumulative cases in NSW.

# In[ ]:


summarised_cases['cumulative']=summarised_cases.loc[::-1, 'Count'].cumsum()[::-1]
summarised_cases[['notification_date','cumulative']].head(1).style


# In[ ]:


fig = px.line(summarised_cases, x="notification_date", y="cumulative", title='Cumulative cases in NSW over time')
fig.update_layout(xaxis_title="Date", yaxis_title="Count")
fig.show()


# Most recent rate of case increase

# In[ ]:


summarised_cases['Rate of Change']=summarised_cases['cumulative'].pct_change()*-100
summarised_cases = summarised_cases.dropna()
summarised_cases[['notification_date','Rate of Change']].head(1).style


# In[ ]:


fig = px.line(summarised_cases, x="notification_date", y="Rate of Change", title='Rate of change of new cases by day in NSW over time')
fig.update_layout(xaxis_title="Date", yaxis_title="Rate")
fig.show()


# In[ ]:


#summarise by suburbs

summarised_suburb = nswmaster.groupby('locality').agg({'locality':'count','lga_name19':'first'}).rename(columns={'locality':'Count'}).reset_index()
summarised_suburb = summarised_suburb.sort_values(by=['Count'], ascending=False)
summarised_suburb = summarised_suburb.replace('BEN BUCKLER','BONDI/BEN BUCKLER')
summarised_suburb_histogram = summarised_suburb.head(20)

data  = go.Data([
            go.Bar(
              y = summarised_suburb_histogram.Count,
              x = summarised_suburb_histogram.locality,
              orientation='v',
        )])
layout = go.Layout(
        title = "NSW Coronavirus Cases By 20 Most Affected Suburbs"
)


# In[ ]:


fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


summarised_locals = nswmaster.groupby('lhd_2010_name').agg({'lhd_2010_name':'count'}).rename(columns={'lhd_2010_name':'Count'}).reset_index()
summarised_locals = summarised_locals.sort_values(by=['Count'], ascending=False)

data  = go.Data([
            go.Bar(
              y = summarised_locals.Count,
              x = summarised_locals.lhd_2010_name,
              orientation='v'
        )])
layout = go.Layout(
        title = "NSW Coronavirus Cases By Region"
)
fig  = go.Figure(data=data, layout=layout)
py.iplot(fig)


#  # Cases within Sydney Metro Region

# In[ ]:


#Prepare Sydney Metro Cases
nswmastermap = nswmaster[nswmaster.long != 0.000000]
nswmastermap = nswmastermap[nswmastermap['postcode'].between(2000, 2234)]
nswmastermap = nswmastermap.groupby('lga_name19').agg({'lga_name19':'count','long':'first','lat':'first'}).rename(columns={'lga_name19':'Count'}).reset_index()


# In[ ]:


#Map Sydney Metro Cases
fig = px.scatter_mapbox(nswmastermap, lat="lat", lon="long", hover_name="lga_name19", hover_data=["Count"],
                        color_discrete_sequence=["red"], zoom=9, height=500, size="Count",size_max=25)
fig.update_layout(mapbox_style="stamen-terrain")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# And by Suburb

# In[ ]:


nswmastermap = nswmaster[nswmaster.long != 0.000000]
nswmastermap = nswmastermap[nswmastermap['postcode'].between(2000, 2234)]
nswmastermap = nswmastermap.groupby('locality').agg({'locality':'count','long':'first','lat':'first'}).rename(columns={'locality':'Count'}).reset_index()

#Map Sydney Suburbs
fig = px.scatter_mapbox(nswmastermap, lat="lat", lon="long", hover_name="locality", hover_data=["Count"],
                        color_discrete_sequence=["red"], zoom=9, height=500, size="Count",size_max=25)
fig.update_layout(mapbox_style="stamen-terrain")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[ ]:


sydneycases = nswmaster[nswmaster.long != 0.000000]
sydneycases = sydneycases[sydneycases['postcode'].between(2000, 2234)]
sydneycases = sydneycases.groupby('lhd_2010_name').agg({'lhd_2010_name':'count'}).rename(columns={'lhd_2010_name':'Count'}).reset_index()
sydneycases = sydneycases.sort_values(by=['Count'], ascending=False)

data  = go.Data([
            go.Bar(
              y = sydneycases.Count,
              x = sydneycases.lhd_2010_name,
              orientation='v'
        )])
layout = go.Layout(
        title = "Sydney Cases By Region"
)
fig  = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


sydneycasessub = nswmaster[nswmaster.long != 0.000000]
sydneycasessub = sydneycasessub[sydneycasessub['postcode'].between(2000, 2234)]
sydneycasessub = sydneycasessub.groupby('locality').agg({'locality':'count','lga_name19':'first'}).rename(columns={'locality':'Count'}).reset_index()
sydneycasessub = sydneycasessub.sort_values(by=['Count'], ascending=False)
sydneycasessub = sydneycasessub.replace('BEN BUCKLER','BONDI/BEN BUCKLER')
sydneycasessub_histogram = sydneycasessub.head(20)

data  = go.Data([
            go.Bar(
              y = sydneycasessub_histogram.Count,
              x = sydneycasessub_histogram.locality,
              orientation='v',
        )])
layout = go.Layout(
        title = "Sydney Coronavirus Cases By 20 Most Affected Suburbs"
)
fig  = go.Figure(data=data, layout=layout)
py.iplot(fig)


# # Cases outisde Sydney Metro Region - Mapped

# In[ ]:


#Prepare Non Syd Metro Cases
nswmastermap = nswmaster[nswmaster.long != 0.000000]
nswmastermap = nswmastermap[nswmastermap['postcode'].between(2235, 2999)]
nswmastermap = nswmastermap.groupby('lga_name19').agg({'lga_name19':'count','long':'first','lat':'first'}).rename(columns={'lga_name19':'Count'}).reset_index()


# In[ ]:


fig = px.scatter_mapbox(nswmastermap, lat="lat", lon="long", hover_name="lga_name19", hover_data=["Count"],
                        color_discrete_sequence=["red"], zoom=5, height=500, size="Count",size_max=15)
fig.update_layout(mapbox_style="stamen-terrain")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# Those cases broken down by region:

# In[ ]:


nonsydneycases = nswmaster[nswmaster.long != 0.000000]
nonsydneycases = nonsydneycases[nonsydneycases['postcode'].between(2235, 2999)]
nonsydneycases = nonsydneycases.groupby('lhd_2010_name').agg({'lhd_2010_name':'count'}).rename(columns={'lhd_2010_name':'Count'}).reset_index()
nonsydneycases = nonsydneycases.sort_values(by=['Count'], ascending=False)

data  = go.Data([
            go.Bar(
              y = nonsydneycases.Count,
              x = nonsydneycases.lhd_2010_name,
              orientation='v'
        )])
layout = go.Layout(
        title = "Non Sydney Cases By Region"
)
fig  = go.Figure(data=data, layout=layout)
py.iplot(fig)


# Those cases broken down by suburb

# In[ ]:


nonsydneycasessub = nswmaster[nswmaster.long != 0.000000]
nonsydneycasessub = nonsydneycasessub[nonsydneycasessub['postcode'].between(2235, 2999)]
nonsydneycasessub = nonsydneycasessub.groupby('locality').agg({'locality':'count','lga_name19':'first'}).rename(columns={'locality':'Count'}).reset_index()
nonsydneycasessub = nonsydneycasessub.sort_values(by=['Count'], ascending=False)
nonsydneycasessub_histogram = nonsydneycasessub.head(20)

data  = go.Data([
            go.Bar(
              y = nonsydneycasessub_histogram.Count,
              x = nonsydneycasessub_histogram.locality,
              orientation='v',
        )])
layout = go.Layout(
        title = "Non Sydney Coronavirus Cases By 20 Most Affected Suburbs"
)
fig  = go.Figure(data=data, layout=layout)
py.iplot(fig)


# # Other Data Points

# This analysis focuses on age groups and likely sources of infection.

# # Age Groups

# The most prevalent Age group is the 70+ range. However perhaps surprisingly, the next two ranges exist win the 20-29 age group.

# In[ ]:


summarised_ages = nswages.groupby('age_group').agg({'age_group':'count'}).rename(columns={'age_group':'Count'}).reset_index()
summarised_ages = summarised_ages.sort_values(by=['Count'], ascending=False)
summarised_ages[['age_group','Count']].head(3).style


# In[ ]:


data  = go.Data([
            go.Bar(
              y = summarised_ages.Count,
              x = summarised_ages.age_group,
              orientation='v'
        )])
layout = go.Layout(
        title = "Cases by Age Group"
)
fig  = go.Figure(data=data, layout=layout)
py.iplot(fig)


# # Likely Source of Infection

# Cases sourced overseas are by far the greatest contributor to cases in NSW.

# In[ ]:


summarised_factors = nswfactors.groupby('likely_source_of_infection').agg({'likely_source_of_infection':'count'}).rename(columns={'likely_source_of_infection':'Count'}).reset_index()
summarised_factors  = summarised_factors.sort_values(by=['Count'], ascending=False)
summarised_factors[['likely_source_of_infection','Count']].head(3).style


# In[ ]:


data  = go.Data([
            go.Bar(
              y = summarised_factors.Count,
              x = summarised_factors.likely_source_of_infection,
              orientation='v'
        )])
layout = go.Layout(
        title = "Cases by Likely Source of Infection"
)
fig  = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In[ ]:


nswmasterx = nswmaster[['lhd_2010_name','locality']]
nswmasterx['Case ID'] = np.arange(len(nswmasterx))+1
nswmasterx['Case ID'] = nswmasterx['Case ID'].apply(str)
nswmasterx = nswmasterx.dropna()
nswmasterx['Total'] = 1
print(nswmasterx)


# Concepts of tracing can start to be useful, particularly where network analysis is involved. Code is slow to run in notebook, so feel free to take a copy and remove commenting below where it needs to be rendered. I try to keep the static image up to date.

# In[ ]:


casenetwork = Network(height="600px", width="100%", font_color="black", notebook=True)
casenetwork.set_edge_smooth("discrete")

sources = nswmasterx['locality']
targets = nswmasterx['Case ID']

weights = nswmasterx['Total']

edge_data = zip(sources, targets, weights)



for e in edge_data:
    src = e[0]
    dst = e[1]
    w = e[2]
    casenetwork.add_node(src, src, title=src, color="#0000ff", value=10)
    casenetwork.add_node(dst, dst, title=dst, color ="#ff0000", value=2)
    casenetwork.add_edge(src, dst, value=w, color='#00ff00')

#casenetwork.show("covidnswgraph.html")


# ![](https://pbs.twimg.com/media/EVhtYUfUwAIBT8y?format=jpg&name=medium)
