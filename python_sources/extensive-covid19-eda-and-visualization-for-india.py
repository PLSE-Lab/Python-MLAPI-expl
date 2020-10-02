#!/usr/bin/env python
# coding: utf-8

# Detailed Analysis on India's Covid19 status. Analysis include daily analysis of Confirmed/Cured/Death cases, comparison of various healthcare facilities in different Indian states, understanding of different age groups affected by this pandemic, and overall picture of how India is prepared to tackle Covid19.

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


# In[ ]:


Testcenter = pd.read_csv("/kaggle/input/covid19-in-india/ICMRTestingLabs.csv")


# In[ ]:


from IPython.display import HTML

Testcenter.to_csv('submission.csv')

def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='submission.csv')


# In[ ]:


import plotly as py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
pd.set_option('display.max_rows', None)


# In[ ]:


covid = pd.read_csv("/kaggle/input/covid19-in-india/covid_19_india.csv")
covid.tail()


# In[ ]:


plt.figure(figsize=(23,10))
plt.bar(covid.Date, covid.Confirmed,label="Confirmed")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(frameon=True, fontsize=12)
plt.title('Confrim',fontsize=30)
plt.show()

plt.figure(figsize=(23,10))
plt.bar(covid.Date, covid.Cured,label="Cured")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.legend(frameon=True, fontsize=12)
plt.title('Cured',fontsize=30)
plt.show()

plt.figure(figsize=(23,10))

plt.bar(covid.Date, covid.Deaths,label="Deaths")
plt.xlabel('Date')
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.legend(frameon=True, fontsize=12)
plt.title('Deaths',fontsize=30)
plt.show()





# In[ ]:


plt.figure(figsize=(23,10))
plt.bar(covid.Date, covid.Confirmed,label="Confirmed")
plt.bar(covid.Date, covid.Cured,label="Cured")
plt.bar(covid.Date, covid.Deaths,label="Deaths")
plt.xlabel('Date')
plt.xticks(rotation=90)
plt.ylabel("Count")
plt.legend(frameon=True, fontsize=12)
plt.title('Confrimed vs Cured vs Deaths',fontsize=30)
plt.show()

f, ax = plt.subplots(figsize=(23,10))
ax=sns.scatterplot(x="Date", y="Confirmed", data=covid,
             color="black",label = "Confirm")
ax=sns.scatterplot(x="Date", y="Cured", data=covid,
             color="red",label = "Cured")
ax=sns.scatterplot(x="Date", y="Deaths", data=covid,
             color="blue",label = "Deaths")

plt.xticks(rotation=90)
plt.plot(covid.Date, covid.Confirmed,zorder=1,color="black")
plt.plot(covid.Date, covid.Cured,zorder=1,color="red")
plt.plot(covid.Date, covid.Deaths,zorder=1,color="blue")


# In[ ]:


covidage=pd.read_csv("/kaggle/input/covid19-in-india/AgeGroupDetails.csv")
covidage


# In[ ]:


plt.figure(figsize=(23,10))
plt.bar(covidage.AgeGroup, covidage.TotalCases,label="Age Group")
plt.xlabel('Age Group')
plt.ylabel("Cases")
plt.legend(frameon=True, fontsize=25)
plt.title('Affected Age Group',fontsize=30)
plt.show()


# In[ ]:


currdate=pd.datetime.now().date().strftime('%-d/%m/%y')


# In[ ]:


from datetime import datetime, timedelta
days_to_subtract = 1
d = datetime.today() - timedelta(days=days_to_subtract)
d = pd.datetime.now().date() - timedelta(days=1)
currdate=d.strftime('%d/%m/%y')


# In[ ]:


covid1=covid[covid['Date'] == currdate]
covidcurr=covid1[['Date', 'State/UnionTerritory', 'Cured', 'Deaths', 'Confirmed']].copy()
covidcurr.plot()


# In[ ]:


#covidpivot=pd.pivot_table(covid,['Cured','Confirmed', 'Deaths'], 'State/UnionTerritory', aggfunc=pd.Series.nunique)


# In[ ]:


cm = sns.light_palette("orange", as_cmap=True)
covidcurr.style.background_gradient(cmap=cm)


# In[ ]:


covidstatest=pd.read_csv("/kaggle/input/covid19-in-india/StatewiseTestingDetails.csv")
covidstatest.tail()


# In[ ]:


covidstatestpivot=pd.pivot_table(covidstatest,['TotalSamples','Negative', 'Positive'], 'State', aggfunc=sum)
covidstatestpivot1=covidstatestpivot.reset_index('State')


# In[ ]:



fig = px.bar(covidstatestpivot1, x='State', y='TotalSamples',
             hover_data=['Negative', 'Positive'], color='Positive',labels={'TotalSamples':'Total Samples'},
              height=400)
fig.show()


# In[ ]:


covidtest=pd.read_csv("/kaggle/input/covid19-in-india/ICMRTestingLabs.csv")
covidtest.tail()


# In[ ]:


covidtestgrp=covidtest.groupby('state').count()
covidtestgrp=covidtestgrp.reset_index('state')


# In[ ]:


import plotly.graph_objects as go

fig = go.Figure([go.Bar(x=covidtestgrp['state'], y=covidtestgrp['lab'])])
fig.update_layout(title_text='Number of Testing Labs in Each State')
fig.show()


# In[ ]:


covidhosp=pd.read_csv("/kaggle/input/covid19-in-india/HospitalBedsIndia.csv")
covidhosp1=covidhosp.drop([1,36], axis=0)
covidhosp1.tail()


# In[ ]:


fig = go.Figure()
fig.add_trace(go.Bar(
    x=covidhosp1['State/UT'],
    y=covidhosp1['NumUrbanHospitals_NHP18'],
    name='Urban Hospitals',
    marker_color='indianred'
))
fig.add_trace(go.Bar(
    x=covidhosp1['State/UT'],
    y=covidhosp1['NumRuralHospitals_NHP18'],
    name='Rural Hospitals',
    marker_color='lightsalmon'
))
fig.add_trace(go.Bar(
    x=covidhosp1['State/UT'],
    y=covidhosp1['TotalPublicHealthFacilities_HMIS'],
    name='Total Public Hospitals',
    marker_color='green'
))


fig.update_layout(barmode='group', xaxis_tickangle=-45, title_text='Number of Urban, Rural and Total Public Hospitals in Each State')
fig.show()


# In[ ]:


covidindi=pd.read_csv("/kaggle/input/covid19-in-india/IndividualDetails.csv")
covidindigrp=covidindi.groupby(['current_status'])
covidindigrp.tail()


# In[ ]:


covidpopu=pd.read_csv("/kaggle/input/covid19-in-india/population_india_census2011.csv")
covidpopu.sort_values(by=['Population'],ascending=False)


# In[ ]:


covidpop=pd.read_csv("/kaggle/input/covid19-india-coordinates/Indiadata1.csv")
covidpop1=covidpop[['State/UT','Latitude','Longitude','Total Public Health Facility']].copy()
covidpop1=covidpop1.dropna()


# In[ ]:



import folium 
from folium import plugins
map = folium.Map(location=[20, 80], zoom_start=2,tiles='Stamen Toner')

for lat, lon, value, name in zip(covidpop1['Latitude'], covidpop1['Longitude'], covidpop1['Total Public Health Facility'], covidpop1['State/UT']):
    folium.CircleMarker([lat, lon],
                        radius=value*0.01,
                        popup = ('<strong>State</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Public Hospitals</strong>: ' + str(value) + '<br>'), color='blue',                        
                        fill_color='red',                  
                        fill_opacity=0.3 ).add_to(map)
map

