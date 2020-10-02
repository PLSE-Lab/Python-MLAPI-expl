#!/usr/bin/env python
# coding: utf-8

# #### **Watch the video for better understanding of NCV:** https://www.youtube.com/watch?v=mOV1aBVYKGA#action=share
# 
# **Coronaviruses (CoV) are a large family of viruses that cause illness ranging from the common cold to more severe diseases such as Middle East Respiratory Syndrome (MERS-CoV) and Severe Acute Respiratory Syndrome (SARS-CoV). A novel coronavirus (nCoV) is a new strain that has not been previously identified in humans.  **
# 
# 

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


# #### _**Importing libaries for visualizaton**_

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
import plotly.express as px
import plotly.graph_objects as go
import folium 


# #### **Reading CSV here as well importing warnings**

# In[ ]:


#virus_data holds information about cases by Corona virus
virus_data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")
#world_cords holds information about co-ordinates of countries
world_cords=pd.read_csv("../input/world-coordinates/world_coordinates.csv")
import warnings
warnings. filterwarnings('ignore')


# In[ ]:


#getting the view about the virus_data  and world_cords data frames using info()
print(virus_data.info())
print("-"*50)
print(world_cords.info())


# In[ ]:


#getting the view about the virus_data data frame using info()
print(virus_data.head())
print("/"*50)
print(world_cords.head())


# In[ ]:


#As we can see Sno is not needed so lets drop it and Code from world_cords is not needed 
virus_data.drop(columns=["SNo","Last Update"],inplace=True)
world_cords.drop("Code",axis=1,inplace=True)
world_cords.rename(columns={"Country":"Country/Region"},inplace=True)


# In[ ]:


#checking if the Sno and Last Update are dropped or not from virus_data and Code from world_cords 
print(list(virus_data.columns))
print(list(world_cords.columns))


# In[ ]:


#replacing mainland china as china 
virus_data['Country/Region'].replace({'Mainland China':'China'},inplace=True)


# In[ ]:


#grouping according to date
virus_data_date_wise=virus_data.groupby("ObservationDate")["Confirmed","Deaths","Recovered"].sum()


# In[ ]:


#visualization of the corona virus cases across world along with time 
#Red is for confirmed and green is recovery
fig=go.Figure()
fig.add_trace(go.Scatter(x=virus_data_date_wise.index,y=virus_data_date_wise.Confirmed,mode="lines+markers",name="Confirmed"))
fig.add_trace(go.Scatter(x=virus_data_date_wise.index,y=virus_data_date_wise.Deaths,mode="lines+markers",name="Deaths"))
fig.add_trace(go.Scatter(x=virus_data_date_wise.index,y=virus_data_date_wise.Recovered,mode="lines+markers",name="Recovered"))
fig.update_layout(title='Confirmed vs Recovered trend of Corona virus across the world',
                   xaxis_title='Dates',
                   yaxis_title='Cases')
fig.show()


# **As we can see the confirmed cases are growing rapidly than the recovery cases across world  from the above graph but recovery is going good than death rate **

# In[ ]:


#As we can see that china has most confirmed cases and more deaths lets see latest date details
virus_data.ObservationDate.max()


# In[ ]:


#Filtering data based on date
virus_data_latest=virus_data[(virus_data['ObservationDate']==virus_data['ObservationDate'].max())]
virus_data_latest.head().style.set_properties(**{'background-color': 'black',
                           'color': 'lawngreen',
                           'border-color': 'white'})


# In[ ]:


#lets group the latest data country wise 
virus_country_wise=virus_data_latest.groupby("Country/Region",as_index=False)["Confirmed","Deaths","Recovered"].sum()
virus_country_wise=virus_country_wise.merge(world_cords,on="Country/Region")


# In[ ]:


# create map and display it for better visual and interactive understanding 
world_map = folium.Map(location=[10, -20], zoom_start=2,tiles='Stamen Toner')

for lat, lon, value1,value2,value3,name in zip(virus_country_wise['latitude'],virus_country_wise['longitude'],virus_country_wise['Confirmed'],virus_country_wise['Deaths'],virus_country_wise['Recovered'],virus_country_wise['Country/Region']):
    folium.CircleMarker([lat, lon],
                        radius=8,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed </strong>: ' + str(value1) + '<br>'
                                '<strong>Death </strong>: ' + str(value2) + '<br>'
                                '<strong>Recovered </strong>: ' + str(value3) + '<br>'),
                        color='black',
                        
                        fill_color='red',
                        fill_opacity=1,
                        ).add_to(world_map)
world_map


# In[ ]:


#lets see data for these three countries China,Italy,India
virus_selected_country=virus_data[(virus_data['Country/Region']=='China')|(virus_data['Country/Region']=='Italy')|(virus_data['Country/Region']=='India')]
virus_selected_country=virus_selected_country.groupby(["Country/Region","ObservationDate"],as_index=False)["Confirmed","Deaths","Recovered"].sum()
virus_selected_country


# In[ ]:


fig=px.line(virus_selected_country,x='ObservationDate',y='Confirmed',color='Country/Region')
fig.show()


# In[ ]:


fig=px.line(virus_selected_country,x='ObservationDate',y='Deaths',color='Country/Region')
fig.show()


# In[ ]:


fig=px.line(virus_selected_country,x='ObservationDate',y='Recovered',color='Country/Region')
fig.show()


# **So prevention is better than cure if infection rate can be controlled sooner then it would be easy to help with recovery ,so instructions can be followed as mentioned in the image below to stay safe from novel corona virus Please follow safety measures properly or else other nations will face trouble like Italy **
# ![image.png](attachment:image.png)

# In[ ]:




