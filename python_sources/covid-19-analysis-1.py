#!/usr/bin/env python
# coding: utf-8

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


get_ipython().system('pip install geocoder')


# In[ ]:


# Imports 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import folium
from folium import plugins
import geocoder
import geopy
import ipywidgets 
import json
matplotlib.rc("xtick",labelsize=10)
matplotlib.rc("ytick",labelsize=10)


# In[ ]:


# Data File having Confirmed Cases around the globe
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")
# Data File having Confirmed Cases of India till 18th March 2020.
train_India = pd.read_csv("/kaggle/input/covid19-corona-virus-india-dataset/complete.csv")
# Data File having Confirmed Cases of India till 21st March 2020.
# This data has been collected by me from https://www.mohfw.gov.in/
train_Updated_India = pd.read_csv("/kaggle/input/updateddata/new_data.csv")


# In[ ]:


train.head()


# ## Plotting between Confirmed Cases with respect to dates .

# In[ ]:


confirmed_cases = [c for c in train['ConfirmedCases'].values if c!=0]  


# In[ ]:


# Getting the corresponding dates
dates = [train.loc[train["ConfirmedCases"]!=0,"Date"]]


# In[ ]:


# Getting dates in format dd/mm/YY.
new_dates = [pd.to_datetime(d) for d in dates]
new_dates  = [d.dt.strftime("%d/%m/%Y") for d in new_dates]
# Converting dates into string 
new_dates = new_dates[0].to_string(index=False)
# Converting string into list 
new_dates = new_dates.split("\n")


# Getting the dates where the confirmed cases were more than <b>60000</b>.

# In[ ]:


imp_dates = [pd.to_datetime(d).dt.strftime("%d/%m/%Y").to_string(index=False).split("\n") for d in [train.loc[train["ConfirmedCases"]>60000,"Date"]]]
imp_dates


# In[ ]:


# Getting the dates where most number of confirmed Cases were recorded i.e 67800
train.loc[train["ConfirmedCases"]==67800,"Date"]


# In[ ]:


# Having got the values of dates where corresponding Confirmed Cases were non-zero 
# We are ready to plot our observations
plt.plot(range(len(confirmed_cases)),confirmed_cases,c="darkblue")
plt.ylim(0,max(confirmed_cases))
# Manual Plotting the date when the confirmed cases were 67800.
plt.xticks([1600],["18-03-2020"],rotation=90)
plt.show()


# In[ ]:


# Here I have manually figured out the date at which confirmed cases were 67800.


# ### As we can see from the graph we have confirmed cases more than 60000 for a particular country/region and as we can see below the country is CHINA.

# In[ ]:


# If India is in data then printing the confirmed cases
train.loc[train["Country/Region"]=="India","ConfirmedCases"]


# In[ ]:


# Getting total number of confirmed cases in India
confirmed_India = train.loc[train["Country/Region"]=="India","ConfirmedCases"]
confirmed_India.to_string(index=False).split("\n")
confirmed_India = [int(e) for e in confirmed_India]
print(sum(confirmed_India))


# ## Visualizing the Number of confirmed cases(with dates) in India Only

# In[ ]:


dates_India = [pd.to_datetime(d).dt.strftime("%d/%m/%Y").to_string(index=False).split("\n") for d in [train.loc[train["Country/Region"]=="India","Date"]]]


# In[ ]:


dates_India = dates_India[0]
dates_India


# In[ ]:


plt.figure(figsize=(60,20))
matplotlib.rc("xtick",labelsize=30)
matplotlib.rc("ytick",labelsize=30)
plt.bar(range(len(dates_India)),confirmed_India)
loc,labels = plt.xticks()
plt.xticks(ticks=range(0,len(dates_India)),labels=dates_India,rotation=90)
plt.ylabel("Confirmed Cases India",fontsize=50,labelpad=30)
plt.xlabel("Dates",fontsize=50,labelpad=30)
plt.show()


# ### Making World Map Giving Info about the Confirmed Cases 

# ### Visualizing Map of India for COVID-19 Confirmed Cases

# In[ ]:


with open("/kaggle/input/indiadataset/india.json") as f:
    geo_data = json.load(f)


# In[ ]:


india_map = folium.Map(location=[20.5937,78.9629],zoom_start=4.48)
folium.TopoJson(geo_data,"objects.india_pc_2014",name="geojson").add_to(india_map)
folium.LayerControl().add_to(india_map)
india_map


# In[ ]:


#Making a choropleth Map
india_map1 = folium.Map(location=[20.5937,78.9629],zoom_start=4.4,tiles="CartoDB Positron")
folium.Choropleth(geo_data=geo_data,data=train_India,columns=["Name of State / UT","Total Confirmed cases (Indian National)"],
                  fill_color="YlGn",topojson="objects.india_pc_2014").add_to(india_map1)
india_map1


# #### Below is the district wise visualization of India . If You have Data Regarding districts of India then you can use this code to visualize that .

# In[ ]:


india_map2 = folium.Map(location=[20.5937,78.9629]) 
folium.GeoJson("/kaggle/input/indiageojson/india_district.geojson",name="Geojson").add_to(india_map2)
india_map2


# In[ ]:


# Making Choropleth for districts of India
india_map4 = folium.Map(location=[20.5937,78.9629],zoom_start=4.4)
with open("/kaggle/input/indiageojson/india_district.geojson") as f:
    india_geojson = json.load(f)

with open("/kaggle/input/indiadataset/india.json") as f:
    india_json = json.load(f)

folium.Choropleth(geo_data=india_json,
                    data_out="data.json",
                 data = train_India,
                 topojson="objects.india_pc_2014",
                 fill_color="YlGn",
                 columns=["Name of State / UT","Total Confirmed cases (Indian National)"],
                 legend_name="Confirmed Cases",
                 name="COVID-19",
                 highlight=True).add_to(india_map4)
india_map4


# ### Since the updated dataset does not contain information about latitudes and longitudes of different states hence creating one.

# In[ ]:


latitudes = [15.9129,22.09042035,28.6699929,22.2587,28.45000633,31.10002545,12.57038129,8.900372741,21.30039105,19.25023195,19.82042971,11.93499371,31.51997398,26.44999921,12.92038576,18.1124,30.71999697,34.29995933,34.152588,27.59998069,30.32040895,22.58039044,																]
longitudes = [79.7400,82.15998734,77.23000403,71.1924,77.01999101,77.16659704,76.91999711,76.56999263,76.13001949,73.16017493,85.90001746,79.83000037,75.98000281,74.63998124,79.15004187,79.0193,76.78000565,74.46665849,77.577049,78.05000565,78.05000565,88.32994665]


# In[ ]:


len(longitudes)


# In[ ]:


train_Updated_India["Name of State / UT"]


# In[ ]:


temp_India = train_Updated_India


# In[ ]:


temp_India["latitude"] = latitudes
temp_India['longitude'] = longitudes


# In[ ]:


with open("/kaggle/input/statesjson/india-states.json") as f:
    india_states = json.load(f)   


# In[ ]:


# Map that has markers at different states giving info about confirmed cases corresponding states.
new = folium.Map(location=[20.5937,78.9629],zoom_start=4.48)
new.choropleth(geo_data=india_states,data_out="data.json",fill_color="YlGn",data=temp_India,columns=["Name of State / UT","Total Confirmed cases (Indian National)"],
        topojson="objects.IND_adm1")
for i in temp_India.itertuples():
  folium.Marker(location=[i.latitude,i.longitude],popup=f"State:{i._2} \n Confirmed Cases:{i._3}").add_to(new)
new


# In[ ]:


# Saving our map
new.save("India_COVID-19-2103.html")


# In[ ]:


sub = temp_India.to_csv("submission.csv")


# In[ ]:




