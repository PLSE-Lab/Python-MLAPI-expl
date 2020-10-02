#!/usr/bin/env python
# coding: utf-8

# This is my attempt to visualize the air pollution in Seoul, South Korea. I am using the data from [air-pollution-in-seoul](https://www.kaggle.com/bappekim/air-pollution-in-seoul) provided by [@bappekim](https://www.kaggle.com/bappekim).

# ## Importing Libraries and Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
import folium
import warnings
warnings.filterwarnings('ignore')


# There are multiple csv files available. Lets open the measurement summary

# In[ ]:


pol_data = pd.read_csv("/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv")
pol_data.head()


# In[ ]:


pol_data.shape


# There are 11 columns and 647511 rows in the dataset

# In[ ]:


pol_data.isnull().sum()


# There are no null values in the data. Lets see the distribution.

# In[ ]:


pol_data[['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5']].describe()


# Here we can see that the minimum value was -1 in some cases. That is not an acceptable value as there is nothing like negative pollution. This could be a measurement error. Lets count the number of occurrances of this.

# In[ ]:


print("We have", pol_data['SO2'].loc[(pol_data['SO2']<0)].count(),"negative values for SO2")
print("We have", pol_data['NO2'].loc[(pol_data['NO2']<0)].count(),"negative values for NO2")
print("We have", pol_data['O3'].loc[(pol_data['O3']<0)].count(),"negative values for O3")
print("We have", pol_data['CO'].loc[(pol_data['CO']<0)].count(),"negative values for CO")
print("We have", pol_data['PM10'].loc[(pol_data['PM10']<0)].count(),"negative values for PM10")
print("We have", pol_data['PM2.5'].loc[(pol_data['PM2.5']<0)].count(),"negative values for PM2.5")


# In[ ]:


# https://www.kaggle.com/ramontanoeiro/seoul-air-pollution
data = [go.Scatter(x=pol_data['Measurement date'],
                   y=pol_data['SO2'], name='SO2'),
        go.Scatter(x=pol_data['Measurement date'],
                   y=pol_data['NO2'], name='NO2'),
        go.Scatter(x=pol_data['Measurement date'],
                   y=pol_data['CO'], name='CO'),
        go.Scatter(x=pol_data['Measurement date'],
                   y=pol_data['O3'], name='O3')]
       
##layout object
layout = go.Layout(title='Gases Levels',
                    yaxis={'title':'Level (ppm)'},
                    xaxis={'title':'Date'})
                    
## Figure object

fig = go.Figure(data=data, layout=layout)

## Plotting
py.iplot(fig)


# In[ ]:


data = pol_data[pol_data['SO2']<0]


# In[ ]:


data[['SO2','NO2','O3','CO','PM10','PM2.5']].describe()


# It looks like most of this are occurring in same date, as we can see that the count is same and the mean is almost near to -1 in most of the columns. We can use imputation to replace these values with the mean.

# In[ ]:


#https://www.kaggle.com/bappekim/visualizing-the-location-of-station-using-folium
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=-1, strategy='mean')
df_imputed = pd.DataFrame(imp.fit_transform(pol_data[["SO2","NO2","O3","CO","PM10","PM2.5"]]))
df_imputed.columns = pol_data[["SO2","NO2","O3","CO","PM10","PM2.5"]].columns
df_imputed.index = pol_data.index
remain_df = pol_data[pol_data.columns.difference(["SO2","NO2","O3","CO","PM10","PM2.5"])]
df = pd.concat([remain_df, df_imputed], axis=1)
df.head()


# In[ ]:


# #https://pandas.pydata.org/pandas-docs/version/0.17.0/generated/pandas.core.groupby.GroupBy.last.html
#TODO : Implement the time series with folium
last_entry = df.groupby('Station code').max() #here max is used just to get all type of pointers in the maps
# # last_entry.apply(lambda x: x.sample())
last_entry


# Now we need to know about the levels of the above chemicals that are good and bad.

# In[ ]:


safe_limit = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')
safe_limit


# The get color function return a color based on the level of polution of each chemical

# In[ ]:


#https://stackoverflow.com/a/16729808
def get_colors(data, safe_limit, item):
    item_row = safe_limit.loc[safe_limit['Item name'] == item]
    if (data > item_row.iloc[0]['Very bad(Red)']):
        return 'red'
    elif (data > item_row.iloc[0]['Bad(Yellow)']):
        return 'yellow'
    elif (data > item_row.iloc[0]['Normal(Green)']):
        return 'green'
    else:
        return 'blue'


# We are adding additional columns in the last_entry dataframe for representation purpose.

# In[ ]:


last_entry['SO2 Color'] = last_entry['SO2'].apply(get_colors, args =(safe_limit, 'SO2' )) 
last_entry['NO2 Color'] = last_entry['NO2'].apply(get_colors, args =(safe_limit, 'NO2' )) 
last_entry['O3 Color'] = last_entry['O3'].apply(get_colors, args =(safe_limit, 'O3' )) 
last_entry['CO Color'] = last_entry['CO'].apply(get_colors, args =(safe_limit, 'CO' )) 
last_entry['PM10 Color'] = last_entry['PM10'].apply(get_colors, args =(safe_limit, 'PM10' )) 
last_entry['PM2.5 Color'] = last_entry['PM2.5'].apply(get_colors, args =(safe_limit, 'PM2.5' )) 
last_entry


# Lets plot the map showing the level of SO2

# ## Pollution of SO2

# In[ ]:


# This creates the map object
m = folium.Map(
    location=[37.541, 126.981], # center of where the map initializes
    #tiles='Stamen Toner', # the style used for the map (defaults to OSM)
    zoom_start=11, # the initial zoom level
    title = "Pollution level of SO2") 
for ind in last_entry.index: 
    #print(row[1][0])
    folium.Marker([last_entry['Latitude'][ind], last_entry['Longitude'][ind]], popup=ind, icon=folium.Icon(color=last_entry['SO2 Color'][ind], icon='info-sign')).add_to(m)

# Diplay the map
m


# ## Pollution of NO2

# In[ ]:


# This creates the map object
m = folium.Map(
    location=[37.541, 126.981], # center of where the map initializes
    #tiles='Stamen Toner', # the style used for the map (defaults to OSM)
    zoom_start=11, # the initial zoom level
    title = "Pollution level of NO2") 
for ind in last_entry.index: 
    #print(row[1][0])
    folium.Marker([last_entry['Latitude'][ind], last_entry['Longitude'][ind]], popup=ind, icon=folium.Icon(color=last_entry['NO2 Color'][ind], icon='info-sign')).add_to(m)

# Diplay the map
m


# ## Pollution of O3

# In[ ]:


# This creates the map object
m = folium.Map(
    location=[37.541, 126.981], # center of where the map initializes
    #tiles='Stamen Toner', # the style used for the map (defaults to OSM)
    zoom_start=11, # the initial zoom level
    title = "Pollution level of O3") 
for ind in last_entry.index: 
    #print(row[1][0])
    folium.Marker([last_entry['Latitude'][ind], last_entry['Longitude'][ind]], popup=ind, icon=folium.Icon(color=last_entry['O3 Color'][ind], icon='info-sign')).add_to(m)

# Diplay the map
m


# ## Pollution of CO

# In[ ]:


# This creates the map object
m = folium.Map(
    location=[37.541, 126.981], # center of where the map initializes
    #tiles='Stamen Toner', # the style used for the map (defaults to OSM)
    zoom_start=11, # the initial zoom level
    title = "Pollution level of CO") 
for ind in last_entry.index: 
    #print(row[1][0])
    folium.Marker([last_entry['Latitude'][ind], last_entry['Longitude'][ind]], popup=ind, icon=folium.Icon(color=last_entry['CO Color'][ind], icon='info-sign')).add_to(m)

# Diplay the map
m


# ## Pollution of PM10

# In[ ]:


# This creates the map object
m = folium.Map(
    location=[37.541, 126.981], # center of where the map initializes
    #tiles='Stamen Toner', # the style used for the map (defaults to OSM)
    zoom_start=11, # the initial zoom level
    title = "Pollution level of PM10") 
for ind in last_entry.index: 
    #print(row[1][0])
    folium.Marker([last_entry['Latitude'][ind], last_entry['Longitude'][ind]], popup=ind, icon=folium.Icon(color=last_entry['PM10 Color'][ind], icon='info-sign')).add_to(m)

# Diplay the map
m


# ## Pollution of PM2.5

# In[ ]:


# This creates the map object
m = folium.Map(
    location=[37.541, 126.981], # center of where the map initializes
    #tiles='Stamen Toner', # the style used for the map (defaults to OSM)
    zoom_start=11, # the initial zoom level
    title = "Pollution level of PM2.5") 
for ind in last_entry.index: 
    #print(row[1][0])
    folium.Marker([last_entry['Latitude'][ind], last_entry['Longitude'][ind]], popup=ind, icon=folium.Icon(color=last_entry['PM2.5 Color'][ind], icon='info-sign')).add_to(m)

# Diplay the map
m


# - TODO : Add time slider in folium maps
