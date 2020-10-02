#!/usr/bin/env python
# coding: utf-8

# # Australian Bushfire - Map analysis
# 
# The Australian bushfire has led to a massive loss to wildlife, forest area and has even caused human casualties, including firefighters from the U.S. It has even affected the air quality in nearby cities such as Sydney and Melbourne. We will take a look at fire data obtained from NASA satellite's MODIS and VIIRS.
# 
# What is covered - 
# - Regions with Highest recorded fire radiation in a day
# - Dates on which bushfires were at a peak.
# - Timeline of bushfire - barplot
# - A heat map with time - for Australian bushfire
# - Canberra Fire over the last 10 days
# - Kangaroo island fire
# 
# Note :
# - Also since the loading time could be high we will only consider data for the last 3 months - Nov 1, 2019 to Jan 31, 2020.

# ## Install dependencies and set file path

# Some [issues](https://github.com/python-visualization/folium/issues/812) with branca and folium HeatMapWithTime were resolved recenlty (by [destein](https://github.com/dstein64) and [sknzl](https://github.com/sknzl)),
# update them using the following - (keep internet option on in settings)

# In[ ]:


get_ipython().system('pip install git+https://github.com/python-visualization/branca')
get_ipython().system('pip install git+https://github.com/sknzl/folium@update-css-url-to-https')


# In[ ]:


#dependencies
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plotting
import seaborn as sns #for beatiful visualization
import folium
from folium import plugins

#set file path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


folium.__version__


# In[ ]:


folium.branca.__version__


# ## Load the data

# In[ ]:


fire_nrt_m6 = pd.read_csv("../input/australian-bush-fire-satellite-data-nasa/fire_nrt_M6_101673.csv")
fire_archive_m6 = pd.read_csv("../input/australian-bush-fire-satellite-data-nasa/fire_archive_M6_101673.csv")
fire_nrt_v1 = pd.read_csv("../input/australian-bush-fire-satellite-data-nasa/fire_nrt_V1_101674.csv")
fire_archive_v1 = pd.read_csv("../input/australian-bush-fire-satellite-data-nasa/fire_archive_V1_101674.csv")

type(fire_nrt_v1)


# Since VIIRS provides more spatial resolution(375m), We will be using VIIRS for further visualization and analysis.

# ## Merge archive and nrt data
# 
# Archive data is between Sep 1,2019 to Dec 31,2019.
# Nrt is between jan 1,2020 to jan 31,2020.
# 
# We will be merging both the data

# In[ ]:


df_merged = pd.concat([fire_archive_v1,fire_nrt_v1],sort=True)
data = df_merged
data.head()


# In[ ]:


data.info()


# We will be concentrating particularly on frp(Fire radiation power) which can detect bushfires

# ## Filter the data
# We will consider only 4 fields - latitude, longitude,acq_date and frp (fire radiation power) for this analysis.

# In[ ]:


df_filter = data.filter(["latitude","longitude","acq_date","frp"])
df_filter.head()


# - **Also since most of the fire activity occurred after September/November, and the complete data takes time to load in this notebook, we will filter the data between Nov 1, 2019 to Jan 31, 2020**

# In[ ]:


df = df_filter[df_filter['acq_date']>='2019-11-01']
df.head()


# ## Regions with Highest recorded fire radiation in a day

# In[ ]:


data_topaffected = df.sort_values(by='frp',ascending=False).head(10)
data_topaffected


# By reverse geocoding we can obtain the locations(Mentioned in Conclusion at the end).
# 
# **Below is the map marking the regions which were highest affected in a day**

# In[ ]:


#Create a map
m = folium.Map(location=[-35.0,144], control_scale=True, zoom_start=3,attr = "text some")
df_copy = data_topaffected.copy()

# loop through data to create Marker for each hospital
for i in range(0,len(df_copy)):
    
    folium.Marker(
    location=[df_copy.iloc[i]['latitude'], df_copy.iloc[i]['longitude']],
    #popup=popup,
    tooltip="frp: " + str(df_copy.iloc[i]['frp']) + "<br/> date: "+ str(df_copy.iloc[i]['acq_date']),
    icon=folium.Icon(color='red',icon='fire',prefix="fa"),
    ).add_to(m)
        
m


# ## Dates on which bushfires were at peak

# In[ ]:


dfdate = df[['acq_date','frp']].set_index('acq_date')
dfdate_highest = dfdate.groupby('acq_date').sum().sort_values(by='frp',ascending=False)
dfdate_highest.head(10)


# ## Timeline of bushfire - barplot

# - Note : this may take sometime to execute
# 

# In[ ]:


plt.figure(figsize=(10,5))
sns.set_palette("pastel")
ax = sns.barplot(x='acq_date',y='frp',data=df)
for ind, label in enumerate(ax.get_xticklabels()):
    if ind % 10 == 0:  # every 10th label is kept
        label.set_visible(True)
    else:
        label.set_visible(False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.xlabel("Date")
plt.ylabel('FRP (fire radiation power)')
plt.title("time line of bushfire in Australia")
plt.tight_layout()


# - The above barplot represents the progress of fire from Nov 1, 2019 to jan 31, 2020
# - You can notice three big spikes after 30th Dec, representing highest frp activity
# 

# ## Heat map with time - for Australian bushfire

# In[ ]:


from folium.plugins import HeatMapWithTime
# A small function to get heat map with time given the data

def getmap(ip_data,location,zoom,radius):
    
    #get day list
    dfmap = ip_data[['acq_date','latitude','longitude','frp']]
    df_day_list = []
    for day in dfmap.acq_date.sort_values().unique():
        df_day_list.append(dfmap.loc[dfmap.acq_date == day, ['acq_date','latitude', 'longitude', 'frp']].groupby(['latitude', 'longitude']).sum().reset_index().values.tolist())
    
    # Create a map using folium
    m = folium.Map(location, zoom_start=zoom,tiles='Stamen Terrain')
    #creating heatmap with time
    HeatMapWithTime(df_day_list,index =list(dfmap.acq_date.sort_values().unique()), auto_play=False,radius=radius, gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}, min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(m)

    return m
getmap(df,[-27,132],3.5,3)


# - The above map gives heatmap with time
# - Play it at higher fps to increase speed

# ## Canbbera Fire over last 10 days

# In[ ]:


#df tail for the latest data
df_10days = df.tail(21500)
#Using getmap function to obtain map from above, location set to canberra
getmap(df_10days,[-35.6,149.12],8,3)


#  - You can see the red spot appearing in Canberra over last 4 days, indicating fire activity
# 

# ## Kangaroo Island fire

# In[ ]:


#Using getmap function to obtain map from above, location set to kangaroo island
getmap(df,[-36, 137.22],8.5,3)


# ## Conclusion:
# - Dates on which Bushfires were at a peak in Australia - 
#     - 4th January 2020.
#     - 30th december 2019.
#     - 8th January 2020. 
# - Regions with highest recorded fire radiation power in a day -
#     1. Abbeyard, Victoria on 8th Jan.
#     2. Flinders chase in Kangaroo island on 8th Jan
#     3. Ravine road, flinders chase in Kangaroo Island
#     4. Cobberas, Victoria on 4th Jan
#     5. West bay road, Flinders chase in kangaroo island on 8th Jan.
#    
# - Observations from the map - 
#    - Most of the fire activity in kangaroo island falls between 20th December to 10th January
#    - Fire activity near Capital Canberra has been observed from 25th January onwards
# ****
