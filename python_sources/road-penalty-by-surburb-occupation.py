#!/usr/bin/env python
# coding: utf-8

# > **LOADED DATA LIST**

# In[162]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import os
from IPython.display import HTML, display,Image
import json

import geopandas as gpd
from geopandas.tools import sjoin

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import shapely
from shapely.geometry import Point

import unicodedata
import pysal as ps


get_ipython().run_line_magic('matplotlib', 'inline')


# In[163]:


data1 = pd.read_csv('../input/penalty_data_set_sorted_rev2.csv')




# > **DATA HEAD OF PENALTIES**

# In[164]:


data1.head(20)


# > **FILTERING AND SORTING SPEEDING OFFENCE**
# *  Filter by 'SPEED_IND', Column will have value 'Y' for all speed camera related offence.

# In[165]:


df = pd.DataFrame(data1, columns= ['LOCATION_CODE','LOCATION_DETAILS','SPEED_IND','TOTAL_NUMBER'])
df1 = df[df.SPEED_IND.notnull()]

df1.head(20)


# >*  Sorted top 5 location codes which have most speeding penalties.
# >*  Grouped by 'LOCTION_CODE' to get Total number of penalties.
# >*  Highlighted to show clarity.

# In[166]:


#pd.merge(df1, on = ['LOCATION_CODE'])
#df1.set_index('LOCATION_CODE').stack()
df1_ar = df1.groupby(['LOCATION_CODE'], as_index=False).sum()
#df1_ar.set_index('LOCATION_CODE')
df1_rank = df1_ar.sort_values('TOTAL_NUMBER', ascending=False)
df1ten = df1_rank.head(10)


def highlight_column(x):
    y = 'background-color: cyan'
    df1ten_color = pd.DataFrame('', index=x.index, columns=x.columns)
    df1ten_color.iloc[:5, :2] = y
    return df1ten_color
df1ten.style.apply(highlight_column, axis=None)


# > *  Specified the Location Details which are based on Location Code
# > *  Coordinates were specified from website. (List of referencing site will be provided below)

#     1.  9666 - GREAT WESTERN HIGHWAY MOUNT VICTORIA WESTBOUND (-33.579721, 150.222532)
#     2.  9635 - EASTERN DISTRIBUTOR DARLINGHURST NORTHBOUND (-33.879264, 151.215236)
#     3.  9623 - GEORGES RIVER ROAD CROYDON PARK WESTBOUND (-33.897478, 151.099810)
#     4.  9838 - CLEVELAND STREET MOORE PARK (Z) WESTBOUND (-33.895468, 151.221482)
#     5.  9614 - CROSS CITY TUNNEL EAST SYDNEY WESTBOUND (-33.874218, 151.216292)
#     6.  9656 - LANE COVE TUNNEL LANE COVE WEST WESTBOUND (-33.806265, 151.152842)
#     7.  7295 - 7295 - ELIZABETH STREET SYDNEY NORTHBOUND (-33.873315, 151.209916)
#     8.  9817 - BOTANY ROAD ROSEBERY (Z) SOUTHBOUND (-33.925314, 151.195645)
#     9.  9597 - M1 PRINCES MOTORWAY GWYNNEVILLE NORTHBOUND (-34.414871, 150.881320)
#     10. 9655 - 9656 - LANE COVE TUNNEL LANE COVE WEST WESTBOUND (-33.806265, 151.152842)
#     
# 
#     

# > **FILTERING RED LIGHT CAMERA OFFENCE**
# > *  Filtered by 'RED_LIGHT_CAMERA_IND' which always have value 'Y', it won't have any values if not for Red Light Camera

# In[167]:


df = pd.DataFrame(data1, columns= ['LOCATION_CODE','LOCATION_DETAILS','RED_LIGHT_CAMERA_IND','TOTAL_NUMBER'])
df2 = df[df.RED_LIGHT_CAMERA_IND.notnull()]
df2.head(20)


# > *  Specifying Locations based on data - Red Light Cameras with most penalites
# > *  Sorted data by 'Total Number'
# > *  Highlighted Top 5 data

# In[168]:


df2_ar = df2.groupby(['LOCATION_CODE'], as_index=False).sum()
df2_rank = df2_ar.sort_values('TOTAL_NUMBER', ascending=False)
df2ten = df2_rank.head(10)

def highlight_column(x):
    y = 'background-color: cyan'
    df2ten_color = pd.DataFrame('', index=x.index, columns=x.columns)
    df2ten_color.iloc[:5, :2] = y
    return df2ten_color
df2ten.style.apply(highlight_column, axis=None)


# > *  Specified the Location Details which are based on Location Code
# > *  Coordinates were specified from website. (List of referencing site will be provided below)

#     1.  7244 - GEORGE STREET HAYMARKET EASTBOUND (-33.882848, 151.204195)
#     2.  7248 - STACEY STREET BANKSTOWN NORTHBOUND (-33.916676, 151.041053)
#     3.  7144 - WOODVILLE ROAD GRANVILLE NORTHBOUND (-33.827973, 151.004927)
#     4.  7305 - FALCON STREET NEUTRAL BAY WESTBOUND (-33.829647, 151.213025)
#     5.  7295 - ELIZABETH STREET SYDNEY NORTHBOUND (-33.873315, 151.209916)
#     6.  7297 - EPPING ROAD LANE COVE WESTBOUND (-33.810760, 151.163728)
#     7.  7127 - WOODVILLE ROAD VILLAWOOD SOUTHBOUND (-33.883777, 150.976361)
#     8.  7317 - PRINCES HIGHWAY ST PETERS NORTHBOUND (-33.907194, 151.181590)
#     9.  7315 - HUME HIGHWAY LIVERPOOL SOUTHBOUND (-33.928713, 150.918102)
#     10. 7177 - PENNANT HILLS ROAD THORNLEIGH NORTHBOUND (-33.730801, 151.081238)

# > **PLOT TOP 5 LOCATIONS ONTO MAP**
# >*  Each top 5 locations for Speed Cameras and Red Light Cameras.
# >*  iframe plugin used for popup styles - to utilise HTML format.

# In[ ]:


import folium
from folium.plugins import MarkerCluster
from folium.map import *
map_osm = folium.Map(location=[-33.877632, 151.082277])
map_osm.save('/tmp/map.html')




m = folium.Map(location=[-33.877632, 151.082277], tiles='Stamen Toner', zoom_start=11)
 
# SPEED Camera marker
S1="""
    <h1> </h1><br>
    GREAT WESTERN HIGHWAY, MT VICTORIA
    <p>
    <code>
        SPEED CAMERA<br>
        RANK: 1<br>
        PENALTIES: 19168
    </code>
    </p>
    """
iframeS1 = folium.IFrame(html=S1, width=350, height=150)
popupS1 = folium.Popup(iframeS1, max_width=1000)

S2="""
    <h1> </h1><br>
    STACEY STREET, BANKSTOWN
    <p>
    <code>
        SPEED CAMERA<br>
        RANK: 2<br>
        PENALTIES: 18077
    </code>
    </p>
    """
iframeS2 = folium.IFrame(html=S2, width=350, height=150)
popupS2 = folium.Popup(iframeS2, max_width=1000)

S3="""
    <h1> </h1><br>
    WOODVILLE ROAD, GRANVILLE
    <p>
    <code>
        SPEED CAMERA<br>
        RANK: 3<br>
        PENALTIES: 9703
    </code>
    </p>
    """
iframeS3 = folium.IFrame(html=S3, width=350, height=150)
popupS3 = folium.Popup(iframeS3, max_width=1000)

S4="""
    <h1> </h1><br>
    FALCON STREET, NEUTRAL BAY
    <p>
    <code>
        SPEED CAMERA<br>
        RANK: 4<br>
        PENALTIES: 8621
    </code>
    </p>
    """
iframeS4 = folium.IFrame(html=S4, width=350, height=150)
popupS4 = folium.Popup(iframeS4, max_width=1000)

S5="""
    <h1> </h1><br>
    ELIZABETH STREET, SYDNEY
    <p>
    <code>
        SPEED CAMERA<br>
        RANK: 5<br>
        PENALTIES: 8593
    </code>
    </p>
    """
iframeS5 = folium.IFrame(html=S5, width=350, height=150)
popupS5 = folium.Popup(iframeS5, max_width=1000)

folium.Marker([-33.579721, 150.222532], 
              popup=popupS1,
              icon=folium.Icon(color='blue')
             ).add_to(m)
folium.Marker([-33.879264, 151.215236], 
              popup=popupS2,
              icon=folium.Icon(color='blue')
             ).add_to(m)
folium.Marker([-33.897478, 151.099810], 
              popup=popupS3,
              icon=folium.Icon(color='blue')
             ).add_to(m)
folium.Marker([-33.895468, 151.221482], 
              popup=popupS4,
              icon=folium.Icon(color='blue')
             ).add_to(m)
folium.Marker([-33.874218, 151.216292], 
              popup=popupS5,
              icon=folium.Icon(color='blue')
             ).add_to(m)
    
    
    
# Red Light Camera marker
R1="""
    <h1> </h1><br>
    GEORGE STREET, HAYMARKET
    <p>
    <code>
        RED LIGHT CAMERA<br>
        RANK: 1<br>
        PENALTIES: 6855
    </code>
    </p>
    """
iframe1 = folium.IFrame(html=R1, width=350, height=150)
popup1 = folium.Popup(iframe1, max_width=1000)

R2="""
    <h1> </h1><br>
    STACEY STREET, BANKSTOWN
    <p>
    <code>
        RED LIGHT CAMERA<br>
        RANK: 2<br>
        PENALTIES: 6215
    </code>
    </p>
    """
iframe2 = folium.IFrame(html=R2, width=350, height=150)
popup2 = folium.Popup(iframe2, max_width=1000)

R3="""
    <h1> </h1><br>
    WOODVILLE ROAD, GRANVILLE
    <p>
    <code>
        RED LIGHT CAMERA<br>
        RANK: 3<br>
        PENALTIES: 4414
    </code>
    </p>
    """
iframe3 = folium.IFrame(html=R3, width=350, height=150)
popup3 = folium.Popup(iframe3, max_width=1000)

R4="""
    <h1> </h1><br>
    FALCON STREET, NEUTRAL BAY
    <p>
    <code>
        RED LIGHT CAMERA<br>
        RANK: 4<br>
        PENALTIES: 3965
    </code>
    </p>
    """
iframe4 = folium.IFrame(html=R4, width=350, height=150)
popup4 = folium.Popup(iframe4, max_width=1000)

R5="""
    <h1> </h1><br>
    ELIZABETH STREET, SYDNEY
    <p>
    <code>
        RED LIGHT CAMERA<br>
        RANK: 5<br>
        PENALTIES: 3930
    </code>
    </p>
    """
iframe5 = folium.IFrame(html=R5, width=350, height=150)
popup5 = folium.Popup(iframe5, max_width=1000)

folium.Marker([-33.882848, 151.204195], 
              popup=popup1,
              icon=folium.Icon(color='red')
             ).add_to(m)
folium.Marker([-33.916676, 151.041053], 
              popup=popup2,
              icon=folium.Icon(color='red')
             ).add_to(m)
folium.Marker([-33.827973, 151.004927], 
              popup=popup3,
              icon=folium.Icon(color='red')
             ).add_to(m)
folium.Marker([-33.829647, 151.213025], 
              popup=popup4,
              icon=folium.Icon(color='red')
             ).add_to(m)
folium.Marker([-33.873315, 151.209916], 
              popup=popup5,
              icon=folium.Icon(color='red')
             ).add_to(m)



m


# In[ ]:


data2 = pd.read_csv('../input/TZP2016 Employment by industry and travel zone 2011-2056.csv')


# >**DATA HEAD OF OCCUPATION**

# In[ ]:


data2.head(20)


# >**FILTERING AND SORTING DATA - MOUNT VICTORIA**
# >*  Mount Victoria has been analysed as a surburb with most speeding penalties

# In[ ]:


dfa = pd.DataFrame(data2, columns= ['TZ_NAME11','Industry','EMP_2016'])
dfa1 = dfa[dfa.TZ_NAME11.isin(['Mount Victoria Station'])]
dfa1



# > *  Employment in 2016 (EMP_2016) is the most recent dataset that we can refer to.
# > *  EMP_2016's data type set as string in default, therefore, changed to float for correct sorting.
# > *  Highlighted Top 4 Occupation from the table which used in py graph within rest of data values.

# In[ ]:


dfa1_ar = dfa1.sort_values('EMP_2016', ascending = False)
dfa1_rank = dfa1_ar.iloc[dfa1_ar['EMP_2016'].astype(float).argsort()]
dfa1_rank.head(35)


def highlight_column(x):
    y = 'background-color: cyan'
    dfa1_color = pd.DataFrame('', index=x.index, columns=x.columns)
    dfa1_color.iloc[range (29, 33), 2] = y
    return dfa1_color
dfa1_rank.style.apply(highlight_column, axis=None)


# In[ ]:


totala1 = dfa1_ar['EMP_2016'].astype(float).sum()
totala1


# **CONCLUSION - SPEEDING AT MOUNT VICTORIA**
# - Mount Victoria is a surburb where the Great Western Highway is passing through. Highways without traffic lights could encourage a temptation for speeding.
# - More than 40% of local people are working for Transport, Postal and Warehousing. These occupations require abilities of time management for delivery.

# In[ ]:


#Py for Mount Victoria
fig, ax = plt.subplots(figsize=(15, 7.5), subplot_kw=dict(aspect="equal"))

recipe1 = ["74.56804 Transport-Postal-Warehousing",
          "40.24344 Accommodation-Food_Services",
          "36.65257 Education-Training",
          "32.54229 Construction",
          "114.72754999999997 Rest"]

data3 = [float(x.split()[0]) for x in recipe1]
ingredients1 = [x.split()[-1] for x in recipe1]


def func(pct, allvals):
    absolute1 = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute1)


wedges, texts, autotexts = ax.pie(data3, autopct=lambda pct: func(pct, data3),
                                  textprops=dict(color="w"))

ax.legend(wedges, ingredients1,
          title="Occupation List",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

ax.set_title("OCCUPATION AT MT VICTORIA")

plt.show()


# >**FILTERING AND SORTING DATA - HAYMARKET**
# >*  Haymarket has been analysed as a surburb with most red light penalties

# In[ ]:


dfb = pd.DataFrame(data2, columns= ['TZ_NAME11', 'SA2_NAME11','Industry','EMP_2016'])
dfb1 = dfb[dfb.SA2_NAME11.isin(['Sydney - Haymarket - The Rocks'])]
# dfb2 = dfb1[dfb1.TZ_NAME11.isin(['Hay'])]
# dfb2 = dfb1[(dfb1.SA2_NAME11 == 'Sydney - Haymarket - The Rocks') & (dfb1.TZ_NAME11 == 6)]

dfb1.head(20)


# > *  Narrowing down ther filter by using a keyword 'Chinatown

# In[ ]:


dfb2 = dfb1[dfb1['TZ_NAME11'].str.contains("Chinatown")]
dfb2_ar = dfb2.sort_values('EMP_2016', ascending = False)
# dfb2_group = dfb2_ar.groupby(['Industry'], as_index=False).sum()  #THIS DOESN\"T WORK, IT WILL CONCACTENATE THE STRINGS. NEED TO FIND A WAY TO ADD EMP_2016 INTO NEW COLUMN WITH FLOAT TRANSLATED. This is only to give blue highlights as well.
dfb2_ar.head(25)


# >*  EMP_2016's data type is set to spring from default, selectively changed to numeric by using a function 'pd.to_numeric'

# In[ ]:


dfb2_ar[['EMP_2016']] = dfb2_ar[['EMP_2016']].apply(pd.to_numeric)
dfb2_ar
dfb2_rank = dfb2_ar.sort_values('EMP_2016', ascending=False)
dfb2_rank


# > *  Grouped data by 'INDUSTRY' - Location name would not be a matter anymore since data has been already filtered only for Haymarket.
# > *  Used a fuction, '.sum()' - Only works for numeric data types, in this case, EMP_2016.

# In[ ]:


dfb2_group = dfb2_rank.groupby(['Industry'], as_index=False).sum()
dfb2_group


# >*  Sorted data by EMP_2016
# >*  Highlighted Top 4 Occupation from the table which used in py graph within rest of data values

# In[ ]:


dfb2_sorted = dfb2_group.sort_values('EMP_2016', ascending=False)
dfb2_sorted

def highlight_column(x):
    y = 'background-color: cyan'
    dfb1_color = pd.DataFrame('', index=x.index, columns=x.columns)
    dfb1_color.iloc[:4, :2] = y
    return dfb1_color
dfb2_sorted.style.apply(highlight_column, axis=None)


# In[ ]:


totalb1 = dfb2_sorted['EMP_2016'].astype(float).sum()
totalb1


# >**CONCLUSION - NOT STOPPING AT HAYMARKET**
# >- Haymarket is a surburb where all sort of shops and services are. Within its massive floating population, the traffic congestion is your best friend when you are driving down the streets of Haymarket. Spending half an hour in the traffic would definitely make you feel hesitated.
# >- More than 40% of local people are working for Transport, Postal and Warehousing. These occupations require abilities of time management for delivery.

# In[ ]:


#Py for Haymarket
fig, ax = plt.subplots(figsize=(15, 7.5), subplot_kw=dict(aspect="equal"))

recipe2 = ["687.159 Accommodation-Food_Services",
          "175.325 Retail-Trade",
          "158.261 Professional-Scientific-Technical_Services",
          "98.3247 Health_Care-Social_Assistance",
          "308.34503 Rest"]

data4 = [float(x.split()[0]) for x in recipe2]
ingredients2 = [x.split()[-1] for x in recipe2]


def func(pct, allvals):
    absolute2 = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d})".format(pct, absolute2)


wedges, texts, autotexts = ax.pie(data4, autopct=lambda pct: func(pct, data4),
                                  textprops=dict(color="w"))

ax.legend(wedges, ingredients2,
          title="Occupation List",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

plt.setp(autotexts, size=8, weight="bold")

ax.set_title("OCCUPATION AT HAYMARKET")

plt.show()


# >**REFLECTION : THINGS SHOULD BE CONSIDERED TO DERIVE A BETTER ANALYSIS**
# >- Location of Cameras do not exactly indicate the people's destination (Where they are from, where they are going)
# >- Sometimes people do speeding over the limits and not stoping at Red lights - The dataset we are utilising may not specify this scenario and recorded promptly
# >- It is hard to specify the precise coordinate of cameras becasue the described locations of camera in referred website were not clear enough - ex. Great Western Highway, between Mitchells Lookout Road and Ambermere Drive of Mount Victoria
# >- Cannot prove that crimes were coming from local people or travellers - I assumed that local people must be acknowledged well about cameras, therefore, there might be more chance for travellers who did not abide the road rules. However, Mount Victoria's occupational research provided different data.
# >- Latest Occupation data is based on Year 2016. Penalty data of Year 2015-2016 or 2016-2017 should be utilised rather than 2017-2018.

# >**REFERENCING WEBSITES**
# >*  Transport Road Safety Website to specify Speed Camera Locations      
#     (http://roadsafety.transport.nsw.gov.au/speeding/speedcameras/current-locations.html)
# >*  NSWRed Light Cameras Website to specify Red Light Camera Locations 
#     (http://redlightcamera.tripod.com/nsw.html) (http://www.photoenforced.com/Sydney.html#.WwmmGoiFOUk)
# >*  Google Map to specify Coordinate of Cameras
#     (http://maps.google.com.au)
# >*  All the magical spells were found from the sea, Stack Overflow
#     (http://www.stackoverflow.com/questions)

# 2018 Some rights reserved | Ewan Chong

# In[ ]:




