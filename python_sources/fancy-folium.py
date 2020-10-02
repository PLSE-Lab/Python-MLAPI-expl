#!/usr/bin/env python
# coding: utf-8

# I've been wanting to play around with some mapping for a while now and thought this data set would be a good place.
# I can say I've been on a real journey with this! From knowing virtually nothing about mapping, having just played around with some plotly for a couple of MOOCs.
# <br>
# My main aim was to produce interactive maps and I started off with libraries that used shapefiles such as geoviews (http://geo.holoviews.org/), this was a great library that produced some nice interactive choropleth maps. However I soon learnt that by using shapefiles for Great Britain from the data.gov (https://data.gov.uk/data/) these were encoded in OSGB 1936 (https://en.wikipedia.org/wiki/Ordnance_Survey_National_Grid), where the coordinates are given in metres and many mapping libraries expect the shapefile to be WGS84, where coordinates are in degrees, and that was before dealing with projections...
# <br>
# I therefore settled on using folium, which didn't have to deal with projections as the map was provided, and you put your information into this, and plotting directly into Bokeh, which uses GeoJSON which avoids the issues with shapefiles. This was a great learning curve in the JSON format, and provided some great interactive choropleth maps. I will upload this as a separate notebook.

# Import Libraries

# In[ ]:


import folium
import branca
import pandas as pd
print(folium.__file__)
print(folium.__version__)


# Read in one of the csv files. As these are rather big (this one has almost half a million lines), then I'll only import the columns of interest.

# In[ ]:


df_2009_2011 = pd.read_csv('../input/2000-16-traffic-flow-england-scotland-wales/accidents_2009_to_2011.csv',
                           usecols=['Longitude','Latitude','Number_of_Vehicles',
                           'Number_of_Casualties','LSOA_of_Accident_Location',
                           'Day_of_Week','Light_Conditions','Weather_Conditions',
                           'Road_Surface_Conditions','Year','Date','Time'])
df_2009_2011.info()


# Again as there's a lot of data I will just focus on one specific area of interest. The LSOA is associated with a post code location so I will will narrow the DataFrame down to one of these locations, which happens to be central London (SW1A 1AA) and is around Buckingham Palace and the Mall.

# In[ ]:


df = df_2009_2011[(df_2009_2011['Year']==2010) & (df_2009_2011['LSOA_of_Accident_Location']=='E01004736')]
print(len(df))
df.head()


# The DataFrame contains a lat/long point for each reported accident, plus other information logged about this accident.
# <br>
# To start with simply each accident location can be added to a folium map.

# In[ ]:


#location is the mean of every lat and long point to centre the map.
location = df['Latitude'].mean(), df['Longitude'].mean()

#A basemap is then created using the location to centre on and the zoom level to start.
m = folium.Map(location=location,zoom_start=15)

#Each location in the DataFrame is then added as a marker to the basemap points are then added to the map
for i in range(0,len(df)):
    folium.Marker([df['Latitude'].iloc[i],df['Longitude'].iloc[i]]).add_to(m)
        
m


# Taking a minute to look at this map it appear sensible, roads that would expected to have a large volume of traffic such as Picadilly and Trafalgar Square have lots of markers whereas The Mall has very few.
# 
# It's also possible to add a popup with text to each location.

# In[ ]:


location = df['Latitude'].mean(), df['Longitude'].mean()
m = folium.Map(location=location,zoom_start=15)

for i in range(0,len(df)):
       
    popup = folium.Popup('Accident', parse_html=True) 
    folium.Marker([df['Latitude'].iloc[i],df['Longitude'].iloc[i]],popup=popup).add_to(m)
m


# The color and type of marker can also be changed.

# In[ ]:


#There are a number of accidents with multiple casualties
df['Number_of_Casualties'].value_counts()


# In[ ]:


location = df['Latitude'].mean(), df['Longitude'].mean()
m = folium.Map(location=location,zoom_start=15)

#The num of casulaties for each accident can be determined and the colour assigned then added to the basemap.
for i in range(0,len(df)):
    num_of_casualties = df['Number_of_Casualties'].iloc[i]
    if num_of_casualties == 1:
        color = 'blue'
    elif num_of_casualties == 2:
        color = 'green'
    else:
        color = 'red'
    
    popup = folium.Popup('Accident', parse_html=True) 
    folium.Marker([df['Latitude'].iloc[i],df['Longitude'].iloc[i]],popup=popup,icon=folium.Icon(color=color, icon='info-sign')).add_to(m)

m


# Now you can really go to town with the markers and add html in the popup boxes to make a nicely formated box.
# <br>
# This function is not complete as it would be great to make it more dynamic for other situations, however for a demonstration it serves it's purpose.

# In[ ]:


def fancy_html(row):
    i = row
    
    Number_of_Vehicles = df['Number_of_Vehicles'].iloc[i]                             
    Number_of_Casualties = df['Number_of_Casualties'].iloc[i]                           
    Date = df['Date'].iloc[i]
    Time = df['Time'].iloc[i]                                           
    Light_Conditions = df['Light_Conditions'].iloc[i]                               
    Weather_Conditions = df['Weather_Conditions'].iloc[i]                             
    Road_Surface_Conditions = df['Road_Surface_Conditions'].iloc[i]
    
    left_col_colour = "#2A799C"
    right_col_colour = "#C5DCE7"
    
    html = """<!DOCTYPE html>
<html>

<head>
<h4 style="margin-bottom:0"; width="300px">{}</h4>""".format(Date) + """

</head>
    <table style="height: 126px; width: 300px;">
<tbody>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Number of Vehicles</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Number_of_Vehicles) + """
</tr>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Casualties</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Number_of_Casualties) + """
</tr>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Time</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Time) + """
</tr>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Light Conditions</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Light_Conditions) + """
</tr>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Weather Conditions</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Weather_Conditions) + """
</tr>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Road Conditions</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Road_Surface_Conditions) + """
</tr>
</tbody>
</table>
</html>
"""
    return html


# Of note for the popup boxes above any string passed needs to be parsed as html to work. This is also true for this function, however an extra step has to be taken here with the html passed as an IFrame into the popup.

# In[ ]:


location = df['Latitude'].mean(), df['Longitude'].mean()
m = folium.Map(location=location,zoom_start=15,min_zoom=5)

for i in range(0,len(df)):
    html = fancy_html(i)
 
    iframe = branca.element.IFrame(html=html,width=400,height=300)
    popup = folium.Popup(iframe,parse_html=True)
    
    folium.Marker([df['Latitude'].iloc[i],df['Longitude'].iloc[i]],
                  popup=popup,icon=folium.Icon(color=color, icon='info-sign')).add_to(m)

m


# Now it would be cool to visualise this using other types of maps. Lets start with a heatmap.

# In[ ]:


data_heat = df[['Latitude','Longitude','Number_of_Casualties']].values.tolist()


# This produces a list of lists where each element is a list with the latitude, longitude and if required a weight.
# <br>
# The weight here is set to the number of casualties

# In[ ]:


data_heat[0]


# There are are number of plugins for folium (http://nbviewer.jupyter.org/github/python-visualization/folium/tree/master/examples/).
# These need to be imported first and then the data is simply added as a HeatMap.

# In[ ]:


import folium.plugins as plugins

m = folium.Map(location=location, zoom_start=15)
#tiles='stamentoner'

plugins.HeatMap(data_heat).add_to(m)

m


# This broadly corresponds with the previous maps with a large concentration of accidents around Picadilly Circus and Trafalgar square.
# <br>
# Now lets expand our horizons a little bit.
# <br>
# The LSOA (Lower Layer Super Output Area) is associated with each postcode in the UK, and this is the geographic area used in the accident csv. However the next layer of granulatity, the LAD (Local Authority District) is slightly larger and will cover a whole London Borough.
# <br>
# Fortunately there is a lookup table available at http://geoportal.statistics.gov.uk/datasets/3ecc1f604e0148fab8ea0b007dee4d2e_0

# In[ ]:


df_Areas = pd.read_csv('../input/ukregions/Output_Area_to_Local_Authority_District_to_Lower_Layer_Super_Output_Area_to_Middle_Layer_Super_Output_Area_to_Local_Enterprise_Partnership_April_2017_Lookup_in_England_V2.csv',
                       usecols=['LAD16CD','LSOA11CD','LAD16NM'])
df_Areas.head()
df_Areas = pd.DataFrame(df_Areas[df_Areas['LAD16NM']=='Westminster']['LSOA11CD'])


# The cell above returns all the LSOA11CD associated with the Westminter LAD. We can then take the accidents from 2010, rename the columns to allow the two DataFrames to be merged.

# In[ ]:


df_West = pd.DataFrame(df_2009_2011[df_2009_2011['Year']==2010])
df_West.rename(columns={'LSOA_of_Accident_Location':'LSOA11CD'},inplace=True)
df_West = pd.merge(df_West,df_Areas,on='LSOA11CD')
print(len(df_West))
df_West.head()


# Finally reset the location of the map to the centre of Westminster and create the data as for the previous heat map.

# In[ ]:


location = location = df_West['Latitude'].mean(), df_West['Longitude'].mean()
data = df_West[['Latitude','Longitude','Number_of_Casualties']].values.tolist()


# In[ ]:


m = folium.Map(location=location, zoom_start=13)

plugins.HeatMap(data).add_to(m)

m


# Wow, that's a big red blob. By zooming in it's possible to gain some insight, however it may be more useful to display the map in a slightly different way to allow a clearer picture. Folium allows a time dimension to be added to map.
# <br>
# Firstly if we convert the date column to datetime and then convert each date to a month, which will return an integer value between 1-12.

# In[ ]:


df_West['Date'] = pd.to_datetime(df_West['Date'])
df_West['Month'] = df_West['Date'].apply(lambda time: time.month)


# Then add each accident for a particular month to a list of lists.

# In[ ]:


data = [df_West[df_West['Month']==df_West['Month'].unique()[i]][['Latitude','Longitude']].values.tolist() 
        for i in range(len(df_West['Month'].unique()))]


# This list of lists contains 12 elements for each month, with each element containing a list of all the accident locations for that month.
# <br>
# The month numbers can then be assigned a more sensible value for the index of the animation.

# In[ ]:


monthDict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
            7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

index = [monthDict[i] for i in sorted(df_West['Month'].unique())]


# The heatmap is constructed in a similar fashion to the previous examples.
# <br>
# The difference is HeatMapWithTime is called with the new data list and the index as arguments.

# In[ ]:


m = folium.Map(location=location,zoom_start=12)
hm = plugins.HeatMapWithTime(data=data,index=index)

hm.add_to(m)

m


# I hope you enjoyed this notebook. Any questions or comments are more than welcome, as I'd be keen to look at ways to improve the outputs.

# In[ ]:





# In[ ]:




