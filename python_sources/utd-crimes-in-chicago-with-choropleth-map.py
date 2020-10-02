#!/usr/bin/env python
# coding: utf-8

# <center> <h1> <b> <font size="+5">Introduction</font></b></h1></center>
# <img src="https://i.imgur.com/Q7jO9Eb.png">
# <img src="https://i.imgur.com/DmpzmPD.png" alt="Smiley face" align="right">
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:blue; padding: 1em;">Hello, we are the famous <b>Chicago Bears football fans</b>. We have been assigned to assist with this presentation, with an explanation of code and analyzing data. This Kaggle was created for a class project and was a great learning tool. By publishing this Kaggle to the public, we hope you find the information provided to be informative and helpful.</p>

# <p><b> <font size="+2"> Table of Contents </font> </b></p>
# <p></p>
# <a href="#Importing_package and_dataset">Importing package and dataset</a>
# <p></p>
# <a href="#Data_Cleansing">Data Cleansing</a>
#     <UL>
#         <LI><a href="#Finding_Duplicate">Finding Duplicate</a>
#         <LI><a href="#Finding_missing_data">Finding Missing Data</a>
#     </UL>
# <a href="#Locating">Locating  crime in Chicago</a>
#     <UL>
#         <LI><a href="#Preparing_data_for_Choropleth_Map">Preparing data for Choropleth_Map</a>
#          <LI><a href="#Creating_Choropleth_Map">Creating 2016 Choropleth Map for Police wards and Districts</a>
#    </UL>
# <a href="#crime_rate_over_time"> Monitoring crimes rate over time</a>
#     <UL>
#          <LI><a href="#Community_areas">Community areas Choropleth Map</a>
#          <LI><a href="#Overview">Overview of crimes in Chicago</a>
#     </UL>
# <a href="#Narcotics">Taking a closer look at Narcotics</a>

# <p id="Importing_package and_dataset"> <b> <font size="+2"> Importing packages and dataset </font> </b></p>
# <p><b>numpy</b> and <b>pandas</b> are pretty standard package in python. There are many tutorial if you need to get familiarize on it.</p>
# <p> <b>folium</b> is a Python library that helps you create interactive leaflet maps</p>
# <p> Concatenate all dataset into a single dataframe</p>

# <img src="https://i.imgur.com/VgE45Xt.png" alt="Smiley face" align="right"><p style="border:4px; border-radius: 15px; border-style:solid; border-color:blue; padding: 1em;"><b>Concatenation</b> is used to appending one dataset to the end of the other dataset.<br>This way we can keep the records somewhat more organized by years.</p>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
import folium
from folium import IFrame, FeatureGroup, LayerControl, Map, Marker, plugins
import seaborn as sns
import matplotlib.pyplot as plt

Chicago_COORDINATES = (41.895140898, -87.624255632)

#crimes1 = pd.read_csv('../input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)
crimes2 = pd.read_csv('../input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv',error_bad_lines=False)
crimes3 = pd.read_csv('../input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv',error_bad_lines=False)
crimes4 = pd.read_csv('../input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv',error_bad_lines=False)
crimedata = pd.concat([crimes2, crimes3, crimes4], ignore_index=False, axis=0)

#Deleting dataframe as they are no longer needed.
del crimes2
del crimes3
del crimes4
#crimes4

crimedata.head()


# <p id="Data_Cleansing"><center> <h1> <b> <font size="+5">Data Cleansing</font></b></h1></center></p>
# 
# <img src="https://i.imgur.com/fwB68pF.png">
# 
# <p><center>Now that all the data has been combined into one dataframe, we will need to do some data cleansing</center></p>

# <p id="Finding_Duplicate"><b> <font size="+2"> Finding Duplicate </font> </b></p>
# 
# Let get a general idea of the shape of the combined dataset.

# In[ ]:


crimedata.shape


# Let find the total value of duplicate with the same <b>ID</b>

# In[ ]:


crimedata.ID.duplicated().sum()


# <img src="https://i.imgur.com/FzsTGPb.png" align="left">
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:orange; padding: 1em;">
# That is alot of duplicate data with the same <b>ID</b>.<br>
# We need to check if multiple crime commited can be listed under the same incidents <b>ID</b>. For example, if a person were to be committed of evading arrest and in possession of illegal substance. We need to know if it would be listed with different <b>ID</b> or the same <b>ID</b>.<br>
# So now we need to use <b>.duplicated()</b> on the whole row, and compared result</p>

# In[ ]:


crimedata.duplicated().sum()


# <img src="https://i.imgur.com/NKCXP7C.png" align="left">
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:orange; padding: 1em;">
# As you can see the number of duplicate <b>ID</b> count and duplicate rows of data count have the same value.<br>
# We can safely assume that multiple crime cannot be recorded under the same <b>ID</b>.<br>
# Let us drop the duplicate data from our crimedata dataframe, by using <b>.drop_duplicates()</b></p>

# In[ ]:


# Droping duplicate data
#(keep='first') allowed us to keep the first duplicate data in the dataframe and remove any duplicate data found after it.

crimedata.drop_duplicates(subset=None, keep='first', inplace=True)
#crimedata.drop_duplicates(subset=['ID', 'Case Number'], inplace=True)
crimedata.shape


# <p>Let re check the shape of the crimedata dataframe now that we have drop the duplicated.</p>
# <p>6017767 - 1681211 = 4336556.</p>
# <p>Look like the number check out</p>

# In[ ]:


null_data = crimedata[crimedata.isnull().any(axis=1)]
null_data.head(5) 


# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:blue; padding: 1em;">
# Missing data can be found in column: <b>Case Number, Location Description, Ward, Community Area, X Coordinate, Y Coordinate, Latitude, Longitude, Location</b><br><br> 
# I decided that column: <b>X Coordinate, Y Coordinate, Latitude, Longitude, Location,</b> will not be needed for our planned analysis and will be drop from the datafram<img src="https://i.imgur.com/DUj1oKw.png" align="right"><br><br>
# Other column that will be drop, as they won't be needed are: <b>'Case Number', 'Block', 'IUCR', 'Location Description', 'Arrest', 'Domestic', 'Beat','Updated On', 'FBI Code'</b><br><br>
# That will just leave us with column: <b>Ward, Community Area</b> with missing data that we have to deal with.</p>

# In[ ]:


# Droping column
crimedata = crimedata.drop(columns=['Unnamed: 0', 'Case Number', 'Block', 'IUCR', 'Arrest',
                                    'Domestic', 'Beat', 'Updated On', 'FBI Code', 'X Coordinate', 'Y Coordinate', 
                                    'Latitude', 'Longitude', 'Location'], axis = 1)
#'Location Description'
crimedata.tail()


# Let us find the percentage of missing data within our dataframe.

# In[ ]:


percent_missing = crimedata.isnull().sum()/ len(crimedata) * 100
percent_missing


# <img src="https://i.imgur.com/hhb6BIb.png" align="right">
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:blue; padding: 1em;">
# Having such low percentage of missing data in <b>District, Ward and Community Area</b>. Removing the row with missing data will have little impact on the overall result.</p>

# In[ ]:


crimedata = crimedata.dropna()
crimedata.isnull().sum()


# <p id="Locating"><center> <h1 id="Locating _crime_in_Chicago"> <b> <font size="+5">Locating  crime in Chicago</font></b></h1></center></p>
# 
# 
# <img src="https://i.imgur.com/UUXwQGg.png">
# 
# <p id="Preparing_data_for_Choropleth_Map"><b> <font size="+2"> Preparing data for Choropleth Map </font> </b></p>
# <p>By creating a choropleh map, we can see how crime rate differ from different areas.</p>
# <p>Before we can create a choropleh map, <b>District, Ward</b> and <b>Community</b> column contain decimal point within thier datas.</p>
# <p>we need to Get rid of decimal value so it can be match with the geojsom file  and be converted to choropleth map</p>
# <p>by converting the column to <b>int</b> type and back to <b>string</b> type, we can removed the decimal point values.</p>

# In[ ]:


crimedata.tail()


# In[ ]:


#getting rid of decimal in District, Ward and Community Area and turning them into string type.
crimedata[['District', 'Ward','Community Area']] = crimedata[['District', 'Ward','Community Area']].astype('int')
crimedata[['District', 'Ward','Community Area']] = crimedata[['District', 'Ward','Community Area']].astype('str')
crimedata.head()


# In[ ]:


# I decided to go with every 2 year for the Chicago community areas
#crimedata2005 = crimedata[crimedata["Year"]==2005]
crimedata2006 = crimedata[crimedata["Year"]==2006]
#crimedata2007 = crimedata[crimedata["Year"]==2007]
crimedata2008 = crimedata[crimedata["Year"]==2008]
#crimedata2009 = crimedata[crimedata["Year"]==2009]
crimedata2010 = crimedata[crimedata["Year"]==2010]
#crimedata2011 = crimedata[crimedata["Year"]==2011]
crimedata2012 = crimedata[crimedata["Year"]==2012]
#crimedata2013 = crimedata[crimedata["Year"]==2013]
crimedata2014 = crimedata[crimedata["Year"]==2014]
crimedata2015 = crimedata[crimedata["Year"]==2015]
crimedata2016 = crimedata[crimedata["Year"]==2016]


# space inbetween community area column is causing some problem, replaceing all space with underscore.

# In[ ]:


crimedata2006.columns = crimedata2006.columns.str.strip().str.lower().str.replace(' ', '_')
crimedata2008.columns = crimedata2008.columns.str.strip().str.lower().str.replace(' ', '_')
crimedata2010.columns = crimedata2010.columns.str.strip().str.lower().str.replace(' ', '_')
crimedata2012.columns = crimedata2012.columns.str.strip().str.lower().str.replace(' ', '_')
crimedata2014.columns = crimedata2014.columns.str.strip().str.lower().str.replace(' ', '_')
crimedata2016.columns = crimedata2016.columns.str.strip().str.lower().str.replace(' ', '_')


# In[ ]:


crimedata2016.head(5)


# In[ ]:


#definition of the boundaries in the map
district_geo = r'../input/boundaries-wards/Boundaries_Wards.geojson'

#calculating total number of incidents per district for 2016
WardData2016 = pd.DataFrame(crimedata2016['ward'].value_counts().astype(float))
WardData2016.to_json('Ward_Map.json')
WardData2016 = WardData2016.reset_index()
WardData2016.columns = ['ward', 'Crime_Count']
 
#creating choropleth map for Chicago District 2016
map1 = folium.Map(location=Chicago_COORDINATES, zoom_start=11)
map1.choropleth(geo_data = district_geo, 
                #data_out = 'Ward_Map.json', 
                data = WardData2016,
                columns = ['ward', 'Crime_Count'],
                key_on = 'feature.properties.ward',
                fill_color = 'YlOrRd', 
                fill_opacity = 0.7, 
                line_opacity = 0.2,
                threshold_scale=[0, 4000, 8000, 12000, 16000, 20000],
                legend_name = 'Number of incidents per police ward 2016')

#WardData2016.sort_values('Ward')


# <p id="Creating_Choropleth_Map"><center><b> <font size="+3"> Creating Choropleth Map </font> </b></center></p>
# 
# <p id="Police_wards_2016"><b> <font size="+2"> Number of incidents 2016 </font> </b></p>
# 
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:orange; padding: 1em;">
# 
# Chicago has drawn the Police ward and district boundries in 2012, but was not implemented untill may of 2015. Therefore we can only show the incident map for 2016. The data for 2017 is incomplete and only has data form the begining of the year. <br><img src="https://i.imgur.com/G7kDtAg.png" align="left"><br>
# <b>Police Ward Boundries data</b> can be found here <a href="https://data.cityofchicago.org/d/sp34-6z76">Boundries Ward</a><br>
# <b>Police District boundaries data</b>can be found here, <a href="https://data.cityofchicago.org/d/fthy-xz3r">Boundries District</a><br>
# <b>Police wards Number</b> can be found here <a href="https://www.sixthward.us/2011/12/back-of-yards-at-center-of-ward-remap.html">Boundries Ward Number</a></p>

# In[ ]:


map1


# In[ ]:


#definition of the boundaries in the map
district_geo = r'../input/chicago-police-district/Boundaries_Police_Districts.geojson'

district_data = pd.DataFrame(crimedata2016['district'].value_counts().astype(float))
district_data.to_json('District_Map.json')
district_data = district_data.reset_index()
district_data.columns = ['district', 'Crime_Count']

#creation of the choropleth
map2 = folium.Map(location=Chicago_COORDINATES, zoom_start=11)
map2.choropleth(geo_data = district_geo,  
                data = district_data,
                columns = ['district', 'Crime_Count'],
                key_on = "feature.properties.dist_num",
                fill_color = 'YlOrRd', 
                fill_opacity = 0.7, 
                line_opacity = 0.2,
                threshold_scale=[0, 4000, 8000, 12000, 16000, 20000],
                legend_name = 'Number of incidents per district 2016')


# In[ ]:


map2


# In[ ]:


from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)   
#display_side_by_side(WardData2016.sort_values('Crime_Count', ascending=True).tail(5),district_data.sort_values('Crime_Count', ascending=True).tail(5))


# <img src="https://i.imgur.com/NKCXP7C.png" align="left">
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:orange; padding: 1em;">
# You can see that <b>District</b> map is much darker than <b>Ward</b> map.<br>
# The reason for this is that the number of crime is being condense from 50 ward to 25 police district.</p>

# In[ ]:


WardData2016[['ward']] = WardData2016[['ward']].astype('int')
district_data[['district']] = district_data[['district']].astype('int')
display_side_by_side(WardData2016.sort_values('ward', ascending=True),district_data.sort_values('district', ascending=True))


# Just checking if 2016 Police district crime count is the same as 2016 Ward crime count.

# In[ ]:


Total = district_data['Crime_Count'].sum()
print (Total)


# In[ ]:


Total = WardData2016['Crime_Count'].sum()
print (Total)


# <p id="crime_rate_over_time"><center> <h1> <b> <font size="+5">Monitoring crime rate over time</font></b></h1></center></p>
# 
# <img src="https://i.imgur.com/bKlGTZU.png">
# <p><center>In recent years, overall recorded crime levels in chicago have seen a sustained decrease, falling to their lowest level. </center></p>

# In[ ]:


sns.countplot(x='Year',data=crimedata, color=('BLUE'))
fig = plt.gcf()
plt.ylabel('No of Crimes')
fig.set_size_inches(15,7)

plt.show()


# <img src="https://i.imgur.com/NKCXP7C.png" align="left">
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:orange; padding: 1em;">
# We can see that the crime rate has been falling each year, but have a dramatic drop off from the year 2008 to 2014. We need to investigate further to find the reason for this steep decline.<br>

# <p id="Community_areas"><b> <font size="+2"> Number of incidents per Community areas </font> </b></p>
# 
# <img src="https://i.imgur.com/VgE45Xt.png" align="right">
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:blue; padding: 1em;">
# The Social Science Research Committee at the University of Chicago have been defined community areas during the late 1920s.<br><br>
# <b>Community areas data</b> can be found here <a href="https://data.cityofchicago.org/d/cauq-8yn6">Community areas</a></p>

# In[ ]:


Community_Areas_geo = r'../input/chicago-community-areas/Chicago_Community_Areas.geojson'
# Community_Areas map 2016
Community_Areas_data2016 = pd.DataFrame(crimedata2016['community_area'].value_counts().astype(float))
Community_Areas_data2016.to_json('Community_Area_Map2016.json')
Community_Areas_data2016 = Community_Areas_data2016.reset_index()
Community_Areas_data2016.columns = ['community_area', 'Crime_Count']


map2016 = folium.Map(location=Chicago_COORDINATES, zoom_start=11)
#map2016.add_child(feature_group)
#map8.add_children(folium.map.LayerControl())

map2016.choropleth(geo_data = Community_Areas_geo,
                data = Community_Areas_data2016,
                columns = ['community_area', 'Crime_Count'],
                key_on = "feature.properties.area_numbe",
                fill_color = 'YlOrRd', 
                fill_opacity = 0.7, 
                line_opacity = 0.2,
                threshold_scale=[0, 4000, 8000, 12000, 16000, 20000],
                legend_name = 'Number of incidents per community area 2016')
               

folium.TileLayer('cartodbdark_matter').add_to(map2016)
folium.TileLayer('Stamen Terrain').add_to(map2016)
folium.TileLayer('Stamen Toner').add_to(map2016)
folium.TileLayer('Mapbox Bright').add_to(map2016)


#Marker coordnate for each Comunity area

# Next time use loop, you dummy
# you you Super Dummy
feature_group = FeatureGroup(name='Comunity area number')
feature_group.add_child(Marker([42.01,-87.67],'Comunity area 1, ROGERS PARK'))
feature_group.add_child(Marker([42.0, -87.70],'Comunity area 2, WEST RIDGE'))
feature_group.add_child(Marker([41.965,-87.655],'Comunity area 3, UPTOWN'))
feature_group.add_child(Marker([41.975, -87.685],'Comunity area 4, LINCOLN SQUARE'))
feature_group.add_child(Marker([41.95, -87.685],'Comunity area 5, NORTH CENTER'))
feature_group.add_child(Marker([41.94, -87.655],'Comunity area 6, LAKE VIEW'))
feature_group.add_child(Marker([41.92, -87.655],'Comunity area 7, LINCOLN PARK'))
feature_group.add_child(Marker([41.9, -87.632],'Comunity area 8, NEAR NORTH SIDE'))
feature_group.add_child(Marker([42.006, -87.815],'Comunity area 9, EDISON PARK'))
feature_group.add_child(Marker([41.987, -87.8],'Comunity area 10, NORWOOD PARK'))
feature_group.add_child(Marker([41.98, -87.769],'Comunity area 11, JEFFERSON PARK'))
feature_group.add_child(Marker([41.987, -87.752],'Comunity area 12, FOREST GLEN'))
feature_group.add_child(Marker([41.985, -87.72],'Comunity area 13, NORTH PARK'))
feature_group.add_child(Marker([41.965, -87.72],'Comunity area 14, ALBANY PARK'))
feature_group.add_child(Marker([41.95, -87.764],'Comunity area 15, PORTAGE PARK'))
feature_group.add_child(Marker([41.954, -87.725],'Comunity area 16, IRVING PARK'))
feature_group.add_child(Marker([41.945, -87.808],'Comunity area 17, DUNNING'))
feature_group.add_child(Marker([41.927, -87.8],'Comunity area 18, MONTCLARE'))
feature_group.add_child(Marker([41.925, -87.765],'Comunity area 19, BELMONT CRAGIN'))
feature_group.add_child(Marker([41.925, -87.73501],'Comunity area 20, HERMOSA'))
feature_group.add_child(Marker([41.938, -87.71],'Comunity area 21, AVONDALE'))
feature_group.add_child(Marker([41.923, -87.7],'Comunity area 22, LOGAN SQUARE'))
feature_group.add_child(Marker([41.9, -87.725],'Comunity area 23, HUMBOLDT PARK'))
feature_group.add_child(Marker([41.9, -87.685],'Comunity area 24, WEST TOWN'))
feature_group.add_child(Marker([41.89, -87.761],'Comunity area 25, AUSTIN'))
feature_group.add_child(Marker([41.878, -87.729],'Comunity area 26, WEST GARFIELD PARK'))
feature_group.add_child(Marker([41.878, -87.705],'Comunity area 27, EAST GARFIELD PARK'))
feature_group.add_child(Marker([41.874, -87.665],'Comunity area 28, NEAR WEST SIDE'))
feature_group.add_child(Marker([41.861, -87.714],'Comunity area 29, NORTH LAWNDALE'))
feature_group.add_child(Marker([41.84, -87.714],'Comunity area 30, SOUTH LAWNDALE'))
feature_group.add_child(Marker([41.85, -87.664],'Comunity area 31, LOWER WEST SIDE'))
feature_group.add_child(Marker([41.876, -87.627],'Comunity area 32, LOOP'))
feature_group.add_child(Marker([41.8555, -87.6199],'Comunity area 33, NEAR SOUTH SIDE'))
feature_group.add_child(Marker([41.84, -87.633],'Comunity area 34, ARMOUR SQUARE'))
feature_group.add_child(Marker([41.834, -87.6199],'Comunity area 35, DOUGLA'))
feature_group.add_child(Marker([41.824, -87.602],'Comunity area 36, OAKLAND'))
feature_group.add_child(Marker([41.811, -87.632],'Comunity area 37, FULLER PARK'))
feature_group.add_child(Marker([41.811, -87.617],'Comunity area 38, GRAND BOULEVARD'))
feature_group.add_child(Marker([41.809, -87.595],'Comunity area 39, KENWOOD'))
feature_group.add_child(Marker([41.792, -87.617],'Comunity area 40, WASHINGTON PARK'))
feature_group.add_child(Marker([41.792, -87.595],'Comunity area 41, HYDE PARK'))
feature_group.add_child(Marker([41.78, -87.595],'Comunity area 42, WOODLAWN'))
feature_group.add_child(Marker([41.763, -87.575],'Comunity area 43, SOUTH SHORE'))
feature_group.add_child(Marker([41.738, -87.615],'Comunity area 44, CHATHAM'))
feature_group.add_child(Marker([41.742, -87.589],'Comunity area 45, AVALON PARK'))
feature_group.add_child(Marker([41.739, -87.548],'Comunity area 46, SOUTH CHICAGO'))
feature_group.add_child(Marker([41.728, -87.597],'Comunity area 47, BURNSIDE'))
feature_group.add_child(Marker([41.73, -87.575],'Comunity area 48, CALUMET HEIGHTS'))
feature_group.add_child(Marker([41.709, -87.619],'Comunity area 49, ROSELAND'))
feature_group.add_child(Marker([41.703, -87.598],'Comunity area 50, PULLMAN'))
feature_group.add_child(Marker([41.692, -87.568],'Comunity area 51, SOUTH DEERING'))
feature_group.add_child(Marker([41.71, -87.535],'Comunity area 52, EAST SIDE'))
feature_group.add_child(Marker([41.672, -87.628],'Comunity area 53, WEST PULLMAN'))
feature_group.add_child(Marker([41.658, -87.603],'Comunity area 54, RIVERDALE'))
feature_group.add_child(Marker([41.65, -87.54],'Comunity area 55, HEGEWISCH'))
feature_group.add_child(Marker([41.792, -87.77],'Comunity area 56, GARFIELD RIDGE'))
feature_group.add_child(Marker([41.809, -87.726],'Comunity area 57, ARCHER HEIGHTS'))
feature_group.add_child(Marker([41.815, -87.70],'Comunity area 58, BRIGHTON PARK'))
feature_group.add_child(Marker([41.83, -87.672],'Comunity area 59, MCKINLEY PARK'))
feature_group.add_child(Marker([41.836, -87.648],'Comunity area 60, BRIDGEPORT'))
feature_group.add_child(Marker([41.809, -87.657],'Comunity area 61, NEW CITY'))
feature_group.add_child(Marker([41.792, -87.726],'Comunity area 62, WEST ELSDON'))
feature_group.add_child(Marker([41.795, -87.695],'Comunity area 63, GAGE PARK'))
feature_group.add_child(Marker([41.778, -87.77],'Comunity area 64, CLEARING'))
feature_group.add_child(Marker([41.77, -87.726],'Comunity area 65, WEST LAWN'))
feature_group.add_child(Marker([41.77, -87.695],'Comunity area 66, CHICAGO LAWN'))
feature_group.add_child(Marker([41.775, -87.665],'Comunity area 67, WEST ENGLEWOOD'))
feature_group.add_child(Marker([41.775, -87.644],'Comunity area 68, ENGLEWOOD'))
feature_group.add_child(Marker([41.764, -87.622],'Comunity area 69, GREATER GRAND CROSSING'))
feature_group.add_child(Marker([41.744, -87.708],'Comunity area 70, ASHBURN'))
feature_group.add_child(Marker([41.742, -87.658],'Comunity area 71, AUBURN GRESHAM'))
feature_group.add_child(Marker([41.716, -87.673],'Comunity area 72, BEVERLY'))
feature_group.add_child(Marker([41.716, -87.648],'Comunity area 73, WASHINGTON HEIGHTS'))
feature_group.add_child(Marker([41.694, -87.708],'Comunity area 74, MOUNT GREENWOOD'))
feature_group.add_child(Marker([41.688, -87.67],'Comunity area 75, MORGAN PARK'))
feature_group.add_child(Marker([41.98, -87.91],'Comunity area 76, OHARE'))
feature_group.add_child(Marker([41.985, -87.665],'Comunity area 77, EDGEWATER'))

map2016.add_child(feature_group)
map2016.add_child(folium.map.LayerControl())
#map2016


# 
# <img src="https://i.imgur.com/fzorqjI.png" align="left">
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:orange; padding: 1em;">
# We finaly got folium marker and TileLayer to work!<br>
# You can now click on the marker to find out the <b>comuntiy area number</b> and <b>name</b>.<br>
# TileLayer can change the map display. It doesn't add any value to our anayltic, but it look super cool.</p>

# <img src="https://i.imgur.com/FzsTGPb.png" align="left">
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:orange; padding: 1em;">Dealing with date time is a real hassle and reading this short explanation doesn't really help. I highly recommend just messing around with the code on your own time.</p>

# In[ ]:


crimedata['Date'] = pd.to_datetime(crimedata['Date'],format='%m/%d/%Y %I:%M:%S %p')


# In[ ]:


import calendar
crimedata['Month']=(crimedata['Date'].dt.month).apply(lambda x: calendar.month_abbr[x])


# In[ ]:


crimedata['Month'] = pd.Categorical(crimedata['Month'] , categories=['Jan','Feb','Mar','Apr','May',
                                                'Jun','Jul','Aug','Sep','Oct','Nov','Dec'], ordered=True)

months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']


# 
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:blue; padding: 1em;">
# Here we are just converting <b>Date</b> column to <b>datetime</b> and than sorting it into month category. <br><img src="https://i.imgur.com/VgE45Xt.png" align="right">such as jan = 1 and dec = 12.<br><br>
# From there we added a <b>Month</b> column. This will be used to make our visual later.<br><br>
# Below you can see our newly made <b>Month</b> column</p>

# In[ ]:


crimedata.head(5)


# In[ ]:


crimedata.groupby(['Month','Year'])['ID'].count().unstack().plot(marker='o', figsize=(15,10))
plt.xticks(np.arange(12),months)
plt.ylabel('No of Crimes')

plt.show()


# <img src="https://i.imgur.com/NKCXP7C.png" align="left">
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:orange; padding: 1em;">
# When looking at the time-series graph, you'll notice two seasonal variations.<br><br>
# 1. February will have a serve drop in crime. We concluded that weather may have played a factor in the drop in crime. There is a strong correlation with lower temperature resulting in the lower crime rate.<br><br>
# 2. During summer time will have the most active crime recorded
# </p>

# In[ ]:


sns.set(rc={'figure.figsize':(15,10)})
sns.countplot(y='Primary Type',data=crimedata,order=crimedata['Primary Type'].value_counts().index, color=('BLUE'))
plt.xticks(rotation='vertical')
plt.xlabel('No of Crimes')
plt.ylabel('Type of Crimes')
plt.show()


# <img src="https://i.imgur.com/VgE45Xt.png" align="right">
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:blue; padding: 1em;">
# Here we can see the top 6 crimes being commited in chicago is: <b>THEFT, BATTERY, CRIMINAL DAMAGE, NARCOTICS, BURGLARY</b> and<b>  ASSAULT</b><br>
# By viewing when the crime is committed within the 24-hours day, we can better understand the behaviour and characteristics of each crime.</p>

# In[ ]:


df_crime=crimedata[(crimedata['Primary Type']=='THEFT')|(crimedata['Primary Type']=='BATTERY')|
                 (crimedata['Primary Type']=='CRIMINAL DAMAGE')|(crimedata['Primary Type']=='NARCOTICS')|
                 (crimedata['Primary Type']=='BURGLARY')|(crimedata['Primary Type']=='ASSAULT')]


# In[ ]:


df_crime.groupby([df_crime['Date'].dt.hour,'Primary Type',])['ID'].count().unstack().plot(marker='o')
plt.ylabel('Number of Crimes')
plt.xlabel('Hours of the day')
plt.xticks(np.arange(24))
plt.show()


# <img src="https://i.imgur.com/G7kDtAg.png" align="left">
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:orange; padding: 1em;">
# When looking at the time-series graph, you'll notice several trends in crime committed.<br><br>
# Most crimes will start decline at 1 AM<br>
# Lowest number of crime commited is between 5 AM and 6AM<br>
# Highest number of theft reported is in the afternoon. "Stealing in broad daylight"<br>
# Narcotics is shown to peak twice, once around the afternoon and again between 7 PM and 9 PM</p>

# <p id="Narcotics"><center> <h1> <b> <font size="+3">Taking a closer look at Narcotics</font></b></h1></center></p>
# <img src="https://i.imgur.com/4j9y1cF.png"><br>
# <p>Crime for <b>NARCOTICS</b> behave much differently than the rest of the crime. we will be taking a closer look at the data.</p>

# In[ ]:


df_drug = crimedata[crimedata['Primary Type'] == 'NARCOTICS']


# In[ ]:


plt.figure(figsize = (15, 12))
sns.countplot(y = df_drug['Description'],color=("Blue"))
plt.xlabel('Number of Crimes')
plt.ylabel('Type of Crimes')


# In[ ]:


df_narcotic=crimedata[(crimedata['Description']=='POSS: CANNABIS 30GMS OR LESS')|(crimedata['Description']=='POSS: HEROIN(WHITE)')|
                    (crimedata['Description']=='POSS: CRACK')|(crimedata['Description']=='POSS: CANNABIS MORE THAN 30GMS')]


# In[ ]:


df_narcotic.groupby([df_narcotic['Date'].dt.year,'Description',])['ID'].count().unstack().plot(marker='o')
plt.ylabel('Number of Crimes')
plt.xlabel('Year')
#plt.xticks(np.arange(1))
plt.show()


# <img src="https://i.imgur.com/NKCXP7C.png" align="left">
# <p style="border:4px; border-radius: 15px; border-style:solid; border-color:orange; padding: 1em;">
# As you can see, the reported number of marijuana crimes has been dropping throughout the year.
# Currently, Medical marijuana is legal in Illinois, but the recreational use of marijuana is not.
# The State of Illinoise has pass a law to decriminalization possestion of marijuana.
# </p>

# In[ ]:



na2016 = df_narcotic[df_narcotic["Year"]==2016]
na2016.columns = na2016.columns.str.strip().str.lower().str.replace(' ', '_')


# In[ ]:


na2016.shape


# In[ ]:


# Community_Areas map 2016
Community_Areas_data2016 = pd.DataFrame(na2016['community_area'].value_counts().astype(float))
Community_Areas_data2016.to_json('Community_Area_nMap2016.json')
Community_Areas_data2016 = Community_Areas_data2016.reset_index()
Community_Areas_data2016.columns = ['community_area', 'Crime_Count']


nmap2016 = folium.Map(location=Chicago_COORDINATES, zoom_start=11)
#map2016.add_child(feature_group)
#map8.add_children(folium.map.LayerControl())

nmap2016.choropleth(geo_data = Community_Areas_geo,
                data = Community_Areas_data2016,
                name='choropleth',
                columns = ['community_area', 'Crime_Count'],
                key_on = "feature.properties.area_numbe",
                fill_color = 'YlGn', 
                fill_opacity = 0.7, 
                line_opacity = 0.2,
                threshold_scale=[0, 500, 1000, 1500, 2000, 2500],
                legend_name = 'Number of incidents per community area 2016')
               

folium.TileLayer('cartodbdark_matter').add_to(nmap2016)
folium.TileLayer('Stamen Terrain').add_to(nmap2016)
folium.TileLayer('Stamen Toner').add_to(nmap2016)
folium.TileLayer('Mapbox Bright').add_to(nmap2016)

nmap2016.add_child(feature_group)
nmap2016.add_child(folium.map.LayerControl())
#map2016


# <p><b> <font size="+2"> Conclusion </font> </b></p>
# <img src="https://i.imgur.com/iescrbs.png" align="left">

# In[ ]:




