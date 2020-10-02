#!/usr/bin/env python
# coding: utf-8

# ## Analysis in this notebook
#  - __Question:__ What is the global distribution of recent eruptions and what type of volcano is associated with each type
#  - __Data Source:__ https://www.ngdc.noaa.gov/nndc/servlet/ShowDatasets?dataset=102557&search_look=50&display_look=50
#  - __ Data format:__  Original exported format was tab-delimited, it was converted to CSV prior to uploading into Kaggle
#  - __ Deliverable: __  An interactive terrain map with color coded markers based on volcano type with pop up info of volcano properties
#  - __ Fun: __  Zoom in to some of these volcanoes, especially ones located near island chain hot spots (Indonesia and Hawaii)

# > ## Import libraries

# In[40]:


# -*- coding: utf-8 -*-
"""
Created on Sun May  6 20:29:19 2018

@author: David O'Dell
"""

import pandas as pd
import csv
import folium


# > ## Import CSV and inspect raw data
# __Note:__  Since the data source website only provides tab-delimited output files - and Kaggle prefers CSV files - we can convert the file here in the Kaggle notebook, download it and then upload it as a new version of the dataset.  Here is the code used to do so in the comments:
# 

# In[52]:


#'results' file was the raw tab delimited file from the data source website
#raw_data = r"../input/results"
#csv_file = r"volcano_data_2010.csv"

#input_file = csv.reader(open(raw_data, "r"), delimiter = '\t')
#output_csv = csv.writer(open(csv_file, "w"))
#output_csv.writerows(input_file)
#now you need to manually upload the file back into Kaggle

df1 = pd.read_csv("../input/volcano_data_2010.csv")
print(df1.shape)
print(df1.columns)
df1.head(n = 5)
df1.tail(n = 5)


# > ## Subset the dataframe and verify

# In[42]:


#subset df1 to only get all rows and only columns we need

df2 = df1.loc[:, ("Year", "Name", "Country",
                 "Latitude", "Longitude", "Type")]

#check the first and last 5 rows
df2.head(n = 5)
df2.tail(n = 5)


# > ## Check for missing or NA values in subset data

# In[43]:


#since this data is from the web check to see if there are any Nan
#df2.isnull().sum().sum()

#or run an if statement to see if there are any and print a warning, else pass and continue the script
#df2.isnull().any().any()

if df2.isnull().any().any() is True:
    print("Sorry there is at least 1 NaN value")
else:
    pass


# > ## Check numeric type and format decimals

# In[44]:


#established by now that it's clean data we need to first subset the lat long
df_ll = df2.loc[:, ("Latitude", "Longitude")]

#find out what type of object the lat long numbers are in by sampling one of each
#mainly to just check and see if they are strings hiding as numbers
print(type(df_ll.iloc[0,0]))
print(type(df_ll.iloc[0,1]))

#We see they are floating numbers, and we really don't need so many digits.
#The end result would be too many digits to view on the marker pop up.
#Let's apply round to 2 digits on each element of the column.

df_ll['Latitude'] = df_ll['Latitude'].apply(lambda x: round(x,2))
df_ll['Longitude'] = df_ll['Longitude'].apply(lambda x: round(x,2))

#take the dataframe values for lat long and make a list of lists
df_ll_list = df_ll.values.tolist()


# > ## Start formatting the map

# In[45]:


#use terrain map layer to actually see volcano terrain
map = folium.Map(location = None, tiles = "Stamen Terrain")


# > ## Iterate through volcano list
#  - associate volcano type to marker color
#  - assign volcano content to marker

# In[46]:


#insert multiple markers, iterate through list
#each marker popup will display info from the list
#add a different color marker associated with type of volcano

#subset desired properties to display in popup
df_year = df2.loc[:, ("Year")]
df_name = df2.loc[:, ("Name")]
df_country = df2.loc[:, ("Country")]
df_type = df2.loc[:, ("Type")]


i = 0
for coordinates in df_ll_list:

#assign a color marker for the type of volcano, Strato being the most common
    if df_type[i] == "Stratovolcano":
        type_color = "green"
    elif df_type[i] == "Complex volcano":
        type_color = "blue"
    elif df_type[i] == "Shield volcano":
        type_color = "orange"
    else:
        type_color = "black"


#now place the markers with the popup labels and data
    map.add_child(folium.Marker(location = coordinates,
                            popup =
                            "Year: " + str(df_year[i]) + '<br>' +
                            "Name: " + str(df_name[i]) + '<br>' +
                            "Country: " + str(df_country[i]) + '<br>'
                            "Type: " + str(df_type[i]) + '<br>'
                            "Coordinates: " + str(df_ll_list[i]),
                            icon = folium.Icon(color = "%s" % type_color)))
    i = i + 1


# > ## Complete and save the map as HTML file

# In[50]:


#save map as HTML
map.save("volcanoes2010.html")


# > ## Access shell and verify file was written

# In[48]:


get_ipython().run_cell_magic('bash', '', 'ls -l')


# > ## Display the map

# In[49]:


#call the map to display
map


# In[ ]:




