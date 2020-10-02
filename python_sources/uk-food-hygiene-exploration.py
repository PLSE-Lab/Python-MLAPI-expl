#!/usr/bin/env python
# coding: utf-8

# # Exploration of Poole (UK) food hygiene ratings
# 
# Unlike Wales, food establishments in England aren't legally required to display their food hygiene rating on their building. The result of this is that only establishments that have a good rating display it. However, the ratings of all establishments are freely available online, so we can map all this data! I haven't seen this done before, and figured it could be useful for people looking for somewhere to eat in a new city.
# 
# ## What?
# 
# This is a quick visualisation of food hygeine ratings data in Poole, UK. 
# 
# The food hygiene ratings are explained here: https://www.food.gov.uk/safety-hygiene/food-hygiene-rating-scheme
# 
# The data can be found here: http://ratings.food.gov.uk/open-data/en-GB

# ## Import libraries and read data

# In[ ]:


import numpy as np 
import pandas as pd 
import folium
from folium import IFrame

df = pd.read_csv("../input/poole2018.csv")


# ## Create a map of the establishment's hygiene ratings
# Note that the coordinates aren't the best - sometimes when establishments are nearby they will be given the same coordinates, and when plotted on this map the markers will simply overlap.
# Also, I'm aware of many resteraunts that don't appear in the dataset (not sure why the food standards agency haven't included them).

# In[ ]:


#colours for each marker
color_key={'5':'darkblue','4':'blue','3':'purple','2':'lightred','1':'darkred','0':'black','AwaitingInspection':'lightgray','Exempt':'lightgray'}

#initialise the map
map_hyg = folium.Map(location=[50.7299,-1.9615],zoom_start = 13) 

#add a marker to the map for each establishment
for index,row in df.iterrows():
    folium.Marker([row['Lat'], row['Long']],
                  popup= "<b>"+row['Name']+"</b> <br> Type: "+row['Type']+"<br> <br> Rating: "+row['Rating'],
                  icon=folium.Icon(color=color_key[row['Rating']])
                 ).add_to(map_hyg)
#show the map
map_hyg


# ## Create a heatmap of the establishments

# In[ ]:


from folium import plugins
from folium.plugins import HeatMap

#initialise the heatmap
heatmap = folium.Map(location=[50.7299,-1.9615],zoom_start = 13) 

# make the data for the heatmap by collecting the coordinates in a list
heat_data = [[row['Lat'],row['Long']] for index, row in df.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(heatmap)

# Display the map
heatmap


# ## Visualising food hygiene by establishment type

# In[ ]:


#get counts for each rating for each est. type
counts=[]
for est_type in np.unique(df['Type']):
    est_type_data={}
    df_type=df[df['Type']==est_type].copy()
    for rating in np.unique(df_type['Rating']):
        est_type_data[rating]=(df_type.Rating == rating).sum()
    counts.append(est_type_data)
#create and show the resulting dataframe
df_counts=pd.DataFrame(counts,index=np.unique(df['Type'])).fillna(0)
df_counts


# In[ ]:


#plot a stacked bar chart of the dataframe
import seaborn as sns
from matplotlib.colors import ListedColormap

sns.set(rc={'figure.figsize':(30,15)},font_scale=3)
df_counts.iloc[:,0:5].plot(kind='bar', stacked=True,colormap=ListedColormap(sns.color_palette("GnBu", 5)),)


# Looks like takeaways/sandwich shops have a higher proportion of low (<4) hygiene ratings than resteraunts/cafes/canteens.
# Also hospitals/childcare/caring premises, retailers and schools/universities seem to have proportionally very high hygeiene ratings, as you'd hope.
# Most suprising to me is that mobile caterers also have a high proportion of 5 rated establishments, although the sample size may be too small to draw too many conclusions.

# ## Create time-series heatmap of establishments between 2013 and 2018

# In[ ]:


from folium import plugins 

#initialise the heatmap
heatmap = folium.Map(location=[50.7299,-1.9615],zoom_start = 13)

#colelct the data for the heatmap in a list of lists (one for each year)
heat_data=[]
years=np.arange(2013,2019)
for year in years:
    file="../input/poole"+str(year)+".csv"
    print(file)
    df = pd.read_csv(file)
    heat_data.append([[row['Lat'],row['Long']] for index, row in df.iterrows()])

# Plot it on the map
hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)
hm.add_to(heatmap)

# Display the map 
heatmap

#Note the buttons to control the heatmap are on the bottom right. Not sure why they arent visible, but if you hover over them the commands can be seen

