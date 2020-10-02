#!/usr/bin/env python
# coding: utf-8

# # Introduction
# In this Notebook, i will be exploring the dataset through various visualization tools in hope to uncover interesting insights that could be useful at the later stage. We will look into the data and understand briefly what the data encompasses. I will also show how we can make use of a library call **Folium**, that enable us to visualize the geographical distribution of the bird species through the use of interactive maps, from various perspective.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
import json
import urllib

sns.set_style('darkgrid')


# ### Reading Data

# In[ ]:


PATH = '../input/birdsong-recognition/'

train_df = pd.read_csv(f'{PATH}train.csv')
test_df = pd.read_csv(f'{PATH}test.csv')

print("Train shape: ", train_df.shape, '\t', 'Test shape: ', test_df.shape)


# In[ ]:


train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df['xc_id'].nunique()


# All recordings from training set are unique

# ### Missing Values

# In[ ]:


print("Columns with missing rows")
print(train_df.isnull().sum().sort_values(ascending=False).head())


# ## Univariate Analysis

# ### Bird Species

# In[ ]:


print("Total number of unique Bird species: ", train_df['ebird_code'].nunique())
print("Distribution of Bird species in Training set: ")
print(train_df.groupby(['ebird_code']).size().sort_values(ascending=False))


# We can see the distribution of the bird species in our training data span from 100 recordings to merely just 9 recordings for *redhea* species

# In[ ]:


plt.figure(figsize=(18,8))
plt.xticks(rotation=90)
plt.title("Distribution of species across countries")
sns.countplot(data=train_df, x='country')


# #### Where are these bird species from training set mostly retrieved from

# In[ ]:


print("Top 5 countries: \n")
print(train_df.groupby(['country']).size().sort_values(ascending=False).head(5))


# ### Geographical Analysis
# I will be using the [folium](https://python-visualization.github.io/folium/) library for geographical visualization of the bird species. Folium is a python library that makes it easy to visualize geographical data with the use of interactive maps. Lets first create a world map that can help us visualize all the location of the training rows. The use of MarkerCluster in folium allows us to view a cluster that will change as we zoom in and out. A single instances will be shown as a red little dot 

# In[ ]:


world = folium.Map(location=[27.623924, -30.471619], zoom_start=2, min_zoom=2)
mc= MarkerCluster()

for i in range(0,len(train_df)):
    if (train_df.iloc[i]['longitude'] != 'Not specified'):
       mc.add_child(folium.Circle(
          location=[float(train_df.iloc[i]['latitude']), float(train_df.iloc[i]['longitude'])],
          radius=5000,
          color='crimson',
          fill=True,
          fill_color='crimson'
       ))
    
world.add_child(mc)
world


# ### Plotting different Bird Species
# We have gathered the information earlier that the count for each bird species range from 9 ~ 100. So lets visualize these bird species individually on the map to see if they are found in similar or different region of the world. Lets choose the top and bottom 2 species: *killde, greegr, buffle, redhea*. The use of layer control allows us to the geographical distribution of each bird species individually
#   

# In[ ]:


#base map
bird = folium.Map(location=[27.623924, -30.471619], zoom_start=3, min_zoom=2)

killde_df = train_df.loc[train_df['ebird_code'] == 'killde']
greegr_df = train_df.loc[train_df['ebird_code'] == 'greegr']
buffle_df = train_df.loc[train_df['ebird_code'] == 'buffle']
redhea_df = train_df.loc[train_df['ebird_code'] == 'redhea']

#Create feature group
killde = folium.FeatureGroup(name='killde')
greegr = folium.FeatureGroup(name='greegr')
buffle = folium.FeatureGroup(name='buffle')
redhea = folium.FeatureGroup(name='redhea')

def add_point(df, fg, color):
    for i in range(0,len(df)):
        if (df.iloc[i]['longitude'] != 'Not specified'):
           fg.add_child(folium.CircleMarker(
              location=[float(df.iloc[i]['latitude']), float(df.iloc[i]['longitude'])],
              radius=3,
              color=color,
              fill=True,
              fill_color=color
           ))

#Add each species as an overlay 
add_point(killde_df, killde, "red")
add_point(greegr_df, greegr, "green")
add_point(buffle_df, buffle, "blue")
add_point(redhea_df, redhea, "black")

#Add overlay to base map
killde.add_to(bird)
greegr.add_to(bird)
buffle.add_to(bird)
redhea.add_to(bird)

#Add layer control
lc = folium.LayerControl(collapsed=False)
lc.add_to(bird)

bird


# ### Plot a Choropleth map for United States
# We have seen from our countplot earlier that USA has the most number of observations in our training data. In this section we will like to visualize the distribution of the bird species within USA itself, across the different states, using a choropleth. To create a choropleth, we will first need the boundary of all the US states, usually contained within a GEOJson format. This is so that with the boundaries specified by the json file, we can then easily mapped the state boundary on our real map in folium.

# In[ ]:


# download GEOJson for mapping to choropleth
url = 'https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json'
urllib.request.urlretrieve(url ,'temp.json')


# Lets append all the states in the json file into a list, so that we can use it to match with it in our *location* column

# In[ ]:


with open('temp.json') as f:
  data = json.load(f)

state_lst = []

for i in data['features']:
    state_lst.append(i['properties']['name'])   #get a list of states from the json


# In[ ]:


def statein(lst, sent):      #function for state name from 'location' column
    for i in lst:
        if i in sent:
            return i
        
us_df = train_df[train_df['country']== "United States"]
us_df['state'] = us_df['location'].apply(lambda x: statein(state_lst, x)) #extract state from location column into new column

usa_choro = us_df.groupby(['state']).size().reset_index()  #get the numbers of training rows for each USA state
usa_choro.columns = ['state','count']


# In[ ]:


choro = folium.Map(location=[37.0902, -95.7129], zoom_start=4, min_zoom=2)

folium.Choropleth(
    geo_data='temp.json',
    name='choropleth',
    data=usa_choro,
    bins=9,
    columns=['state','count'], #'state' is the col name required to match with the key from json, 'count' is the value
    key_on='feature.properties.name', #the name property in json will be matched to the 'state'. The names must matched for choropleth to work 
    fill_opacity=0.8,
    line_opacity=0.5,
    fill_color='BuPu',
    legend_name="Distribution of Birds Species in USA").add_to(choro)

folium.LayerControl().add_to(choro)

choro


# We see most of the training observations collected from **California** and **Arizona**. As a curious minded individual, i went to did some research to understand what is so special about the geographical characteristics in California and Arizona that made the researchers concentrating their collection efforts in these few major states. What i found out is that the regions like California and Arizona has some of the largest birds diversity in USA. And probably because of the wide diversity of species in these regions, that made data collection more attractive in these few states. <br>
# ![bird_diversity](https://biodiversitymapping.org/wordpress/wp-content/uploads/2016/11/Birds_USA_total_richness_large-1024x687.jpg)
# Image credits: https://biodiversitymapping.org/wordpress/index.php/usa-birds/

# There is a very interesting [poster](https://www.stateofthebirds.org/2016/wp-content/uploads/2016/05/SotB_16-04-26-ENGLISH-BEST.pdf) on the diversity of birds in USA which i thought could help us better understand the situation in US

# ### Recordings Duration

# In[ ]:


plt.figure(figsize=(12,8))
plt.title("Distribution of Recordings duration")
sns.distplot(train_df['duration'])


# In[ ]:


print("Duration of Recordings (in seconds): \n")
print(train_df['duration'].describe())


# Apparently, there is a recording that is 2283s long! ~38min

# In[ ]:


train_df.loc[train_df['duration']==2283]


# ### Date

# In[ ]:


with plt.style.context('seaborn-darkgrid'):
    plt.figure(figsize=(18, 10))
    plt.title('Date')
    train_df['date'].value_counts().sort_index().plot()


# In[ ]:


print("Top 10 dates with most recordings:\n ")
print(train_df['date'].value_counts().sort_values(ascending=False).head(10))


# Something interesting we can see from the top 10 dates with the most recordings; they are usually in the month of **May** or **June**.

# ### Rating

# In[ ]:


plt.figure(figsize=(12,8))
plt.title("Distribution of ratings")
sns.countplot(data=train_df,x='rating')


# ### Bird Seen

# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(data=train_df,x='bird_seen')


# ### Sampling Rate

# In[ ]:


plt.figure(figsize=(12, 8))
print("Minimum Sampling Rate: ", train_df['sampling_rate'].min(), '\t', "Maximum Sampling Rate: ", train_df['sampling_rate'].max())
train_df['sampling_rate'].value_counts().sort_index().plot()


# ### Please Upvote this notebook if it has helped you in any ways :)

# That sums up the basic exploration of our dataset. I will continue to update and add more insightful analysis along the way, and also building up into the next section of the workflow. I hope you have enjoyed this notebook. Do ping me up if you have any feedbacks or doubts.  
# Thank you for reading :)

# In[ ]:




