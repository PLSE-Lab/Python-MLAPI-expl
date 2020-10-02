#!/usr/bin/env python
# coding: utf-8

# # Context
# 
# ### We have a pickle file containing Texas counties information related to road incidents like a traffic jam, accidents, flooding, road construction, etc. There are thirty-two different incidents of information on the data set with time, place, coordinates and many more. The overall data size is more than thirty million observations 

# # Objective
# 
# ### We have to analyze the incidents and predict the accident risk involve at those locations where the accident and non-accidents incident were happened together at different times on the same location in the past.

# # Preprocessing
# 
# ### We will explore the pickle file first.
# 
# ### Then we create two more data frames, one for those places where accidents happened. The other for non-accident incidents.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Displaying Full Data Frame
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# importing operating system
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# importing more libraries
import json # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
#from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library

print('Libraries imported.')


# In[ ]:


# Importing pickle library to read pickle file
import pickle


# In[ ]:


df=pd.read_pickle("../input/time_converted_df.pickle")
#df.head()


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df['LAT']=df[['LAT']].astype(float)
df['LON']=df[['LON']].astype(float)


# In[ ]:


# Dropping empty files 
df=df.dropna(subset=['LAT', 'LON','CITY'])
df.isnull().sum()


# # Visualization
# 
# ### The purpose of creating more data frames is to see the visualization of common places
# 
# ### We will examine the first 1000 rows of each data frame to see the resemblance. Whether the presence of other incidents on the accident location is there.
# 
# ### We have to split 'EVENT_TYPE' features into two categories accident places and other places. The purpose is to see the common places, which will help us to further analyze the accidents risks.
# 
# ### We will create four maps, showing Texas, accident locations,non-accident locations and finally both locations on the same map in the fourth map. 
# 
# ### To distinguivish the two incidents, we will use coloring. Red color will show non-accident places where yellow for accident places.

# In[ ]:


# Texas Coordinates
latitude = 31.8160381
longitude = -99.5120986


# In[ ]:


# create map of Texas using latitude and longitude values
map_texas = folium.Map(location=[latitude,longitude], zoom_start=6,tiles='Map Box Control Room')
map_texas


# In[ ]:


df['EVENT_TYPE'].unique().tolist()


# In[ ]:


df_accident=df[df['EVENT_TYPE']=='accident'].reset_index(drop=True)
print (df_accident.shape)
df_accident.head(2)


# In[ ]:


df_without_accident=df[df['EVENT_TYPE']!='accident'].reset_index(drop=True)
print (df_without_accident.shape)
df_without_accident.head(2)


# In[ ]:


df_accident1=df_accident.head(1000)
df_without_accident1=df_without_accident.head(1000)


# In[ ]:


# create map of Downtown Toronto using latitude and longitude values
map_texas = folium.Map(location=[latitude,longitude], zoom_start=6, tiles='Map Box Control Room')

# instantiate a feature group for the incidents in the dataframe
incidents_accident = folium.map.FeatureGroup()
latitudes = list(df_accident1.LAT)
longitudes = list(df_accident1.LON)
labels = list(df_accident1.EVENT_TYPE)

for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.CircleMarker([lat, lng], popup=label).add_to(map_texas)    
    
# add incidents to map
map_texas.add_child(incidents_accident)
map_texas


# In[ ]:


# create map of Downtown Toronto using latitude and longitude values
map_texas = folium.Map(location=[latitude,longitude], zoom_start=6, tiles='Map Box Control Room')#, tiles='Stamen Terrain')

# instantiate a feature group for the incidents in the dataframe
incidents_withoutaccidents = folium.map.FeatureGroup()
latitudes = list(df_without_accident1.LAT)
longitudes = list(df_without_accident1.LON)
labels = list(df_without_accident1.EVENT_TYPE)

for lat, lng, label in zip(latitudes, longitudes, labels):
    folium.CircleMarker([lat, lng], popup=label).add_to(map_texas)    
    
# add incidents to map
map_texas.add_child(incidents_withoutaccidents)
map_texas


# ### Now we want to see the two categories, 'accident' and 'without accident'

# In[ ]:


# Creating a cloumn featuring binary values on the basis of accident risks
df['EVENT'] =[1 if "accident" in x  else 0 for x in df['EVENT_TYPE']]
df.head(2)


# In[ ]:


df1=df[['EVENT_TYPE','LAT','LON', 'EVENT']]


# In[ ]:


df1.dtypes


# In[ ]:


df1= df1.head(1000)


# In[ ]:


# Create dict for 'EVENT' binary values, so that we can view two categories
colordict = {0: 'red', 1: 'yellow'}


# In[ ]:


# create map of dallas using latitude and longitude values
map_dallas = folium.Map(location=[32.791163,-96.749703], zoom_start=10, tiles='openstreetmap')
incidents = folium.map.FeatureGroup()
for lat, lon, traffic_q, label,  in zip(df1['LAT'], df1['LON'], df1['EVENT'], df1['EVENT_TYPE']):
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup = (label ),
        color='r',
        key_on = traffic_q,
        threshold_scale=[0,1],
        fill_color=colordict[traffic_q],
        fill=True,
        fill_opacity=0.7
        ).add_to(map_dallas)

map_dallas.add_child(incidents)
map_dallas


# # Freature Engineering
# 
# ###  It is cleared now after visualizing the two data frames that the locations are common in incidents. It means we have to find each coordinate against the incidents
# 
# ### Creating a new data frame consisting of events and coordinates
# 
# ### We will create separate data frames for both latitude and longitude because latitude may be unique but longitudes may be common. Let see

# In[ ]:


df_sample=df[['EVENT_TYPE', 'LAT']]
df_sample.head(2)


# In[ ]:


df_sample.set_index(['LAT'],inplace=True)
df_sample.head(2)


# ### Merging all the events aganist each latitude

# In[ ]:


event_result=df_sample.groupby(level=['LAT'], sort=False).agg(','.join)


# In[ ]:


event_result.head(2)


# In[ ]:


event_result=event_result.reset_index()
event_result.head()


# In[ ]:


event_result.shape


# ### Now creating a column 'EVENT' om the basis of accident chances in the 'EVENT_TYPE' Feature

# In[ ]:


event_result['EVENT'] =[1 if "accident" in x  else 0 for x in event_result['EVENT_TYPE']]


# In[ ]:


event_result.head(2)


# ### Now for Longitude

# In[ ]:


df2_sample=df[['EVENT_TYPE', 'LON']]
#df2_sample.head(2)
df2_sample.set_index(['LON'],inplace=True)
event_result2=df2_sample.groupby(level=['LON'], sort=False).agg(','.join)
event_result2.head(2)
event_result2=event_result2.reset_index()
event_result2.head(2)


# In[ ]:


event_result2.shape


# ## Comparing the size of two new dataframes based on Latitude and Longitude, its obvious that Longitudes are common, thats why the size of that dataframe is large. We will only persue data for latitude and then merges with longitudes of real data. The latitude dataframe will be enough to predict the accident risks.

# In[ ]:


event_result.rename(columns={'EVENT_TYPE':'EVENT TYPE','LAT':'lat', 'EVENT':'ACCIDENT_RISK'},inplace=True)
frame1=[df,event_result]
frames_main=pd.concat(frame1, axis=1, sort=False)
frames_main.head(2)


# In[ ]:


#droping
frames_main.drop(['lat', 'EVENT_TYPE'], axis=1, inplace=True)
#frames_main.head(2)


# ## Now have created new features, which we call feature engoneering. We will take only two features for rest of the work that is 'EVENT TYPE' & 'ACCIDENT_RISK'

# In[ ]:


main=frames_main[['EVENT TYPE', 'ACCIDENT_RISK']]
main.head(2)


# In[ ]:


main.rename(columns={'EVENT TYPE':'EVENT_TYPE'},inplace=True)
main.head(1)


# In[ ]:


# missing data
main.isnull().sum()


# In[ ]:


main=main.dropna(subset=['EVENT_TYPE', 'ACCIDENT_RISK'])
main.isnull().sum()


# In[ ]:


#!pip install --upgrade pixiedust
#import pixiedust


# # Model Designning & Evaluation
# 
# 
# ### Converting the 'EVENT_TYPE' feature into a category, as it has text information.
# 
# ### We will change text information into a numeric one. Therefore we can visualize the respective data too
# 
# ### Normalize the data for training and evaluation the data
# 
# ### 80% data for training and 20% for testing through train-test split(The sci-kit learn package)
# 
# ### We then apply Deep Learning Module, Neural Network for fitting and evaluate the data

# In[ ]:


main['EVENT_TYPE'] = main['EVENT_TYPE'].astype('category')
main['EVENT_TYPE'] = main['EVENT_TYPE'].cat.codes


# In[ ]:


main.head(2)


# In[ ]:


main['EVENT_TYPE']=main['EVENT_TYPE'].astype(int)


# In[ ]:


# Normalize 'EVENT_TYPE' feature for good view
x=main['EVENT_TYPE']/max(main['EVENT_TYPE'])
#print (x)


# In[ ]:


# Ploting two different graphs for visualization aganist the same data frame
import seaborn as sns; sns.set()
ax = sns.scatterplot(x=x, y="ACCIDENT_RISK",  hue="ACCIDENT_RISK", data=main)
ax.set_title("EVENTS VS RISKS")


# In[ ]:


# we can also see residual plot
ax=sns.residplot(x=x, y='ACCIDENT_RISK', data=main)
ax.set_title("EVENTS VS RISKS")


# In[ ]:


sns.residplot(x='EVENT_TYPE', y='ACCIDENT_RISK', data=main)


# In[ ]:


# split into input and output variables
X = main['EVENT_TYPE'].values
Y = main['ACCIDENT_RISK'].values


# In[ ]:


print (X)
print (Y)


# In[ ]:


# Normalize
X=X/max(X)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam,sgd
from sklearn.model_selection import train_test_split


# In[ ]:


# seed for reproducing same results
seed = 20
np.random.seed(seed)

# split the data into training (80%) and testing (20%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.20, random_state=seed)


# In[ ]:


# create the model
model = Sequential()
model.add(Dense(1, input_dim=1, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

# fit the model
history=model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=50, batch_size=512)#, verbose=0)


# ## Final PLOT

# In[ ]:


import matplotlib.pyplot as plt
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # Observation & Recommendation
# 
# ### We observed that there are many places where multiple incidents happened. It means there are chances of accident risk is involved. Like traffic jam in hazard areas, where plenty of chances of a minor accident can happen there.
# 
# ### We didn't consider the time where an incident has happened. The time factor may also be a good feature to analyze further. It could give us a better probability. 
# 
# 
# 

# # Conclusion
# 
# ### The accuracy of the model is above 90% percent. It means the model fits well. We can concentrate at those common places which are sharing the common coordinates. It will help us to minimize the accident risks.
