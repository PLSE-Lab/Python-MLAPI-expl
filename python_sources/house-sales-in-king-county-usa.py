#!/usr/bin/env python
# coding: utf-8

# | Name | Description | Date
# | :- |-------------: | :-:
# |Reza Hashemi| IBM Data Analysis with Python.  | On 3rd of August 2019

# > This Prerequisite on **Folium data science**, programming in Python, and *data visualization*.The result is a map of housing data, which displays popup details and which gives further details through the size and color of each dot.
# 
# I've **Anaconda** to manage my platforms, and do most of my coding in a **Jupyter Notebook**. 
# 
# The point of using **Folium to visualize data is to improve upon matplotlib's basemap, to present an interactive (and much more visually appealing) map of current data**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# !conda install -c conda-forge basemap-data-hires --yes "You should install <-> !conda install -c conda-forge basemap --yes"
# and !conda install -c conda-forge folium --yes

import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# !conda install -c conda-forge basemap-data-hires --yes
# !conda install -c conda-forge folium --yes

# !conda update scikit-learn --yes

import folium
from folium.plugins import HeatMap
import webbrowser
import matplotlib
from pandas.tools.plotting import scatter_matrix
matplotlib.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results will write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/kc_house_data.csv')


# In[ ]:


# Observing Data


# In[ ]:


# Creating the dataframe for the 1000 most expensive homes
most_exp_df = df.filter(['price','lat','long', 'grade'], axis=1)
most_exp_df = most_exp_df.nlargest(1000, "price")
most_exp_df[400:600]


# In[ ]:


df.head()


# In[ ]:


df.dtypes


# # Checking Data Types

# In[ ]:


df.describe()


# # Observing diffrent Columns Data

# # ** Preparing Data as we dont need colums such as "id","Unnamed" so we will Drop unwanted columns **

# In[ ]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[ ]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# In[ ]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[ ]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[ ]:



sns.boxplot(df["waterfront"],df["price"])


# In[ ]:


sns.regplot(df[["sqft_above"]],df["price"],data=df,ci=None)
plt.ylim(0,)


# In[ ]:


df.corr()['price'].sort_values()


# # ** Clunky Map **

# In[ ]:


from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


lat = most_exp_df['lat'].values
lon = most_exp_df['long'].values
price = most_exp_df['price'].values


# In[ ]:


# Draw the map background
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='h', 
            lat_0=47.56009, lon_0=-122.21398,
            width=100000, height=100000)

m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# Scatter price data
m.scatter(lon, lat, latlon=True,
          alpha=0.5)

plt.show()


# # ** Map Time! **

# In[ ]:


df.corr()['price'].sort_values(ascending=False)


# # importing LinearRegression from sklearn

# In[ ]:



import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# # Training Linear Regression model between df["long"] and df["price"] 
# Calculating R^2 value

# In[ ]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm
lm.fit(X,Y)
lm.score(X, Y)


# # Training Linear Regression with df["sqft_living"] and df["price"]
# Calculating R^2

# In[ ]:


lm1=LinearRegression()
lm1.fit(df[["sqft_living"]],df["price"])
lm1.score(df[["sqft_living"]],df["price"])


# # Checking R^2 Value for Multiple Linear Regression

# In[ ]:


features =df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]


# In[ ]:


lm2=LinearRegression()
lm2.fit(features,df["price"])
lm2.score(features,df["price"])


# # Importing method for Splitting data into Training and Testing dataset

# In[ ]:


from sklearn.model_selection import train_test_split
print("done")


# In[ ]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features ]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# # Performing Ridge Regression

# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


Rige=Ridge(alpha=.1)
Rige.fit(x_train,y_train)
Rige.score(x_test,y_test)


# In[ ]:


pk=PolynomialFeatures(degree=2)
x_train_pk=pk.fit_transform(x_train)
x_test_pk=pk.fit_transform(x_test)
rig=Ridge(alpha=.1)
rig.fit(x_train_pk,y_train)
rig.score(x_test_pk,y_test)


# # **Now Let's do The Mapping .. -=^.^=-**

# In[ ]:


# Level 1: My initial, basic map in Folium!

# Creating the map, focused on the average of the lat/long values in the data
m1 = folium.Map(location=[47.56009, -122.21398],
                zoom_start=9.25, prefer_canvas=True)

# Changing the background map type
folium.TileLayer("Mapbox Bright").add_to(m1)

# Adding each home as a marker to the map
for index, row in most_exp_df.iterrows():
    folium.CircleMarker([row['lat'], row['long']],
                        radius=1,
                        fill=True).add_to(m1)

m1


# In[ ]:


# Level 2: The same map, but now with popover text

# Creating the map
m2 = folium.Map(location=[47.56009, -122.21398],
                zoom_start=9.25, prefer_canvas=True)

# Setting the background map type
folium.TileLayer("Mapbox Bright").add_to(m2)

# Adding each home as a marker to the map
for index, row in most_exp_df.iterrows():

    # Adding popup text, so clicking each point shows details about each home
    popup_text = "Price: {}<br> Latitude: {}<br> Longitude: {}"
    popup_text = popup_text.format(row["price"],
                                   row["lat"],
                                   row["long"])

    # Adding each home to the map
    folium.CircleMarker([row['lat'], row['long']],
                        radius=1,
                        fill=True,
                        popup=popup_text).add_to(m2)

m2


# In[ ]:


# Level 3: The same map, but now each dot corresponds with the price at which
# each home was sold (bigger dot = more expensive)

# Creating the map
m3 = folium.Map(location=[47.56009, -122.21398],
                zoom_start=9.25, prefer_canvas=True)

# Setting the background map type
folium.TileLayer("Mapbox Bright").add_to(m3)

# Adding each home as a marker to the map
for index, row in most_exp_df.iterrows():

    # Adding popup text, so clicking each point shows details about each home
    popup_text = "Price: {}<br> Latitude: {}<br> Longitude: {}"
    popup_text = popup_text.format(row["price"],
                                   row["lat"],
                                   row["long"])

    # Adding each home to the map, but this time the radius of the dot will
    # be proportional to the price (divided by 1 million)
    folium.CircleMarker([row['lat'], row['long']],
                        radius=(row["price"]/1000000),
                        fill=True,
                        popup=popup_text).add_to(m3)

m3


# In[ ]:


# Level 4: Changing the color of each dot to reflect price buckets

# Creating the map
m4 = folium.Map(location=[47.56009, -122.21398],
                zoom_start=9.25, prefer_canvas=True)

# Setting the background map type
folium.TileLayer("Mapbox Bright").add_to(m4)

# Adding each home as a marker to the map
for index, row in most_exp_df.iterrows():

    # Adding popup text, so clicking each point shows details about each home
    popup_text = "Price: {}<br> Latitude: {}<br> Longitude: {}"
    popup_text = popup_text.format(row["price"],
                                   row["lat"],
                                   row["long"])

    # Changing the color based on buckets of cost
    if row["price"] < 1300000:
        color = "#85CB33" #green
    elif row["price"] >= 1300000 and row["price"] < 2000000:
        color = "#F9B700" #yellow
    else:
        color = "#E01A4F" #hot pink
    
    # Adding each home to the map
    folium.CircleMarker([row['lat'], row['long']],
                        radius=(row["price"]/1000000),
                        fill=True,
                        color=color,
                        popup=popup_text).add_to(m4)

m4


# In[ ]:


# Side quest! Adding a heat map based on price

# Creating the map
m5 = folium.Map(location=[47.56009, -122.21398],
                zoom_start=9.25, prefer_canvas=True)

# Setting the background map type
folium.TileLayer("Mapbox Bright").add_to(m5)

# Plotting the heatmap
heat_data = [[row['lat'],row['long']] for index, row in most_exp_df.iterrows()]

# Adding the heatmap to the map
HeatMap(heat_data).add_to(m5)

m5


# # Just incase you didn't like my work .. > FUCK YOU :D

# In[ ]:




