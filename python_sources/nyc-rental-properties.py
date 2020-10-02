#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.head()


# **Explore data for better understanding**

# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# Interpretation:
# Let's start with fields with Null values
# 1. Last Review: This is the date when last review was posted. We can drop this field as it has no significance in our analysis.
# 2. Review_per_month: We can simply replace null values with 0. Because it shows no reviews per posted for the listing.
# 3. Name and Host_name: Well, these variables are also insignificant, so I will just drop them.

# In[ ]:


df.drop(['name','host_name','last_review'], axis=1, inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.fillna({'reviews_per_month':0}, inplace=True)
df.isnull().sum()


# In[ ]:


df.describe().T


# **Interpretation:**
# 1. We can observe mean price in NYC is 152. And it ranges between 69 to 175 dollars.
# 2. On average customer stays for 7 nights.
# 3. Each AIRBNB host has 7 listings on average!
# 4. And all listing are mostly available all year long.

# # **Some Visualizations******

# In[ ]:


label = df['neighbourhood_group'].unique()
sizes = df['neighbourhood_group'].value_counts()
colors = ['lightcoral','indianred', 'tomato','orangered', 'salmon']
fig, ax = plt.subplots(1,1,figsize=(10,10))
plt.pie(sizes,explode=[0.009,0.006,0,0,0], labels = label, shadow = False, labeldistance = 1.1,autopct='%1.3f%%', startangle = 90, colors=colors)
ax.set_title('Listings share as per Neighbourhood')
plt.show()


# In[ ]:


#color2 = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
fig2, ax2 = plt.subplots(1,1, figsize=(10,10))
sns.countplot('neighbourhood_group',data=df, order=df['neighbourhood_group'].value_counts().index)
ax2.set_title('Number of Listing - Bar Chart')


# **HeatMap for rental propertiies in NYC**

# In[ ]:


import folium
from folium.plugins import HeatMap
m=folium.Map([40.7128,-74.0060],zoom_start=11)
HeatMap(df[['latitude','longitude']].dropna(),radius=8,gradient={0.3:'green',0.6:'yellow',1.0:'red'}).add_to(m)
display(m)


# Here we have added NYC coordinates in Map() as we know this data is for NYC. Otherwise, we could use LocateControl() to locate data on World Map.
# Gradient colors: Red, Yellow, Green shows the intensity of data points, red being most dense.  

# **HeatMap of Price**

# In[ ]:


df2 = df[['latitude','longitude','price']].groupby(['latitude','longitude']).mean().copy()
df2.head()
df3 = df[['neighbourhood_group']]
# df2.reset_index()
# df2['price'][1]


# In[ ]:


m2=folium.Map([40.7128,-74.0060],zoom_start=11, titles = "HeatMap of Price")
HeatMap(df[['latitude','longitude','price']].groupby(['latitude','longitude']).mean().reset_index().values.tolist(),radius=10,gradient={0.25:'green',0.6:'yellow',0.8:'blue',1.0:'red'}).add_to(m2)
# folium.CircleMarker(df2[['latitude','longitude'][1]], popup = 'some').add_to(m2)
# m2.add_child(folium.ClickforMarker(popup='Awesome'))
# m2.add_child(folium.ClickForMarker(df2[['latitude','longitude'][1]], popup='some'))
# MarkerCluster(df[['latitude','longitude','price']].groupby(['latitude','longitude']).mean().reset_index().values.tolist(), popups = df[['latitude','longitude','price']].groupby(['latitude','longitude']).mean().reset_index().values.tolist(), overlay=True, control=True, show=True).add_to(m2)
# BoatMarker(df[['latitude','longitude','price']].groupby(['latitude','longitude']).mean().reset_index().values.tolist(),radius=10,gradient={0.25:'green',0.6:'yellow',0.8:'blue',1.0:'red'}).add_to(m2)
# width = df[['latitude']], height = df[['longitutde']], radius =
display(m2)


# In[ ]:


import plotly.graph_objects as go
# df['text'] = df['neighbourhood'] + ':' + df['price'] 
# + ', ' + df['state'] + '' + 'Arrivals: ' + df['cnt'].astype(str)
fig = go.Figure(data=go.Scattergeo(
        lon = df['longitude'],
        lat = df['latitude'],
        mode = 'markers',
#         text = df['text'],
        marker_color = df['price'],
#         marker_size = df['price']
        ))

fig.update_layout(
        title = 'Price Scatter Plot',
        geo_scope='usa',
    )
fig.show()


# **HeatMap of Room Availability**

# In[ ]:


m3=folium.Map([40.7128,-74.0060],zoom_start=11)
HeatMap(df[['latitude','longitude','availability_365']].groupby(['latitude','longitude']).mean().reset_index().values.tolist(),radius=10,gradient={0.25:'green',0.6:'yellow',0.8:'blue',1.0:'red'}).add_to(m3)
display(m3)
#0.25:'green',0.6:'yellow',0.8:'orange',1.0:'red'


# In[ ]:


#Next we will check price distribution in these neighbourhood groups
#creating a sub-dataframe with no extreme values / less than 500
sub_df=df[df.price < 300]
fig3 = plt.subplots(figsize=(10,10))
viz=sns.violinplot(data=sub_df, x='neighbourhood_group', y='price', scale="count", hue = 'room_type')
viz.set_title('Distribution of prices for each neighberhood_group')


# In[ ]:


df_cat = df.groupby(['neighbourhood_group','room_type'])['price'].mean().reset_index()
df_cat


# Here the width of violin shape is defined by the count of rental places. Hence, you could see lean violin shape for Staten Island and Bronx.
# Here the white dot represents Mean and the lenght of black rectangle represent +/- 1 SD.
# 

# In[ ]:


#Scatter plot between availability and price
sx = df['availability_365']
sy = df['price']
plt.scatter(sx,sy,alpha=0.5)
plt.show()


# Well! clearly, no linear relation is there between price and room availability.

# # Regression Model

# In[ ]:


#transform data
from sklearn import preprocessing

enc = preprocessing.LabelEncoder()
enc.fit(df['neighbourhood_group'])
df['neighbourhood_group']=enc.transform(df['neighbourhood_group'])    

enc = preprocessing.LabelEncoder()
enc.fit(df['neighbourhood'])
df['neighbourhood']=enc.transform(df['neighbourhood'])

enc = preprocessing.LabelEncoder()
enc.fit(df['room_type'])
df['room_type']=enc.transform(df['room_type'])

df.drop(['id', 'host_id'], axis = 1, inplace=True)


# In[ ]:


plt.figure(figsize=(30, 30))
sns.pairplot(df, height=3, diag_kind="hist")


# We can see there is no linear relation between price and other variables. 
# Hence, it is does not qualify for linear regression model.
# I will run the regression model to prove the point. 

# In[ ]:


df.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
col_to_scale = ['neighbourhood','room_type','availability_365','latitude','longitude','number_of_reviews','calculated_host_listings_count','minimum_nights','neighbourhood_group','reviews_per_month']
df[col_to_scale] = StandardScaler().fit_transform(df[col_to_scale])
df.head()


# In[ ]:


#Regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
lm = LinearRegression()
#'minimum_nights',,'id''neighbourhood_group','reviews_per_month',
X = df[['neighbourhood','room_type','availability_365','latitude','longitude','number_of_reviews','calculated_host_listings_count']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

lm.fit(X_train,y_train)

from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
pred = lm.predict(X_test)

print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
        np.sqrt(metrics.mean_squared_error(y_test, pred)),
        r2_score(y_test,pred) * 100,
        mean_absolute_error(y_test,pred)
        ))


# We can observe from R2 value, that this model only explains 8% of change in price through given variables. Hence, this data (and parameters) is not sufficient to provide any prediction model.
# 
