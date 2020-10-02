#!/usr/bin/env python
# coding: utf-8

# This is my very first kernel on kaggle. Please upvote if you like it.

# **IMPORTING ALL NECESSARY LIBRARIES**

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#import cufflinks as cf
import plotly.offline as py
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected= False)
#cf.go_offline()
from geopy.geocoders import Nominatim
import folium
from folium.plugins import HeatMap


# In[ ]:


import pandas as pd
df= pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')


# In[ ]:


print("dataset contains {} rows and {} columns ".format(df.shape[0], df.shape[1]))


# In[ ]:


df.info()


# **SHOWING SNIPPET OF THE DATASET**

# In[ ]:


df.head()


# **EXPLORATORY DATA ANALYSIS:-**

# **Which are the top restaurant chains in Bangaluru?**

# In[ ]:


plt.figure(figsize=(10, 7))
sns.set_style('darkgrid')
chains= df['name'].value_counts()[:20]
#online= df['']
sns.barplot(x= chains, y = chains.index, palette= 'dark')
sns.despine()
plt.title('Top 20 restaurant chains in Bangaluru')
plt.xlabel('Number of outlets')
plt.ylabel('Name of Restaurants')


# **WE OBSERVE THE FOLLOWING:**
# 1. Cafee Cofee Day has the most number of outlets in the city of bangaluru. 
# 2. McDonald's is on the 17th number according to number of outlets.

# **RESTAURAANTS ACCEPTING vs NOT ACCEPTING ONLINE ORDERS IN BANGALURU**

# In[ ]:


x= df['online_order'].value_counts()

pieplot= go.Pie(labels= x.index, values= x)
layout= go.Layout(title='ACCEPTING ONLINE ORDERS IN BANGALURU', width= 500, height= 400)
fig= go.Figure(data= [pieplot], layout= layout)
py.iplot(fig, filename='pieplot')


# It is clear that almost 60% retaurants accept online order in Banguluru.

# **RESTAURANTS PROVIDING ONLINE TABLE BOOKING**

# In[ ]:


y= df['book_table'].value_counts()
pieplot= go.Pie(labels= y.index, values= y)
layout=go.Layout(title= 'Table Booking', width= 500, height= 400)
fig= go.Figure(data= [pieplot],layout= layout)
py.iplot(fig)


# **MOST COMMON RESTAURANT TYPES IN BANGULURU**

# In[ ]:


rest_type= df['rest_type'].value_counts()[:20]
plt.figure(figsize=(8, 6))
sns.set_style('darkgrid')
sns.barplot(x= rest_type, y = rest_type.index)
plt.title('Top 20 restaurant types in Banguluru')
plt.xlabel('Count')
plt.ylabel('Restaurant type')


# WE OBSERVE THAT:
#     1. BANGULURU being a busy city, most common type of restaurants are Quick Bites restaurants.
#     2. Takeaway or delivery is less popular.  
#     3. Fine dining is very rare. 

# **RATING DISTRIBUTION**

# In[ ]:


rating= df['rate']
rating= rating.dropna().apply(lambda x : float(x.split('/')[0]) if (len(x)>3) else np.nan).dropna()
plt.figure(figsize=(7,6))
sns.set_style('darkgrid')
sns.distplot(rating, bins = 20,  color= 'red')
plt.title('Rating distribution of the restaurants')
plt.xlabel('Ratings')


# WE OBSERVE THAT:
#     1. Maximum restaurants have ratings between 3 and 4.
#     2. Restaurants with rating higher than 4.5 are very rare.

# **COST DISTRIBUTION FOR TWO PEOPLE:**

# In[ ]:


cost= df['approx_cost(for two people)']
cost= cost.dropna().apply(lambda x : int(x.replace(',', '')))
plt.figure(figsize=(6,6))
sns.set_style('darkgrid')
sns.distplot(cost, color= 'green')
plt.title('Approx. cost for two people in city of Banguluru')
plt.xlabel('Cost for two people')


#  WE OBSERVE THAT:
#      1. Maximum restaurants in banguluru cost less than INR 1000 for two people.

# **COST vs RATING**

# In[ ]:


crdf= df[['rate', 'approx_cost(for two people)', 'online_order']].dropna()
crdf['rate']= crdf['rate'].apply(lambda x : float(x.split('/')[0]) if (len(x)>3) else 0)
crdf['approx_cost(for two people)']= crdf['approx_cost(for two people)'].apply(lambda x : int(x.replace(',', '')))

plt.figure(figsize=(11, 8))
sns.set_style('darkgrid')
sns.scatterplot( x= 'rate', y = 'approx_cost(for two people)', hue= 'online_order', data= crdf )
plt.title('Cost vs Rating comparison')
plt.xlabel('Rating between 0 to 5')
plt.ylabel('Approx. cost for 2 people')


# **VOTE DISTRIBUTION**

# In[ ]:


online_yes= df[df['online_order']== 'Yes']['votes']

online_no= df[df['online_order']== 'No']['votes']
trace1= go.Box(y= online_yes, name= 'Accepting online orders')
trace2= go.Box(y= online_no, name= 'Not accepting online orders')
layout= go.Layout(title= 'Vote disrtibution', width= 800, height= 400)
data= [trace1, trace2]
fig= go.Figure(data = data, layout= layout)
py.iplot(fig)


# We observe that :
#     1. Restaurants accepting online orders get more umber of votes.
#     2. Median number of votes are different in both categoies.

# **APPROX. COST COMPARISON:**

# In[ ]:


crdf= df[['approx_cost(for two people)', 'online_order']].dropna()
crdf['approx_cost(for two people)']= crdf['approx_cost(for two people)'].apply(lambda x : int(x.replace(',', '')))
crdf_yes= crdf[crdf['online_order']== 'Yes']['approx_cost(for two people)']
crdf_no= crdf[crdf['online_order']== 'No']['approx_cost(for two people)']
trace1= go.Box(y= crdf_yes, name= 'Accepting online orders')
trace2= go.Box(y= crdf_no, name= 'Not accepting online orders')
layout= go.Layout(title= 'APPROX. COST COMPARISON', width= 800, height= 500)
data= [trace1, trace2]
fig= go.Figure(data= data, layout= layout)
py.iplot(fig)


# We observe that:
#     1. The cost is significantly less when restaurants accept orders online.
#   ****2. Does this strongly justify the ongoing outcry of the restaurant owners in INDIA against Zomato's pricing sceme?****

# **MOST FAMOUS AREAS IN BANGULURU FOR FOOD LOVERS ( FOODIE AREAS )**

# In[ ]:


areas= df['location'].value_counts()[:20]

plt.figure(figsize=(7, 6))
sns.set_style('darkgrid')
sns.barplot(x= areas, y= areas.index, palette='rocket')
plt.title('Top 20 areas with most number of restaurants in Banguluru')
plt.xlabel('Number of restaurants')
plt.ylabel('Location in city')


# WE OBSERVE THAT:
#     1. BTM, HSR, Koramangala 5th Block has the most number of restaurants.
#     2. BTM dominates by having more than 5000 restaurants.
#     3. MG Road which is a very popular area in Banguluru has less than 1000 restaurants.

# **MOST FAMOUS CUISINES IN BANGULURU:**
#     

# In[ ]:


cuisines= df['cuisines'].value_counts()[:10]
plt.figure(figsize=(6,5))
sns.set_style('darkgrid')
sns.barplot(x= cuisines, y= cuisines.index, palette='dark')
plt.title('Top 10 most famous cuisines in Banguluru')
plt.xlabel('Count')
plt.ylabel('Cuisines')


# WE OBSERVE THAT:
#     1. North Indian, South Indian, Chinese are very popular in Banguluru.

# **GEOCODING**

# In[ ]:


locations=pd.DataFrame({"Name":df['location'].unique()})
locations['Name']=locations['Name'].apply(lambda x: "Bangalore " + str(x))
lat_lon=[]
geolocator=Nominatim(user_agent="app")
for location in locations['Name']:
    location = geolocator.geocode(location, timeout= 20)
    if location is None:
        lat_lon.append(np.nan)
    else:    
        geo=(location.latitude,location.longitude)
        lat_lon.append(geo)


locations['geo_loc']=lat_lon
locations.to_csv('locations.csv',index=False)


# In[ ]:


locations["Name"]=locations['Name'].apply(lambda x :  x.replace("Bangalore","")[1:])
locations.head()


# **Heatmap of all the restaurants in Bangaluru**

# In[ ]:


rest_count= pd.DataFrame(df['location'].value_counts().reset_index())
rest_count.columns= ['Name', 'count']
rest_count=rest_count.merge(locations, on = "Name", how = "left" ).dropna()
rest_count.head()


# In[ ]:


maps = folium.Map(location= [12.97 , 77.59], zoom_start= 12)
folium.Marker(
location= [12.97 , 77.59], popup= 'geographical center of Bangaluru', 
    icon= folium.Icon(color= 'green', icon= 'ok-sign')).add_to(maps)
lat , lon= zip(*np.array(rest_count['geo_loc']))
rest_count['lat']= lat
rest_count['lon']= lon

HeatMap(rest_count[['lat', 'lon', 'count']].values.tolist()).add_to(maps)
maps


# We Observe That:
# 1. Restaurants are concentrated towards the center of Bangaluru.
# 2. Central Bangaluru is the best place for starting new restautrants.
# 

# **FINDING NORTH INDIAN RESTAURANTS IN BANGALURU**

# In[ ]:


north_data=  pd.DataFrame(df[df['cuisines']== 'North Indian'].groupby(['location'], as_index= False)['url'].agg('count'))
north_data.columns= ['Name', 'count']
#north_data.head()
north_data = north_data.merge(locations, on = 'Name', how = 'left' ).dropna()
#north_data.head()
lat , lon= zip(*np.array(north_data['geo_loc']))
north_data['lat'] = lat
north_data['lon'] = lon
north_data.head()


# In[ ]:


maps= folium.Map(location = [12.97 , 77.59], zoom_start= 12)
folium.Marker(
location= [12.97 , 77.59], popup= 'geographical center of Bangaluru', 
    icon= folium.Icon(color= 'green', icon= 'ok-sign')).add_to(maps)
HeatMap(north_data[['lat', 'lon', 'count']].values.tolist()).add_to(maps)
maps


# WE OBSERVE THAT:
#     1. North Indian Restaurants are more popular in South Bangaluru region.

# ** FINDING SOUTH INDIAN RESTAURANTS IN BANGALURU**

# In[ ]:


south_data=  pd.DataFrame(df[df['cuisines']== 'South Indian'].groupby(['location'], as_index= False)['url'].agg('count'))
south_data.columns= ['Name', 'count']
south_data = south_data.merge(locations, on = 'Name', how = 'left' ).dropna()
lat , lon= zip(*np.array(south_data['geo_loc']))
south_data['lat'] = lat
south_data['lon'] = lon
south_data.head()


# In[ ]:


maps= folium.Map(location = [12.97 , 77.59], zoom_start= 12)
folium.Marker(
location= [12.97 , 77.59], popup= 'geographical center of Bangaluru', 
    icon= folium.Icon(color= 'green', icon= 'ok-sign')).add_to(maps)
HeatMap(south_data[['lat', 'lon', 'count']].values.tolist()).add_to(maps)
maps

