#!/usr/bin/env python
# coding: utf-8

# ![](https://img4.cityrealty.com/neo/i/p/mig/airbnb_guide.jpg)

# >  **Loading of Libraries**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#This notebook is inpired from the work done by: Erik Bruin in the following link
#https://www.kaggle.com/erikbruin/airbnb-the-amsterdam-story-with-interactive-maps
#Also Please upvote the great work done by Erik 



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from folium import plugins
from folium.plugins import FastMarkerCluster
from folium.plugins import HeatMap
from scipy.stats import pearsonr

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# * * **Data Loading into a dataset**
# 
# 

# In[ ]:


#Loading of Files<a></a>
dataset = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
dataset.head()


# Now we need to check how many rows and columns the dataset is having. It is also necessary to check how many unique values are present for each column. For this we will also use the nunique.
# 
# So we have the 48895 rows and 16 columns.

# In[ ]:


print('\nRows : ',dataset.shape[0])
print('\nColumns :', dataset.shape[1])
print('\nColumns:',dataset.columns.to_list())
print(' \nUnique:\n',dataset.nunique())


# **Plotting the dataset in a map **
# 
# Since according to the dataset we have been provided with the latitude and longitude so we will use folium libraries to present the data into a map. This map can be zoomed and the individual locations can be viewed in detail. I have used the zoom start to 9 so that the clusters can be visible.

# In[ ]:



Long=-73.80
Lat=40.80
locations = list(zip(dataset.latitude, dataset.longitude))

map1 = folium.Map(location=[Lat,Long], zoom_start=9)
FastMarkerCluster(data=locations).add_to(map1)
map1





# **For Nan Values in column reviews_per_month,replacing Nan with 0 **

# In[ ]:


dataset.fillna({'reviews_per_month':0},inplace=True)


# **Scatter Plot to show the neighbourhood group based on Latitude and Longitude**
# 
# We will plot the same latitude and longitude in  a scatter plot to have the cluster of the location, the same we did with the folium maps.

# In[ ]:


plt.figure(figsize=(12,8))
sns.scatterplot(x=dataset.longitude,y=dataset.latitude,hue=dataset.neighbourhood_group)
plt.show()


# **Unique Values**
# 
# It is very important to understand and analyze the Unique values, this gives a lot of insight to the data and the user preference for a particular choice. We will take Room Type and Neighbourhood group
# 
# From the below, we have 3 types of room in the dataset and 5 different neighbourhood group. In the next section we will explore more on these two categories to understand the user distribution.

# In[ ]:


print('Unique value for room_type are :',dataset.room_type.unique())
print('Unique value for neighbourhood_group are :',dataset.neighbourhood_group.unique())


# **Room Types and Neighbourhood Group**
# 
# We will first check the distribution of the room type by grouping the data. From the below its clear the Apartment and Private data is more than that of shared rooms. 
# In general, Shared rooms costs less and can be very useful for travellers who moves from one city to another city quite frequently. Though the shared rooms data is less, we will still try to uncover as much details as we can.
# 

# In[ ]:


dataset['room_type'].value_counts().plot(kind='bar',color=['r','b','y'])
plt.show()


# **Top 10 Apartment listings**
# 
# Below i am displaying the Entire Home/Apartment renting and it looks odd that Sonder(NYC) have itself is too high. We will also list out only the details for the user if they are genuine hotels or there is any discrepancy in the data. We will use the latitude and longitude of the data to find out.
# 
# 

# In[ ]:


apt = dataset[dataset['room_type']=='Entire home/apt']
list_apt = apt.groupby(['host_id','host_name','neighbourhood','neighbourhood_group']).size().reset_index(name='apartment').sort_values(by=['apartment'],ascending=False)
list_apt.head(10)




# **Lets see the Sonder (NYC) **
# 
# Seeing the below latitude and longitude it is clear that it is in the same building.

# In[ ]:


sonder_data = dataset[dataset['host_name']=='Sonder (NYC)']
sonder_data_by = sonder_data[['host_id','host_name','neighbourhood','latitude','longitude']]
sonder_data_by.head(5)


# **Top 10 Private room**
# 
# We will do it for both Private room and Shared room and also will check for the top hoteliers if it is the same location or it is spread out.

# In[ ]:


private = dataset[dataset['room_type']=='Private room']
list_private = private.groupby(['host_id','host_name','neighbourhood']).size().reset_index(name='private').sort_values(by=['private'],ascending=False)
list_private.head(10)


# **Location wise Private room**

# In[ ]:


private_data = dataset[dataset['host_name']=='John']
private_data_by = private_data[['host_id','host_name','neighbourhood','latitude','longitude']]
private_data_by.head()


# **Shared Room Exploration**

# In[ ]:


private = dataset[dataset['room_type']=='Shared room']
list_private = private.groupby(['host_id','host_name','neighbourhood']).size().reset_index(name='shared').sort_values(by=['shared'],ascending=False)
list_private.head(10)


# > **Exploration of Neighbourhood Group*
# 
# Let's explore the neighbourhood group now to see the data distribution. From the below it looks like Manhattan and Brooklyn has more number of listing that the Queens,Bronx and Staten island.
# 

# In[ ]:


dataset['neighbourhood_group'].value_counts().plot(kind='bar',color=['r','b','y','g','m'])
plt.show()


# In[ ]:


private = dataset[dataset['neighbourhood_group']=='Manhattan']
list_private = private.groupby(['host_id','host_name','neighbourhood','neighbourhood_group']).size().reset_index(name='count').sort_values(by=['count'],ascending=False)
list_private.head(10)


# **Price Exploration**
# 
# We will check if there is any null value presentin the price column and from the below, it looks like we don't have any null value to take care of.
# 

# In[ ]:


dataset.price.isna().sum()


# Let's have a quick summary of the price data. In according to the summary statistics it is clear that the Price ranges from $0-$180. But there also exists price which has a maximum of $10000. This we cannot discard as an outlier because there are many scenarios in which price differs. The price varies on different factors which includes location,room type, neighbourhood , season etc. 
# Also, we can see from the below there are few few values with 0, which can be due to dynamic pricing or the willingness of not to share the price with the Airbnb. 
# 
# We  also plot a boxplot to understand how the data is spread out for high ranges for the price irrespective of region.

# In[ ]:


dataset['price'].describe()


# In[ ]:


figsize=(12,8)
sns.boxenplot(x='price',data=dataset)


# **Average room rent for locality**
# 
# For any traveller, the most important thing is the price since this sets the budget of his/her trip. So in the below we will figure out what is the average price per night. We will check for different room type and based on neighbourhood group to figure out what is the average per night stay. Staying at a Apartment is always an expensive stay than shared room/private rooms for any location. This is so because  Entire room is rented out by family for nice stay where privacy is also one of the major factor. Whereas Stay at Shared rooms are being preferred by travellers who generally don't wish to stay for long time at a particular place and moves around places quickly.
# 
# So looking at the  plot it is clear :
# 
# a. Shared room at staten Island is the most cheapest stay per night whereas Renting a Entire apartment/Home at Manhattan per night is the most expensive.
# 
# b. Average price for Private room is also considerably expensive at manhattan so is the shared room at Manhattan is expensive than other private rooms of the neighbourhood. This clearly states that Manhattan is the expensive stay than any other locality.
# 
# c. Bronx is the most cheapest stay in terms of neighbourhood group comparison in respect to room type.
# 
# d. Though Shared room at Staten Island is the cheapest whereas Apartment renting is not cheapest at Staten Island. This can be due to the location of a perfect gateway from the rush of the city for a quality time with family get together , let me know what you think :).
# 
# We will also list out the average price for each type of room per neighbourhood so that tourists can plan based on the budget.
# 
# 

# In[ ]:


dataset.head()
plt.figure(figsize=(12,8))
df = dataset[dataset['minimum_nights']==1]
df1 = df.groupby(['room_type','neighbourhood_group'])['price'].mean().sort_values(ascending=True)
df1.plot(kind='bar')
plt.title('Average Price for rooms in neighbourhood group')
plt.ylabel('Average Daily Price')
plt.xlabel('Neighbourhood Group')
plt.show()
print('List of Average Price per night based on the neighbourhood group')
pd.DataFrame(df1).sort_values(by='room_type')


# **Expensive Neighbourhood **
# 
# Till so far we have checked on the location group, but we have not came across each neighbourhood. This is very much important in terms of price to understand which locality has the highest price margin in terms of night being spend by traveller. Let's dig further to understand further.
# 
# So we will now plot the most expensive neighbourhood, and we will plot only Top 15 neighbourhood and Bottom 15 with respect to average price. This will help a traveller to choose the appropriate neighbourhood based on his budget.
# 
# So according to the below plot Fort Wadsworth is the most expensive in terms of neighbourhood. Whereas Bull's head locality is the least expensive to stay.
# 

# In[ ]:



print('Top 20 most expensive locality in Airbnb listing are :')
df4 = dataset.dropna(subset=["price"]).groupby("neighbourhood")[["neighbourhood", "price"]].agg("mean").sort_values(by="price",
                                                                                                              ascending=False).rename(index=str, columns={"price": "Average price per night based on neighbourhood"}).head(15)

df4.plot(kind='bar')
plt.show()
pd.DataFrame(df4)




# In[ ]:


print('Least expensive neighbourhood according to Airbnb listing are')
df4 = dataset.dropna(subset=["price"]).groupby("neighbourhood")[["neighbourhood", "price"]].agg("mean").sort_values(by="price",
                                                                                                              ascending=False).rename(index=str, columns={"price": "Average price per night based on neighbourhood"}).tail(15)

df4.plot(kind='bar')
plt.show()
pd.DataFrame(df4)


# **Most number of locality listed**
# 
# We will now try to figure out how many number of neighbourhood has been posted based on the count. We can see Williamsburg has most number of listing count where as Fort Wadesworth has one of the least listing i.e 1. 
# 
# I have listed below the listing of top 15 as well as least 15 based on the neighbourhood.
# 
# If we recall this neighbourhood is one of the highest stay based on price so this can be one of the reason that the price is increased due to the less number of listing.
# 

# In[ ]:



df5 = dataset.groupby('neighbourhood')[['neighbourhood','host_name']].agg(['count']
                                                                   )['host_name'].sort_values(by='count',ascending=False).rename(index=str,columns={'Count':'Listing Count'})

df5.head(15).plot(kind='barh')
plt.show()
pd.DataFrame(df5.head(15))


# In[ ]:


print('Least Listing number of count')
df5 = dataset.groupby('neighbourhood')[['neighbourhood','host_name']].agg(['count']
                                                                   )['host_name'].sort_values(by='count',ascending=False).rename(index=str,columns={'Count':'Listing Count'})

df5.tail(15).plot(kind='barh')
plt.show()
pd.DataFrame(df5.tail(15))


# **Location and Review Score**
# 
# Review is the one of the important criteria with online activity these days. This gives a lot of insights to a particular place for tourist and they can swing mood when it comes to online booking. A cheap place with bad review can drive a tourist for not booking and an expensive place with nicest review can shell a tourist more than what he have thought initially. So we will try to figure out the review , how each neighbourhood is doing in respect to review. Since there is a limited data with review we will try to figure out as much as we can.
# 
# First criteria of our review is we will consider only those who have a review more than 50, so that we can have an insight of the data.
# 
# So according to the below plot, Brooklyn got most review in comparison to Manhattan and that is an interesting find. Also Staten Island which is cheaper has less review than the other neighbourhood group. We cannot proceed further to understand why is that case since we have a limited data.
# 

# In[ ]:


fig = plt.figure(figsize=(12,4))
review_50 = dataset[dataset['number_of_reviews']>=50]
df2 = review_50['neighbourhood_group'].value_counts()
df2.plot(kind='bar',color=['r','b','g','y','m'])
plt.title('Location and Review Score(Min of 50)')
plt.ylabel('Number of Review')
plt.xlabel('Neighbourhood Group')
plt.show()
print(' Count of Review v/s neighbourhood group')
pd.DataFrame(df2)


# **Top 5 host**
# 
# Based on the review score(Minimum 50) we will plot, who is our top 5 Host, this increases the confidence of tourist before booking.
# 
# Also lets plot based on the lattitude and location of our review data.
# 
# 

# In[ ]:


map1=folium.Map([40.7128,-74.0080],zoom_start=9.8)
location = ['latitude','longitude']
df = review_50[location]
HeatMap(df.dropna(),radius=8,gradient={.4: 'blue', .65: 'lime', 1: 'red'}).add_to(map1)
map1


# Below plot shows the Top 5 host and it looks like Michael has received more reviews for topping the chart .

# In[ ]:


plt.figure(figsize=(12,6))
review_50.head(2)
df1 = review_50['host_name'].value_counts()[:5].plot(kind='bar',color=['r','b','g','y','m'])
#sns.barplot(x=df1.index,y=df1.values)


# **Plot Price based on the Availability 365**
# 
# We will plot a scatterplot to understand if there is any price increase based on the availability and looking below the plot its hardly to infer. But looks like with availability with 365 the price increases to 10K.

# In[ ]:


plt.figure(figsize=(15,8))
sns.scatterplot(y=dataset['price'],x=dataset['availability_365'])


# **Average Listing for each Neighbourhood group**

# In[ ]:


df6 = review_50.groupby(['neighbourhood_group','room_type']).mean()
df6 = df6.drop(['id','calculated_host_listings_count','reviews_per_month'],axis=1)
pd.DataFrame(df6).sort_values('neighbourhood_group')

