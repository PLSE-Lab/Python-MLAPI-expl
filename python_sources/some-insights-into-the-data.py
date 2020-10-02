#!/usr/bin/env python
# coding: utf-8

# This data file includes all needed information to find out more about hosts, geographical availability, necessary metrics to make predictions and draw conclusions. 
# 
# First, let's import data and create some visualizations, to better understand the data and patterns within. 

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()


# We see that data has 16 columns and 48895 entries. Moreover, we can check the number of null values in each column. 

# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# We can print out some statistic information about the data columns. 

# In[ ]:


df.describe()


# The histograms of numerical columns except the id, latitude, and longitude columns, show us very valuable information. 

# In[ ]:


plt.figure(figsize=(12,8))
i=0
for col in ['price','minimum_nights','number_of_reviews','reviews_per_month','calculated_host_listings_count','availability_365']:
    plt.subplot(2,3,i+1)
    plt.hist(df[col],normed=True,bins=100)
    plt.title(col)
    i+=1


# We create a new dataset for the listings that are available (means that the available days are not zero), priced less than 500 and the minimum nights of listing is less than 366 (1 year). The reason for this process is to remove the outliers, and to be able to see the trends in the data more easily. 

# In[ ]:


df_first=df[(df['price']<=500) & (df['availability_365']>0) & (df['minimum_nights']<366)]


# We have 5 different neighbourhood groups. Lets examine the data regarding these different groups. 

# In[ ]:


df['neighbourhood_group'].unique()


# In the following, we can see the mean price of each neighbourhood, number of listings per neighbourhood, mean number of reviews per neighbourhood, and mean availability per neighbourhood. 

# In[ ]:


from collections import Counter
cont=Counter(df_first['neighbourhood_group'])


# In[ ]:


mean_price_per_neighbor=df_first.groupby('neighbourhood_group').apply(np.mean)
plt.figure(figsize=(12,7))
plt.subplot(221)
plt.bar(mean_price_per_neighbor.index,mean_price_per_neighbor['price'],width=0.3,alpha=0.5)
plt.title('Mean Price of Neighbourhoods')

number_of_listings=cont.values()
plt.subplot(222)
plt.bar(mean_price_per_neighbor.index,number_of_listings,width=0.3,alpha=0.5)
plt.title('Number of listings of Neighbourhoods')

plt.subplot(223)
plt.bar(mean_price_per_neighbor.index,mean_price_per_neighbor['number_of_reviews'],width=0.3,alpha=0.5)
plt.title('Mean Number of Reviews of Neighbourhoods')

plt.subplot(224)
plt.bar(mean_price_per_neighbor.index,mean_price_per_neighbor['availability_365'],width=0.3,alpha=0.5)
plt.title('Mean Availability of Neighbourhoods');


#  We can also visualize the number of listings in each neighbourhood on a pie chart. 

# In[ ]:


cont1 = Counter(df_first['neighbourhood_group'])
labels = cont1.keys()
sizes = cont1.values()

explode = (0, 0, 0.1, 0.2 , 0.5)  
fig1, ax1 = plt.subplots(figsize=(6,6))
wedges, texts, _ = ax1.pie(sizes, explode=explode, labels=labels,radius=2, autopct='%1.1f%%',pctdistance=0.8,labeldistance=1.1)
ax1.axis('equal')  
plt.title('Number of Listings')
plt.tight_layout()
plt.show()


# Here, we also examine the listings that are expensive, meaning that the price is greater than 500. This pie chart shows the number of expensive listings in each neighbourhood.

# In[ ]:


df_second=df[(df['price']>=500) & (df['availability_365']>0)]


# In[ ]:


cont2= Counter(df_second['neighbourhood_group'])
labels = cont2.keys()
sizes = cont2.values()

explode = (0, 0, 0.1, 0.2 , 0.5)  
fig2, ax2 = plt.subplots(figsize=(6,6))
wedges, texts, _ = ax2.pie(sizes, explode=explode, labels=labels,radius=2, autopct='%1.1f%%',pctdistance=0.8,labeldistance=1.1)
ax2.axis('equal')  
plt.title('Number of Expensive Listings')
plt.tight_layout()
plt.show()


# We can plot all the listings (inexpensive listings) on a map. The color of each point represents the price of the listing. 

# In[ ]:


lons = df_first['longitude'].tolist()
lats = df_first['latitude'].tolist()

import geopandas


geo_df = geopandas.read_file(geopandas.datasets.get_path('nybb'))
geo_df = geo_df.to_crs({"init": "epsg:4326"})
geo_df.plot(figsize=(15, 15), alpha=0.4, edgecolor='k',column='BoroName',
            legend=True,legend_kwds={'loc': 'upper left','frameon':False})
plt.scatter(lons, lats, alpha=0.8, c=df_first['price'],s=6)
plt.colorbar(shrink=0.6)
plt.title('Distributaion of Inexpensive Listings and Their Prices');


# Let's do the same with the expensive listings as well. 

# In[ ]:


lons = df_second['longitude'].tolist()
lats = df_second['latitude'].tolist()

import geopandas


geo_df = geopandas.read_file(geopandas.datasets.get_path('nybb'))
geo_df = geo_df.to_crs({"init": "epsg:4326"})
geo_df.plot(figsize=(15, 15), alpha=0.4, edgecolor='k',column='BoroName',
            legend=True,legend_kwds={'loc': 'upper left','frameon':False})
plt.scatter(lons, lats, alpha=0.8, c=df_second['price'],s=6)
plt.colorbar(shrink=0.6)
plt.title('Distributaion of Expensive Listings and Their Prices');


# All the data we were working with so far, were available listings. We can also examine the unavailable data. 

# In[ ]:


df_unavailable=df[(df['availability_365']==0) & (df['price']<=500)]


# In[ ]:


cont3= Counter(df_unavailable['neighbourhood_group'])
labels = cont3.keys()
sizes = cont3.values()

explode = (0, 0, 0.1, 0.2 , 0.5)  
fig3, ax3 = plt.subplots(figsize=(6,6))
wedges, texts, _ = ax3.pie(sizes, explode=explode, labels=labels,radius=2, autopct='%1.1f%%',pctdistance=0.8,labeldistance=1.1)
ax3.axis('equal')  
plt.title('Number of Unavailable Listings')
plt.tight_layout()
plt.show()


# In[ ]:


lons = df_unavailable['longitude'].tolist()
lats = df_unavailable['latitude'].tolist()

import geopandas


geo_df = geopandas.read_file(geopandas.datasets.get_path('nybb'))
geo_df = geo_df.to_crs({"init": "epsg:4326"})
geo_df.plot(figsize=(15, 15), alpha=0.4, edgecolor='k',column='BoroName',
            legend=True,legend_kwds={'loc': 'upper left','frameon':False})
plt.scatter(lons, lats, alpha=0.8, c=df_unavailable['price'],s=6)
plt.colorbar(shrink=0.6)
plt.title('Distributaion of Unavailable Listings and Their Prices');


# We also check the listings with high minimum nights (more than a month). 

# In[ ]:


df_highminnight=df[df['minimum_nights']>30]
cont4= Counter(df_highminnight['neighbourhood_group'])
labels = cont4.keys()
sizes = cont4.values()

explode = (0, 0, 0.1, 0.2 , 0.5)  
fig4, ax4 = plt.subplots(figsize=(6,6))
wedges, texts, _ = ax4.pie(sizes, explode=explode, labels=labels,radius=2, autopct='%1.1f%%',pctdistance=0.8,labeldistance=1.1)
ax4.axis('equal')  
plt.title('Nimber of Listings with Minimum Nights of More Than a Month')
plt.tight_layout()
plt.show()


# In[ ]:


lons = df_highminnight['longitude'].tolist()
lats = df_highminnight['latitude'].tolist()

import geopandas


geo_df = geopandas.read_file(geopandas.datasets.get_path('nybb'))
geo_df = geo_df.to_crs({"init": "epsg:4326"})
geo_df.plot(figsize=(15, 15), alpha=0.4, edgecolor='k',column='BoroName',
            legend=True,legend_kwds={'loc': 'upper left','frameon':False})
plt.scatter(lons, lats, alpha=0.8, c=df_highminnight['minimum_nights'],s=6)
plt.colorbar(shrink=0.6)
plt.title('Distributaion of Listings with High Minimum Nights');


# Let's see what listings have more than 100 reviews. 

# In[ ]:


df_highreview=df[df['number_of_reviews']>=100]
cont5= Counter(df_highreview['neighbourhood_group'])
labels = cont5.keys()
sizes = cont5.values()


explode = (0, 0, 0.1, 0.2 , 0.5)  
fig5, ax5 = plt.subplots(figsize=(6,6))
wedges, texts, _ = ax5.pie(sizes, explode=explode, labels=labels,radius=2, autopct='%1.1f%%',pctdistance=0.8,labeldistance=1.1)
ax5.axis('equal')  
plt.title('Number of Listings with More Than 100 Reviews')
plt.tight_layout()
plt.show()


# Let see what percentage of all the listings in each neighbourhood have more than 100 reviews. Brooklyn Wins!

# In[ ]:


labels = cont5.keys()
num = []
for item in labels:
    num.append(cont5[item]*100/cont1[item])
    
plt.bar(labels,num,width=0.3,alpha=0.5)
plt.title('Percentage of High Reviewed Listings Over all Listings in Each Neighbourhood');


# In[ ]:


lons = df_highreview['longitude'].tolist()
lats = df_highreview['latitude'].tolist()

import geopandas


geo_df = geopandas.read_file(geopandas.datasets.get_path('nybb'))
geo_df = geo_df.to_crs({"init": "epsg:4326"})
geo_df.plot(figsize=(15, 15), alpha=0.4, edgecolor='k',column='BoroName',
            legend=True,legend_kwds={'loc': 'upper left','frameon':False})
plt.scatter(lons, lats, alpha=0.8, c=df_highreview['number_of_reviews'],s=6)
plt.colorbar(shrink=0.6)
plt.title('Distributaion of Listings with More than 100 Reviews');


# Let's examine the types of listings which are Private Room, Entire Home/Apt, and Shared Room. 

# In[ ]:


types=Counter(df['room_type'])
plt.figure()
plt.bar(types.keys(),types.values(),width=0.3,alpha=0.5)
plt.title('Number of Listings of Each Type of Listing');


# We can easily find out which neighbourhood offers more of which type of listing.

# In[ ]:


df_types=df.groupby(['neighbourhood_group', 'room_type']).size()
df_types


# In[ ]:


df_bronx=df_types['Bronx']
labels = df_bronx.index
sizes = df_bronx.values

explode = (0, 0, 0.1)  
plt.figure(figsize=(12,12))
plt.subplot(321)
wedges, texts, _ = plt.pie(sizes, explode=explode, labels=labels,radius=2, autopct='%1.1f%%',pctdistance=0.8,labeldistance=1.1)
plt.axis('equal')  
plt.tight_layout()
plt.title('Bronx')

df_brooklyn=df_types['Brooklyn']
labels = df_brooklyn.index
sizes = df_brooklyn.values

 
plt.subplot(322)
wedges, texts, _ = plt.pie(sizes, explode=explode, labels=labels,radius=2, autopct='%1.1f%%',pctdistance=0.8,labeldistance=1.1)
plt.axis('equal')  
plt.title('Brooklyn')
plt.tight_layout()

df_manhattan =df_types['Manhattan']
labels = df_manhattan.index
sizes = df_manhattan.values

 
plt.subplot(323)
wedges, texts, _ = plt.pie(sizes, explode=explode, labels=labels,radius=2, autopct='%1.1f%%',pctdistance=0.8,labeldistance=1.1)
plt.axis('equal')  
plt.tight_layout()
plt.title('Manhattan')

df_queens =df_types['Queens']
labels = df_queens.index
sizes = df_queens.values

 
plt.subplot(324)
wedges, texts, _ = plt.pie(sizes, explode=explode, labels=labels,radius=2, autopct='%1.1f%%',pctdistance=0.8,labeldistance=1.1)
plt.axis('equal')  
plt.tight_layout()
plt.title('Queens')

df_staten =df_types['Staten Island']
labels = df_staten.index
sizes = df_staten.values

 
plt.subplot(325)
wedges, texts, _ = plt.pie(sizes, explode=explode, labels=labels,radius=2, autopct='%1.1f%%',pctdistance=0.8,labeldistance=1.1)
plt.axis('equal') 
plt.title('Staten Island');
plt.tight_layout()


# At the end, we can examine which words were beign used to describe listings more frequently.

# In[ ]:


df_name=df.copy()
df_name['name']=df_name['name'].str.lower()
df_name['name']=df_name['name'].str.replace('[^\w\s]',' ')
df_name['name']=df_name['name'].str.strip()
#df_name['name']=df_name['name'].str.split(' ')
#df_name['name']=df_name['name'].apply(remove_empty)
df_name['name']=df_name['name'].fillna('nan')

from nltk.corpus import stopwords
stop = stopwords.words('english')

df_name['name']=df_name['name'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

all_words=" ".join(df_name['name'])


# In[ ]:


def remove_empty(row):
    new_row=[]
    try:
        for item in row: 
            if item !='':
                new_row.append(item)
    except Exception:
        pass
    return new_row


# In[ ]:


word_counts = Counter(all_words.split(" "))
word_counts.most_common(35)


# # Conclusion
# 
# Using a lot more insights and visualizations of the data we can find a lot of hidden relations and patterns. 
# Further we can investigate the relation between the words used to describe a listing and the number of reviews it recieves. We can use this information to generate automatically new describtions, that will attract more people, and increse the number of reviews as a result. 
# We can use the data investigated here, to automatically assigne an optimized price, considering the location, the type, number of reviews and other factors. 

# In[ ]:




