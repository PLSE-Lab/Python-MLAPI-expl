#!/usr/bin/env python
# coding: utf-8

# > # ***Importing packages from python libraries***

# In[ ]:


### Import packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')                  # To apply seaborn whitegrid style to the plots.
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Set the display column option**

# In[ ]:


pd.options.display.max_columns = 20 ##set column widths to display the data


# ## Read data from the CSV

# In[ ]:


NewyorkCityDf = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv') ## read the CSV file from kaggle


# In[ ]:


NewyorkCityDf.shape


# In[ ]:


NewyorkCityDf.describe()


# In[ ]:


NewyorkCityDf.info()


#  ## 01. Find the top 10 expensive Host names and price range

# In[ ]:


hostname_DF = NewyorkCityDf.loc[NewyorkCityDf.price>1500][['name','host_name', 'price']][:11].set_index('host_name').sort_values(by = 'price', ascending = False)
print(hostname_DF)


# In[ ]:


## Find the top 10 expensive Host's listing and host names

##NewyorkCityDf.setIndex(['host_name'])
hostname_DF = NewyorkCityDf.loc[NewyorkCityDf.price>1500][['host_name', 'price']][:11].set_index('host_name').sort_values(by = 'price', ascending = False).plot(kind = 'bar', figsize = (12,5))
plt.xlabel('host names')
plt.ylabel('price')
##hostname_DF.set_index('host_name')
print(hostname_DF)
##NewyorkCityDf.loc[NewyorkCityDf.price>1500][['host_name', 'price']][:11].sort_values(by = 'price', ascending = False).plot(kind = 'bar', xticks = 'host_name')


# ### Conclusion: 01
# **Top 10 expensive hostnames are plotted on the graph.The 10 most expensive properties list are ranging from 1700-6000$. The most expensive host name is jay and Liz**

#  ## 02. a : Find the top 10 expensive listing names and their neighbourhood

# In[ ]:



hostname_DF = NewyorkCityDf.loc[NewyorkCityDf.price>8000][['name', 'price', 'neighbourhood_group']][:11].sort_values(by = 'price', ascending = False)
plt.figure(figsize=(12,7))
sns.barplot(y="name", x="price", data=hostname_DF, hue = 'neighbourhood_group',palette= 'gist_earth')


# ### Conclusion: 02. a
# ** Top 10 expensive listing names are plotted on the graph.**
# >> The 10 most expensive properties list are ranging from 8000-10000$. 
# > > * Most expensive properties are in Manhatten

# ## 02.b Find the top 10 cheapest listing names and their neighbourhood group

# In[ ]:


cheapesthostname_DF = NewyorkCityDf.loc[NewyorkCityDf.price<50][['name', 'price', 'neighbourhood_group']][:11].sort_values(by = 'price')
plt.figure(figsize=(12,7))
ax = sns.barplot(y="name", x="price", data=cheapesthostname_DF, hue = 'neighbourhood_group',palette= 'gist_earth')


# > **Conclusion # 02.b :** The most cheapeast listing is Clean and quiet located in Brookly which will cost around 35$. other 3 cheapest listings which are located in Staten island

# In[ ]:


NewyorkCityDf.loc[NewyorkCityDf.price>8000][['host_name', 'price']][:11].sort_values(by = 'price', ascending = False)


# In[ ]:


NewyorkCityDf.columns.values.tolist()


# ### 03: How many different room types in Newyork airbnp listed

# In[ ]:


## Hom many different types of rooms types are there in airbnb hosted for newyork
# Using matplotlib to add labels and title to the plot. 
plt.figure(figsize=(12,7))
plt.xlabel('Room Types')
plt.ylabel('Number of Items')
plt.title('Bar Chart showing the Number of rooms in each room type value')

# In order to save your plot into an image on your system, use the following command.
# The image will be saved in the directory of this notebook.
sns.countplot(x='room_type',  data=NewyorkCityDf)


# **Conclusion #3 In Newyork most of the rooms are entire home/apt type or Private rooms. Shared rooms are very less comparatively**

# ## 04: How many different types of rooms types are hosted on airbnb per neighbourhood in Newyork

# In[ ]:


## Hom many different types of rooms types are there in airbnb hosted for newyork
NewyorkCityDf['neighbourhood_group'].value_counts().sort_values().plot(kind = 'bar',colormap='BrBG', figsize=(12,5), fontsize = 15) 
# Using matplotlib to add labels and title to the plot. 
# Pandas and matplotlib are linked with each other in the notebook by the use of this line in the Imports: %matplotlib inline

plt.xlabel('Neighbourhood Group', fontsize=15)
plt.ylabel('Number of listing', fontsize=15)
plt.title('Bar Chart showing the Number of listing per neighbourhood_group', fontsize=15)


# **Conclusion #4: Majority of the apartments are in manhattan and brooklyn. There are very few lsiting in staten island**

# **## 05: Find out how the private room, entire apt, shared rooms are distributed across neighbourhoods**

# In[ ]:


roomtypecount = pd.Series(NewyorkCityDf.groupby(['neighbourhood_group'])['room_type'].value_counts())


# In[ ]:


roomtypecount


# In[ ]:


NewyorkCityDf.groupby(['neighbourhood_group'])['room_type'].value_counts().sort_values().plot(kind = 'bar', figsize=(12,5), colormap = 'Dark2')


# **Conclusion # 5: Manhatten has the most highest number of entire apt type listing compare to private room or shared room. in Brokkly the entire apt and private apt are some what closer in number**

# 
# 

# **## 06: In which neighbourhood group the average listing prices are high **

# In[ ]:


NewyorkCityDf.groupby(['neighbourhood_group'])['price'].mean().plot(kind = 'bar', figsize=(12,5))


# In[ ]:


NewyorkCityDf.groupby(['neighbourhood_group','room_type'])['price'].mean().sort_values(ascending = False)


# ** Conclusion # 6: In manhatten the listing price is high for all types of rooms where in brookly shared rooms are cheapest compared to other neighbourhood states.**

# **## 7: How are rooms are distributed in each neighbourhood_group**

# In[ ]:


plt.figure(figsize=(12,8))
ytickrange = np.arange(0, 14000, 500) 
ax = sns.countplot(x='room_type', hue="neighbourhood_group", data=NewyorkCityDf)
ax.set_yticks(ytickrange)


# **Conclusion # 7: In Manhatten majority of the rooms are entire home type where in Brooklyn majority of the rooms are private rooms. In Staten island and bronx the shared rooms are very very less**

# **## Find the low cost and middle cost properties per neighbourhood group**

# In[ ]:


def groupPrice(price):
    if price < 100:
        return "Low Cost"
    elif price >=100 and price < 200:
        return "Middle Cost"
    else:
        return "High Cost"
      
price_group = NewyorkCityDf['price'].apply(groupPrice)
NewyorkCityDf.insert(10, "price_group", price_group, True)
NewyorkCityDf.head(5)


# In[ ]:


g = sns.catplot(x="neighbourhood_group", hue="room_type",col="price_group", data=NewyorkCityDf, kind="count", height=5, aspect=1)
plt.show()


# **Conclusion # 8 : The plot reveals that Brooklyn might be a good choice for an individual traveler who aims for a low cost private room. Either Manhattan or Brooklyn might be a desirable destination for a family/group trip because this is relatively easier to get an entire home/apartment unit with a reasonable middle price range that can be shared by a group of travelers.**

# In[ ]:


BBox = ((NewyorkCityDf.longitude.min(),NewyorkCityDf.longitude.max(),NewyorkCityDf.latitude.min(),NewyorkCityDf.latitude.max()))


# In[ ]:


BBox


# In[ ]:


newyorkMap = plt.imread("/kaggle/input/nycjpg/NYC_UV.jpg")


# In[ ]:


import folium
folium_map = folium.Map(location=[40.738, -73.98],
                        zoom_start=13,
                        tiles="CartoDB dark_matter")

folium.CircleMarker(location=[40.738, -73.98],fill=True).add_to(folium_map)
folium_map


# In[ ]:


ig, ax = plt.subplots(figsize = (18,20))
ax = sns.scatterplot(data=NewyorkCityDf, x='longitude', y='latitude', hue='neighbourhood_group')
##(NewyorkCityDf.longitude, NewyorkCityDf.latitude, zorder=1, alpha= 0.5, c='b', s=10)
ax.set_title('Plotting Spatial Data on Newyork Map')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(newyorkMap, zorder=0, extent = BBox, aspect= 'equal', alpha = 0.5, cmap  = 'winter')


# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(y="neighbourhood", x="price", data=NewyorkCityDf.nlargest(10,['price']))
plt.ioff()


# In[ ]:


data=NewyorkCityDf.nlargest(10,['price'])


# In[ ]:


data


# In[ ]:




