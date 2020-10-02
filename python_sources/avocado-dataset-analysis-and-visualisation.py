#!/usr/bin/env python
# coding: utf-8

# <h1>Avocado Database Analysis and Data Visualization</h1>
# 
# This notebook contains python script for analyse and visualization of the Avocado dataset. Analysis is focused on the Average Price per city and Total Volume per city.
# The first step is to download necessary libraries. 

# In[ ]:


import numpy as np
import pandas as pd
import os
import seaborn as sns
sns.set(style="ticks")

import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt


# Read the csv file into the notebook. 

# In[ ]:


avo=pd.read_csv("../input/notebook-datasets/avocado.csv",encoding="utf-8")


# In[ ]:


#get general info about the dataset
avo.info()


# As we can see there are 14 columns and 18249 rows. This dataset contains no null values. But let's check once again to be sure. 

# In[ ]:


#check for missing values
avo.isnull().sum()


# Value range helps to get understanding of the data's scale. 

# In[ ]:


for col in avo.columns:
    if type(avo[col][1])!=str:
        print(col, ' min: ', avo[col].min(), ' max: ', avo[col].max())


# In[ ]:


#all columns with numbers are floats and all columns with words are strings
#transform data column to datetime format
avo["Date"]=pd.to_datetime(pd.Series(avo["Date"]), format="%m/%d/%Y")
len(avo.Date.unique()) # it includes 169 dates


# In[ ]:


#let's see the string columns to make sure that values there are in line
len(avo['region'].unique())


# In[ ]:


for i in avo['region'].unique():
    print(i)


# As we can see, region column contains very different types of regions. Some of them represent one or two city and the others represent the whole country or big regions. Lets just focus on the one or two city names and get rid of the bigger regions. 
# 

# In[ ]:


not_cities=['TotalUS','West','California','Midsouth','Northeast','SouthCarolina','SouthCentral','Southeast','GreatLakes','NothernNewEngland']
cities=avo[avo['region'].isin(not_cities)==False]


# In[ ]:


cities['month']=[x.month for x in cities.Date]
cities['year']=[x.year for x in cities.Date]


# The dataset is not groupped by linear time. It is groupped by city and shows certain time period which is not ascending.
# Let'f create pivot table to show total volume by date and by cities.

# In[ ]:


pivot_cities=pd.pivot_table(cities,index=['Date','region'],values=['AveragePrice','Total Volume'])
pivot_cities=pivot_cities.reset_index()
pivot_cities=pivot_cities.rename(columns={'region':'city'})


# Now we can take a look at the prices and total volume range tendency over the years.
# The general tendency in prices shows growth and lowering of the prices over the time span from 2015 till 2018. Despite the huge changes in prices, the price in the beginning of 2015 and in the end of 2018 seem to be the same with a little shift up in 2018. The same happened with the total sales volume. Interestingly, the two plots show that as prices went down, total volume went up and vise versa. Both charts show that cretail cities are have very huge avocado sales volume in comparison the the rest of the cities with moderate level of sales. The same we can see in the price chart. 

# In[ ]:


plt.figure(figsize=(20,15))  
g=sns.lineplot(x='Date', y='AveragePrice', hue='city',data=pivot_cities)
g=g.set_xlim(pivot_cities['Date'].min(), pivot_cities['Date'].max())
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.title("Avocado Average Price 2015 - 2018",pad=20, fontsize=30)
plt.ylabel('Average Price', fontsize=20)
plt.xlabel('Date', fontsize=20)
plt.show(g)


# In[ ]:


plt.figure(figsize=(20,15))  
g=sns.lineplot(x='Date', y='Total Volume', hue='city',data=pivot_cities)
g=g.set_xlim(pivot_cities['Date'].min(), pivot_cities['Date'].max())
plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.title("Avocade Sales Total Volume 2015 - 2018",pad=20, fontsize=30)
plt.ylabel('Total Volume', fontsize=20)
plt.xlabel('Date', fontsize=20)
plt.show(g)


# It is interesting to see how total volume and prices fluctuations in each city. Since the both columns have very different scales, I decided to devide the Total Volume columne by 1,000,000 and to compbine the two charts.
#     We can see that Los Angeles, New York, Plains, Phenix, New Mexico and Dallas are among the leaders of avocado consumption. Total volume did not fluctuare much in the rest of the cities. Whereas prices show very active fluctuation in all cities over this period of time. Prices went seriously up and down in San Francisco, Seattlem Boise, Atlanta and some other city. Chicago, Dallas, Denver, Houston, Los Angeles, New York, Phoenix, Plains, New Mexico show clar corelation in prices and volume - the lower price, the higher volumes and vice versa. 

# In[ ]:


# or another approach add column that will scale Total Volume to be able to show in one chart
pivot_cities['TV']=pivot_cities['Total Volume']/1000000
g=sns.FacetGrid(pivot_cities, col='city', col_wrap=5,height=1.5, aspect=1.5)
g=g.map(plt.plot, 'Date','AveragePrice').set_xticklabels([str(x)[:10] for x in pivot_cities['Date']],rotation=90)
g.map(plt.plot, 'Date', 'TV',color='r').set_xticklabels([str(x)[:10] for x in pivot_cities['Date']],rotation=90).set_ylabels('Volume & Price')
g.add_legend()
g.set_titles('{col_name}')


# There are two types of the avocado - organic and conventional. As we can see the average price of organic avocado is higher from 0.5 to 3.5 with the median on 1.75. Non organic avocado prices are between 0.4 and 2.1 with the medium of 1. As to the volumes, it is clearly seen that people prefer more non organic avocados and the organic avocado total volume share is very small in the total sales volume. 
# 

# In[ ]:


#create new pivot table to be able to see the avocado type difference
pivot_cities1=pd.pivot_table(cities,index=['Date','region','type'],values=['AveragePrice','Total Volume'])
pivot_cities1=pivot_cities1.reset_index()

g = sns.FacetGrid(pivot_cities1, col="type",height=4, aspect=2)
g.map(plt.hist, "AveragePrice")

g = sns.FacetGrid(pivot_cities1, col="type",height=4, aspect=2)
g.map(plt.hist, "Total Volume")


# The code below calculates total volume of the avocados per city. I just added all values per city per date to get the general big number. Los Angelas, Plains and New York are top consumers of avocado.

# In[ ]:


#calculate total volume per city for the whole period

sum_tot={}
for c in pivot_cities.city:
    a=pivot_cities[pivot_cities['city']==c]
    tot=a['Total Volume'].sum()
    sum_tot[c]=round(tot,2)

cities_list=[i for i in sum_tot.keys()]
val=[i for i in sum_tot.values()]

plt.figure(figsize=(3,10))
sns.barplot(val,cities_list, palette="ch:.25")
plt.title("Total Avocado Consumption per City", pad=20, fontsize=20)


# To show the total volume on map, I found longitude and latitude of each city in google and stored it in the separate csv file coord.csv.

# In[ ]:



coord = pd.read_csv("../input//notebook-datasets/coordinates.csv")


# As you can see it contains three columns name, latitude and longitude.

# In[ ]:


coord.info()


# I need to remove GreatLakes and NorthernNewEngland as these are the big regions of the USA not cities. 

# In[ ]:


coord=coord[(coord['name']!='GreatLakes') & (coord['name']!='NorthernNewEngland')].reset_index()


# Create column total to be able to use it for the matplotlib bubble map.

# In[ ]:


c_dic=dict(zip(cities_list, val))
coord['total']=coord['name'].map(c_dic)
coord['total']=[round(i,2) for i in coord['total']]


# The matplotlib bubble map shows that avocado is most popular among the thouthern cities of the USA and New York.

# In[ ]:


#plot map

plt.figure(figsize=(50,50))
#style = dict(size=10, color='gray')
#plt.imshow(usa_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5)
ax=coord.plot(kind="scatter", x="longitude", y="latitude",
     c=coord['total'],s=coord['total']/100000, cmap=plt.get_cmap("jet"),
    colorbar=True, alpha=0.2, figsize=(15,10),
)
for i, txt in enumerate(coord['name']):
    ax.annotate(txt, (coord['longitude'][i]+0.3, coord['latitude'][i]+0.3))
    #plt.text(x+.03, y+.03, txt, fontsize=9)

plt.title("Total Avocado Consumption Bubble Map", pad=20, fontsize=20)
plt.ylabel("Latitude", fontsize=14)
plt.legend()
plt.show(ax)


# Code below creates a plotly map chart woth the time frame. 

# In[ ]:


import plotly.express as px


# In[ ]:


coord_lat_dic=dict(zip(coord['name'],coord['latitude']))
coord_lon_dic=dict(zip(coord['name'],coord['longitude']))
pivot_cities['lat']=pivot_cities['city'].map(coord_lat_dic)
pivot_cities['lon']=pivot_cities['city'].map(coord_lon_dic)


# In[ ]:


pivot_cities['date']=[str(x)[:10] for x in pivot_cities['Date']]


# In[ ]:



fig = px.scatter_geo(pivot_cities, color="Total Volume",
                     hover_name="city", size="Total Volume",
                     lon='lon',
                     lat='lat',
                     animation_frame="date",
                     scope='usa',
                     title ={'text':"Avocado Consumption Growth 2015-2018",
                            'xanchor': 'center',
                            'y':0.9,
                            'x':0.5}
                    )
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:




