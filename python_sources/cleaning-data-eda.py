#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")
import os
import warnings
warnings.simplefilter('ignore')
from mpl_toolkits.basemap import Basemap
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/craigslistVehiclesFull.csv')


# In[ ]:


df.head()


# ### Latitude & Longitude
# Lets see where most the data points are originating. I've shuffled the data and taken 10,000 points.

# In[ ]:


df_shuffled = df.sample(frac=1)
df_shuffled.head()
plt.figure(figsize=(12,6))
m = Basemap(projection='mill',
            llcrnrlat = 25,
            llcrnrlon = -170,
            urcrnrlat = 80,
            urcrnrlon = -60,
            resolution='l')
m.drawcoastlines()
m.drawcountries(linewidth=2)
m.drawstates(color='b')
i = 0
for index, row in df_shuffled.iterrows():
    lat = row['lat']
    lon = row['long']
    xpt, ypt = m(lon, lat)
    m.plot(xpt,ypt,'.',markersize=0.2,c="red")
    # stopping criteria
    i = i + 1
    if (i == 10000): break


# ### Price

# In[ ]:


print ("Top 10 most used price points:")
print (df['price'].value_counts().iloc[:10])


# In[ ]:


print (df.price.describe())


# We shouldnt be swayed by some of the figures above. This mean is close to 600k becuase of few car prices that are in the millions. On further anaysis of the prices column, I found that there were 43 cars that were being sold for values >600k. These include a \$1,000,000 Ferrari and a \$1,000,000,000 chevy! I could also see that there were a lot of dirty values such as 999,999,999 ans 123,456,789. 
# We can safely conclude that the data is dirty and its upto us to refine it if we want to derive some valuable insights. 
# Some possible reasons for dirt prices:
# * Users *might* have forgot to put a decial point. For eg: a Jeep Jeep Wrangler Rubicon 2018 starts at around 35,000. I saw a price for the same at 3,200,000.    
# 
# If you have any other reasons, then please do let me know down in the comments below. 

# We have to make some hard decisions now. There are way too many dubious values that hinder us from doind any kind of analysis. 
# I saerch online for averge car prices this year. [Here](https://mediaroom.kbb.com/2018-02-01-Average-New-Car-Prices-Rise-Nearly-4-Percent-For-January-2018-On-Shifting-Sales-Mix-According-To-Kelley-Blue-Book) is an excellent link on the topic.
# Considering the worst case, we can take the *High-end Luxury Car* atrribute. Cars in this section start at an averge of arounf ~100k. 
# I am willing to go to upto 200k for my analysis and droppping all other values.

# In[ ]:


df.shape


# In[ ]:


df.drop(df[df.price > 150000].index, inplace = True)


# In[ ]:


df.shape


# In[ ]:


plt.figure(figsize=(3,6))
sns.boxplot(y='price', data=df,showfliers=False);


# Note that in the above box plot, the outlies havent been show. If you want to see the plot with the outliers or even the violin plot of the df['prices'] you can un comment the following lines. Cavet: the plots are extremly skewed becuase of the outliers.

# In[ ]:


# sns.boxplot(y='price', data=df);
# sns.violinplot(y='price', data=df);


# ### City

# In[ ]:


print ("Number of cities :" + str(len(df['city'].unique())))


# In[ ]:


print ('Top 10 cities:')
print (df['city'].value_counts().iloc[:10])


# Anchorage seems to have a peculiarly high number. Almost twice the one on the second place!

# In[ ]:


# print ('Listing for the least 5 cities:')
# print (df['city'].value_counts().iloc[-5:])


# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(x='city',data=df,order=df['city'].value_counts().index);
ax.set_xticklabels(ax.get_xticklabels(), fontsize=0);


# ### Year

# In[ ]:


print ('Top 10 car manufacturing years:')
print (df['year'].value_counts().iloc[:10])


# By reading through the wikipedia page of [Automotive industry in the United States](http://en.wikipedia.org/wiki/Automotive_industry_in_the_United_States) It seems that the car sales grew to large numbers during the 1960s. I will reduce the dataset further here and ignore years <1960.

# In[ ]:


init = df.shape


# In[ ]:


df.drop(df[df.year < 1960].index, inplace = True)


# In[ ]:


removed_rows = init[0] - df.shape[0];
# print (removed_rows)
print ("No. of rows removed = " + str(removed_rows))


# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(x='year',data=df);
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",fontsize=6);


# The graph indicates a decrese in listings of 2009 car models. Could this be due to the recession? Maybe people didnt buy many cars in 2009 due to economic troubles and consequently, there are lesser cars made in 2009 for sale .

# ### Manufacturer

# In[ ]:


print ("The unique manufacturers we have are:")
print (df['manufacturer'].unique())


# In[ ]:


plt.figure(figsize=(10,6))
ax = sns.countplot(x='manufacturer',data=df);
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",fontsize=8);
plt.title("Manufacturers vs no. of listings")


# In[ ]:


# (df.groupby(['manufacturer'])['price']).mean().sort_values(ascending=False)
plt.figure(figsize=(10,6))
ax = sns.barplot(x='manufacturer', y='price', data=df);
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right",fontsize=8);
plt.title("Car manufacturer vs avergae price");


# In[ ]:


df_shuffled = df.sample(frac=1)
df_shuffled.head()


# In[ ]:


plt.figure(figsize=(12,6))
m = Basemap(projection='mill',
            llcrnrlat = 25,
            llcrnrlon = -170,
            urcrnrlat = 80,
            urcrnrlon = -60,
            resolution='l')
m.drawcoastlines()
m.drawcountries(linewidth=2)
m.drawstates(color='b')
i = 0
for index, row in df_shuffled.iterrows():
    lat = row['lat']
    lon = row['long']
    xpt, ypt = m(lon, lat)
    m.plot(xpt,ypt,'.',markersize=0.2,c="red")
    # stopping criteria
    i = i + 1
    if (i == 10000): break

