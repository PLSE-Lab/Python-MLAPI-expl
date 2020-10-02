#!/usr/bin/env python
# coding: utf-8

# Hello! 
# This is my first kernel. Let's see how it goes!
# 
# This is a **State-wise Analysis** of the data with respect to different parameters.

# In[40]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from numpy import array
from mpl_toolkits.basemap import Basemap
from functools import reduce
from subprocess import check_output

print(check_output(["ls", "../input/cities_r2.csv"]).decode("utf8"))
inputfile = pd.read_csv('../input/cities_r2.csv')
inputfile.head(2)


# In[41]:


print("The number of states in India are: ",inputfile['state_code'].nunique()) 


# In[42]:


inputfile['latitude'] = inputfile['location'].apply(lambda x: x.split(',')[0])
inputfile['longitude'] = inputfile['location'].apply(lambda x: x.split(',')[1])
inputfile.head(1)


# ## Top 10 Populated Cities of India ##

# In[43]:


print("The Top 10 Cities sorted according to the Total Population (Descending Order)")
top_pop_cities = inputfile.sort_values(by='population_total',ascending=False)
top10_pop_cities=top_pop_cities.head(10)
top10_pop_cities


# In[44]:



plt.subplots(figsize=(20, 15))
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary(fill_color='lightblue')
map.fillcontinents(color='orange')
map.drawcountries(color='black')
map.drawcoastlines(linewidth=0.5,color='black')  

lg=array(top10_pop_cities['longitude'])
lt=array(top10_pop_cities['latitude'])
pt=array(top10_pop_cities['population_total'])
nc=array(top10_pop_cities['name_of_city'])

x, y = map(lg, lt)
plt.plot(x, y, 'ro', markersize=10)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')

plt.title('Top 10 Populated Cities in India',fontsize=20)


# ## Top 10 Literate Cities of India ##

# In[45]:


print("The Top 10 Cities sorted according to the Literacy Rate (Descending Order)")
top10_lit_cities = inputfile.sort_values(by='effective_literacy_rate_total',ascending=False).head(10)
top10_lit_cities


# In[46]:


plt.subplots(figsize=(20, 15))
map = Basemap(width=1200000,height=900000,projection='lcc',resolution='l',
                    llcrnrlon=67,llcrnrlat=5,urcrnrlon=99,urcrnrlat=37,lat_0=28,lon_0=77)

map.drawmapboundary(fill_color='lightblue')
map.fillcontinents(color='green')
map.drawcountries(color='black')
map.drawcoastlines(linewidth=0.5,color='black')  

lg=array(top10_lit_cities['longitude'])
lt=array(top10_lit_cities['latitude'])
pt=array(top10_lit_cities['population_total'])
nc=array(top10_lit_cities['name_of_city'])

x, y = map(lg, lt)
plt.plot(x, y, 'ro', markersize=10)


for ncs, xpt, ypt in zip(nc, x, y):
    plt.text(xpt+60000, ypt+30000, ncs, fontsize=10, fontweight='bold')

plt.title('Top 10 Literate Cities in India',fontsize=20)


# ## State-wise distribution of Top Cities of India##

# In[47]:


print("The States with the number of Top Cities in them")
states_no_of_top_cities=inputfile["state_name"].value_counts().sort_values(ascending=False)
plt.figure(figsize=(25, 10))
states_no_of_top_cities.plot(title='States with the number of Top Cities in them',kind="bar", fontsize=20)


#  - The State of Uttar Pradesh has the highest number of Top Cities.
#  - It is followed by West Bengal and Maharashtra. 
# 
# **This is obviously in direct relation with the size of the states.**

# ##Literacy Rate of States of India ##

# In[48]:


print("The States with the Average Literacy Rate of their Top Cities")
litratesort=inputfile.groupby(['state_name'])['effective_literacy_rate_total'].mean().sort_values(ascending=False)
plt.figure(figsize=(25, 10))
litratesort.head(29).plot(title='Average Literacy Rate of States',kind='bar', fontsize=20)


# ## Population of States of India ##

# In[49]:


print("The Population of States considering the Top Cities in it")
pop_state=inputfile.groupby(['state_name'])['population_total'].sum().sort_values(ascending=False)
plt.figure(figsize=(25, 10))
pop_state.head(29).plot(title='Population of States considering the Top Cities in it',kind='bar', fontsize=20)


# ## Sex-Ratio of States of India ##

# In[50]:


print("The Average Sex Ratio of the States considering all the Top Cities in it")
sex_ratio_states=inputfile.groupby(['state_name'])['sex_ratio'].mean().sort_values(ascending=False)
plt.figure(figsize=(25, 10))
sex_ratio_states.head(29).plot(title='Average Sex Ratio of the States considering all the Top Cities in it',kind='bar', fontsize=20)


# ## Graduate Ratio of States of India ##

# In[51]:


inputfile['graduate_ratio']=inputfile['total_graduates']/(inputfile['population_total']-inputfile['0-6_population_total'])
inputfile.head(2)


#  - Although, as just the '0-6_population_total' is known, the Graduate Ratio wouldn't be an accurate measure but can give a rough idea. 

# In[52]:


print("The Graduates Ratio of the States considering all the Top Cities in it")
grad_ratio_states=inputfile.groupby(['state_name'])['graduate_ratio'].mean().sort_values(ascending=False)
plt.figure(figsize=(25, 10))
grad_ratio_states.head(29).plot(title='Graduates Ratio of the States considering all the Top Cities in it',kind='bar', fontsize=20)


# ## Top Developing States of India ##

# In[53]:


print("Top States having better Total Literacy Rates, Sex-Ratio and Graduation Ratio are as follows:")
los = [litratesort.head(15),sex_ratio_states.head(15),grad_ratio_states.head(15)]
pd.concat(los, axis=1, join='inner')


# To conclude, it is safe to say that the following states have a **better** Literacy Rate, Sex-Ratio and Graduate Ratio than the other states of India:
# 
#  - Manipur
#  - Meghalaya
#  - Puducherry
#  - Orissa
#  - Kerela
#  - Assam
# 
# 
#  - 3/6 states are from the North-East India. 
# 
#  - Also, 6 out of the Top 10 Literate cities are from South-India.
# 
# *Comments/Suggestions are welcome!
# 
# Thank you.*
