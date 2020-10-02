#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


# import numpy as np 
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

# plt.style.use('default')
color_pallete = ['#182952', '#ff0000']
sns.set_palette(color_pallete)

import plotly.express as px


# # Importing Data

# In[ ]:


chennai=pd.read_csv('../input/temperature-of-different-cities-of-india/Chennai.csv')
kolkata=pd.read_csv('../input/temperature-of-different-cities-of-india/Kolkata.csv')
mumbai=pd.read_csv('../input/temperature-of-different-cities-of-india/Mumbai.csv')
delhi=pd.read_csv('../input/temperature-of-different-cities-of-india/Delhi.csv')


# # Preprocessing

# In[ ]:


# Adding new column

chennai['City']='Chennai'
kolkata['City']='Kolkata'
mumbai['City']='Mumbai'
delhi['City']='Delhi'


# In[ ]:


# combining dataframes to a new dataframe
frames=[chennai, kolkata, mumbai, delhi]
cities = pd.concat(frames)


# In[ ]:


cities.info()


# In[ ]:


# droping recent year data, since it is note complete
cities =  cities.drop(cities[cities.YEAR == 2019].index)


# In[ ]:


# rearranging columns
cities = cities[['City', 'DAY', 'MONTH', 'YEAR', 'TEMPERATURE']]
cities.head()


# In[ ]:


# creating a new date column
cities['Date'] = pd.to_datetime(cities[['YEAR', 'MONTH', 'DAY']])
cities.head()


# In[ ]:


# changing column names 
cities.columns = ['City', 'Day', 'Month', 'Year', 'Temp', 'Date']
cities.head()


# In[ ]:


# converting Faranheit to Celcious
cities['Temp'] = round((cities['Temp']-32)*(5/9),1)
cities.head()


# In[ ]:


# finding yearwise mean
city_temp_by_year = pd.DataFrame(cities.groupby(['City', 'Year'])['Temp'].mean())
city_temp_by_year = city_temp_by_year.reset_index()
city_temp_by_year.head()


# In[ ]:


plt.figure(figsize=(18, 6))
ax=sns.lineplot(x="Year", y="Temp",hue="City", data=city_temp_by_year, linewidth=2)
ax.xaxis.set_major_locator(plt.MaxNLocator(24))
ax.set_xlim(1995, 2019)


# In[ ]:


fig = px.line(city_temp_by_year, x="Year", y="Temp", color='City')
fig.show()


# In[ ]:


fig = px.line(city_temp_by_year, x="Year", y="Temp", color='City', facet_col="City", 
              category_orders={"City": ["Delhi", "Kolkata", "Mumbai", "Chennai"]})
fig.show()


# In[ ]:




