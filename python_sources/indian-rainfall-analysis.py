#!/usr/bin/env python
# coding: utf-8

# # Analysis of Indian Rainfall

# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from numpy import median
#import gmaps
get_ipython().run_line_magic('matplotlib', 'inline')

import os
print(os.listdir("../input"))


# In[ ]:


import gmaps


# # Indian Rainfall Exploratory Analysis

# ### Here, I have also used gmaps to visualize the rainfall on map. But it does not work in this Jupyter Notebook.
# ### So i have included that code as a Markdown with its output pictures.

# Let's have a look at the data first.

# In[ ]:


district_wise_rainfall = pd.read_csv("../input/district wise rainfall normal.csv")
district_wise_rainfall.head()


# In[ ]:


# rainfall in India csv file has NA values, so let's fill NA as zero
india_rainfall = pd.read_csv("../input/rainfall in india 1901-2015.csv").fillna(0)
india_rainfall.head()


# In[ ]:


After seeing the data, following are the things that can be explores:
    - Statistical results like minimum, maximum, mean rainfall
    - What is the annual rainfall in each state
    - How is monthly and seasonal rainfall distributed over the years in all states
        - Which places have highest and lowest rainfalls
    - What is annual rainfall trend in whole country
        - Which year has highest and lowest rainfall
    - Rainfall patterns in each state for different months and seasons


# ## Statistical results like minimum, maximum, mean rainfall

# In[ ]:


district_wise_rainfall.info()


# In[ ]:


district_wise_rainfall.describe()


# In[ ]:


india_rainfall.info()


# In[ ]:


india_rainfall.describe()


# ## What is the annual rainfall in each state

# Let's find number of states we have

# In[ ]:


subdivision = india_rainfall['SUBDIVISION'].unique()
subdivision


# In[ ]:


states = district_wise_rainfall['STATE_UT_NAME'].unique()
states


# In[ ]:



plt.style.use('ggplot')

fig = plt.figure(figsize=(18, 28))
ax = plt.subplot(2,1,1)
ax = plt.xticks(rotation=90)
ax = sns.boxplot(x='STATE_UT_NAME', y='ANNUAL', data=district_wise_rainfall)
ax = plt.title('Annual rainfall in all States and UT')

ax = plt.subplot(2,1,2)
ax = plt.xticks(rotation=90)
ax = sns.barplot(x='STATE_UT_NAME', y='ANNUAL', data=district_wise_rainfall)


# ## How is monthly and seasonal rainfall distributed over the years in all states
# ### Which places have highest and lowest rainfalls

# Let's find out the total monthly rainfall fn each state by summing the rainfall in all years

# In[ ]:


total_rainfall_in_states = district_wise_rainfall.groupby(['STATE_UT_NAME']).sum()
total_rainfall_in_states['STATE_UT_NAME'] = total_rainfall_in_states.index
total_rainfall_in_states.head()


# In[ ]:


plt.style.use('ggplot')
index = total_rainfall_in_states.index
fig = plt.figure(figsize=(18, 28))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1 = plt.xticks(rotation=90)
ax1 = sns.heatmap(total_rainfall_in_states[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']])
ax1 = plt.title('Total Rainfall Monthly')

ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2 = plt.xticks(rotation=90)
ax2 = sns.heatmap(total_rainfall_in_states[['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']])
ax2 = plt.title('Total Rainfall Seasonal')

ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ax3 = plt.xticks(rotation=90)
ax3 = sns.barplot(x='STATE_UT_NAME', y='ANNUAL', data=total_rainfall_in_states.sort_values('ANNUAL'))
ax3 = plt.title('Total Rainfall in all States and UT in increasing order')


# As you can see in the plots, June to September is the period with highest rainfalls almost every year. This is expected, since India is a monsoon country and those months are the rainy season. Second plot shows Chandigarh and Lakshwadeep has got the least rainfall, while Assam and Uttar Pradesh are the places with highest rainfall.

# ## What is annual rainfall trend in whole country
# ### Which years have highest and lowest rainfall

# Let's aggregate rainfall over years and see the trends in yearly rainfall all over India.

# In[ ]:


yearly_rainfall = india_rainfall.groupby(['YEAR']).sum()
yearly_rainfall['rise_fall'] = np.where(yearly_rainfall['ANNUAL'] > yearly_rainfall['ANNUAL'].shift(1), "Rise", "Fall")
yearly_rainfall['YEAR']= yearly_rainfall.index
yearly_rainfall.head()


# In[ ]:


year = yearly_rainfall.index
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(28, 10))

ax = sns.pointplot(x=year, y="ANNUAL", data=yearly_rainfall, color='grey')
ax = sns.pointplot(x=year, y="ANNUAL", data=yearly_rainfall,hue='rise_fall', markers=["x", "o"], join=False)
ax = plt.xticks(rotation=90)
ax = plt.title('Annual Rainfall Trend in India')


# Above plot shows that rainfall keeps on fluctuating almost every year. There are sometimes certain peak periods of high rainfall and very low rainfall.

# In[ ]:


fig = plt.figure(figsize=(18, 28))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1 = plt.xticks(rotation=90)
ax1 = sns.heatmap(yearly_rainfall[['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']])
ax1 = plt.title('Total Rainfall Monthly in all years')

ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2 = plt.xticks(rotation=90)
ax2 = sns.heatmap(yearly_rainfall[['Jan-Feb','Mar-May','Jun-Sep','Oct-Dec']])
ax2 = plt.title('Total Rainfall Seasonal in all years')

ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ax3 = plt.xticks(rotation=90)
ax3 = sns.barplot(x='YEAR', y='ANNUAL', data=yearly_rainfall)
ax3 = plt.title('Total Rainfall in  all years')


# Above plots show that Jun-Sep is the peak period of rainfall every year. Also, the rainfall keeps on fluctuating every year.
# Next, let's find out which year has highest and lowest rainfall.

# In[ ]:


x1 = yearly_rainfall.sort_values('ANNUAL')
fig = plt.figure(figsize=(20, 8))
ax = plt.xticks(rotation=90)
ax = x1['ANNUAL'].plot.bar(color=['red', 'pink', 'darkred'], edgecolor = 'black')
ax = plt.title('Highest and Lowest rainfalls')


# 1972 and 2002 were the years with least amount of rainfall. 1933 and 1961 were the years with highest rainfall.

# ## Rainfall patterns in each state for different months and seasons

# In[ ]:


total_rainfall_in_states.info()


# In[ ]:


total_rainfall_in_states = total_rainfall_in_states.drop('STATE_UT_NAME', axis=1)
total_rainfall_in_states = total_rainfall_in_states.T
total_rainfall_in_states


# Let's check out total monthly and seasonal rainfall.

# In[ ]:


monthly_total_rainfall = total_rainfall_in_states.head(12)
seasonal_total_rainfall = total_rainfall_in_states.tail(4)


# In[ ]:


# For each state, we visualize the rainfall patterns in different months and season
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(45,100))

for i in range(35):
    plt.subplot(7, 5, i+1)
    t = monthly_total_rainfall[monthly_total_rainfall.columns[i]].plot.bar()
    t.set_title("Monthly Rainfall for " + str(monthly_total_rainfall.columns[i]))
plt.show()


# In[ ]:


# For each state, we visualize the rainfall patterns in different months and season
plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(45,100))

for i in range(35):
    plt.subplot(7, 5, i+1)
    t = seasonal_total_rainfall[seasonal_total_rainfall.columns[i]].plot.bar()
    t.set_title("Seasonal Rainfall for " +str(seasonal_total_rainfall.columns[i]))
plt.show()


# I tried to add google map visualization of the rainfall on India's map. But it seems like gmaps does not work in kaggle kernel. So, I have just added the code below for reference.

# ## Markdown and output for gmaps code

# import gmaps.geojson_geometries
# gmaps.configure(api_key="AIzaSyBYrbp34OohAHsX1cub8ZeHlMEFajv15fY") # Your Google API key
# gmaps.geojson_geometries.list_geometries()

# > import gmaps
# 
# > import gmaps.geojson_geometries
# 
# > gmaps.configure(api_key="AIzaSyBYrbp34OohAHsX1cub8ZeHlMEFajv15fY") # Your Google API key
# 
# > gmaps.geojson_geometries.list_geometries()

# #### Output

# dict_keys(['countries', 'countries-high-resolution', 'england-counties', 'us-states', 'us-counties', 'india-states', 'brazil-states'])

# india_geojson = gmaps.geojson_geometries.load_geometry('india-states') # Load GeoJSON of countries
# 

#  #### Load India maps
# 
# > india_geojson = gmaps.geojson_geometries.load_geometry('india-states') # Load GeoJSON of countries

# #### check India map
# 
# > fig = gmaps.figure()
# 
# > gini_layer = gmaps.geojson_layer(india_geojson, fill_color='blue')
# 
# > fig.add_layer(gini_layer)
# 
# > fig

# fig = gmaps.figure()
# gini_layer = gmaps.geojson_layer(india_geojson, fill_color='blue')
# fig.add_layer(gini_layer)
# fig

# https://github.com/saxenakrati09/Indian_Rainfall/blob/master/map%20(1).png

# In[ ]:


total_rainfall_in_states = total_rainfall_in_states.T
total_rainfall_in_states.index


# In[ ]:


total_rainfall_in_states['Latitude'] = [11.66702557, 14.7504291, 27.10039878,26.7499809, 25.78541445, 
                                        30.71999697, 22.09042035, 20.26657819, 20.3974, 28.6699929,
                                       15.491997, 23.00, 28.45000633, 31.10002545, 34.29995933,
                                       23.80039349, 12.57038129, 8.900372741, 10.56257331, 21.30039105,
                                       19.25023195, 24.79997072, 25.57049217, 23.71039899, 25.6669979,
                                       19.82042971, 11.93499371, 31.51997398,26.44999921, 27.3333303,
                                       12.92038576, 23.83540428, 27.59998069, 30.32040895, 22.58039044]

total_rainfall_in_states['Longitude'] = [92.73598262, 78.57002559, 93.61660071, 94.21666744, 87.4799727, 
                                         76.78000565, 82.15998734, 73.0166178, 72.8328, 77.23000403,
                                        73.81800065, 72.00, 77.01999101, 77.16659704, 74.46665849,
                                        86.41998572, 76.91999711, 76.56999263, 72.63686717, 76.13001949,
                                        73.16017493, 93.95001705, 91.8800142, 92.72001461, 94.11657019,
                                        85.90001746, 79.83000037, 75.98000281, 74.63998124, 88.6166475,
                                        79.15004187, 91.27999914, 78.05000565, 78.05000565, 88.32994665]


# In[ ]:


total_rainfall_in_states.head()


# In[ ]:


# Let's plot on the map
annual_rainfall_states = total_rainfall_in_states[['Latitude', 'Longitude', 'ANNUAL']]
annual_rainfall_states


# > fig = gmaps.figure()
# 
# > gini_layer = gmaps.geojson_layer(india_geojson, fill_color='white')
# 
# > fig.add_layer(gini_layer)
# 
# > heatmap_layer = gmaps.heatmap_layer(
# >                 annual_rainfall_states[['Latitude', 'Longitude']], weights = annual_rainfall_states['ANNUAL'],
# >                 max_intensity = 100000, point_radius = 30.0)
# 
# > fig.add_layer(heatmap_layer)
# 
# > fig

# ![]https://github.com/saxenakrati09/Indian_Rainfall/blob/master/map%20(2).png
# 

# ![](https://github.com/saxenakrati09/Indian_Rainfall/blob/master/map%20(2).png)

# In[ ]:




