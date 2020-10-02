#!/usr/bin/env python
# coding: utf-8

# I will analyze the crimes in Boston during the last 3 years.
# My analysis is divided in three parts :
# 
# I) Geographical analysis 
# II) Temporal analysis  
# III) Presentation of different types of crimes

# In[ ]:


#Imporation of usefull libraries

import numpy as np 
import pandas as pd #data structure
import seaborn as sns
import matplotlib.pyplot as plt
import folium #maps
from folium.plugins import HeatMap
import os


# #Loading data
# 
# The database of crimes goes from june 2015 to september 2018. We will focus on the entiere years, that is to say 2016 and 2017. 

# In[ ]:


dataCrime=pd.read_csv('../input/crime.csv', encoding="latin-1")
crime=dataCrime.loc[dataCrime['YEAR'].isin([2016,2017])] #Remove years 2015 and 2018
crime.head() #Quick glance to the data


# So we got a database of crimes in Boston which contains in particulary for every crime : Offense Description, Date and Hour,Disctrict and Coordonates. We will make an analysis based on those data. First of all, let's have a view on Boston. 

# **I. Geographical analysis**

# In[ ]:


Boston=folium.Map(location=[42.340,-71.05], #Initialization of the map
               zoom_start=4
)

folium.CircleMarker([42.340,-71.05],
                        radius=20,
                        fill_color="#b22222"
                       ).add_to(Boston)

Boston


# In[ ]:


#Quick coordonates cleaning
crime.Lat.replace(-1, None, inplace=True)
crime.Long.replace(-1, None, inplace=True)


# In[ ]:


f1=plt.figure(1) #Figure of all different crimes
crime.plot(kind="scatter", x="Long", y="Lat", alpha=0.01, s=1)
plt.title('Representation of crimes in Boston between 2016 and 2017')

f2=plt.figure(2) #Figure of ballistic crimes
ballistic_crimes=crime.loc[crime.OFFENSE_CODE_GROUP=='Ballistics']
ballistic_crimes=ballistic_crimes[['Lat','Long']]
ballistic_crimes.plot(kind="scatter", x="Long", y="Lat", s=1)
plt.title('Reprensation of ballistic crimes in Boston from 2015 to 2018')
plt.show()


# Let's see where the ballistic crimes are located on a map.

# In[ ]:


ballistic_crimes.Lat.fillna(0, inplace = True) #Remove NaN coordonate values
ballistic_crimes.Long.fillna(0, inplace = True) 

map=folium.Map(location=[42.320,-71.05], #Initiate map on Boston city
               zoom_start=12
)

HeatMap(data=ballistic_crimes, radius=16).add_to(map)


map


# What are the most sensitive discrits ?

# In[ ]:


data_district=crime.copy() 
data_district.DISTRICT.value_counts() #Observation of the number of crimes per district


# In[ ]:



data_district=data_district.replace(['D14','A15','A7','E13','E18','E5'],'Other Districts') #Regroup the districts in which there is the less crimes
crime_per_district=data_district.DISTRICT.value_counts()


plt.figure(figsize=(8,8))
colors=['cornflowerblue','lightcoral','orange','gold','peachpuff','lightgreen','skyblue']
plt.pie(crime_per_district.values, labels=crime_per_district.index, autopct='%1.1f%%', startangle=90, colors=colors)
plt.title('Repartition of crimes by Districts')
plt.show()


# In[ ]:


sns.scatterplot(data=data_district, x='Long', y='Lat', hue='DISTRICT', alpha=0.003, palette=colors, hue_order=['Other Districts','B2','C11','D4','B3','A1','C6'])


# **II) Temporal analysis**

# Now we can wonder what are the hours during which there is the most and the less crimes.

# In[ ]:


sns.countplot(x='HOUR', data=crime)


# Note : The majority of crimes is realised at the beginning of the night and the minority between 2AM and 6AM. Same for the days of the week and the months.

# In[ ]:


labels=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
count_days=crime.DAY_OF_WEEK.value_counts()
plt.figure(figsize=(8,4))
ax=sns.barplot(count_days.index, count_days.values, palette=colors)
ax.set_xlabel('Day of Week')
ax.set_xticklabels(labels, rotation=50)
ax.set_ylabel('Number of crimes')
plt.tight_layout()


# In[ ]:


labels_months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
count_months=crime.MONTH.value_counts()
plt.figure(figsize=(8,4))
ax=sns.barplot(count_months.index, count_months.values)
ax.set_xlabel('Months')
ax.set_xticklabels(labels_months)
plt.tight_layout()


# 
# 
# We can also wonder what could be the evolution of number of crimes over the years, but 2 years of data is too few to observe a real trend.
# 
# 

# **III) Types of crimes in Boston**

# In[ ]:


crime_type=crime.OFFENSE_CODE_GROUP.value_counts()
crime_type 


# In[ ]:


labels_months=crime_type.index
count_type=crime.OFFENSE_CODE_GROUP.value_counts()
plt.figure(figsize=(10,10))
ax=sns.barplot(count_type.index, count_type.values)
ax.set_xlabel('Offense code group')
ax.set_xticklabels(labels_months, rotation=90)
plt.tight_layout()


# 
# 
# Thank you for reading me, it was my first Kernel so don't hesitate to tell me your advices ! 
# 
# 
