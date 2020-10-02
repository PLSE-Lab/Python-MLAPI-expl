#!/usr/bin/env python
# coding: utf-8

# # EDA on Global Missing Migrants

# This notebook aims to understand the pheonmena of missing migrants from 2014 to early 2019.
# 
# It would study the following questions:
# 1. Any change in the trend of migrants? 
# 2. Where did the death/ missing migrants mostly happen? 
# 3. What are the causes of the death of migrants? 
# 4. Any causes that led to the higher death rate of the migrants? 

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from datetime import datetime

import folium
from folium.plugins import HeatMap
from folium.plugins import FastMarkerCluster
from IPython.display import HTML, display

import random

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


#read data
df0 = pd.read_csv("../input/MissingMigrants-Global-2019-03-29T18-36-07.csv")


# # Index

# [1. Snapshtot on data](#1.-Snapshtot-on-data)
# 
# [2. Data Cleansing](#2.-Data-Cleansing)
# 
# [3. Exploratory data analysis](#3.-Exploratory-data-analysis)
#    * [3.1 Trend of migrants and corresponding death/missing over Time](#3.1-Trend-of-migrants-and-corresponding-death/missing-over-Time)
#    * [3.2 Illustration of number of dead/missing on map](#3.2-Illustration-of-number-of-dead/missing-on-map)
#    * [3.3 Cause of deaths of migrants](#3.3-Cause-of-deaths-of-migrants)
#    * [3.4 Illustration of death rate of top causes](#3.4-Illustration-of-death-rate-of-top-causes)

# # 1. Snapshtot on data

# In[ ]:


df0.sample(10)


# ### Info on types of data:

# In[ ]:


df0.info()


# Utilize only useful columns for EDA

# In[ ]:


df = df0[['Web ID','Region of Incident',
         'Reported Date', 
         'Total Dead and Missing',
         'Number of Survivors', 
        'Cause of Death',
        'Location Description',
        'Information Source',
        'Location Coordinates',
        'Migration Route',
        'URL','UNSD Geographical Grouping']]


# ## 2. Data Cleansing

# Data cleansing will involve the following steps:
#     1. Remove duplicated rows if any
#     2. Find out and process N/A row in column 'Location Coordinates'
#     3. Change the format of column 'Reported Date' to datetime format
#     4. Process odd entries(i.e. 0 victims under column 'Total Dead and Missing' but Number of Survivors is unknown )

# 

# 1. Check if there are duplicated rows and remove if any

# In[ ]:


df[df['Web ID'].duplicated()]


# 2. Find out N/A entry under column 'Location Coordinates'

# In[ ]:


df[df['Location Coordinates'].isna()]


# Since there is only one row and with only 1 casualty, I will just remove the row

# In[ ]:


df = df[df['Location Coordinates'].notna()]


# In[ ]:


#also split the column for later use
df['Location Coordinates'] = df['Location Coordinates'].str.split(",")
df['Location Coordinates_lon'] = pd.to_numeric(df['Location Coordinates'].str[0][:])
df['Location Coordinates_lat'] = pd.to_numeric(df['Location Coordinates'].str[1][:])
df.drop('Location Coordinates',axis = 1)


# 3. Change 'Reported Date' column to datetime format

# In[ ]:


df['Reported Date'] = pd.to_datetime(df['Reported Date'], infer_datetime_format=True)


# 4. Inspect rows with no 'Total Dead and Missing'

# In[ ]:


df[df['Total Dead and Missing']==0]


# Fill 'Total Dead and Missing' according to information from available URL

# In[ ]:


df['Total Dead and Missing'][4599]=11
df['Total Dead and Missing'][5013]=6


# Drop rows that cannot be explained

# In[ ]:


#df.drop(index=df[df['Total Dead and Missing']==0].index,inplace = True)


# Add a row to sum 'Total Dead and Missing' and 'Number of Survivors'

# In[ ]:



df.insert(6,'Total Migrants',0)
df['Total Migrants'] = df['Total Dead and Missing']+df['Number of Survivors']


# Snapshot on Dataframe after cleansing

# In[ ]:


df.head()


# *For rows with unknown 'Number of Survivors' the total migrants are still unknown

# # 3. Exploratory data analysis

# ## 3.1 Trend of migrants and corresponding death/missing over Time

# In[ ]:


df_temp = df.groupby('Reported Date')['Total Dead and Missing'].sum().reset_index()
df_temp['Reported_Month']=df_temp['Reported Date'].dt.strftime('%Y-%m')
df_temp = df_temp.groupby('Reported_Month')['Total Dead and Missing'].sum().reset_index()


# In[ ]:


df_temp = df.groupby('Reported Date')['Total Dead and Missing','Total Migrants'].sum().reset_index()
df_temp['Reported_Month']=df_temp['Reported Date'].dt.strftime('%Y-%m')
df_temp = df_temp.groupby('Reported_Month')['Total Dead and Missing','Total Migrants'].sum().reset_index()


# In[ ]:


fig,ax =plt.subplots(figsize=(18,10));
ax = sns.lineplot(x="Reported_Month", y="Total Dead and Missing", data=df_temp, color="r").set_title('Total Dead and Missing Migrants over time');
plt.xticks(rotation='vertical');

ax2 = sns.lineplot(x="Reported_Month", y="Total Migrants", data=df_temp, color="b");

ax.figure.legend(['Total Dead and Missing','Total Migrants'],fontsize='large');

plt.xlabel('Date');
plt.ylabel('No. of Migrants');


# ## 3.2 Illustration of number of dead/missing on map

# In[ ]:


m = folium.Map(location=[30, 20], zoom_start=3)

m.add_children(HeatMap(zip(df['Location Coordinates_lon'],
                           df['Location Coordinates_lat'], 
                           df['Total Dead and Missing']),
                           min_opacity = 0.2))

FastMarkerCluster(data=list(zip(df['Location Coordinates_lon'].values, df['Location Coordinates_lat'].values))).add_to(m)
folium.LayerControl().add_to(m)


display(m)


# ## 3.3 Cause of deaths of migrants

# Here below are the full list of causes of the death/ missing migrants:

# In[ ]:


df['Cause of Death'].sort_values().unique()


# ### Graphic Illustration on Top cause for dead or missing vs Total migrants:

# In[ ]:


df_cause_of_death=df.drop(index=df[df['Total Migrants'].isna()].index)
df_cause_of_death=df_cause_of_death.groupby('Cause of Death').sum()[['Total Dead and Missing','Total Migrants']].sort_values(
    by = 'Total Dead and Missing',ascending = False).reset_index() 


# In[ ]:


#define sorter
sorter_cause_of_death= list(df_cause_of_death['Cause of Death'].iloc[0:30])


# In[ ]:


df_cause_of_death1 = pd.melt(df_cause_of_death,id_vars=['Cause of Death'],
        value_vars = ['Total Dead and Missing','Total Migrants'])
df_cause_of_death1 = df_cause_of_death1[df_cause_of_death1['Cause of Death'].isin(sorter_cause_of_death)]
#change to log value
df_cause_of_death1.rename({'value': 'Total(log10)'}, axis=1, inplace=True)
df_cause_of_death1['Total(log10)'] = np.log10(df_cause_of_death1['Total(log10)'])


# In[ ]:


chart = sns.catplot(kind="bar", data = df_cause_of_death1,
            x='Cause of Death',y='Total(log10)',hue = 'variable',
            hue_order = ['Total Migrants','Total Dead and Missing'],
            order = sorter_cause_of_death, height=8,aspect = 1.5)
chart.fig.suptitle('Top Causes for Dead or Missing Migrants')

for axes in chart.axes.flat:
    axes.set_xticklabels(axes.get_xticklabels(), rotation=60,horizontalalignment='right')

plt.ylabel('Number of Migrants');


# ### Figures on Causes of Death happened in regions

# In[ ]:


df_location = df[df['Cause of Death'].isin(sorter_cause_of_death)]

df_location1 = df_location[['Region of Incident','Cause of Death','Total Dead and Missing']]

df_location1 = df_location1.groupby(['Region of Incident','Cause of Death']).sum().unstack(level=-1)
df_location1.columns = df_location1.columns.droplevel() 
df_location1 = df_location1.fillna(0)


# In[ ]:


fig, ax = plt.subplots(figsize=(18,10));
sns.heatmap(df_location1,cmap = 'Wistia',annot=True, annot_kws={"size": 7},linewidths=0.25,ax=ax).set_title('Total Dead and Missing by Cause of Death and Region of Incident');


# ### Illustration on map with sampling size of 5000:

# In[ ]:


colors =  ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
             'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
             'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
             'gray', 'black', 'lightgray']


# In[ ]:


m = folium.Map(location=[30, 20], zoom_start=3)

df_location2 = df_location.sample(1500)
for cause,color in zip(sorter_cause_of_death[0:15],colors):
    for x,y,z in zip(df_location2[df_location2['Cause of Death']==str(cause)]['Location Coordinates_lon'],
                     df_location2[df_location2['Cause of Death']==str(cause)]['Location Coordinates_lat'],
                     df_location2[df_location2['Cause of Death']==str(cause)]['Total Dead and Missing']):
        folium.Circle(radius=z*1000,
                  location=[x,y],
                  color = color,
                  popup='{0}, no. of death/missing: {1}'.format(cause,z),
                  fill=True).add_to(m)

display(m)


# ## 3.4 Illustration of death rate of top causes

# In[ ]:


sorter_cause_of_death1= list(df_cause_of_death['Cause of Death'].iloc[0:15])
df_location2 = df[df['Cause of Death'].isin(sorter_cause_of_death1)]
df_location2 = df_location2[['Region of Incident','Cause of Death','Total Dead and Missing','Total Migrants']]
df_location2['Total Migrants'].fillna(df_location2['Total Dead and Missing'],inplace=True)


# In[ ]:


df_location2_1 = df_location2.groupby('Cause of Death').count()['Total Dead and Missing'].reset_index()
df_location2_1 = df_location2_1.rename({'Total Dead and Missing':'Count of Cases'},axis=1)
df_location2_2 = df_location2.groupby('Cause of Death').sum().reset_index()
df_location2_2 = df_location2_2.rename({'Total Dead and Missing':'Sum of Total Dead and Missing',
                                        'Total Migrants':'Sum of Total Migrants'},axis=1)
df_location2_3 = pd.merge(df_location2_2,df_location2_1,on=['Cause of Death'],how='left')


# In[ ]:


df_location2_3.sort_values(by = 'Sum of Total Dead and Missing', ascending = False)


# In[ ]:


df_location2_3['Sum of Total Dead and Missing'] = np.log10(df_location2_3['Sum of Total Dead and Missing'])
df_location2_3['Sum of Total Migrants'] = np.log10(df_location2_3['Sum of Total Migrants'])
df_location2_3.rename({'Sum of Total Dead and Missing':'Sum of Total Dead and Missing(log10)','Sum of Total Migrants':'Sum of Total Migrants(log10)' },axis=1,inplace=True)


# ### Visualized Illustration

# In[ ]:


fig, ax = plt.subplots(figsize=(18,10));
ax = sns.scatterplot(x="Sum of Total Migrants(log10)", y="Sum of Total Dead and Missing(log10)",
                     hue="Cause of Death", size="Count of Cases", 
                     sizes=(100, 2000),alpha = 0.5,legend='brief',data=df_location2_3)

for line in range(0,df_location2_3.shape[0]):
     ax.text(x=df_location2_3['Sum of Total Migrants(log10)'][line],
             y= df_location2_3['Sum of Total Dead and Missing(log10)'][line],
             s=df_location2_3['Cause of Death'][line],
             horizontalalignment='center', size='small', color='black', weight='semibold',linespacing=5)
ax.set_title('Sum and Count of Total Dead and Missing Migrants by Cause of Death');


# * Causes with high gradient means higher death rate(e.g. easier to cause death by Dehydration than Asphyxiation)

# ## It's my first post in Kaggle.<br>Please feel free to give me advice or upvote!
