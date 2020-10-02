#!/usr/bin/env python
# coding: utf-8

# **911 Emergency Calls  - Data Analysis & Spatial Visualisation**

# In[ ]:


#Importing Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Loading Dataset
url = '../input/data-analysis-of-911-emergency-calls/911.csv'
data = pd.read_csv(url, header='infer')


# **Data Exploration & Clean up**

# In[ ]:


data.shape


# In[ ]:


#checking for null / missing values
data.isna().sum()


# In[ ]:


#dropping index with missing or null values
data = data.dropna()


# In[ ]:


data.reset_index(inplace=True, col_level=1, drop=True)


# In[ ]:


data.head()


# Looking at the dataset above, the following columns serves no purpose for my analysis
# * desc
# * zip
# * e
# 
# Hence I shall drop these columns.

# In[ ]:


data = data.drop(columns=['desc','zip','e'], axis=1)


# In my analysis, I'm more interested in "type" of emergencies rather than "details" of those emergencies. Hence creating a function to extract the emergency type from the column = title

# In[ ]:


# Function to extract emergency type
def extract_emergency_type(txt):
    #res = re.findall(r"^\w+",txt)
    result = ''.join(re.findall(r"^\w+",txt))
    return result

#Applying the function to column = title
data['type'] = data['title'].apply(extract_emergency_type)


# In[ ]:


data.head()


# Now, performing the following actions in that sequence:
# * Dropping the title column
# * Renaming the twp column
# * Re-arraning the columns
# * Changing text to lower case

# In[ ]:


# drop column = title
data = data.drop(columns='title',axis=1)


# In[ ]:


# rename the column = twp
data = data.rename(columns={'twp':'area'})


# In[ ]:


# re-arrange the columns
data = data[['timeStamp','type','area','addr','lat','lng']]


# In[ ]:


#changing to lower case
data['type'] = data['type'].str.lower()
data['area'] = data['area'].str.lower()
data['addr'] = data['addr'].str.lower()


# In[ ]:


data.head()


# **Processing the Time Stamp Column**

# In[ ]:


# -- Processing the TimeStamp column

data.timeStamp = pd.to_datetime(data.timeStamp, format='%Y-%m-%d %H:%M:%S')
data['year'] = data.timeStamp.apply(lambda x: x.year)
data['month'] = data.timeStamp.apply(lambda x: x.month)
data['day'] = data.timeStamp.apply(lambda x: x.day)
data['hour'] = data.timeStamp.apply(lambda x: x.hour)


# In[ ]:


data.head()


# In[ ]:


#Taking backup
data_backup = data.copy()


# **Analysis & Visualisation of Year 2016**

# ***Emergencies per Month in 2016***

# In[ ]:


# Emergencies per Month in 2016
emrgncy_monthly = pd.DataFrame(data[data['year']==2016].groupby('month').size())
emrgncy_monthly['MEAN'] = data[data['year']==2016].groupby('month').size().mean()

plt.figure(figsize=(18,6))
data[data['year']==2016].groupby('month').size().plot(label='Emergencies per month')
emrgncy_monthly['MEAN'].plot(color='red', linewidth=2, label='Average', ls='--')
plt.title('Total Monthly Emergencies in 2016', fontsize=14)
plt.xlabel('Month')
plt.xticks(np.arange(1,9))
plt.ylabel('Number of emergencies')
plt.tick_params(labelsize=10)
plt.legend(prop={'size':10})


# **Analysis:**
# * January & June-July had "above average" number of emergencies
# * Number of emergencies fell to "low" in August

# ***Emergency Types per Month in 2016***

# In[ ]:


# Emergency Types per Month in 2016
emrgncy_type_yrly = data[data['year']==2016].pivot_table(values='area', index='month', columns='type', aggfunc=len).plot(figsize=(15,6), linewidth=2)
plt.title('Emergency Types per Month in 2016', fontsize=16)
plt.xlabel('Month')
plt.xticks(np.arange(1,9))
plt.legend(prop={'size':10})
plt.tick_params(labelsize=10)


# **Analysis:**
# * The hihgest type of emergencies in 2016 were EMS (Emergency Medical Services) & Traffic
# * Hihgest EMS & Traffic emergencies were in July 2016
# 

# ***Daily Emergencies in July 2016***

# In[ ]:


#Daily Emergencies in July 2016
emrgncy_daily = pd.DataFrame(data[(data['year']==2016) & (data['month']==7)].groupby('day').size())
emrgncy_daily['MEAN'] = data[(data['year']==2016) & (data['month']==7)].groupby('day').size().mean()

plt.figure(figsize=(18,6))
data[(data['year']==2016) & (data['month']==7)].groupby('day').size().plot(label='Emergencies per day in July')
emrgncy_daily['MEAN'].plot(color='red', linewidth=2, label='Average', ls='--')
plt.title('Daily Emergencies in July 2016', fontsize=14)
plt.xlabel('Day')
plt.xticks(np.arange(1,31))
plt.ylabel('Number of emergencies')
plt.tick_params(labelsize=10)
plt.legend(prop={'size':10})


# **Analysis:**
# * Hihgest number of emergencies was between 25 - 26 July 2016
# * Lowest number of emergencies was between 2- 4 July 2016

# ***Emergency Types in month of July 2016***

# In[ ]:


# Emergency Types in month of July 2016
emrgncy_type_mnthly = data[(data['year']==2016) & (data['month']==7)].pivot_table(values='area', index='day', columns='type', aggfunc=len).plot(figsize=(15,6), linewidth=2)
plt.title('Emergency Types in month of July 2016', fontsize=16)
plt.xlabel('Day')
plt.xticks(np.arange(1,31))
plt.legend(prop={'size':10})
plt.tick_params(labelsize=10)


# **Analysis:**
# * Fire & Traffic emergencies were highest on 25 July 2016
# * Traffic emergencies were lowest on 24 July 2016, however there was a sharp rise on 25 July 2016
# * Traffc emergencies were also highest on 13 July 2016

# ***Hourly Emergencies on 25 July 2016***

# In[ ]:


#Hourly Emergencies on 25 July 2016
emrgncy_hourly = pd.DataFrame(data[(data['year']==2016) & (data['month']==7) & (data['day']==25)].groupby('hour').size())
emrgncy_hourly['MEAN'] = data[(data['year']==2016) & (data['month']==7) & (data['day']==25)].groupby('hour').size().mean()

plt.figure(figsize=(18,6))
data[(data['year']==2016) & (data['month']==7) & (data['day']==25)].groupby('hour').size().plot(label='Emergencies on 25 July 2016')
emrgncy_hourly['MEAN'].plot(color='red', linewidth=2, label='Average', ls='--')
plt.title('Hourly Emergencies on 25 July 2016', fontsize=14)
plt.xlabel('Hours')
plt.xticks(np.arange(1,24))
plt.ylabel('Number of emergencies')
plt.tick_params(labelsize=10)
plt.legend(prop={'size':10})


# **Analysis:**
# 1. Highest number of emergencies between 6 - 8 PM
# 2. Lowest number of emergencies at 2 AM in morning

# ***Emergency Types on 13th & 25th July 2016***

# In[ ]:


# Emergency Types on 25th July 2016
emrgncy_type_hourly = data[(data['year']==2016) & (data['month']==7) & (data['day']==25)].pivot_table(values='area', index='hour', columns='type', aggfunc=len).plot(figsize=(15,6), linewidth=2)
plt.title('Emergency Types on 25th July 2016', fontsize=16)
plt.xlabel('Hours')
plt.xticks(np.arange(1,24))
plt.legend(prop={'size':10})
plt.tick_params(labelsize=10)



# In[ ]:


# Emergency Types on 13th July 2016
emrgncy_type_hourly = data[(data['year']==2016) & (data['month']==7) & (data['day']==13)].pivot_table(values='area', index='hour', columns='type', aggfunc=len).plot(figsize=(15,6), linewidth=2)
plt.title('Emergency Types on 13th July 2016', fontsize=16)
plt.xlabel('Hours')
plt.xticks(np.arange(1,24))
plt.legend(prop={'size':10})
plt.tick_params(labelsize=10)


# **Analysis:**
# On 25 Jul 2016, Fire & Traffic emergencies were highest between 5 - 8 PM
# On 25 Jul 2016, Medical emergencies were lower compared to Fire & Traffic
# 
# On 13 Jul 2016, Traffic emergencies were highest between 4 - 5 PM
# On 13 Jul 2016, Fire emergencies were lower compared to Traffic & EMS

# Since the dataset has data only for December for 2015 & Jan - Aug for 2016, it is not possible to compare the yearly stats for the emergencies. 
# 
# Therefore I shall focus on the transition period i.e. Dec 2015 - Jan 2016.

# ***Emergencies per Type During New Year Time [Dec 2015- Jan 2016]***

# In[ ]:


# Emergencies per Type During New Year Time [Dec 2015- Jan 2016]

#creating a specific dataframe
ny_df = data[(data['year'].isin(['2015','2016'])) & (data['month'].isin(['12','1']))]

ny_emergencies = ny_df.pivot_table(values='area', index='type', columns=['month','year'], aggfunc=len)
ny_emergencies.columns = ['Jan-2016','Dec-2015']


# Using seaborn heatmap
plt.figure(figsize=(6,6))
plt.title('Emergencies per Type During New Year Time [Dec 2015- Jan 2016]', fontsize=14)
plt.tick_params(labelsize=10)
sns.heatmap(ny_emergencies, cmap='icefire', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".0f")


# **Analysis:**
# EMS Emergencies increased by 56.74% from Dec-2015 to Jan 2016
# Fire Emergencies increased by 74.29% from Dec-2015 to Jan 2016
# Traffic Emergencies increased by 78.83% from Dec-2015 to Jan 2016
# 
# There was a drastic increase in Traffic Emergencies between Dec-2015 & Jan-2016

# ***Percentage Change in Emergency Calls per Type in 2016***

# In[ ]:


#Creating a pivot table for Year = 2016
percChng_2016 = data[data['year']==2016].pivot_table(values='area', index='month', columns='type', aggfunc=len)


# In[ ]:


#Calculating the percentage change
percChng_2016 = percChng_2016.pct_change()


# In[ ]:


#dropping the index with NULL values
percChng_2016 = percChng_2016.dropna()


# In[ ]:


# Plotting the Heatmap - Percentage Change in Emergency Calls per Type in 2016
plt.figure(figsize=(6,6))
plt.title('Percentage Change in Emergency Calls per Type in 2016', fontsize=14)
plt.tick_params(labelsize=10)
ax = sns.heatmap(percChng_2016, cmap='Blues', linecolor='grey',linewidths=0.1, cbar=False, annot=True, fmt=".2%")


# **Analysis:**
# * Certainly a drop in emergencies (all type) in August 2016
# * A big increase in FIRE Emergencies in June 2016
# * A very minor drop in TRAFFIC Emergencies in July 2016
# 

# **Spatial Visualisation**

# In[ ]:


# Geovisualization library
import folium
from folium.plugins import CirclePattern, HeatMap, HeatMapWithTime, FastMarkerCluster


# In[ ]:


# Function to create a base map

def generateBaseMap(default_location=[40.0655815,-75.2430282], default_zoom_start=10):
    base_map = folium.Map(location=default_location, control_scale=True, zoom_start=default_zoom_start,width='50%', height='50%', tiles="cartodbpositron")
    return base_map

#Calling the function
base_map = generateBaseMap()


# In[ ]:


# Spatial Visualisation of Emergencies on 25 Jul 2016
spt_df_25Jul2016 = data[(data['year']==2016) & (data['month']==7) & (data['day']==25)]

for i in range(0,len(spt_df_25Jul2016)):
    
    if spt_df_25Jul2016.iloc[i]['type'] == 'traffic':
        folium.CircleMarker(location=[spt_df_25Jul2016.iloc[i]['lat'], spt_df_25Jul2016.iloc[i]['lng']],
                      popup=spt_df_25Jul2016.iloc[i]['type'],radius = 2, color='#2ca25f',fill=True,    #Green
                      fill_color='#2ca25f').add_to(base_map)
    elif spt_df_25Jul2016.iloc[i]['type'] == 'ems':
        folium.CircleMarker(location=[spt_df_25Jul2016.iloc[i]['lat'], spt_df_25Jul2016.iloc[i]['lng']],
                      popup=spt_df_25Jul2016.iloc[i]['type'],radius = 2, color='#2b8cbe',fill=True,   #Blue 
                      fill_color='#2b8cbe').add_to(base_map)
    else:
         folium.CircleMarker(location=[spt_df_25Jul2016.iloc[i]['lat'], spt_df_25Jul2016.iloc[i]['lng']],
                      popup=spt_df_25Jul2016.iloc[i]['type'],radius = 2, color='#f03b20',fill=True,   #Red
                      fill_color='#f03b20').add_to(base_map)       
    
#Calling the base_map function
base_map


# In[ ]:


# Addiing Cluster Layer 
FastMarkerCluster(data=list(zip(spt_df_25Jul2016['lat'].values, spt_df_25Jul2016['lng'].values))).add_to(base_map)
folium.LayerControl().add_to(base_map)

#Calling the function
base_map


# **Analysis:**
# * Highest Emergency calls were made from Norristown on 25 July 2016
# * Least Emergency calls were made from Skippack on 25 July 2016
# 
