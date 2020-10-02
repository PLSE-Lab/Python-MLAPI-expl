#!/usr/bin/env python
# coding: utf-8

# # Air Pollution in India 2020
# 
# Before we start, we should know what air pollution means? <br><br>
# It is a mixture of various gases and solid particles in the air originated from various sources like chemicals from factories, car emissions, dust, gases emitting devices etc. Air pollution varies in different parts of the country depending upon its industrial area, population etc. <br><br>
# Now, the amount of air pollution is measured using the **AQI (Air quality Index)**, it is like a measuring index which shows the changes in the amount of pollution present in the air. It is specifically used for the same purpose.<br><br>
# **"What is Air Quality Index?"/ Business Standard**<br> <br>
# It is a measure of how air pollution affects one's health within a short time period. The purpose of the AQI is to help people know how the local air quality impacts their health. The Environmental Protection Agency (EPA) calculates the AQI for five major air pollutants, for which national air quality standards have been established to safeguard public health.
#  
# 1. Ground-level ozone
# 2. Particle pollution/particulate matter (PM2.5/pm 10)
# 3. Carbon Monoxide
# 4. Sulfur dioxide
# 5. Nitrogen dioxide
# 
# ![](https://w.ndtvimg.com/sites/3/2019/12/18122812/air_pollution_standards_cpcb.png)
# 
# 
# https://www.business-standard.com/about/what-is-air-quality-index
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import plotly.express as px
from IPython.display import HTML,display

warnings.filterwarnings("ignore")

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Exploring the Data
# Here, we will be exploring the data in the given city wise csv file and make data frames according to our needs, i.e one data frame between the duration of 2015-2019(pre corona) and the other one of 2020 (post corona)

# In[ ]:


city_day = pd.read_csv("/kaggle/input/air-quality-data-in-india/city_day.csv")


# In[ ]:


city_day.shape
city_day.head()
#city_day.columns


# The gases data that we have are: ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2','O3', 'Benzene', 'Toluene', 'Xylene', 'AQI', 'AQI_Bucket']

# In[ ]:


city_day.mean()


# **Replacing the Null values with means**

# In[ ]:


gases = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']

city_day = city_day.fillna(city_day.mean())
city_day.head()


# ## Creating a Pre-covid 19 city wise air quality data frame

# In[ ]:


req_df_precovid = city_day.loc[(city_day['Date'] > '2014-12-31') & (city_day['Date'] < '2020-01-01')]
req_df_precovid = req_df_precovid.groupby("City").mean()
req_df_precovid.head()
#req_df_precovid.shape
#print (list(req_df_precovid.index))


# ### The output here:
# **Shape:** (21, 13) <br/><br/>
# **Index list:**['Ahmedabad', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Delhi', 'Gurugram', 'Guwahati', 'Hyderabad', 'Jaipur', 'Jorapokhar', 'Kolkata', 'Lucknow', 'Mumbai', 'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram']

# ## Creating a Post-covid 19 city wise air quality data frame

# In[ ]:


req_df_postcovid = city_day.loc[(city_day['Date'] > '2019-12-31') & (city_day['Date'] < '2020-06-01')]
req_df_postcovid = req_df_postcovid.groupby("City").mean()
req_df_postcovid.head()
#req_df_postcovid.shape
#print (list(req_df_postcovid.index))


# ### The output here:
# **Shape:** (24, 13) <br/><br/>
# **Index list:**['Ahmedabad', 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru', 'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad', 'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow', 'Mumbai', 'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram']

# ## Visualizing and compairing Pre-Covid and Post-Covid Atmosphere

# **1) Compairing SO2 levels**

# In[ ]:


precovidchartSO2 = sns.barplot(x=req_df_precovid.index, y=req_df_precovid['SO2'], palette = "bright")
precovidchartSO2.set_xticklabels(precovidchartSO2.get_xticklabels(), rotation=90)
plt.title("Pre Covid SO2 levels in the atmosphere")


# In[ ]:


postcovidchartSO2 = sns.barplot(x=req_df_postcovid.index, y=req_df_postcovid['SO2'], palette = "bright")
postcovidchartSO2.set_xticklabels(postcovidchartSO2.get_xticklabels(), rotation=90)
plt.title("Post Covid SO2 levels in the atmosphere")


# The main source of sulfur dioxide in the air is the industrial activity that processes materials that contain sulfur, eg the generation of electricity from coal, oil, or gas that contains sulfur.<br><br>
# We can see here that the SO2 levels of some cities like Ahmedabad, Amaravati, Bhopal, Hyderabad, etc have reduced while the levels of cities like Amritsar, Bangalore, etc still were increasing. <br><br>This is because they come under the Industrial regions of India that produces electricity and their activities must have increased with the new year's demand at the start of 2020, causing an increase in the AQI of SO2, but with the passing time, even these cities have recorded an all-time low AQI levels post-COVID lockdown.

# **2) Specific city wise comparison of particular Gases in Various Cities**

# In[ ]:


maincities = ['Bengaluru', 'Delhi', 'Mumbai', 'Patna']
maingases = ['NH3', 'O3', 'NO2']
req_df_precovid_mod = req_df_precovid.loc[(req_df_precovid.index.isin(maincities))]
req_df_precovid_mod = req_df_precovid_mod[maingases]
#predf = pd.melt(req_df_precovid_mod, id_vars = req_df_precovid_mod.index , var_name="Gases", value_name="Gas Levels")
print(req_df_precovid_mod)
#req_df_precovid_mod = req_df_precovid_mod.set_index([req_df_precovid_mod.index, '']).value
#req_df_precovid_mod.unstack().plot(kind='bar')
#precovidchart = sns.barplot(x = req_df_precovid_mod.index, y=req_df_precovid_mod['SO2'])


# In[ ]:


barWidth = 0.25
 
# set height of bar
bars1, bars2, bars3 = list(req_df_precovid_mod.NH3), list(req_df_precovid_mod.O3), list(req_df_precovid_mod.NO2)
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#994BBB', width=barWidth, edgecolor='black', label='NH3')
plt.bar(r2, bars2, color='#F12290', width=barWidth, edgecolor='black', label='O3')
plt.bar(r3, bars3, color='#F7E44A', width=barWidth, edgecolor='black', label='NO2')
 
# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], list(req_df_precovid_mod.index))
 
# Create legend & Show graphic
plt.legend()
plt.show()


# ## Visualizing Post-Covid Atmosphere

# In[ ]:


req_df_postcovid_mod = req_df_postcovid.loc[(req_df_postcovid.index.isin(maincities))]
req_df_postcovid_mod = req_df_postcovid_mod[maingases]

# set height of bar
bars1, bars2, bars3 = list(req_df_postcovid_mod.NH3), list(req_df_postcovid_mod.O3), list(req_df_postcovid_mod.NO2)
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#994BBB', width=barWidth, edgecolor='black', label='NH3')
plt.bar(r2, bars2, color='#F12290', width=barWidth, edgecolor='black', label='O3')
plt.bar(r3, bars3, color='#F7E44A', width=barWidth, edgecolor='black', label='NO2')
 
# Add xticks on the middle of the group bars
plt.xlabel('group', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], list(req_df_precovid_mod.index))
 
# Create legend & Show graphic
plt.legend()
plt.show()


# ### HeatMaps for a better understanding
# #### Pre-Covid

# In[ ]:


plt.title("Gas Levels in different cities Pre-Covid")
plt.figure(figsize=(14,7))
# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=req_df_precovid, annot=False, cmap='inferno', fmt='g')


# #### Post - Covid

# In[ ]:


plt.title("Gas Levels in different cities Post-Covid")
plt.figure(figsize=(14,7))
# Heatmap showing average arrival delay for each airline by month
sns.heatmap(data=req_df_postcovid, annot=False, cmap='inferno', fmt='g')


# City wise Comparison between 2019 and 2020

# **Mumbai**

# In[ ]:


df_2019 = city_day.loc[(city_day['Date'] > '2019-03-31') & (city_day['Date'] < '2019-04-15')]
df_2020 = city_day.loc[(city_day['Date'] > '2020-03-31') & (city_day['Date'] < '2020-04-15')]
compare_df = pd.concat([df_2019, df_2020])
compare_df.head()
compare_df.shape

df_mumbai = compare_df[compare_df['City'] == "Mumbai"]
df_mumbai[['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'SO2', 'NH3', 'AQI', "Date"]].style.background_gradient(cmap='Blues')


# **Delhi**

# In[ ]:


df_delhi = compare_df[compare_df['City'] == "Delhi"]
df_delhi[['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'SO2', 'NH3', 'AQI', "Date"]].style.background_gradient(cmap='Purples')


# **Banglore**

# In[ ]:


df_bang = compare_df[compare_df['City'] == "Bengaluru"]
df_bang[['PM2.5', 'PM10', 'NO2', 'NOx', 'CO', 'SO2', 'NH3', 'AQI', "Date"]].style.background_gradient(cmap='Reds')


# In[ ]:


compare_df['Year'] = pd.DatetimeIndex(compare_df['Date']).year
compo3 = sns.catplot(x="City", y="O3", hue="Year", data = compare_df, height=10, aspect=1.8, kind="bar", palette="bright")
compo3.set_xticklabels(rotation=90)


# In[ ]:


compnh3 = sns.catplot(x="City", y="NH3", hue="Year", data = compare_df, height=10, aspect=1.8, kind="swarm", palette="bright")
compnh3.set_xticklabels(rotation=90)


# Hence, I would like to conclude by saying that the pollution levels have obviously decreased and can be more clearly identified from the April month of this year, when the whole country went under lockdown.
# 
