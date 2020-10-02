#!/usr/bin/env python
# coding: utf-8

# ![](http://)

# # Welcome to Chicago Crime Mapping
# ### A simple map and graph based interaction between crimes in chicago and the coordinates.
# 
# ![](https://i.imgur.com/ZZNGFv9.png)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/Chicago_Crimes_2012_to_2017.csv')
df.head()


# In[ ]:


import folium
import matplotlib.pyplot as plt
import seaborn as sns


# # Having a look at the null values in our dataset

# In[ ]:


plt.figure(figsize=(10,7))
sns.heatmap(df.isnull(), cbar = False, cmap = 'viridis')


# In[ ]:


df = df.dropna()
df = df.drop(columns=['Unnamed: 0', 'ID', 'Case Number', 'Block', 'IUCR','Domestic', 'Beat', 'District', 'Ward','X Coordinate', 'Y Coordinate','Updated On', 'FBI Code'], axis = 1)


# # A little self explanatory data exploration from our side.
# Mostly focussing on the top 10 value counts available to us in terms of Location Description and the Primary Type of Crime

# In[ ]:


pd.value_counts(df['Location Description'])[:10]


# In[ ]:


pd.value_counts(df['Primary Type'])[:10]


# # Location Description and it's semantics

# In[ ]:


plt.figure(figsize = (15, 10))
sns.countplot(y= 'Location Description', data = df, order = df['Location Description'].value_counts().iloc[:10].index)


# In[ ]:


chicago_map = folium.Map(location=[41.864073,-87.706819],
                        zoom_start=11,
                        tiles="CartoDB dark_matter")


# In[ ]:


locations = df.groupby('Community Area').first()


# In[ ]:


new_locations = locations.loc[:, ['Latitude', 'Longitude', 'Location Description', 'Arrest']]


# In[ ]:


new_locations.head()


# In[ ]:


popup_text = """Community Index : {}<br
                Arrest : {}<br>
                Location Description : {}<br>"""


# # Preparing the first map. 
# 
# ### Using one location each in a particular community area

# In[ ]:


for i in range(len(new_locations)):
    lat = new_locations.iloc[i][0]
    long = new_locations.iloc[i][1]
    popup_text = """Community Index : {}<br>
                Arrest : {}<br>
                Location Description : {}<br>"""
    popup_text = popup_text.format(new_locations.index[i],
                               new_locations.iloc[i][-1],
                               new_locations.iloc[i][-2]
                               )
    folium.CircleMarker(location = [lat, long], popup= popup_text, fill = True).add_to(chicago_map)


# In[ ]:


chicago_map


# In[ ]:


unique_locations = df['Location'].value_counts()


# In[ ]:


unique_locations.index


# # A simple Criminal Rate Index DataFrame

# In[ ]:


CR_index = pd.DataFrame({"Raw_String" : unique_locations.index, "ValueCount":unique_locations})
CR_index.index = range(len(unique_locations))
CR_index.head()


# In[ ]:


def Location_extractor(Raw_Str):
    preProcess = Raw_Str[1:-1].split(',')
    lat =  float(preProcess[0])
    long = float(preProcess[1])
    return (lat, long)


# In[ ]:


CR_index['LocationCoord'] = CR_index['Raw_String'].apply(Location_extractor)


# In[ ]:


CR_index  = CR_index.drop(columns=['Raw_String'], axis = 1)


# # A Simple Chicago Mapping showing the total criminal rates.
# 
# ( As per the number of total criminal rates)

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nchicago_map_crime = folium.Map(location=[41.895140898, -87.624255632],\n                        zoom_start=13,\n                        tiles="CartoDB dark_matter")\n\nfor i in range(500):\n    lat = CR_index[\'LocationCoord\'].iloc[i][0]\n    long = CR_index[\'LocationCoord\'].iloc[i][1]\n    radius = CR_index[\'ValueCount\'].iloc[i] / 45\n    \n    if CR_index[\'ValueCount\'].iloc[i] > 1000:\n        color = "#FF4500"\n    else:\n        color = "#008080"\n    \n    popup_text = """Latitude : {}<br>\n                Longitude : {}<br>\n                Criminal Incidents : {}<br>"""\n    popup_text = popup_text.format(lat,\n                               long,\n                               CR_index[\'ValueCount\'].iloc[i]\n                               )\n    folium.CircleMarker(location = [lat, long], popup= popup_text,radius = radius, color = color, fill = True).add_to(chicago_map_crime)')


# In[ ]:


chicago_map_crime


# # Having a closer look at the thefts 

# In[ ]:


df_theft = df[df['Primary Type'] == 'THEFT']


# In[ ]:


plt.figure(figsize = (15, 7))
sns.countplot(y = df_theft['Description'])


# In[ ]:


df_theft_data = pd.DataFrame({"Counts": df_theft['Description'].value_counts(), "Description" : df_theft['Description'].value_counts().index})


# In[ ]:


df_theft_data.reset_index(inplace=True)


# In[ ]:


df_theft_data = df_theft_data.drop(columns=['index'], axis = 1)
df_theft_data.head()


# # Maybe a sorted array of counts would look good

# In[ ]:


plt.figure(figsize = (15, 7))
sns.barplot(y ="Description", x = "Counts", data = df_theft_data, palette="jet_r")


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_theft['Date'] = pd.to_datetime(df_theft['Date'])")


# In[ ]:


df_theft['Month'] = df_theft['Date'].apply(lambda x : x.month)


# In[ ]:


theft_in_months = pd.DataFrame({"thefts" : df_theft['Month'].value_counts(), "month" : df_theft["Month"].value_counts().index}, index = range(12))


# In[ ]:


theft_in_months.fillna(0, inplace=True)
theft_in_months = theft_in_months.sort_values(['month'], ascending=[1])


# In[ ]:


theft_in_months.head()


# # An overall monthly trend presented in a plate

# In[ ]:


plt.figure(figsize = (15,7))
plt.plot(theft_in_months['month'],theft_in_months['thefts'], label = 'Total In Month')
plt.plot(theft_in_months['month'],theft_in_months['thefts'].rolling(window = 2).mean(),color='red', linewidth=5, label='2-months Moving Average' )

plt.title('Thefts per month', fontsize=16)
plt.xlabel('Months')
plt.legend(prop={'size':16})
plt.tick_params(labelsize=16);


# In[ ]:


print(max(df_theft['Date']))
print(min(df_theft['Date']))


# In[ ]:


df_theft['Date'].iloc[0].date()


# In[ ]:


df_theft_dates = df_theft['Location']
df_theft_dates.index = df_theft['Date']
resampled = df_theft_dates.resample('D')
df_theft_dates['MEAN'] = resampled.size().mean()
df_theft_dates['STD'] = resampled.size().std()


# In[ ]:


UCL = df_theft_dates['MEAN'] + 3 * df_theft_dates['STD']
LCL = df_theft_dates['MEAN'] -  3 * df_theft_dates['STD']


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nplt.figure(figsize=(20, 7))\nresampled.size().plot(label = "Thefts on a daily basis", color = \'red\')\n# plt.plot(y = UCL,x = resampled.index, color=\'red\', ls=\'--\', linewidth=1.5, label=\'UCL\')\n\n# LCL.plot(color=\'red\', ls=\'--\', linewidth=1.5, label=\'LCL\')\n# df_theft_dates[\'MEAN\'].plot(color=\'red\', linewidth=2, label=\'Average\')\n\nplt.title(\'Total crimes per day\', fontsize=16)\nplt.xlabel(\'Day\')\nplt.ylabel(\'Number of crimes\')\nplt.tick_params(labelsize=14)\nplt.legend(prop={\'size\':16})')


# In[ ]:


resampled.size().std()


# # As you may have noticed, the yearly crime statistics follow a general trend.
# 
# Here, the noticable trend is a rise in curve at the start of the year and achieveing the peak at the mid point. somehwhere at **June - July** . After that it has an equally sharp drop to the initial number of crimes as the year started!

# # Having a look at Public Peace Violations

# In[ ]:


df_public_peace =  df[df['Primary Type'] == 'PUBLIC PEACE VIOLATION']


# In[ ]:


df_public_data = pd.DataFrame({"Counts": df_public_peace['Description'].value_counts(), "Description" : df_public_peace['Description'].value_counts().index})
df_public_data.reset_index(inplace=True)
df_public_data = df_public_data.drop(columns=['index'], axis = 1)
df_public_data.head()


# In[ ]:


plt.figure(figsize = (15, 7))
sns.barplot(y ="Description", x = "Counts", data = df_public_data, palette="cool")


# # Focussing on Reckless Conduct, one can see how it out numbers our threats 
# Sadly, Bomb and Arson Threats still are a major problem in society and it's obvious to notice that most of these threats are based in Schools or Public Places.
# 
# But , can we predict where will be next Bombing threats? Let's check out that scenario.

# In[ ]:


unique_locations_bombs = df_public_peace['Location'].value_counts()


# In[ ]:


PB_index = pd.DataFrame({"Raw_String" : unique_locations_bombs.index, "ValueCount":unique_locations_bombs})
PB_index.index = range(len(unique_locations_bombs))
PB_index.head()


# In[ ]:


PB_index['LocationCoord'] = PB_index['Raw_String'].apply(Location_extractor)
PB_index  = PB_index.drop(columns=['Raw_String'], axis = 1)


# In[ ]:


chicago_crime_pp = folium.Map(location=[41.895140898, -87.624255632],
                        zoom_start=13)


# In[ ]:


for i in range(500):
    lat = PB_index['LocationCoord'].iloc[i][0]
    long = PB_index['LocationCoord'].iloc[i][1]
    radius = PB_index['ValueCount'].iloc[i] / 3
    
    if PB_index['ValueCount'].iloc[i] > 30:
        color = "#FF4500"
    else:
        color = "#008080"
    
    popup_text = """Latitude : {}<br>
                Longitude : {}<br>
                Peace Disruptions : {}<br>"""
    popup_text = popup_text.format(lat,
                               long,
                               PB_index['ValueCount'].iloc[i]
                               )
    folium.CircleMarker(location = [lat, long], popup= popup_text,radius = radius, color = color, fill = True).add_to(chicago_crime_pp)


# In[ ]:


folium.TileLayer('cartodbpositron').add_to(chicago_crime_pp)


# # These are the actual figures of pubic disruptions 

# In[ ]:


chicago_crime_pp


# 

# In[ ]:




