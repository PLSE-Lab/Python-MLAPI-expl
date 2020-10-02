#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install pycountry-convert


# In[ ]:


pip install geopy


# In[ ]:


pip install folium


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


oil_df = pd.read_csv("/kaggle/input/ntt-data-global-ai-challenge-06-2020/Crude_oil_trend_From1986-10-16_To2020-03-31.csv")
oil_df['Price_diff'] = oil_df['Price'].diff(1)
oil_df = oil_df[oil_df['Date'] >= "2010-01-01"]
#oil_df = oil_df.sort_values(ascending = False, by = "Date")


# 

# In[ ]:


# Import my necessary libraries
import os
import re
import pandas as pd
import numpy as np
import datetime as dt

import imageio
import matplotlib.pyplot as plt
import folium
from folium import plugins
from folium.plugins import MarkerCluster


# In[ ]:


#function to get longitude and latitude data from country name
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent = "myGeocoder")
def geolocate(country):
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return (loc.latitude, loc.longitude)
    except:
        # Return missing value
        return (np.nan, np.nan)


# In[ ]:





# ## How COVID-19 spread worldwide
# 
# The coronavirus pandemic, also refered to as COVID-19, is an ongoing pandemic caused by SARS-CoV-2 (Severe Acute Respiratory Syndrome). It is thought to have originated in Wuhan, China, in December 2019 and up until today it has reached 216 countries, infecting over 6.5 million people. The map below shows which are the territories affected by COVID-19, as new cases started to be recorded.

# In[ ]:


# Read in the covid data
covid_df = pd.read_csv("/kaggle/input/ntt-data-global-ai-challenge-06-2020/COVID-19_train.csv")
covid_df['Price_diff'] = covid_df['Price'].diff(1)
covid_df['Price_diff_cum'] = covid_df['Price_diff'].cumsum()


# In[ ]:


# Create a df with *_total_cases columns
total_cases_cols = [col for col in covid_df if col.endswith("_total_cases")]

total_cases_cols.insert(0, "Date")
total_cases_cols.insert(len(total_cases_cols)+1, "Price")
total_cases_df = covid_df[total_cases_cols]

countries = [col.split("_")[0] for col in total_cases_df if col.endswith("_total_cases")]
countries = [re.sub(r'\B([A-Z])', r' \1', i) for i in countries]

total_columns = countries.copy()
total_columns.insert(0, "Date")
total_columns.insert(len(total_columns)+1, "Price")

total_cases_df.columns = new_columns
#total_cases_df.head()

# change the df structure to plot it acorss time
total_cases_df = total_cases_df.drop(columns = ["Price","World"])

time_total_cases = total_cases_df.melt(id_vars = ['Date'],
                                   var_name = "country",
                                   value_name = "total_cases")

tm_total_cases = time_total_cases.groupby(["Date","country"]).sum().reset_index()


tm_total_cases_df = pd.merge(tm_total_cases, coord_df, how="left", left_on = "country", right_on="country")

# Select all country positions with confirmed cases
non_neg_total_cases = tm_total_cases_df[tm_total_cases_df.total_cases > 0]
ls_non_neg_total_cases = non_neg_total_cases.groupby('Date')['coord'].apply(list)


# Generate the map
ls_non_neg_total_cases
date_index = ls_non_neg_total_cases.index.to_list()
data = ls_non_neg_total_cases.values.tolist()


gradient = {0:"blue", 0.5:"orange", 1: "red"}

world_map= folium.Map(location = [42.6384261, 12.674297],tiles="cartodbpositron", zoom_start =2)

hm = plugins.HeatMapWithTime(
    data = data,
    index=date_index,
    auto_play=True,
    radius=10,
    gradient=gradient
)

hm.add_to(world_map)

#show the map
world_map


# The pandemic has caused social and economic disruption, including the largest global recession since the Great Depression. Sports, cultural, religious events were cancelled, schools were closed, companies either interrupted their activity or encouraged employees to work from home. The free movement of people was highly limited both inside and outside their city, travelling was restricted to the point that flight companies went backrupt and the oil price plunged below 0 because of the decrease in demand. By inspecting the price movement of oil for the last 10 years, it is safe to say that the drop in the last three months, the same coinciding with the spread of COVID-19, is not something unseen before. Oil price is defined by such fluctuations. The pandemic, however, through the political decisions limiting people movement, accelerated the price drop.

# In[ ]:


fig, ax1 = plt.subplots(figsize = (20,5))

plt.xticks(np.arange(0, len(oil_df['Price_diff']),20), rotation="vertical", fontsize = 7)
plt.axvline(2510)

#l1, = ax1.plot(covid_df_fr['Date'], covid_df_fr['Price'], color = 'tab:red')
l1, = ax1.plot(oil_df['Date'], oil_df['Price_diff'], color = 'tab:red')
ax1.set_ylabel("Oil price change ($)", color = 'tab:red')

plt.title("Oil price fluctuations between 2010 - 2020")
plt.show()


# Below, one can see the negative trend of the oil price for the three months of the year, in relation to the continuous increase in total worldwide COVID-19 reported number of cases.

# In[ ]:


fig, ax1 = plt.subplots(figsize = (15,5))

plt.xticks(rotation="vertical")

#l1, = ax1.plot(covid_df_fr['Date'], covid_df_fr['Price'], color = 'tab:red')
l1, = ax1.plot(covid_df['Date'], covid_df['Price'], color = 'tab:red')
ax1.set_ylabel("Oil price ($)", color = 'tab:red')

ax2 = ax1.twinx()

l2, = ax2.plot(covid_df['Date'], covid_df['World_total_cases'])
ax2.set_ylabel("Number of cases")

fig.legend((l1, l2),('Oil price', 'Total cases'),'upper right')
plt.title("Daily world cases vs. Oil price evolution")
plt.show()


# At a closer inspection of the data, something catches my eyes. 15th of January is the first day when COVID-19 is reported to have spread to another country. If the oil price was, until now, exhibiting some small, normal fluctuations, during the next week, as cases arise in 5 neighbouring countries, the oil price is obviously affected. 
# On the 27th of January the virus reaches Europe and Canada, causing another higher drop in price.
# Between 20th and 24th of February the oil extracting countries started to report COVID-19 cases as well. 
# The plunge on the 9th of March comes after a weekend of new annonuncements, as airports are starting to forbid inbound flights.
# 
# The dramatic increase of the worldwide reported new cases on the 13th of February doesn't seem to shake the oil price too much. The reason might be that all these extra cases were recorded in China and the oil prices is more sensitive to world wide changes.

# In[ ]:


fig, ax1 = plt.subplots(figsize = (15,5))

plt.xticks(rotation="vertical")

#l1, = ax1.plot(covid_df_fr['Date'], covid_df_fr['Price'], color = 'tab:red')
l1, = ax1.plot(covid_df['Date'], covid_df['Price_diff'], color = 'tab:red')
ax1.set_ylabel("Oil price change ($)", color = 'tab:red')

ax2 = ax1.twinx()

l2, = ax2.plot(covid_df['Date'], covid_df['World_new_cases'])
ax2.set_ylabel("Number of worldwide new cases")

fig.legend((l1, l2),('Oil price fluctuations', 'New cases'),'upper right')
plt.title("World daily new reported cases vs. Oil price fluctuations")
plt.show()


# The picture of the pandemic changed quite a lot between February and March. If at first, one could pinpoint the worldwide increase in cases to one source (2020-02-13), by the end of March, the contributor was the entire world.

# In[ ]:


# Create a df with *_new_cases columns
new_cases_cols = [col for col in covid_df if col.endswith("_new_cases")]

new_cases_cols.insert(0, "Date")
new_cases_cols.insert(len(new_cases_cols)+1, "Price")
new_cases_df = covid_df[new_cases_cols]

countries = [col.split("_")[0] for col in new_cases_df if col.endswith("_new_cases")]
countries = [re.sub(r'\B([A-Z])', r' \1', i) for i in countries]

new_columns = countries.copy()
new_columns.insert(0, "Date")
new_columns.insert(len(new_columns)+1, "Price")

new_cases_df.columns = new_columns


# In[ ]:


# # For each country, get the coordinates:
# coord = [list(geolocate(i)) for i in countries]
# coord_df = pd.DataFrame({"country": countries,
#                          "coord": coord})


# In[ ]:


# change the df structure to plot it acorss time
new_cases_df = new_cases_df.drop(columns = "Price")

time_new_cases = new_cases_df.melt(id_vars = ['Date'],
                                   var_name = "country",
                                   value_name = "new_cases")

tm_new_cases = time_new_cases.groupby(["Date",'country']).sum().reset_index()

tm_new_cases_df = pd.merge(tm_new_cases, coord_df, how="left", left_on = "country", right_on="country")


# ### The picture of new cases on 2020-02-13

# In[ ]:


# Look at how the virus spreading progresses:
new_cases = new_cases_df[new_cases_df['Date'] == "2020-02-13"].values[0][1:]
new_cases_02_13 = pd.DataFrame({"country": countries,
                               "new_cases": new_cases})

#coord = [geolocate(i) for i in new_cases_02_13["country"]]
new_cases_02_13["latitude"] = [i[0] for i in coord]
new_cases_02_13["longitude"] = [i[1] for i in coord]


# Create a world map to show distributions of infected people 
df = new_cases_02_13[new_cases_02_13['new_cases'] >0][:-2]
total = np.sum(df['new_cases'])
#empty map
world_map= folium.Map(location = [42.6384261, 12.674297], tiles="cartodbpositron", zoom_start =2)

marker_cluster = MarkerCluster().add_to(world_map)
#for each coordinate, create circlemarker of cases

for i in range(len(df)):
    lat = df.iloc[i]['latitude']
    long = df.iloc[i]['longitude']
    radius= (df.iloc[i]['new_cases']/total)*10
    popup_text = """country : {}<br>
                 new cases : {}<br>"""
    popup_text = popup_text.format(df.iloc[i]['country'],
                               df.iloc[i]['new_cases']
                               )
    try:
        if df.iloc[i]['new_cases'] > 1000:
            col = "red"
        elif (df.iloc[i]['new_cases'] > 500) & (df.iloc[i]['new_cases'] <= 1000):
            col = "orange"
        else:
            col = "green"
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True, fill_color = col, color=col).add_to(world_map)
    except ValueError:
        continue

#show the map
world_map


# ### The picture of new cases on 2020-03-27

# In[ ]:


# Select a date towards the end, when the pandemic has reached all corners of the world
new_cases = new_cases_df[new_cases_df['Date'] == "2020-03-27"].values[0][1:]
new_cases
new_cases_03_27 = pd.DataFrame({"country": countries,
                               "new_cases": new_cases})
# Get the country coordinates

coord = [geolocate(i) for i in new_cases_03_27["country"]]
new_cases_03_27["latitude"] = [i[0] for i in coord]
new_cases_03_27["longitude"] = [i[1] for i in coord]

# Create a world map to show distributions of users 
df = new_cases_03_27[new_cases_03_27['new_cases'] >0][:-2]
total = np.sum(df['new_cases'])

#empty map
world_map= folium.Map(location = [42.6384261, 12.674297], tiles="cartodbpositron", zoom_start =2)

#marker_cluster = MarkerCluster().add_to(world_map)
#for each coordinate, create circlemarker of cases

for i in range(len(df)):
    lat = df.iloc[i]['latitude']
    long = df.iloc[i]['longitude']
    radius= (df.iloc[i]['new_cases']/total)*30
    popup_text = """country : {}<br>
                 new cases : {}<br>"""
    popup_text = popup_text.format(df.iloc[i]['country'],
                               df.iloc[i]['new_cases']
                               )
    try:
        if df.iloc[i]['new_cases'] > 1000:
            col = "red"
        elif (df.iloc[i]['new_cases'] > 500) & (df.iloc[i]['new_cases'] <= 1000):
            col = "orange"
        else:
            col = "green"
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True, fill_color = col, color=col).add_to(world_map)
    except ValueError:
        continue

#show the map
world_map


# In[ ]:


#calculate the PEARSON CORRELATION COEFFICIENT
from numpy import cov
import scipy.stats
print("covariance {}".format(cov(covid_df['World_new_cases'][1:], covid_df['Price_diff'][1:])[0][0])) # positive
scipy.stats.pearsonr(covid_df['World_new_cases'][1:], covid_df['Price_diff'][1:])


# ### As an example for reading the satellite images, let's consider France

# In[ ]:


TIF = "/kaggle/input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/"
tif_list = os.listdir("/kaggle/input/ntt-data-global-ai-challenge-06-2020/NTL-dataset/tif/")
france_list = [i for i in tif_list if i.startswith("France")]
france_list.sort()


# In[ ]:


fig = plt.figure(figsize = (20,10))
num=1
for each in france_list:
    image = imageio.imread(os.path.join(TIF, each))
    ax = fig.add_subplot(3,5,num)
    ax.imshow(image)
    
    interval = str.split(each, ".tif")[0]
    ax.set_title(interval)
    num += 1

