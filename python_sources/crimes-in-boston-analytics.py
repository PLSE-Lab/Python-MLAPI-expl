#!/usr/bin/env python
# coding: utf-8

# # 1. Road Data
#  * ### Note that Encode type!

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Where am I? find working dir.

# In[ ]:


#current working directory
print(os.getcwd())


# In[ ]:


Code = pd.read_csv("../input/crimes-in-boston/offense_codes.csv", encoding = 'ISO-8859-1')
print(Code.shape, Code, sep = '\n')


# * We can find the number of "CODE column's records".
# 
# * If all records are unique, the result will be True.

# In[ ]:


len(Code["CODE"].unique()) == len(Code["CODE"])


# * But False, that means One code contains several crimes

# In[ ]:


# But False, that means One code contains several crimes
print(len(Code["CODE"].unique()), len(Code["CODE"]))


# In[ ]:


Crime = pd.read_csv("../input/crimes-in-boston/crime.csv", encoding = 'ISO-8859-1')
print(Crime.shape, Crime, sep = '\n')


# # 2. Find missing value
#  1. Seaborn
#  2. Missingno

# In[ ]:


Crime.info()
# from [RangeIndex: 319073 entries, 0 to 319072] we can find total 319073 values.
# There are several missing values in [DISTRICT, SHOOTING, UCR_PART, STREET, Lat, Long] columns.


# * Visualize it
# 1. 4 missingno method

# ### 1) missingno - Matrix

# In[ ]:


import missingno as msno
msno.matrix(Crime)
#If the dataset is heavy, you can choose a sample and give it as input.
# like this -> msno.matrix(Crime.sample(500))


# * In the district column, there are very few missing values.
# * In the case of the shooting column, most of them are missing values.
# * And there are some missing values in the STREET, Lat, and Long columns.

# ### 2) missingno - Bar

# In[ ]:


msno.bar(Crime)
#We also can use log scale --> msno.bar(Crime, log = True)


# * Each bar represents the total amount of values excluding missing values.

# ### 3) missingno - Heatmap (Correlation)

# In[ ]:


msno.heatmap(Crime)


# * Variables that have no missing or empty values are automatically removed because they do not have a meaningful correlation. (White space in the heatmap)
# * If the correlation is 1 as above [Lat ~ Long], it can be seen that the same data(Long, Lat ==> Longitude and Latitude).
# * This is because longitude and latitude always exist in bundles.
# 
# * And the value of STREET has some correlation with Lat(Long). (<- Is an obvious result!)

# ### 4) missingno - Dendrogram

# In[ ]:


msno.dendrogram(Crime)


# * Dendrograms are tree diagrams that show groups formed by clustering observations at each stage and their level of similarity.

# ### 5) seaborn - Heatmap

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
figure = plt.figure(figsize = (15, 8))
sns.heatmap(Crime.isnull(), yticklabels = '', cbar = True)


# * The seaborn heatmap is similar to the matrix in missingno.

# # 3. Get Insight from each columns' relationship

# In[ ]:


Crime = Crime.drop(columns = "SHOOTING")
# We will not use SHOOTING column because it has many missing values.
Crime = Crime.dropna(axis = 0)
# Remove data that contains missing values.


# ### Crime Statistics by Year, Month, Day, and Hour
# 1. Hour of Day, Day of Week
# 
# -> The graph shows the number of crimes over time

# In[ ]:


plt.figure(figsize = (13, 7))
sns.countplot(x = Crime.HOUR)

plt.figure(figsize = (13, 7))
sns.countplot(x = Crime.DAY_OF_WEEK)
# plt.show()


# 2. Month of Year
# 
# -> In summer, when the [discomfort index] is high, be careful not to make people angry.
# 
# * For this plot, we have to create new column for 1 ~ 12 month.

# In[ ]:


plt.figure(figsize = (10, 10))

Months = "Janurary", "Fabuary", "March", "April", "May", "Jun", "July", "August", "September",           "October", "November", "December"
MonthCount = []

for x in range(len(Months)):
    x += 1
    MonthCount.append(len(Crime[Crime['MONTH'] == x]))
    
# Use Comprehension!    
Explode = [0.3 if x == MonthCount.index(max(MonthCount)) else 0 for x in range(12)]
    
plt.pie(MonthCount, labels = Months, autopct = '%1.1f%%', shadow = True,        startangle = 90, counterclock= False, explode = Explode)
plt.axis('equal')


# # 

# In[ ]:


plt.figure(figsize = (13, 7))
sns.countplot(x = Crime.DISTRICT, order = Crime['DISTRICT'].value_counts().index)


# In[ ]:


from wordcloud import WordCloud
text = []

for i in Crime.OFFENSE_CODE_GROUP:
    text.append(i)
    
text = ''.join(text)

CLOUD = WordCloud(width = 1600, height = 1200, max_font_size = 300, background_color = 'black').generate(text)
plt.figure(figsize = (20, 10))
plt.imshow(CLOUD, interpolation = 'bicubic')
plt.axis("off")
plt.show()


# ###

# In[ ]:


# Replace -1 values in Lat/Long with Nan
Crime.Lat.replace(-1, None, inplace=True)
Crime.Long.replace(-1, None, inplace=True)

sns.scatterplot(data = Crime, x = 'Lat', y = 'Long', hue = 'DISTRICT', alpha = 0.1)
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2)
plt.figure(figsize = (20, 10))


# In[ ]:


import folium
from folium.plugins import HeatMap
Crime_map = folium.Map(location = [42.35843, -71.05977], tiles = "Stamen Toner", 
                zoom_start = 11)
# Latitude / Longitude of Boston : [42.35843, -71.05977]
# There are several type of tiles : Stamen Terrain, Stamen Toner, Mapbox Bright,
#                                   Open Street Map(default), ...
Crime_map


# In[ ]:


data = Crime[Crime.YEAR == 2017]
data = data[["Lat", "Long"]]

# If the program is heavy, scale the data with samples.
data = data.sample(30000)

data = [[row['Lat'], row['Long']] for index, row in data.iterrows()]

HeatMap(data, radius = 10).add_to(Crime_map)
Crime_map


# # 4. Conclusion
# 
# 1. Crimes are most likely to occur at 17:00 and tends to peak at 05:00.
# 2. Crimes are most likely to occur on Friday and least likely to occur on Sunday.
# 3. Crimes tends to occur mainly between August and October.
# 4. ETC..
