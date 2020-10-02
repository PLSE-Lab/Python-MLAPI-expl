#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

terror = pd.read_csv("../input/globalterrorismdb_0616dist.csv", encoding='ISO-8859-1')

plt.figure(figsize=(15,8))
data = terror[terror["region_txt"] == "South America"]

m = Basemap(projection='mill', llcrnrlat = -60, urcrnrlat = 25, llcrnrlon = -100, urcrnrlon = -20, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
m.drawstates()

x, y = m(list(data["longitude"].astype("float")), list(data["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 8, alpha = 0.8, color = "darkgreen")

plt.title('Terror Attacks on South America (1970-2015)')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
usa = terror[terror["region_txt"] == "North America"]

mapusa = Basemap(projection='mill', llcrnrlat = 0, urcrnrlat = 75, llcrnrlon = -170, urcrnrlon = -55, resolution = 'h')
mapusa.drawcoastlines()
mapusa.drawcountries()
mapusa.drawstates()

x, y = mapusa(list(usa["longitude"].astype("float")), list(usa["latitude"].astype(float)))
mapusa.plot(x, y, "go", markersize = 8, alpha = 0.8, color = "darkblue")

plt.title('Terror Attacks on North America 1970-2015')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
usa = terror[terror["region_txt"] == "Central America & Caribbean"]

m = Basemap(projection='mill', llcrnrlat = 0, urcrnrlat = 30, llcrnrlon = -105, urcrnrlon = -60, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
m.drawstates()

x, y = m(list(usa["longitude"].astype("float")), list(usa["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 8, alpha = 0.8, color = "blue")

plt.title('Terror Attacks on Central America & Caribbean (1970-2015)')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
usa = terror[terror["region_txt"] == "Middle East & North Africa"]

m = Basemap(projection='mill', llcrnrlat = 0, urcrnrlat = 60, llcrnrlon = -35, urcrnrlon = 65, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
m.drawstates()

x, y = m(list(usa["longitude"].astype("float")), list(usa["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 8, alpha = 0.8, color = "#FF0000")

plt.title('Terror Attacks on Middle East & North Africa (1970-2015)')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
data = terror[terror["region_txt"] == "Sub-Saharan Africa"]

m = Basemap(projection='mill', llcrnrlat = -45, urcrnrlat = 60, llcrnrlon = -35, urcrnrlon = 65, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
m.drawstates()

x, y = m(list(data["longitude"].astype("float")), list(data["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 8, alpha = 0.8, color = "#FFFF00")

plt.title('Terror Attacks on Sub-Saharan Africa (1970-2015)')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
data = terror[terror["region_txt"].isin(["Eastern Europe", "Western Europe"])]

m = Basemap(projection='mill', llcrnrlat = 10, urcrnrlat = 75, llcrnrlon = -15, urcrnrlon = 70, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
m.drawstates()

x, y = m(list(data["longitude"].astype("float")), list(data["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 6, alpha = 0.8, color = "#66FF00")

plt.title('Terror Attacks on Europe (1970-2015)')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
data = terror[terror["region_txt"].isin(["South Asia", "Southeast Asia", "Central Asia", "East Asia"])]

m = Basemap(projection='mill', llcrnrlat = -15, urcrnrlat = 70, llcrnrlon = 30, urcrnrlon = 165, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
m.drawstates()

x, y = m(list(data["longitude"].astype("float")), list(data["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 6, alpha = 0.8, color = "#0000FF")

plt.title('Terror Attacks on Asia (1970-2015)')
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))

m = Basemap(projection = 'mill', llcrnrlat = -80, urcrnrlat = 80, llcrnrlon = -180, urcrnrlon = 180, resolution = 'h')
m.drawcoastlines()
m.drawcountries()
m.drawstates()

x, y = m(list(terror["longitude"].astype("float")), list(terror["latitude"].astype(float)))
m.plot(x, y, "go", markersize = 8, alpha = 0.8, color = "darkgreen")

plt.title('Global Terror Attacks (1970-2015)')
plt.show()

