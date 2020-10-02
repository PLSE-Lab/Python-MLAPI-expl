#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
plt.style.use('ggplot')
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()
from mpl_toolkits.basemap import Basemap
from IPython.display import display, HTML


# In[ ]:


from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')


# In[ ]:


data = pd.read_csv("../input/operations.csv")


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


# most popular bombing aircraft
sns.countplot(data["Aircraft Series"], order=data["Aircraft Series"].value_counts().iloc[:5].index)


# ## The most popular bombing aircraft was the B24
# <img src="https://www.sos.wa.gov/legacy/korea65/img/panels/wa2-sm.jpg" width="800px">

# ## The most popular fighter aircraft during WWII was A20
# <img src="http://www.aviation-history.com/douglas/a20-12a.jpg" width="800px">

# In[ ]:


# Top target countries
plt.figure(figsize=(10,4))
sns.countplot(data["Target Country"], palette="coolwarm", order=data["Target Country"].value_counts().iloc[:5].index)


# In[ ]:


#Theatres
plt.figure(figsize=(10,4))
sns.countplot(data["Theater of Operations"], palette="coolwarm", order=data["Theater of Operations"].value_counts().iloc[:5].index)


# In[ ]:


# Who launched the attacks?
colors = sns.color_palette(["#9fafea", "#e66364", "#b1b1b1", "#ef84a8"])
sns.countplot(data["Country"], palette=colors, order=data["Country"].value_counts().iloc[:4].index)


# In[ ]:


# The busiest day
print(data["Mission Date"].value_counts().iloc[:4])
busiest_day = data[data["Mission Date"] == "3/24/1945"]
sns.countplot(busiest_day["Country"], palette=colors, order=busiest_day["Country"].value_counts().index)


# ## March 24, 1945 the Allies launched Operation Varsity
# ### It was the largest one-day airborne operation of all time
# <img src="https://warfarehistorynetwork.com/wp-content/uploads/Lead-Varsity1.jpg" width="800px">

# In[ ]:


#Bombed sites during Operation Varsity
lat = busiest_day["Target Latitude"].values
lon = busiest_day["Target Longitude"].values
fig = plt.figure(figsize=(10,10))
m = Basemap(projection='lcc',resolution='l',lat_0=50.77,
            lon_0=10.06,width=2E6,height=2.2E6)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')
m.scatter(lon, lat, latlon=True,
          cmap='Reds', alpha=0.5)
plt.show()


# In[ ]:


cleaned_data = data.dropna(subset=['Country','Target Longitude','Target Latitude','Aircraft Series','Mission Date'])


# In[ ]:


from itertools import chain

def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')
#World Bombing Map
lat = cleaned_data["Target Latitude"].values
lon = cleaned_data["Target Longitude"].values
fig = plt.figure(figsize=(15,20),edgecolor='w')
m = Basemap(projection='moll',resolution='l',lat_0=0,lon_0=0)
m.scatter(lon, lat, latlon=True,
          cmap='Reds', alpha=0.2, s=4)
draw_map(m)


# In[ ]:


#World Takeoff Bases
bases = cleaned_data.dropna(subset=['Takeoff Latitude', 'Takeoff Longitude'])
typo = bases[bases["Takeoff Latitude"]=="TUNISIA"].index
bases.drop(typo, inplace=True)
lat = bases["Takeoff Latitude"].astype('float64').values
lon = bases["Takeoff Longitude"].astype('float64').values
fig = plt.figure(figsize=(15,20),edgecolor='w')
m = Basemap(projection='moll',resolution='l',lat_0=0,lon_0=0)
m.scatter(lon, lat, latlon=True,
          cmap='Reds', alpha=0.2, s=28)
draw_map(m)


# In[ ]:


popular = cleaned_data[cleaned_data["Aircraft Series"].isin(['B24','B25','B17','A20','A26'])]
popular["Mission Year"] = pd.to_datetime(cleaned_data["Mission Date"]).dt.year
#popular = popular.drop(popular[popular["Altitude (Hundreds of Feet)"]>6000].index)
popular = popular.reset_index(drop=True)


# In[ ]:


f, ax = plt.subplots(figsize=(7,6))
#ax.set_xscale("log")
sns.boxenplot(data=popular, x="Total Weight (Tons)", y="Aircraft Series")


# In[ ]:


f, ax = plt.subplots(figsize=(7,6))
ax.set_yscale("log")
sns.boxenplot(data=popular, y="Altitude (Hundreds of Feet)", x="Aircraft Series")


# In[ ]:


# Evolution of WWII Bombers
plt.figure(figsize=(18,18))
popular = popular.drop(popular[popular["Altitude (Hundreds of Feet)"]>400].index)
popular = popular.drop(popular[popular["Total Weight (Tons)"]>200].index)
g = sns.FacetGrid(popular, col="Mission Year", hue="Aircraft Series")
g.map(plt.scatter, "Total Weight (Tons)", "Altitude (Hundreds of Feet)")
g.add_legend()


# In[ ]:


data.loc[data["Total Weight (Tons)"].idxmax()]["Aircraft Series"]


# # Through time and technological advancements, aircraft were able to carry more bombs and fly higher altitudes
# ## The heaviest of all was the **Boeing B-29 Superfortress**
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/B-29_in_flight.jpg/1280px-B-29_in_flight.jpg">

# In[ ]:




