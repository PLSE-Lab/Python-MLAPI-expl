#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
import warnings
warnings.filterwarnings('ignore')


# ## Data ##

# In[ ]:


earth_quake = pd.read_csv("../input/database.csv")


# In[ ]:


earth_quake.head()


# In[ ]:


earth_quake.columns


# In[ ]:


earth = earth_quake[["Date","Latitude","Longitude","Magnitude"]]


# In[ ]:


earth.head()


# In[ ]:


earth.tail()


# In[ ]:


earth["Date"] = pd.to_datetime(earth["Date"])


# In[ ]:


earth.shape


# ## Creating a Basemap instance ##

# In[ ]:


m = Basemap(projection="mill")


# ## Converting from spherical to cartesian coordinates ##

# In[ ]:


longitudes = earth["Longitude"].tolist()
latitudes = earth["Latitude"].tolist()
x,y = m(longitudes,latitudes)


# ## Mapping all affected areas ##

# In[ ]:


fig = plt.figure(figsize=(12,10))
plt.title("All affected areas")
m.scatter(x,y, s = 4, c = "blue")
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.show()


# ## The Severity of an Earthquake ##

# ### Minimum and Maximum Magnitude ###

# In[ ]:


minimum = earth["Magnitude"].min()
maximum = earth["Magnitude"].max()
average = earth["Magnitude"].mean()

print("Minimum:", minimum)
print("Maximum:",maximum)
print("Mean",average)


# In[ ]:


(n,bins, patches) = plt.hist(earth["Magnitude"], range=(0,10), bins=10)
plt.xlabel("Earthquake Magnitudes")
plt.ylabel("Number of Occurences")
plt.title("Overview of earthquake magnitudes")

print("Magnitude" +"   "+ "Number of Occurence")
for i in range(5, len(n)):
    print(str(i)+ "-"+str(i+1)+"         " +str(n[i]))


# - Over 16,000 (68.5%) earthquakes magnitude were between 5 and 6
# - Over 40 (0.17%) earthquakes magitude were greater than 8.

# In[ ]:


plt.boxplot(earth["Magnitude"])
plt.show()


# In[ ]:


highly_affected = earth[earth["Magnitude"]>=8]


# In[ ]:


print(highly_affected.shape)


# In[ ]:


longitudes = highly_affected["Longitude"].tolist()
latitudes = highly_affected["Latitude"].tolist()
n = Basemap(projection="mill")
a,b = n(longitudes,latitudes)

fig2 = plt.figure(2, figsize= (12,10))
plt.title("Highly affected areas")
n.scatter(a,b,  c = "blue", s = highly_affected["Magnitude"] *20)
n.drawcoastlines()
n.fillcontinents(color='coral',lake_color='aqua')
n.drawmapboundary()
n.drawcountries()
fig2.show()


# ## Frequency by Month ##

# In[ ]:


earth["Month"] = earth['Date'].dt.month


# In[ ]:


#month_occurrence = earth.pivot_table(index = "Month", values = ["Magnitude"] , aggfunc = )

month_occurrence = earth.groupby("Month").groups
print(len(month_occurrence[1]))

month = [i for i in range(1,13)]
occurrence = []

for i in range(len(month)):
    val = month_occurrence[month[i]]
    occurrence.append(len(val))

print(occurrence)
print(sum(occurrence))


# In[ ]:


fig, ax = plt.subplots(figsize = (10,8))
bar_positions = np.arange(12) + 0.5

# Heights of the bars.  In our case, the average rating for the first movie in the dataset.
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
num_cols = months
bar_heights = occurrence

ax.bar(bar_positions, bar_heights)
tick_positions = np.arange(1,13)
ax.set_xticks(tick_positions)
ax.set_xticklabels(num_cols, rotation = 90)
plt.show()


# - Highly affected month : March
# - Least affected month : June

# ## Frequency by Year ##

# In[ ]:


earth["Year"] = earth['Date'].dt.year


# In[ ]:


year_occurrence = earth.groupby("Year").groups


year = [i for i in range(1965,2017)]
occurrence = []

for i in range(len(year)):
    val = year_occurrence[year[i]]
    occurrence.append(len(val))

maximum = max(occurrence)
minimum = min(occurrence)
print("Maximum",maximum)
print("Minimum",minimum)

#print("Year :" + "     " +"Occurrence")

#for k,v in year_occurrence.items():
    #print(str(k) +"      "+ str(len(v)))

fig = plt.figure(figsize=(10,6))
plt.plot(year,occurrence)
plt.xticks(rotation = 90)
plt.xlabel("Year")
plt.ylabel("Number of Occurrence")
plt.title("Frequency of Earthquakes by Year")
plt.xlim(1965,2017)
plt.show()


# - Least affected year : 1966
# - Highly affected year : 2011
