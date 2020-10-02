#!/usr/bin/env python
# coding: utf-8

# `enter code here`# Earthquakes in Italy

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import seaborn as sns
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')

eqdata = pd.read_csv('../input/italy_earthquakes_from_2016-08-24_to_2016-11-30.csv').set_index('Time')


# **Exploring the data for earthquakes in Italy from late August through September**

# In[ ]:


eqdata.head()


# In[ ]:


eqdata.index = pd.to_datetime(eqdata.index)


# In[ ]:


mag = eqdata.Magnitude
(n, bins, patches) = plt.hist(mag, bins = 4)
plt.tight_layout()
plt.show()
print(bins)
print(n)


# **Most of these earthquakes were minor:**
# 
#  - Over 7000 earthquakes were between 2 and 3 in Magnitude
#  - Over 500 were between a 3 and a 4
#  - Only 19 earthquakes between a 4 and a 5
#  - 5 earthquakes were greater than that

# In[ ]:


eqdata["Magnitude"].resample('2D').mean().plot()
plt.title("Time Series: Average Magnitude")
plt.ylabel("Magnitude")


# **A good amount of earthquake activity occurred from mid October to mid November**

# In[ ]:


print('Highest Magnitude Earthquake:\n','Date/Time:', eqdata.Magnitude.idxmax(),'\n', 'Magnitude:', eqdata.Magnitude.max(),'\n Latitude/Longitude:',eqdata.Latitude[eqdata.Magnitude.idxmax()],',',eqdata.Longitude[eqdata.Magnitude.idxmax()])


# In[ ]:


def drawmap(df, zoom=1):
    z= (10/3)-(1/3)*zoom
    m = Basemap(projection = 'merc',llcrnrlat=df.Latitude.min()-z, urcrnrlat=df.Latitude.max()+z, llcrnrlon=df.Longitude.min()-z, urcrnrlon=df.Longitude.max()+z)
    x,y = m(list(df.Longitude),list(df.Latitude))
    m.scatter(x,y, c = df.Magnitude, cmap = 'seismic')
    m.colorbar()
    m.drawcoastlines()
    #m.drawstates()
    #m.drawcountries()
    m.bluemarble()
    plt.show()
    plt.clf()


# In[ ]:


over4 = eqdata[eqdata.Magnitude >=4.25]


# In[ ]:


drawmap(over4, zoom = -3)
drawmap(over4, zoom = 10)


# **As we can see all of the earthquakes occur toward the center of Italy.**
# 
# **To my understanding, this is actually  a very mountainous area, where there are some large fault lines, which make this area prone to earthquakes.**

# In[ ]:


eqdata.corr()


# In[ ]:


eqs1030 = eqdata[(eqdata.index >= '2016-10-26') & (eqdata.index <= '2016-11-03') & (eqdata.Magnitude >4)]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set(title = 'EQs Days before and after OCT.30 6.5 MAG EQ', xlim = [eqs1030.index.min(), eqs1030.index.max()])
ax.scatter(eqs1030.index,eqs1030.Magnitude)
fig.tight_layout()
fig.autofmt_xdate()
plt.show()


# **Lastly, I plotted the Earthquake activity above a 4.0 magnitude, 4 days before and after the large 6.5 magnitude earthquake. There were two large earthquakes(over 5 mag), just 4 days before the big one, which could have been foreshocks, foreshadowing the large 6.5 magnitude quake.**

# In[ ]:




