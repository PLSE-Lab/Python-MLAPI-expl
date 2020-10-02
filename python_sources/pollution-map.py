#!/usr/bin/env python
# coding: utf-8

# ### Visualize water pollution data on map
# 
# ![Water Pollution 2009-12 ](https://github.com/vinayshanbhag/coordinates/raw/master/india-water-pollution-anim.gif)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


df = pd.read_csv('../input/IndiaAffectedWaterQualityAreas.csv',encoding='latin1',skiprows=[0], names=['state','district','block','panchayat','village','habitation','pollutant','year'])


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from geopy import geocoders
import math
import re
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Cleanup
# 
# Cleanup place names (remove code) so that geolocations can be looked up.

# In[ ]:


df['district'] = df.district.apply(lambda x: str(x).split('(')[0])
df['block'] = df.block.apply(lambda x: str(x).split('(')[0])
df['panchayat'] = df.panchayat.apply(lambda x: str(x).split('(')[0])
df['village'] = df.village.apply(lambda x: str(x).split('(')[0])
df['habitation'] = df.habitation.apply(lambda x: str(x).split('(')[0])

# Keep just the year. All dates are 1/4/20XX anyway
df['year'] = df.year.apply(lambda x: str(x).split('/')[-1])


# ### Pollutant by district and year 

# In[ ]:


g = df.groupby(['pollutant','district','year']).size().unstack()
g.fillna('NA',inplace=True)
g


# ### Plot on the map 
# 
# Geocode locations using google maps api (not included here). Geocoding worked only at the district level. 
# 
# Plot map using basemap. Point size for a pollutant corresponds to number of instances in source data.

# In[ ]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://raw.githubusercontent.com/vinayshanbhag/coordinates/master/India-water-pollution-plot.png", width=700,height=700)

