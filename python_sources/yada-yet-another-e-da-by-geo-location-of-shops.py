#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# A quick examination of `shops.csv` reveals a name of the shop could also contain a city name.  Seeing this hasn't been reported yet, wanted to look through all of the items and generate some potential features.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from operator import itemgetter

# Input data files are available in the "../input/" directory.

import os
input_folder = "../input/competitive-data-science-predict-future-sales"
file_names = (os.listdir(input_folder))


# In[2]:


shops = pd.read_csv(os.path.join(input_folder,'shops.csv'))


# In[3]:


shops['city']=shops.shop_name.apply(str.split).apply(itemgetter(0)).apply(str.strip,args='!')


# The first word of the `shop_name` can be interpreted as a city name

# In[4]:


shops.head()


# ### This code will not run on Kaggle
# This cell requires internet to work, this looks up location for each city, excluding th shops that are "digital"
# ```python
# import geocoder
# shops['coordinates']=[geocoder.yandex(c).latlng for c in shops.city.values]
# shops['lat']=shops.coordinates.drop([9,12,55]).apply(itemgetter(0)).apply(float)
# shops['long']=shops.coordinates.drop([9,12,55]).apply(itemgetter(1)).apply(float)
# 
# ```
# 
# 

# ### Since no internet, load the files
# Load the shared coordinates file, as Kaggle kernels do not have access to the internet, but allow to upload and share dataset

# In[5]:


shops[['lat','long']]=pd.read_csv('../input/geo-info/coordinates.csv',index_col=0,dtype=np.float)


# and the machine-translated city names

# In[6]:


shops['city'] = pd.read_csv('../input/geo-info/city_name_eng.csv',index_col=0,header=None)

shops


# In[7]:


shops_summary=shops[~shops.lat.isna()].groupby('city').agg({'lat':np.mean,'long':np.mean,'shop_id':len}).reset_index()
shops_summary.rename({'shop_id':'city_cnt'},inplace=True,axis=1)
shops_summary


# ## Map the shop locations

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import builtins

plt.figure(figsize=(14,10))

map = Basemap(llcrnrlon=19.5,llcrnrlat=35,urcrnrlon=140,urcrnrlat=75)

map.drawcoastlines()
map.drawcountries()

x, y = map(shops_summary.long, shops_summary.lat)

map.scatter(x, y, marker='D',color='m')
for i in range(len(x)):
    plt.text(x[i],y[i],shops_summary.loc[i,'city'])


plt.show()


# ## Conclusion
# 
# A quick EDA using `shops.csv`  yields these results:
# 
# 1.  Variable `shop_name` seems to contain in it a city name for all except three shops (9, 12 and 55). A quick look at the distribution of the city locations suggests it can be used as a category for encoding shops. 
# 2. In addition, two shops may be duplicates, `shop_id` 10 and 11, with their names almost identical.  
# 3. Shops 39 and 40 may be two braches of the same store, differing only in last word of the name.

# In[ ]:




