#!/usr/bin/env python
# coding: utf-8

# # Charlotte Poverty Map
# 
# This notebook creates a map percentage of people living in poverty in Mecklenburg County, derived ultimately from census data (see download section below).  
# 
# This code is based on the Lauren Oldja's article on [Reading Shapefile Zips from a URL](https://medium.com/@loldja/reading-shapefile-zips-from-a-url-in-python-3-93ea8d727856).

# In[ ]:


get_ipython().system('pip install mapclassify')


# In[ ]:


import geopandas as gpd
import requests
import zipfile
import io
import mapclassify
import matplotlib.pyplot as plt


# In[ ]:


# jupyter "magic" to display plots in notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Download poverty data for Charlotte
# 
# Source:  http://data.charlottenc.gov/datasets/census-poverty-tracts
# 
# To find the URL given in shapefile below, click download, right click shapefile, select "Copy Link Location".

# In[ ]:


shapefile = "https://opendata.arcgis.com/datasets/8697f02bb81c4d2783cdb4bead357490_9.zip?outSR=%7B%22latestWkid%22%3A2264%2C%22wkid%22%3A102719%7D"
local_path = 'tmp/'
print('Downloading shapefile...')
r = requests.get(shapefile)
z = zipfile.ZipFile(io.BytesIO(r.content))
print("Done")
z.extractall(path=local_path) # extract to folder
filenames = [y for y in sorted(z.namelist()) for ending in ['dbf', 'prj', 'shp', 'shx'] if y.endswith(ending)] 
print(filenames)


# In[ ]:


dbf, prj, shp, shx = [filename for filename in filenames]
charlotte = gpd.read_file(local_path + shp)
print("Shape of the dataframe: {}".format(charlotte.shape))
print("Projection of dataframe: {}".format(charlotte.crs))
charlotte.tail() #last 5 records in dataframe


# In[ ]:


ax = charlotte.plot()
ax.set_title("Charlotte Map, Default View)");


# In[ ]:


charlotte["poverty_percentage"]= 100*charlotte["Populati_2"]


# In[ ]:


ax = charlotte.plot(figsize=(15,15), column='poverty_percentage', scheme='quantiles', cmap="tab20b", legend=True)
ax.set_title("Mecklenburg Count Census Tracts by Percentage in Poverty", fontsize='large')
#add the legend and specify its location
leg = ax.get_legend()
leg.set_bbox_to_anchor((0.95,0.20))
plt.savefig("charlotte_poverty.png", bbox_inches='tight')




          

