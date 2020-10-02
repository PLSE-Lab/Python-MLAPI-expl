#!/usr/bin/env python
# coding: utf-8

# ## Copy the geopackage to working folder, this makes it possible to read by geopandas

# In[ ]:


get_ipython().system(' cp /kaggle/input/california-crop-mapping-2014/crop_mapping_2014.gpkg  /kaggle/working/crop_mapping_2014.gpkg')


# ## Read geopackage

# In[ ]:


import geopandas as gpd

gdf = gpd.read_file('/kaggle/working/crop_mapping_2014.gpkg')


# ## Crop area by class

# In[ ]:


crop_areas = gdf[['Crop2014', 'Area_ha']].groupby('Crop2014').mean().sort_values('Area_ha', ascending=False)
crop_areas


# In[ ]:


crop_areas.plot.bar(figsize = (20,5))


# ## Crop area by county

# In[ ]:


gdf[['County', 'Area_ha']].groupby('County').mean().sort_values('Area_ha', ascending=False).plot.bar(figsize = (20,5))


# ## Create colormap

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

cmap = colors.ListedColormap ( np.random.rand ( 256,3))


# ## Crop fields located in Kings, California

# In[ ]:


kings =  gdf[gdf.County == "Kings"]

fig, ax = plt.subplots(figsize=(20,30))

kings.plot(column='Crop2014', categorical=True, legend=True, cmap=cmap,ax=ax)


# ## Crop fields located in Madera, California

# In[ ]:


madera =  gdf[gdf.County == "Madera"]

fig, ax = plt.subplots(figsize=(20,30))

madera.plot(column='Crop2014', categorical=True, legend=True, cmap=cmap,ax=ax)

