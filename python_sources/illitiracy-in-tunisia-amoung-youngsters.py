#!/usr/bin/env python
# coding: utf-8

# In[ ]:


### Let's start by importing basic dependencies for data processing.


import numpy as np 
import pandas as pd 

# Checking the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import glob as glob


# In[ ]:


print(glob.glob("*.csv"))


# In[ ]:


# Importing essential files for map viewing

import pandas as pd
import geopandas as gpd
import geoviews as gv

gv.extension('bokeh')


# In[ ]:


geometries = gpd.read_file('../input/TUN_adm1.shp')


# In[ ]:


geometries.boundary


# In[ ]:


geometries.as_matrix   


# In[ ]:


idr = pd.read_csv('../input/codes_maps_tn.csv')


# In[ ]:


idr.columns


# In[ ]:


gdf = gpd.GeoDataFrame(pd.merge(geometries, idr))


# In[ ]:


gdf.head()


# In[ ]:


plot_opts = dict(tools=['hover'], width=550, height=800, color_index='Valeur',
                 colorbar=True, toolbar=None, xaxis=None, yaxis=None)
gv.Polygons(gdf, vdims=['Region', 'Valeur'], label='illiterates aged 10 and more by Delegation in Tunisia').opts(plot=plot_opts)


# In[ ]:




