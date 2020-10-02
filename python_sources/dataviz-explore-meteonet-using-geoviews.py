#!/usr/bin/env python
# coding: utf-8

# # Data visualization challenge : examples with geoviews

#  <font size="4">This notebook gives you some useful tips about the data visualization challenge (cf task https://www.kaggle.com/katerpillar/meteonet/tasks?taskId=710). You can find here some examples with the **geoviews** library. If you have new ideas, solutions, do not hesitate to share it!  
#     **Note** : the internet connection of this notebook must be *on* to allow the basemap downloading.  
#  </font>

# In[ ]:


get_ipython().system('pip install geoviews --upgrade')


# In[ ]:


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from holoviews.operation.datashader import regrid
import cartopy.crs as crs
import cartopy.feature as cfeature
import geoviews as gv 
import geoviews.feature as gf
from geoviews import opts
import geoviews.tile_sources as gts
gv.extension("bokeh")


# ### 1. Opening the data 

# In[ ]:


zone = "NW"     #geographic zone (NW here for North-West of France)
model = 'arome' #weather model (arome or arpege)
MODEL = 'AROME' #weather model (AROME or ARPEGE)
level = '2m'      #vertical level (2m, 10m, P_sea_level or PRECIP)
date = dt.datetime(2016, 2, 14,0,0) # Day example 
#parameter name in the file (cf cells below to know the parameter names -> exploration of metadata)
if level == '2m':
    param = 't2m'
elif level == '10m':
    param = 'u10'
elif level == 'PRECIP':
    param = 'tp'
else:
    param = 'msl'
directory = '/kaggle/input/meteonet/' + zone + '_weather_models_2D_parameters_' + str(date.year) + str(date.month).zfill(2) + '/' + str(date.year) + str(date.month).zfill(2) + '/'
fname = directory + f'{MODEL}/{level}/{model}_{level}_{zone}_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}000000.nc'
data = xr.open_dataset(fname)  


# We explore the dimension of the file. 
# We can see that this file has 3 dimension : (step,latitude,longitude). 
# We can also see that step is a *timedelta* in nanoseconds (which is not so fun to use). 

# In[ ]:


data.head()


#  ### 2. Preprocessing the data 

# #### First, we swap dimension to get valid_time (which is a datetime) instead of step

# In[ ]:


# Changing the time dimension
data = data.swap_dims({"step":"valid_time"})


#  ### 3. Plotting nice graphs with geoviews! 

# #### Then, we define the tile, the borders and the coastline we will plot on the image

# In[ ]:


coastline = gf.coastline(line_width=3,line_color='white').opts(projection=crs.GOOGLE_MERCATOR,scale='50m')
borders = gf.borders(line_width=3,line_color="black").options(scale='50m')
tile = gts.EsriImagery().opts(width=600, height=700)


# ### Now we use our dataset to extract an image of the temperature. 
# We specify that we will render it using GOOGLE_MERCATOR projection in order to be able to add the tile and the coastline

# In[ ]:


gv_t2m = gv.Dataset(data,vdims=["t2m"],crs=crs.PlateCarree())
t2m_image= gv_t2m.to(gv.Image,['longitude', 'latitude'],"t2m",dynamic=True).opts(opts.Image(projection=crs.GOOGLE_MERCATOR,colorbar=True,cmap="jet",tools=["hover"],alpha=0.8))


# ### Plotting the image using a tile, border and coastline
# In order to add element to a a plot geoview use * operator. 
# So here we produce a plot with our temperature field as well as the tile, borders and coastline

# In[ ]:


t2m_image.opts(opts.Image(title="Temperature (in Kelvin)"))* tile * borders * coastline


# ### Doing the same job but using regrid from datashader. 
# The plot should be more reactive. However as the dataset is very small, we cannot see a major difference here

# In[ ]:


regrid(t2m_image.opts(opts.Image(title="Temperature (in Kelvin)",color_levels=None)),interpolation="nearest")* tile * borders * coastline 


# ### We can also decide to join some levels and to  decide a specific limitation for colorbar (such that colorbar is fix for different time)

# In[ ]:


t2m_image.opts(opts.Image(title="Temperature (in Kelvin)",color_levels=10,clim=(265,290),alpha=1))


# # Adding relative humidity on top of temperature 
# Let's now add a second field to the plot.
# First we define the contours for this variable. 

# In[ ]:


ds_r = gv.Dataset(data,vdims=["r"],crs=crs.PlateCarree())
r_contours = ds_r.to(gv.LineContours,['longitude', 'latitude'],dynamic=True,name="Humidity").opts(cmap='RdBu_r',levels=5,show_legend=False,line_width=3,alpha=0.9,width=600, height=700,tools=["hover"])


# ### Then we plot it 
# When you use the widget on the right, both plots change.

# In[ ]:


(r_contours * t2m_image.opts(alpha=0.5)*tile).opts(title = "Relative humidity (in %) on top of temperature") + r_contours.opts(title = "Relative humidity alone (in %)",show_legend=True)


# In[ ]:




