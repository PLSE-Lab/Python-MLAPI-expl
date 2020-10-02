#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Greetings from the Kaggle bot! This is an automatically-generated kernel with starter code demonstrating how to read in the data and begin exploring. Click the blue "Edit Notebook" or "Fork Notebook" button at the top of this kernel to begin editing.

# ## Exploratory Analysis
# To begin this exploratory analysis, first use `matplotlib` to import libraries and define functions for plotting the data. Depending on the data, not all plots will be made. (Hey, I'm just a kerneling bot, not a Kaggle Competitions Grandmaster!)

# In[ ]:


import matplotlib.pyplot as plt # plotting
import xarray as xr
import cartopy.crs as ccrs 

import os 
import cartopy.feature as cfeature 


# There is 41 netcdf file in this dataset for Total cloud couver and mean sea level pressure. 
# There is also 41 files for temperature, precipitation and wind. 
# 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# The next hidden code cells define functions for plotting data. Click on the "Code" button in the published kernel to reveal the hidden code.

# In[ ]:


# Opening a given file 

ds  = xr.open_dataset("/kaggle/input/msl_ttc_1991.nc")


# In[ ]:


# Check what is inside the dataset. 
# Two variables are present : 
# - msl 
# - tcc

ds


# In[ ]:


# Selecting one time step and plotting for msl 
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,7))
ax = plt.axes(projection=ccrs.PlateCarree())
ds["msl"].isel(time=0).plot.contourf(ax=ax,transform=ccrs.PlateCarree(),levels=20)
ax.coastlines(resolution='50m', color='white', linewidth=2)
ax.add_feature(cfeature.BORDERS.with_scale('50m'),edgecolor='white')
ax.gridlines(draw_labels=True)


# In[ ]:


import matplotlib.pyplot as plt


ds["msl"].isel(time=[0,1,2,3,4,5,6,7]).plot.contourf(levels=20,col='time', col_wrap=4)
ds["tcc"].isel(time=[0,1,2,3,4,5,6,7]).plot.contourf(levels=20,col='time', col_wrap=4)


# In[ ]:


# Seleting one time step and plotting for total cloud cover 
fig = plt.figure(figsize=(20,7))
ax = plt.axes(projection=ccrs.PlateCarree())
ds["tcc"].isel(time=0).plot.contourf(ax=ax,transform=ccrs.PlateCarree(),levels=20)
ax.coastlines(resolution='50m', color='white', linewidth=2)
ax.add_feature(cfeature.BORDERS.with_scale('50m'),edgecolor='white')
ax.gridlines(draw_labels=True)


# In[ ]:


# A bit of exploration using geoview 


# In[ ]:


import geoviews as gv 
import geoviews.feature as gf
from geoviews import opts
import geoviews.tile_sources as gts
gv.extension("bokeh")


# In[ ]:


coastline = gf.coastline(line_width=3,line_color='white').opts(projection=ccrs.GOOGLE_MERCATOR,scale='50m')
borders = gf.borders(line_width=3,line_color="black").options(scale='50m')
tile = gts.EsriImagery().opts(width=600, height=700)


# In[ ]:


ds.msl.min()


# In[ ]:


gv_msl = gv.Dataset(ds,vdims=["msl"],crs=ccrs.PlateCarree())
msl_image= gv_msl.to(gv.Image,['longitude', 'latitude'],"msl",dynamic=True).opts(opts.Image(colorbar=True,
                                                                                            clim=(95000,105000),cmap="jet",tools=["hover"],alpha=0.8))


# In[ ]:


msl_image * tile * borders * coastline


# In[ ]:


gv_tcc = gv.Dataset(ds,vdims=["tcc"],crs=ccrs.PlateCarree())
tcc_image= gv_tcc.to(gv.Image,['longitude', 'latitude'],"tcc",dynamic=True).opts(opts.Image(colorbar=True,
                                                                                            clim=(0,1.1),cmap="jet",
                                                                                            tools=["hover"],alpha=0.8))


# In[ ]:


tcc_image * tile * borders * coastline


# # Showing other variables

# In[ ]:


# Opening a given file 

ds_o  = xr.open_dataset("/kaggle/input/wind_t2m_tp_1991.nc")


# In[ ]:


gv_msl = gv.Dataset(ds_o,vdims=["t2m"],crs=ccrs.PlateCarree())
msl_image= gv_msl.to(gv.Image,['longitude', 'latitude'],"t2m",dynamic=True).opts(opts.Image(colorbar=True,cmap="jet",tools=["hover"],alpha=0.8))
msl_image * tile * borders * coastline


# In[ ]:


gv_tp = gv.Dataset(ds_o,vdims=["tp"],crs=ccrs.PlateCarree())
tp_image= gv_tp.to(gv.Image,['longitude', 'latitude'],"tp",dynamic=True).opts(opts.Image(colorbar=True,cmap="jet",tools=["hover"],alpha=0.8))
tp_image * tile * borders * coastline


# In[ ]:


ds_o.tp


# ## Conclusion
# This concludes your starter analysis! To go forward from here, click the blue "Edit Notebook" button at the top of the kernel. This will create a copy of the code and environment for you to edit. Delete, modify, and add code as you please. Happy Kaggling!
