#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import geopandas as gp 
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
from scipy import ndimage
from shapely.geometry import Point

pylab.rcParams['figure.figsize'] = 8, 6


# # Intro
# 
# ---
# 
# Hi everyone! Welcome to another one of my World Famous Starter Kernels! I usually create a generic starter kernels with a couple of really bad pie charts just to show how to load data and work with it. When working with this dataset, it wasn't long before I realized that I had *no idea* what I was doing.  I'm no GIS expert and this data was on a whole new level for me. It took a lot of time to figure out how to handle shape files and projection systems -- I learned that working with GIS data is not easy. What is a CRS? What packages should I use? Anyway, I had some fun and took the opportunity to learn about it. This kernel is the process I took and I hope it's helpful. Feel free to as questions here or in the forums: [https://www.kaggle.com/center-for-policing-equity/data-science-for-good/discussion](https://www.kaggle.com/center-for-policing-equity/data-science-for-good/discussion)
# 
# Thanks to CPE for sharing and hosting ths challenge!

# In[ ]:


get_ipython().system('ls -l ../input/')


# In[ ]:


get_ipython().system('ls -l ../input/data-science-for-good/cpe-data/Dept_37-00027/37-00027_Shapefiles/')


# ### Sorting out the coordinate system (CRS) and projections seems to be pretty important 
# 
# PRJ files contains data about the projected coordinate system. 
# - It begins with a name for the projected coordinate system.
# - Then it describes the geographic coordinate system. 
# - Then it defines the projection and all the parameters needed for the projection.
# 
# [https://github.com/GeospatialPython/pyshp](https://github.com/GeospatialPython/pyshp)
# 
# 
# 
# The PRJ is this really awful looking file format but it contains information about the coordinate system of the geometry points. Geopandas wasn't able to pick this up automatically for Dept. 37-00049 (Dallas Police Department) so I had to do some investigation...
# 
# This is the PRJ file for Dallas PD
# 
#     GEOGCS["GCS_WGS_1984",
#         DATUM["D_WGS_1984",
#                 SPHEROID["WGS_1984",6378137,298.257223563]],
#         PRIMEM["Greenwich",0],
#         UNIT["Degree",0.017453292519943295]]
#         
# After some research, this appears to be the "ESRI WKT" dialect of the EPSG 4326 representation [http://spatialreference.org/ref/epsg/4326/esriwkt/](http://spatialreference.org/ref/epsg/4326/esriwkt/) (or something... I'm not a GIS expert)
# 
# Regardless, I think the projection is incorrect. If you plot the geometry with EPSG 4326, it lands somewhere crazy.  I finally determined that the correct projection is probably one of the variants of NAD1983 (EPSG 102738).  
# <br><br>
# 
# 
# 
# After some trial and error, I found that using these projections works pretty well.
# 
# ### Department 37-00049
# - Dallas, TX
# - CRS: NAD_1983_StatePlane_Texas_North_Central_FIPS_4202_Feet
# - [epsg:102738](https://www.spatialreference.org/ref/esri/102738/)
# 
# ### Department 37-00027
# - Austin, TX
# - CRS: NAD_1983_StatePlane_Texas_Central_FIPS_4203_Feet 
# - [epsg:102739](http://spatialreference.org/ref/esri/102739/)
# 
# ### Department  35-00103
# - Charlotte-Mecklenburg, North Carolina
# - CRS: WGS84
# - [epsg:4326](http://spatialreference.org/ref/epsg/4326/)

# In[ ]:


states = gp.read_file('../input/us-states-cartographic-boundary-shapefiles/cb_2016_us_state_500k.shp')
states.crs


# # Geopandas and mapping
# 
# Dept. 37-00027 doesn't have a PRJ file  so I'm going to create one. I wasn't sure which projection was correct. It took a lot of trial and error using different projections that I read about at  [http://www.spatialreference.org/](http://www.spatialreference.org/)

# In[ ]:


NAD1983 = 'PROJCS["NAD_1983_StatePlane_Texas_Central_FIPS_4203_Feet",GEOGCS["GCS_North_American_1983",DATUM["North_American_Datum_1983",SPHEROID["GRS_1980",6378137,298.257222101]],PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["False_Easting",2296583.333333333],PARAMETER["False_Northing",9842499.999999998],PARAMETER["Central_Meridian",-100.3333333333333],PARAMETER["Standard_Parallel_1",30.11666666666667],PARAMETER["Standard_Parallel_2",31.88333333333333],PARAMETER["Latitude_Of_Origin",29.66666666666667],UNIT["Foot_US",0.30480060960121924],AUTHORITY["EPSG","102739"]]'

# Create new directory for 37-00027
try:
    get_ipython().system('mkdir -p ./new/dept_37-00027   # Using a magic shell command instead of Python')
    print("This is fine.")
except:
    next
    

# Create new PRJ file
with open('./new/dept_37-00027/APB_DIST.prj', 'w') as outfile:
    outfile.write(NAD1983)


# Copy the rest of the shapefiles
try:
    get_ipython().system('cp ../input/data-science-for-good/cpe-data/Dept_37-00027/37-00027_Shapefiles/* ./new/dept_37-00027/ # Using a magic shell command instead of Python')
except:
    next


# Read shape file
df = gp.read_file('./new/dept_37-00027/APD_DIST.shp')


# Save and reopen with new projection
df.to_file(filename='./new/dept_37-00027/APD_DIST.shp',driver='ESRI Shapefile',crs_wkt=NAD1983)
df = gp.read_file('./new/dept_37-00027/APD_DIST.shp')
print(df.crs)
df = df.to_crs(epsg=4326)
print(df.crs)


# In[ ]:


df.head()


# In[ ]:


# Plot the whole shebang, color by district
df.plot(column="DISTRICT");


# This prepped.csv contains some crime records with a strange header and even stranger X,Y coordinates.
# 
# According to [http://www.earthpoint.us/Convert.aspx](http://www.earthpoint.us/Convert.aspx) 
# 
#     By default, the X coordinate, or Easting is specified first and the Y coordinate, or Northing is specified second. However, the order can be reversed if coordinates are explicitly labeled. Example:
#     1501 2089808.121ftUS 239754.527ftUS	Normal specification - Zone 1501 followed by Easting and Northing in US Survey Feet
#     1501 239754.527YftUS 2089808.121XftUS	Order reversed, Y coordinate is first, X coordinate is second.
#     1501 239754.527NftUS 2089808.121EftUS	Same coordinate, order reversed, Northing followed by Easting.
# 
# I believe these coordinates are in zone 4203 (CRS: NAD_1983_StatePlane_Texas_Central_FIPS_**4203**\_Feet) and are omiited from the X/Y coordinates in the table.
# 
# So for row 1 I think our coordinates are 4203 3100341 and 4203 10030899
# 
# I'd like to make a new column and convert NAD1983 madness to a WGS84 lat/lon that I'm used to seeing.  I found this nice [manual from 1983](https://www.ngs.noaa.gov/PUBS_LIB/ManualNOSNGS5.pdf) that details the State Plane system but thank god someone has already reaad it and figured out how to do it with Python.
# 
# 
# 

# In[ ]:


prepped = pd.read_csv('../input/data-science-for-good/cpe-data/Dept_37-00027/37-00027_UOF-P_2014-2016_prepped.csv', skiprows=1)
prepped = prepped[['Reason Desc', 'Race', 'Latitude','Longitude']].dropna(thresh=3, subset=['Race','Latitude','Longitude'])
prepped.head()


# In[ ]:


prepped['coordinates'] = prepped.apply(lambda x: Point(x['Longitude'],x['Latitude']), axis=1)
prepped = gp.GeoDataFrame(prepped, geometry='coordinates')
prepped.head()


# With the X and Y coords converted to llongitude and latitude I can convert this dataframe to a GeoPandas Dataframe 

# In[ ]:


base = df.plot(column='DISTRICT', edgecolor='white')
base.plot();


# In[ ]:


prepped.plot(column='Reason Desc', markersize=5);


# In[ ]:


fig, ax = plt.subplots()
ax.set_aspect('equal')
df.plot(ax=ax, color='black', edgecolor='white')
prepped.plot(ax=ax, column='Race', markersize=1)
plt.show();


# In[ ]:


def heatmap(d, bins=(100,100), smoothing=1.3, cmap='jet'):
    def getx(pt):
        return pt.coords[0][0]

    def gety(pt):
        return pt.coords[0][1]

    x = list(d.geometry.apply(getx))
    y = list(d.geometry.apply(gety))
    heatmap, xedges, yedges = np.histogram2d(y, x, bins=bins)
    extent = [yedges[0], yedges[-1], xedges[-1], xedges[0]]
    
    logheatmap = np.log(heatmap)
    logheatmap[np.isneginf(logheatmap)] = 0
    logheatmap = ndimage.filters.gaussian_filter(logheatmap, smoothing, mode='nearest')
    
    plt.imshow(logheatmap, cmap=cmap, extent=extent)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()


# In[ ]:


heatmap(prepped, bins=50, smoothing=1.5);

