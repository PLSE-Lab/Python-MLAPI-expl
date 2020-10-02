#!/usr/bin/env python
# coding: utf-8

# # FBI-background-checks-dataset-and-analyse
# **Author** : Claude COSTANTINI, MBA.  
# **Purpose** : get an FBI backgound check dataset, aggregate, prepare plotting and, finaly, plot it.
# 

# In[ ]:


"""
Author : Claude COSTANTINI
Purpose : get an FBI backgound check dataset, aggregate, prepare plotting and, finaly, plot it.
"""
#imports libraries
import pandas as pd
import cartopy.crs as ccrs
import mapclassify as mc
import geopandas as gpd
import geoplot as gplt
from shapely.geometry import Point
from shapely.geometry import Polygon
import shapely.wkt


# In[ ]:


#read the csv file
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        print(filename)
df = pd.read_csv("../input/002-FBI-Background-checks_out.csv")


# After having read the file, df which is a geopandas.DataFrame, stay as-is and is considered as the "source dataset"  
# I use other dataframes to select data and to plot it. So, I can always use the method :  
# >* select and aggregate from df  
# >* prepare for plotting  
# >* plot  
# # aggregate for plotting
# >* **groupby** permits to select all the common columns    
# >* **agg** permits to aggregate and to use a function like sum(), mean(), max()  
# >* **query** permits to filter the data, and to select only the rows you want  

# In[ ]:


# aggregate the data by quarter / reset_index permits to get 'state' and 'quarter' back into the index list
df2plot=df.groupby(['month','state','centralPoint','population','geometry', 'xtext', 'ytext']).agg(
     handgun=pd.NamedAgg(column='handgun', aggfunc=sum),
    )\
.query('month == "2020-03"')\
.reset_index()


# # Create the GeoDataFrame suiting to choropleth maps

# In[ ]:


# setup Coordinate Reference System for accurate projections
crs = {'init': 'epsg:4326'}

# consider the polygons as shapely.polygon not as simple string
geometry=df2plot['geometry'].map(shapely.wkt.loads)

# create GeoDataFrame from DataFrame
gdf2plot = gpd.GeoDataFrame(df2plot, crs=crs, geometry=geometry)

# consider the points as shapely.point not as simple string
gdf2plot['centralPoint'] = gdf2plot['centralPoint'].map(shapely.wkt.loads)


# # Plot the map

# In[ ]:


#prepare the gradient for the color/value match.
scheme = mc.Quantiles(df2plot['handgun'], k=100000)

#plot the map.
ax = gplt.choropleth(
    gdf2plot, 
    hue='handgun',
    edgecolor='black',
    linewidth=0.33,
    cmap='Greens', 
    scheme=scheme,
    projection=gplt.crs.AlbersEqualArea(),
    figsize=(20, 20))

#iterate all the states and annotate all the inner legends.
for i,row in gdf2plot.iterrows():
    bbox_props = dict(boxstyle="round4,pad=0.4", fc="red", ec="g", lw=0.25)
    crs = ccrs.PlateCarree()
    transform = crs._as_mpl_transform(ax)

    ax.annotate(s=gdf2plot.state[i]+"\n"+str(gdf2plot.handgun[i]),
            xy=(gdf2plot.centralPoint[i].x,gdf2plot.centralPoint[i].y),
            xytext=(gdf2plot.xtext[i],gdf2plot.ytext[i]), 
            textcoords="offset points",
            xycoords=transform,
            ha='center', va='center',
            bbox=bbox_props,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='blue', linewidth=0.25),
            color='white')

