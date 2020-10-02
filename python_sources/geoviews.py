#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import geopandas as gpd
import geoviews as gv

gv.extension('bokeh')


# In[ ]:


geometries = gpd.read_file('../input/data-map/TUN_adm1.shp')


# In[ ]:


geometries.as_matrix


# In[ ]:


idr=referendum = pd.read_csv('../input/idr-gouv/idr_gouv.csv')


# In[ ]:


idr.columns


# In[ ]:


gdf = gpd.GeoDataFrame(pd.merge(geometries, idr))


# In[ ]:


gdf.head()


# In[ ]:


plot_opts = dict(tools=['hover'], width=550, height=800, color_index='IDR',
                 colorbar=True, toolbar=None, xaxis=None, yaxis=None)
gv.Polygons(gdf, vdims=['gouvernorat', 'IDR'], label='Regional Development Index in Tunisia (2010)').opts(plot=plot_opts)


# In[ ]:


renderer = gv.renderer('bokeh')


# In[ ]:


g_idr=gv.Polygons(gdf, vdims=['gouvernorat', 'IDR'], label='Regional Development Index in Tunisia').opts(plot=plot_opts)


# In[ ]:


renderer.save(g_idr, 'g_idr')


# In[ ]:




