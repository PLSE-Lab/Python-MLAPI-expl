#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#for easy autocomplete
get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')


# In[4]:


#data load
df_load = []
for x in range(2002,2017):
    df_load.append(pd.read_csv("../input/regions_pl_uro_{}_00_2p.csv".format(x),index_col=0, usecols=list(range(1,14))))

df = pd.concat(df_load, keys = list(range(2002, 2018)))

#create multiindex
index_years = df.index.levels[0]
index_regions = df.index.levels[1].str.strip()
df.index = pd.MultiIndex.from_product([index_years, index_regions], names=['year', 'region'])
df.loc[(slice(2010,2017), 'POMORSKIE'),:]


# In[5]:


# from ipywidgets import interact
import matplotlib.pyplot as plt
from bokeh.plotting import figure, gmap, ColumnDataSource
from bokeh.io import show, output_notebook, push_notebook
from bokeh.models import HoverTool, GMapOptions

output_notebook()


# In[7]:


# @hidden_cell
google_key = 'AIzaSyDRkmIcmSvLNQx7kDWvQ0MILMgvDJwQ6JU'


# In[31]:


regions_lon_lat = pd.read_csv("../input/regions_lon_lat.csv", index_col=0)
regions_lon_lat.index = regions_lon_lat.index.str.upper()

df_lon_lat = pd.merge(df.loc[2010], regions_lon_lat, left_index=True, right_index=True)

source = ColumnDataSource(data=df_lon_lat)


# In[32]:


map_options = GMapOptions(lat=52.0, lng=19, map_type="roadmap", zoom=6)
p = gmap(google_key, map_options, title="Births in Poland", tools=['pan', 'wheel_zoom', 'tap', 'box_zoom'])
#p = figure( x_axis_label='region', y_axis_label='births total')
#p.line(x='region', y=1, color="red", source = source)
#p.line(df[0]['region'], df[0]['1'])
#p.xaxis.major_label_orientation = np.pi/2
circles = p.circle(x="lon", y="lat", name="cities", size=10, fill_color="blue", fill_alpha=0.8, source=source)


show(p)

from bokeh.embed import components
script, div = components(p)


# In[ ]:


print(script)


# In[ ]:


p = figure(title="KUJAWSKO-POMORSKIE", x_axis_label='region', y_axis_label='births total')
p.line(range(2002,2017), df[:][1],color="red")
show(p)


# In[ ]:




