#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from ipywidgets import interact
# bokeh packages
from bokeh.io import output_file,show,output_notebook,push_notebook
from bokeh.plotting import *
from bokeh.models import ColumnDataSource,HoverTool,CategoricalColorMapper
from bokeh.layouts import row,column,gridplot,widgetbox

from bokeh.layouts import layout

from bokeh.embed import file_html

from bokeh.models import Text
from bokeh.models import Plot
from bokeh.models import Slider
from bokeh.models import Circle
from bokeh.models import Range1d
from bokeh.models import CustomJS
from bokeh.models import LinearAxis
from bokeh.models import SingleIntervalTicker

from bokeh.models import Select
from bokeh.palettes import Spectral5,Inferno11
from bokeh.plotting import curdoc, figure
#from bokeh.sampledata.autompg import autompg_clean as df

from bokeh.palettes import Spectral6
output_notebook()
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import cufflinks as cf
cf.set_config_file(world_readable=True,offline=True)


# In[ ]:


# load data to pandas dataframe
df = pd.read_csv('../input/scmp2k19.csv')


# In[ ]:


# get some info about data 
df.info()


# In[ ]:


# explore columns related to the addrress
df.loc[:,['district','mandal','location',]].sample(7,random_state=1)


# In[ ]:


df = df.copy()


# In[ ]:


SIZES = list(range(6, 22, 3))
COLORS = Spectral5
N_SIZES = len(SIZES)
N_COLORS = len(COLORS)


# In[ ]:


# data cleanup
df.district  = df.district.astype(str)
df.humidity_min = df.humidity_min.astype(str)
del df['odate']


# In[ ]:


columns = sorted(df.columns)
discrete = [x for x in columns if df[x].dtype == object]
continuous = [x for x in columns if x not in discrete]


# In[ ]:


TOOLTIPS = [
            ('district','@district'),
            ('mandal', '@mandal'),
            ('location', '@location')
            
           ]


# In[ ]:


def create_figure():
    xs = df[x.value].values
    ys = df[y.value].values
    x_title = x.value.title()
    y_title = y.value.title()

    kw = dict()
    if x.value in discrete:
        kw['x_range'] = sorted(set(xs))
    if y.value in discrete:
        kw['y_range'] = sorted(set(ys))
    kw['title'] = "%s vs %s" % (x_title, y_title)

    p = figure(plot_height=600, plot_width=800, tools='pan,box_zoom,hover,reset',**kw)
    p.xaxis.axis_label = x_title
    p.yaxis.axis_label = y_title



    if x.value in discrete:
        p.xaxis.major_label_orientation = pd.np.pi / 4

    sz = 9
    if size.value != 'None':
        if len(set(df[size.value])) > N_SIZES:
            groups = pd.qcut(df[size.value].values, N_SIZES, duplicates='drop')
        else:
            groups = pd.Categorical(df[size.value])
        sz = [SIZES[xx] for xx in groups.codes]

    c = "#31AADE"
    if color.value != 'None':
        if len(set(df[color.value])) > N_COLORS:
            groups = pd.qcut(df[color.value].values, N_COLORS, duplicates='drop')
        else:
            groups = pd.Categorical(df[color.value])
        c = [COLORS[xx] for xx in groups.codes]

    p.circle(x=xs, y=ys, color=c, size=sz, line_color="white", alpha=0.6, hover_color='Inferno11', hover_alpha=0.5)

    return p


# In[ ]:


def update(attr, old, new):
    layout.children[1] = create_figure()


x = Select(title='X-Axis', value='district', options=columns)
x.on_change('value', update)

y = Select(title='Y-Axis', value='humidity_min', options=columns)
y.on_change('value', update)

size = Select(title='Size', value='None', options=['None'] + continuous)
size.on_change('value', update)

color = Select(title='Color', value='None', options=['None'] + continuous)
color.on_change('value', update)

controls = column([x, y, color, size], width=200)
layout = row(controls, create_figure())

curdoc().add_root(layout)
curdoc().title = "Crossfilter"


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# Row and column
source =  
p1 = figure()
p1.circle(x = "district",y= "Rangareddy",source = source,color="red")
p2 = figure()
p2.circle(x = "district",y= "Warangal",source = source,color="black")
p3 = figure()
p3.circle(x = "district",y= "Khammam",source = source,color="blue")
p4 = figure()
p4.circle(x = "district",y= "Nalgonda",source = source,color="orange")
layout1 = row(p1,p2)
layout2 = row(p3,p4)
layout3= column(layout1,layout2)
show(layout3)


# In[ ]:


show(layout3)


# In[ ]:


# Color mapping
factors = list(df.mandal.unique()) # what we want to color map. I choose genre of games
colors = ["red","green","blue","black","orange","brown","grey","purple","yellow","white","pink","peru"]
mapper = CategoricalColorMapper(factors = factors,palette = colors)
plot =figure()
plot.circle(x= "odate",y = "humidity_min",source=source,color = {"field":"Genre","transform":mapper})
show(plot)
# plot looks like confusing but I think you got the idea of mapping 


# In[ ]:




