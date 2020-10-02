#!/usr/bin/env python
# coding: utf-8

# # Data Visualization with Bokeh

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from bokeh.io import output_file,show,output_notebook,save
from bokeh.plotting import figure


# In[ ]:


df = pd.read_csv("../input/airpassengers-dataframe/AirPassengers.csv")


# In[ ]:


df["Avg"] = df.mean(axis=1,skipna=True)


# In[ ]:


df


# In[ ]:


df["Sum"] = df.iloc[:,:12].sum(axis=1,skipna=True)


# In[ ]:


df


# In[ ]:


df["Year"] = [i for i in range(1949,1961)]


# In[ ]:


df


# In[ ]:


fig = figure(x_axis_label = "Year",y_axis_label = "Average and Max of Air Passengers per Year")


# In[ ]:


fig.circle(df["Year"],df["Avg"])


# In[ ]:


output_notebook()


# In[ ]:


show(fig)


# In[ ]:


save(fig,"circle_glyphs.html")


# In[ ]:


fig.line(df["Year"],df["Avg"])


# In[ ]:


show(fig)


# In[ ]:


df["Max"] =  df.iloc[:,:12].max(axis=1)


# In[ ]:


df


# In[ ]:


fig.x(df["Year"],df["Max"],line_color="red",size=10)


# In[ ]:


show(fig)


# In[ ]:


fig.line(df["Year"],df["Max"],line_color="red")


# In[ ]:


show(fig)


# In[ ]:


output_file("line_circle_glyphs.html")


# In[ ]:


fig_patches = figure(x_axis_label = "Year",y_axis_label = "Air Passengers") 


# In[ ]:


x = [df.iloc[0:,i].values for i in range(12)]


# In[ ]:


y = [df.iloc[0:,13].values for i in range(12)]


# In[ ]:


list(x)


# In[ ]:


fig_patches.patches(y,x)


# In[ ]:


show(fig_patches)


# ### Cars dataset box plot

# In[ ]:


import holoviews as hv
import matplotlib.pyplot as plt
import numpy as np
from bokeh.palettes import Spectral6
from bokeh.transform import linear_cmap
hv.extension('bokeh')


# In[ ]:


output_notebook()


# In[ ]:


cars_df = pd.read_csv("../input/cars93/Cars93.csv")


# In[ ]:


mapper = linear_cmap(field_name='y', palette=Spectral6 ,low=min(np.array(y[0])) ,high=max(np.array(y[0])))


# In[ ]:


box = hv.BoxWhisker(cars_df,"Price","Horsepower", label="Box Plot",color=mapper,fill_color=mapper,fill_alpha=1, size=12)


# In[ ]:


box.options(show_legend=True, width=800)


# In[ ]:


show(box)


# In[ ]:




