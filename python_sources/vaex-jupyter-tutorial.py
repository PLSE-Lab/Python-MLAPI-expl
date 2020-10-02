#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# The vaex-jupyter package contains the building blocks to interactively define an N-dimensional grid, which is then used for visualizations.

# Let us first import the relevant packages, and open the example DataFrame:

# In[ ]:


import vaex
import vaex.jupyter.model as vjm

import numpy as np
import matplotlib.pyplot as plt

df = vaex.example()
df


# We want to build a 2 dimensinoal grid with the number counts in each bin. To do this, we first define two axis objects:

# In[ ]:


E_axis = vjm.Axis(df=df, expression=df.E, shape=140)
Lz_axis = vjm.Axis(df=df, expression=df.Lz, shape=100)
Lz_axis


# In[ ]:


await vaex.jupyter.gather()  # wait until Vaex is done with all background computation
Lz_axis  # now min and max are computed, and bin_centers is set


# An interactive xarray DataArray display
# # 

# In[ ]:


data_array_widget = df.widget.data_array(axes=[Lz_axis, E_axis], selection=[None, 'default'])
data_array_widget  # being the last expression in the cell, Jupyter  will 'display' the widget


# In[ ]:


# NOTE: since the computations are done in the background, data_array_widget.model.grid is initially None.
# We can ask vaex-jupyter to wait till all executions are done using:
await vaex.jupyter.gather()
# get a reference to the xarray DataArray object
data_array = data_array_widget.model.grid
print(f"type:", type(data_array))
print("dims:", data_array.dims)
print("data:", data_array.data)
print("coords:", data_array.coords)
print("Lz's data:", data_array.coords['Lz'].data)
print("Lz's attrs:", data_array.coords['Lz'].attrs)
print("And displaying the xarray DataArray:")
display(data_array)  # this is what the vaex.jupyter.view.DataArray uses


# In[ ]:


df.select(df.x > 0)


# # Interactive plots

# In[ ]:


# NOTE: da is short for 'data array'
def plot2d(da):
    plt.figure(figsize=(8, 8))
    ar = da.data[1]  # take the numpy data, and select take the selection
    print(f'imshow of a numpy array of shape: {ar.shape}')
    plt.imshow(np.log1p(ar.T), origin='lower')

df.widget.data_array(axes=[Lz_axis, E_axis], display_function=plot2d, selection=[None, True])


# In[ ]:


df.select(df.id < 10)


# # Our improved visualization with proper axes and labeling:

# In[ ]:


def plot2d_with_labels(da):
    plt.figure(figsize=(8, 8))
    grid = da.data  # take the numpy data
    dim_x = da.dims[0]
    dim_y = da.dims[1]
    plt.title(f'{dim_y} vs {dim_x} - shape: {grid.shape}')
    extent = [
        da.coords[dim_x].attrs['min'], da.coords[dim_x].attrs['max'],
        da.coords[dim_y].attrs['min'], da.coords[dim_y].attrs['max']
    ]
    plt.imshow(np.log1p(grid.T), origin='lower', extent=extent, aspect='auto')
    plt.xlabel(da.dims[0])
    plt.ylabel(da.dims[1])

da_plot_view_nicer = df.widget.data_array(axes=[Lz_axis, E_axis], display_function=plot2d_with_labels)
da_plot_view_nicer


# In[ ]:


def plot2d_with_selections(da):
    grid = da.data
    # Create 1 row and #selections of columns of matplotlib axes
    fig, axgrid = plt.subplots(1, grid.shape[0], sharey=True, squeeze=False)
    for selection_index, ax in enumerate(axgrid[0]):
        ax.imshow(np.log1p(grid[selection_index].T), origin='lower')

df.widget.data_array(axes=[Lz_axis, E_axis], display_function=plot2d_with_selections,
                     selection=[None, 'default', 'rest'])


# # Modifying a selection will update the figure.

# In[ ]:


df.select(df.id < 10)  # select 10 objects
df.select(df.id >= 10, name='rest')  # and the rest


# Let us introduce another axis, FeH (fun fact: FeH is a property of stars that tells us how much iron relative to hydrogen is contained in them, an idicator of their origin):

# In[ ]:


FeH_axis = vjm.Axis(df=df, expression='FeH', min=-3, max=1, shape=5)
da_view = df.widget.data_array(axes=[E_axis, Lz_axis, FeH_axis], selection=[None, 'default'])
da_view


# # Selection widgets

# In[ ]:


selection_widget = df.widget.selection_expression()
selection_widget


# In[ ]:


await vaex.jupyter.gather()
w = df.widget.counter_selection('default', lazy=True)
w


# # Axis control widgets

# In[ ]:


x_axis = vjm.Axis(df=df, expression=df.Lz)
y_axis = vjm.Axis(df=df, expression=df.E)

da_xy_view = df.widget.data_array(axes=[x_axis, y_axis], display_function=plot2d_with_labels, shape=180)
da_xy_view


# In[ ]:


# wait for the previous plot to finish
await vaex.jupyter.gather()
# Change both the x and y axis
x_axis.expression = np.log(df.x**2)
y_axis.expression = df.y
# Note that both assignment will create 1 computation in the background (minimal amount of passes over the data)
await vaex.jupyter.gather()
# vaex computed the new min/max, and the xarray DataArray
# x_axis.min, x_axis.max, da_xy_view.model.grid


# But, if we want to create a dashboard with Voila, we need to have a widget that controls them:

# In[ ]:


x_widget = df.widget.expression(x_axis.expression, label='X axis')
x_widget


# In[ ]:


from ipywidgets import link
link((x_widget, 'value'), (x_axis, 'expression'))


# In[ ]:


y_widget = df.widget.expression(y_axis, label='X axis')
# vaex now does this for us, much shorter
# link((y_widget, 'value'), (y_axis, 'expression'))
y_widget


# In[ ]:


await vaex.jupyter.gather()  # lets wait again till all calculations are finished


# # A nice container

# In[ ]:


from vaex.jupyter.widgets import ContainerCard

ContainerCard(title='My plot',
              subtitle="using vaex-jupyter",
              main=da_xy_view,
              controls=[x_widget, y_widget], show_controls=True)


# In[ ]:


y_axis.expression = df.vx


# # Interactive plots

# In[ ]:


df = vaex.example()  # we create the dataframe again, to leave all the plots above 'alone'
heatmap_xy = df.widget.heatmap(df.x, df.y, selection=[None, True])
heatmap_xy


# In[ ]:


heatmap_xy.model.x


# The heatmap itself is again a widget. Thus we can combine it with other widgets to create a more sophisticated interface.

# In[ ]:


x_widget = df.widget.expression(heatmap_xy.model.x, label='X axis')
y_widget = df.widget.expression(heatmap_xy.model.y, label='X axis')

ContainerCard(title='My plot',
              subtitle="using vaex-jupyter and bqplot",
              main=heatmap_xy,
              controls=[x_widget, y_widget, selection_widget],
              show_controls=True,
              card_props={'style': 'min-width: 800px;'})


# In[ ]:


heatmap_xy.tool = 'pan-zoom'  # we can also do this programmatically.


# In[ ]:


heatmap_xy.model.x.expression = np.log10(df.x**2)
await vaex.jupyter.gather()  # and we wait before we continue


# In[ ]:


histogram_Lz = df.widget.histogram(df.Lz, selection_interact='default')
histogram_Lz.tool = 'select-x'
histogram_Lz


# In[ ]:


# You can graphically select a particular region, in this case we do it programmatically
# for reproducability of this notebook
histogram_Lz.plot.figure.interaction.selected = [1200, 1300]


# # Creating your own visualizations

# In[ ]:


The primary goal of Vaex-Jupyter is to provide users with a framework to create dashboard and new visualizations. Over time more visualizations will go into the vaex-jupyter package, but giving you the option to create new ones is more important. To help you create new visualization, we have examples on how to create your own:

If you want to create your own visualization on this framework, check out these examples:

ipyvolume example


# ![https://vaex.readthedocs.io/en/latest/example_jupyter_plotly.html](http://)
# 

# In[ ]:


plotly example
alt text


# ![https://vaex.readthedocs.io/en/latest/example_jupyter_plotly.html](http://)

# # END OF THE NOTEBOOK
