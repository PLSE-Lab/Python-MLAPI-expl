#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory


# In[ ]:


from IPython.display import display

pd.options.display.max_columns = None
pd.options.display.max_rows = 15


# # The figure data structure in Python
# 
# ## Overview
# 
# The Plotly Python package is an interface to the Plotly.js library.
# 
# Figures are represented as Python `dicts` or `plotly.graph_objects`. Plotly
# serializes figures into JSON and input them into Plotly.js. Because of this
# flexibility, Plotly is available in other languages like R or Julia.
# 
# Plotly has two main libraries for plotting:
# - `plotly.graph_objects.Figure`
# - `plotly.express`
# 
# Plotly Express is the high-level entry-point to Plotly because it allows you
# to do more with less code. Use Plotly's Figure library if you're already
# comfortable using Plotly and you want to make more advanced plots.

# In[ ]:


# express
fig = px.line(x=['a', 'b', 'c'], y = [1, 3, 2], title='Express line')
fig


# ### What is a figure made of?
# 
# A figure is a __tree__ of attributes. The root attributes are: data, layout, and
# frames.
# 
# https://plotly.com/python/reference/
# 
# _What is a tree?_
# A metaphor for a data type that has:
# - Leaves.
#     - Nodes with no children.
# - Root.
# 

# In[ ]:


# see figures the way Plotly.js sees them
print(fig)


# In[ ]:


# accessing attributes
fig.data[0].line.color


# In[ ]:


# accessing attributes
fig.data[0].line.color = '#FF0000'


# In[ ]:


fig


# ### data
# 
# The `data` attribute is a list of "traces", which are dictionaries containing
#  types of graphs to be plotted in separate subplots.

# ### layout
# 
# The `layout` attribute is a dict that tells Plotly how to position parts of
# the figure.
# - Margins and size.
# - Fonts.
# - Legend and color bars.
# - Subplots
# - Interactive controls

# ### frames
# `Frames` are used in __animated plots__. It is a list of dicts containing
# attributes for each frame in an animation.

# # Creating and updating figures
# 
# __Recommended strategy from Plotly docs__
# 
#     Create entire figures at once using Plotly Express and manipulate the
#     result if you need to create something more complex.

# In[ ]:


# low level: dict
import plotly.io
fig = {
    "data": [{"type": "bar",
              "x": [1, 2, 3],
              "y": [1, 3, 2]}],
    "layout": {"title": {"text": "A Figure made with a Dictionary"}}
}

plotly.io.show(fig)


# In[ ]:


# higher level: graph_objects
import plotly.graph_objects

fig = plotly.graph_objects.Figure(
    data=[plotly.graph_objects.Bar(x=[1, 2, 3], y=[1, 3, 2])],
    layout=plotly.graph_objects.Layout(
        title=plotly.graph_objects.layout.Title(
            text='A Figure made with graph_objects'
        )
    )
)
fig


# In[ ]:


dir(plotly.express.data)


# In[ ]:


iris_data = plotly.express.data.iris()
iris_data


# In[ ]:


iris_data['species'].value_counts()


# In[ ]:


# highest level (recommended)
fig = plotly.express.scatter(data_frame=iris_data, x='sepal_width',
                             y='sepal_length', color='species',
                             title='Iris figure using Plotly Express')
fig


# In[ ]:


plotly.express.scatter(data_frame=iris_data, x='sepal_width',
                     y='sepal_length', color='species', facet_col='species',
                       size='petal_width',
                     title='Iris figure using Plotly Express')


# In[ ]:


iris_data.loc[:, 'sepal_length':'species']


# In[ ]:


plotly.express.scatter_matrix(data_frame=iris_data.loc[:, 'sepal_length':'species'], color='species')


# ## Figure factories
# Figure factories produce graph_object figures for specialized domains.

# In[ ]:


np.meshgrid([1, 2, 3], [1, 2, 3])


# In[ ]:


x, y = np.meshgrid(np.arange(-1, 2, 0.3), np.arange(-1, 4, 0.3))
u = np.cos(x) * y
v = np.sin(x) * y

fig = plotly.figure_factory.create_quiver(x, y, u, v)
fig


# In[ ]:


dir(plotly.express.data)


# In[ ]:


plotly.express.data.wind()


# In[ ]:


plotly.express.data.experiment()


# In[ ]:


plotly.express.data.gapminder()


# In[ ]:


from plotly.express.data import *


# In[ ]:


tips()


# In[ ]:


plotly.express.data.stocks()


# In[ ]:


plotly.express.data.carshare()


# In[ ]:


plotly.express.data.election()


# # Gapminder

# In[ ]:


import plotly.express as px


# In[ ]:


df_gap = px.data.gapminder()
df_gap


# In[ ]:


df_gap.info()


# In[ ]:


df_gap['year']


# In[ ]:


get_ipython().run_line_magic('pinfo', 'pd.to_datetime')


# In[ ]:


pd.to_datetime(df_gap['year'], format='%Y').plot(kind='density')


# In[ ]:


df_gap['year'] = pd.to_datetime(df_gap['year'], format='%Y')
df_gap.info()


# In[ ]:


df_gap.describe()


# In[ ]:


df_gap['year'].min()


# In[ ]:


df_gap['year'].max()


# In[ ]:


df_gap


# ## Life expectancy for each country

# In[ ]:


get_ipython().run_line_magic('pinfo', 'px.bar')


# In[ ]:


df_gap.columns


# In[ ]:


df_gap['lifeExp'].max()


# In[ ]:


fig = px.bar(df_gap, x='country', y='lifeExp')
fig


# In[ ]:


fig = px.scatter(df_gap, y='country', x='lifeExp', facet_row='continent')
fig


# In[ ]:


df_gap['continent'].unique()


# In[ ]:


get_ipython().run_line_magic('pinfo', 'px.scatter')


# In[ ]:


df_gap


# In[ ]:


fig = px.scatter(df_gap.loc[df_gap['continent'] == 'Europe'], y='country', x='lifeExp', 
                 hover_data=['year'])
fig


# In[ ]:


fig = px.scatter(
    df_gap.loc[df_gap['continent'] == 'Europe'], x='year', y='lifeExp', size='lifeExp',
    color='country'
                )
fig


# In[ ]:


get_ipython().run_line_magic('pinfo', 'pd.Series.plot')


# In[ ]:


df_gap['gdpPercap'].loc[(df_gap['gdpPercap'] < 20000) & (df_gap['gdpPercap'] > 2000)].plot(kind='box');


# In[ ]:


# list comprehension
['high' if i > 5000 else 'low' for i in df_gap['gdpPercap']]


# In[ ]:


df_gap['gdp_cat'] = ['high' if i > 5000 else 'low' for i in df_gap['gdpPercap']]


# In[ ]:


fig = px.scatter(
    df_gap.loc[df_gap['continent'] == 'Europe'], x='year', y='lifeExp',
    color='country', facet_col='gdp_cat'
                )
fig


# In[ ]:


df_gap['gdp_cat'] = ['high' if i > 5000 else 'low' for i in df_gap['gdpPercap']]


# In[ ]:


df_gap_am = df_gap.loc[df_gap['continent'] == 'Americas']


# In[ ]:


df_gap_am


# In[ ]:


df_gap_am.describe()


# In[ ]:


df_gap_am['gdp_cat'] = ['high' if i > df_gap_am['gdpPercap'].median() else 'low' for i in df_gap_am['gdpPercap']]


# In[ ]:


df_gap_am


# In[ ]:


fig = px.scatter(
    df_gap_am, x='year', y='lifeExp',
    color='country', facet_col='gdp_cat', size='pop'
                )
fig


# In[ ]:


df_gap_am.columns


# In[ ]:


fig


# In[ ]:


fig = px.scatter(
    df_gap_am, x='year', y='lifeExp',
    color='country', marginal_y='box',  facet_row='gdp_cat'
                )
fig


# In[ ]:


import plotly.figure_factory as ff

df = [dict(Task="Job A", Start='2009-01-01', Finish='2009-02-28'),
      dict(Task="Job B", Start='2009-03-05', Finish='2009-04-15'),
      dict(Task="Job C", Start='2009-02-20', Finish='2009-05-30')]

fig = ff.create_gantt(df)
fig.show()


# In[ ]:


fig = px.line(
    df_gap.loc[df_gap['continent'] == 'Oceania'], x='year', y='lifeExp',
    color='country'
                )
fig


# In[ ]:


fig = px.bar(df_gap, x='country', y='lifeExp', facet_col='continent')
fig


# ## Life exp for each country for each decade
# 

# In[ ]:




