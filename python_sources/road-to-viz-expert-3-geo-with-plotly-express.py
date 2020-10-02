#!/usr/bin/env python
# coding: utf-8

# ## Road to Visualization Expert
# 
# ### Part 3 : Geo with Plotly
# 
# This time it's a visualization of military costs.
# 
# In this kernel, we are going to use map visualization and new visualization tools.
# 
# ![Military](https://thumbor.forbes.com/thumbor/960x0/https%3A%2F%2Fblogs-images.forbes.com%2Fniallmccarthy%2Ffiles%2F2017%2F04%2F20170424_Military_Expenditure.jpg)
# 
# I would like to record my practice to become an expert in data visualization.
# 
# - [Road to Viz Expert (1) - Unusual tools](https://www.kaggle.com/subinium/road-to-viz-expert-1-unusual-tools)
# - [Road to Viz Expert (2) - Plotly & Seaborn](https://www.kaggle.com/subinium/road-to-viz-expert-2-plotly-seaborn)
# 
# **plotly express is amazing.**
# 
# **Table of Contents**
# 
# - EDA & Preprocessing
#     - missingno : color change
#     - go.Table
#     - Pandas Tricks (melt, rename, etc)
# - Map with Scatter&Bubble : px.scatter_geo
#     - default
#     - projection
#     - color
#     - size
#     - animation
# - Choropleth : px.choropleth
#     - default
#     - range_color
#     - animation
# - Network&Line Graph on Map : px.line_geo
#     - default

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import os
print(os.listdir('../input/military-expenditure-of-countries-19602019'))


# ## EDA & Preprocessing

# In[ ]:


data = pd.read_csv('../input/military-expenditure-of-countries-19602019/Military Expenditure.csv')
data.head()


# - [misingno | Advanced Configuration](https://github.com/ResidentMario/missingno/blob/1ef039cda6f77232f78ced4f1cfff15c53e300e8/CONFIGURATION.md)
# 
# You can change missingno matrix color by using `color` parameter.

# In[ ]:


import missingno as msno
data2 = msno.nullity_sort(data, sort='descending')
msno.matrix(data2, color=(0.294, 0.325, 0.125)) #color : Army 


# You can make country list by using `go.Table` + `reshape method`.

# In[ ]:


import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(cells=dict(values=data['Name'].values.reshape(12, 22)))])
fig.update_layout(title=f'Countries Name List ({data.shape[0]})')
fig.show()


# The indicators don't seem to mean much to me. If you look at it, you can see that there is only one content. Let's drop this.

# In[ ]:


print(f"There are {len(data['Indicator Name'].unique())} types of indicator in this dataset.")


# In[ ]:


data.drop(['Indicator Name'], axis=1, inplace=True)
data.head()


# I want to add `continent` column.
# 
# Use `px` default dataset : `px.data.gapminder()`

# In[ ]:


third_data = px.data.gapminder()
third_data.head()


# To merge 2 pandas dataset, preprocessing first.
# I use `drop_duplicates`(remove duplicates raws) + `merge`

# In[ ]:


code_continent = third_data[['iso_alpha', 'continent']].drop_duplicates()
code_continent.rename(columns={'iso_alpha':'Code'}, inplace=True)
code_continent.head()


# In[ ]:


clean_data = pd.merge(data, code_continent , how='left')
clean_data.head()


# In[ ]:


clean_data['continent'] = clean_data['continent'].fillna('unknown')
clean_data = clean_data.fillna(0)


# Draw `Type` countplot with `px.histogram`. 
# 
# > if you don't know how to draw countplot with plotly, 
# Please refer to this article [link]()

# In[ ]:


fig = px.histogram(clean_data, x="Type", y="Type", color="Type")
fig.show()


# The country is overwhelmingly large, the rest very few. If you visualize, the rest will be buried.
# 
# How about continent?

# In[ ]:


fig = px.histogram(clean_data, x="continent", y="continent", color="continent")
fig.update_layout(title='Continent Distribution')
fig.show()


# I can see that more continents are filled than I expected.
# 
# And I will change the year data structure to use this data as **time series data**.
# 
# Use `melt` method. [link](https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-by-melt)

# In[ ]:


data_time = clean_data.melt(id_vars=['Name', 'Code', 'Type', 'continent'])
data_time.rename(columns={'variable' : 'year'}, inplace=True)
data_time


# Let's check null data again.

# In[ ]:


msno.matrix(data_time, color=(0.294, 0.325, 0.125))


# Cool! Let's start visualization with `clean_data` and `data_time`!

# ## Map with Scatter & Bubble : px.scatter_geo
# 
# - [px.scatter_geo]()

# In[ ]:


# type 1 : default graph
# location : country code 
# hover_name : hover information, single column or columns list
import plotly.express as px

fig = px.scatter_geo(clean_data, 
                     locations="Code",
                     hover_name="Name",
                    )
fig.update_layout(title="Simple Map")
fig.show()


# In[ ]:


# type 2 : projection type change

fig = px.scatter_geo(
    clean_data, 
    locations = 'Code',
    hover_name="Name",
    projection="orthographic",
)

fig.update_layout(title='Orthographic Earth')

fig.show()


# In[ ]:


# type 2-2 : projection type change

fig = px.scatter_geo(
    clean_data, 
    locations = 'Code',
    hover_name="Name",
    projection="natural earth",
)

fig.update_layout(title='Natural Earth')

fig.show()


# In[ ]:


# type 3 : change scatter marker size
# size : marker size

fig = px.scatter_geo(
    clean_data, 
    locations = 'Code',
    hover_name="Name",
    size = '2018',
)

fig.show()


# I'm going to delete too large values to see more data.
# 
# Let's look at the distribution before that.

# In[ ]:


fig, ax = plt.subplots(1, 2,figsize=(20, 7))
sns.distplot(clean_data['2018'], color='orange', ax=ax[0])
sns.boxplot(y=clean_data['2018'], color='orange', ax=ax[1])
ax[1].set_title("50% of 2018 non-zero data is under {}".format(clean_data[clean_data['2018'] > 0]['2018'].quantile(0.5)))
plt.show()


# In[ ]:


# type 3-2 : remove outliers

fig = px.scatter_geo(
    clean_data[clean_data['2018'] < 0.3 * 1e12 ], 
    locations = 'Code',
    hover_name="Name",
    size = '2018',
    projection="natural earth",
)

fig.show()


# You can also color to whatever category you want.

# In[ ]:


# type 4 : color

fig = px.scatter_geo(
    clean_data, 
    locations = 'Code',
    hover_name="Name",
    color = 'continent'
)

fig.show()


# I think we can use KNN(K-Nearest Neigbors) to fill continent (except russia)

# In[ ]:


# type 4-2 : color with size

fig = px.scatter_geo(
    clean_data[clean_data['2018'] < 0.2 * 1e12 ], 
    locations = 'Code',
    hover_name="Name",
    size='2018',
    color = 'continent'
)

fig.show()


# In[ ]:


# type 5 : animation with year
fig = px.scatter_geo(data_time, locations="Code", color="continent",
                     hover_name="Name", size="value",
                     animation_frame="year",
                     projection="natural earth")

fig.update_layout(title='Animation but USA...')
fig.show()


# The result I wanted was the following ...
# 
# There's a certain outlier in this dataset.

# In[ ]:


# type 5-2 : animation with default data
# this is population dataset
gapminder = px.data.gapminder()
fig = px.scatter_geo(gapminder, locations="iso_alpha", color="continent",
                     hover_name="country", size="pop",
                     animation_frame="year", 
                     projection="natural earth")
fig.show()


# ## Choropleth : px.choropleth
# 
# > A **choropleth map** is a thematic map in which areas are shaded or patterned in proportion to the measurement of the statistical variable being displayed on the map, such as population density or per-capita income. [Wikipedia](https://en.wikipedia.org/wiki/Choropleth_map)
# 
# Almost same as `px.scatter_geo`
# 
# You can use `range_color` to limit scale. 
# 
# 

# In[ ]:


# type 1 : default choropleth
fig = px.choropleth(clean_data, locations="Code", color="2018",
                     hover_name="Name", 
                    range_color=[0,10000000000],
                     projection="natural earth")

fig.update_layout(title='Choropleth Graph')
fig.show()


# Similar as `scatter_geo`, you can make animation with `year` data (or another time-series feature)

# In[ ]:


# type 2 : choropleth with animation

fig = px.choropleth(data_time, locations="Code", color="value",
                     hover_name="Name", 
                    range_color=[0,1000000000],
                    animation_frame='year')

fig.update_layout(title='Choropleth Graph Animation')
fig.show()


# ## Network & Line : px.line_geo 
# 
# Like Airplane route, it needs line graph on map.
# 
# You can use `px.line_geo`

# In[ ]:


# type 1 : default
fig = px.line_geo(clean_data[clean_data['continent'] !='unknown'], 
                  locations="Code", 
                  color="continent")
fig.show()


# ## TO BE CONTINUE ...
# The kernel is still in progress.

# ## Pre-Conclusion
# 
# - USA is powerful...
