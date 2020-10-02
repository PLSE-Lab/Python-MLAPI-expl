#!/usr/bin/env python
# coding: utf-8

# ## Motivation
# 
# In December, 2019, a local outbreak of pneumonia of initially unknown cause was detected in Wuhan (Hubei, China), and was quickly determined to be caused by a novel coronavirus, namely severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). The outbreak has since spread to every province of mainland China as well as 192 other countries and regions, with more than 339,009 confirmed cases as of Mar 23, 2020.
# 
# In response to this ongoing public health emergency, researchers are releasing datasets available for academic purposes and public use. In this notebook, I will be exploring how to create a simple, yet highly visual map animation demonstrating how the virus spread across the world over the past few weeks using Python, Pandas, and Bokeh.
# 
# The full time-series dataset is continously being updated everyday, and as of Mar 23, 2020 contains 63 columns, the first four columns contain regional information, and the following columns represent each consecutive day - starting from Jan 22, 2020 - that contain the cumulative number of cases. 
# 
# Here is a short clip of what we will get:
# 
# ![Alt Text](https://media.giphy.com/media/MXpbnEhA9ML6ZOJwCm/source.gif)
# 
# Full blog post here: https://horvay.dev/covid19/

# ## Brief on Bokeh
# 
# You've probably used or at least heard of Matplotlib for working with visualization in Python, but in this post, I'll be using a library called Bokeh, so it's worth mentioning how the two differ.
# 
# [Bokeh](https://docs.bokeh.org/en/latest/index.html) is used for creating highly interactive visualizations, while Matplotlib creates static graphics that are meant for simple and quick views into data.
# 
# I went with this library since I want my visualization to be interactive - utilizing tool tips and such to give deeper insights.

# First, I'll explain the imports we are using:

# In[ ]:


import time
import pandas as pd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, push_notebook
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.tile_providers import get_provider
from pyproj import Proj, transform


# * `time` - Is used for updating the map animation, we will update it every 50 milliseconds to show growth
# * `pandas` - Is used for creating and manipulating our dataset into dataframes
# * `numpy` - Is used in conjunction with pandas - in this case for manipulating arrays
# * `bokeh.plotting` - We import `figure` for the plot, and `show` for displaying said plot
# * `bokeh.io` - These imports are used for specifically to show plots inline and updating without starting a special Bokeh server
# * `bokeh.models` - These imports are used for creating tooltips and for sourcing data
# * `bokeh.tile_providers` - Tile providers are used to render base maps of Earth
# * `pyproj` - Is used to convert latitude/longitude to a more welcoming format for Bokeh's spatial libraries

# Next, we can read in the dataset using pandas' `read_csv()` function and display the first five rows of data using the `head()` function on our created dataframe:[](http://)

# In[ ]:


# Read data
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
df.head()


# As explained before, the first four columns contain regional information, and the following columns represent each consecutive day - starting from Jan 22, 2020 - that contain the cumulative number of cases. 
# 
# We may want to engineer a new feature or column that just acts as a placeholder for the cumulative number of cases a.k.a the last day column. We can do this easily using `iloc[]` and which takes all rows as `:` and uses the `-1` indexing trick in Python, be assured we have the last day column.
# 
# You may also notice I went against the current column naming convention by using CamelCase. I am doing this because the other columns use `/` as a seperator, which makes it a bit more difficult to use tool tips when creating an interactive Bokeh plot.
# 
# Let's rename the Provinence/State and Country/Region columns as well using pandas' `rename()` function:

# In[ ]:


# Get total confirm as of today
df['TotalConfirmed'] = df.iloc[:,-1]
# Rename columns for easier access for tooltips
df.rename(columns={'Province/State':'ProvinceState', 'Country/Region':'CountryRegion'}, inplace=True) 
df.head()


# ## Working with Spatial Data
# 
# Bokeh contains built-in tile providers in the `bokeh.tile_providers` module that we will use to represent the map of Earth.
# 
# Bokeh uses a standard web tile format called [Web Mercator](https://en.wikipedia.org/wiki/Web_Mercator_projection) projection for mapping. Our coordinates are stored as latitude/longitude, so we will need to write custom mapping function. 
# 
# Credit to [this StackOverflow question](https://gis.stackexchange.com/questions/247871/convert-gps-coordinates-to-web-mercator-epsg3857-using-python-pyproj) for providing insight in how to do this.

# In[ ]:


# Helper function to convert latitude/longitude to Web Mercator for mapping
# See: https://gis.stackexchange.com/questions/247871/convert-gps-coordinates-to-web-mercator-epsg3857-using-python-pyproj
def to_web_mercator(long, lat):
    try:
        return transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), long, lat)
    except:
        return None, None
    
# Convert all latitude/longitude to Web Mercator and stores in new easting and northing columns
df['E'], df['N'] = zip(*df.apply(lambda x: to_web_mercator(x['Long'], x['Lat']), axis=1))
df['Size'] = 0
source = ColumnDataSource(df)


# ## Map Animation
# 
# ### Creating the Plot
# 
# First lets create the simple things. For the title we want it to convey the date range of our times series. We also want to add tool tips that will display the province / state, country / region, and total confirmed cases as of the last day in the dataset (today). We will also default the plot size to a width of `800px` and height of `500px`.
# 
# We will source the CartoDB Tile Service `CARTODBPOSITRON`. Specifically, we will use `CARTODBPOSITRON_RETINA` as it creates high-resolution (HiDPI) tiles with a size of 512x512px (instead of default 256x256px).
# 
# `output_notebook()` is one of the more important lines. This allows us to output the plot inline in our Jupyter Notebook, typically we would output as an HTML file from Bokeh.

# ### Updating the Plot Real-Time
# 
# We can assign a variable called `handle` that contains the output of displaying the plot. Using a while loop across the total number of days, we can then update our dataframe with the a new size for each region to represent the growth of cases (largest circles represent >= 5000 confirmed cases).
# 
# Using the `.stream()` function we can push the updated data to the plot's datasource and purge the old data after. When we call `push_notebook()` we update the Jupyter Notebook real-time! Last, we increment the day and set a sleep timer for 50 milliseconds to give it a smoother animation cycle.

# In[ ]:


title = "COVID-19 | Growth of confirmed cases {} - {}".format(df.columns[4], df.columns[df.shape[1]-5])
hover = HoverTool(tooltips=[("Province / State", "@ProvinceState"),
                            ("Country / Region", "@CountryRegion"),
                            ("Total confirmed cases as of today", "@TotalConfirmed")])
p = figure(plot_width=800, plot_height=500, title=title, tools=[hover, 'pan', 'wheel_zoom','save', 'reset'])
tile_provider = get_provider('CARTODBPOSITRON_RETINA')
p.add_tile(tile_provider)
p.circle(x='E', y='N', source=source, line_color='grey', fill_color='red', alpha=0.7, size='Size')
output_notebook()

handle = show(p, notebook_handle = True)
day = 4 # Time series data starts at the 4th col
today = df.shape[1]-5 # Time series data stops here since we added 4 new columns
max_size = 20.0 # Represents 5000+ cases
while day < today:
    df['Size'] = np.where(df.iloc[:,day]*.004<=max_size, df.iloc[:,day]*.004, max_size)
    # Push new data
    source.stream(df)
    # Purge old data
    source.data = df
    push_notebook(handle=handle)
    day += 1
    time.sleep(.5)

