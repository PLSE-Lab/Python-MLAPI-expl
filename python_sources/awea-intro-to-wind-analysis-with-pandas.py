#!/usr/bin/env python
# coding: utf-8

# ![www.awea.org](http://awea.files.cms-plus.com/pagelayoutimages/AWEA_Logo_Web.png)
# 
# # AWEA Resource Assessment Workshop - Pre-Conference Python Tutorial
# 
# # About this Tutorial
# * Length: 3-ish hours
# * Prerequisites: 
#     * No experience necessary 
#     * Basic programming concepts and wind assessment methodology helpful
#     * Python very helpful
# * Goals: 
#   * Get exposure to fundamental wind analysis tools
#   * Fork this kernel
#   * Become familiar with pandas, numpy, and data visualization 
#   * Develop analysis
# <br><br>
# 
# # Table of contents:
# - [Introduction](#introduction)
#     - [Python](#python)
#     - [Pandas](#pandas)
#     - [Numpy](#numpy)
#     - [Scipy](#scipy)
# - [Dataset](#data)
#     - [Importing](#importing)
# - [Dataframe](#dataframe)
# 	- General info
#     - [Head and tail](#headtail)
#     - [Slicing data](#slicing)
#         - [Columns](#columns)
#         - [Rows](#rows)
#         - [Rows and columns](#rowscolumns)
# - Split-Apply-Combine
# 	- [Grouped and aggregated calculations](#grouped)
#     - [Resample](#resample)
#     - [Rolling](#rolling)
# - [Shear](#shear)
# - [MCP](#mcp)
# - [Frequency distribution](#freqdist)
#     - [Wind speed](#wsfreqdist)
#     - [Windrose](#windrose)
# - Visualizing data and results
# 	- [Matplotlib](#matplotlib)
#     - [Plotly](#plotly)
# - [Exporting](#exporting)
# - [Anemoi](#anemoi)
#     - [Sensor naming convention](#naming)
#     - [MetMast](#metmast)
#     - Analysis modules
#         - [Shear](#shear)
#         - [Correlate](#correlate)
#     - [Plotting module](#ploting)
# 
# <a id="introduction"></a> 
# # A humble introduction to wind analysis
# ---
# This is a humble introduction to wind analysis using open-source, data science tools. This intro is meant for analysts new to Python-based tools but familiar with wind resource assessment methodology. This notebook is based on the [Pandas for Data Analysis Tutorial](https://youtu.be/oGzU688xCUs), presented by Daniel Chen at SciPy 2017 in Austin, Texas. The full GitHub repository of those tutorial notebooks can be found [here](https://github.com/chendaniely/scipy-2017-tutorial-pandas/tree/master/01-notes).
# 
# My main goal is to remove the barier to entry for an analyst or organization interested in adopting modern, open-source, data-analysis tools. Kaggle is a great environment to learn within because it takes care of the software and package installation needed to get up and running. Kernel notebooks are an effective introductory tool because I can develop an analysis and you can copy, or fork, this notebook to develop your own analysis. Most likely, you'll find it easier to install an environment on your own machine using [Anaconda](https://www.anaconda.com/download/) but for sharing and learning, Kernels are highly effective. 
# 
# <a id="python"></a> 
# ## Python
# 
# > Why would you code in Python, it's slow? Why don't you commute by airplane, its really fast? -Jake Vanderplas PyCon 2017
# 
# We won't go into too much detail about [Python](https://www.python.org/about/) here but if you want to learn more there are plenty of resources available online. In summary, Python is a general-purpose scripting language designed to be easy to read and efficient to develop code. Pure Python is not very helpful for wind analysis without additional packages. It is the ecosystem of data analysis packages that greatly extend Python's capabilities and make it very effective at data analysis. This tutorial will introduce you to some of those packages. For a more in-depth tour of the ecosystem, with respect to wind analysis, you can watch the Working Group's webinar [here](https://register.gotowebinar.com/recording/2399581164311633667). You can also see more in-depth wind analysis examples [here](https://www.kaggle.com/coryjog/example-wind-analysis-in-python) and [here](https://www.kaggle.com/srlightfoote/example-wind-analysis-in-r). 

# In[ ]:


import this


# <a id="pandas"></a> 
# ## Pandas
# 
# The [pandas](https://pandas.pydata.org/) analysis library is a primary reason for Python's adoption by the data science community. There are a bunch of resources [here](https://pandas.pydata.org/talks.html). It can provide quite a bit of value to any organization looking for an easy to adopt tool to analize wind data. One of its strengths is time series analysis and conveniently wind data are mostly in the form of time series.
# 
# <a id="numpy"></a> 
# ## Numpy
# 
# [Numpy](http://www.numpy.org/) is the fundamental package for scientific computing with Python. It is also the primary package upon which Pandas is built and they interface very well with each other. It includes many features that are beneficial to wind analysis and we'll be using the statistics and linear algebra functionality today. You can find a comprehensive, Numpy specific tutorial [here](https://docs.scipy.org/doc/numpy/user/quickstart.html).
# 
# <a id="scipy"></a> 
# ## Scipy
# [Scipy](https://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/general.html) is a collection of mathematical algorithms and convenience functions built on the Numpy extension of Python. The following subpackages are especially helpful for wind data analysis:
# * interpolate - Interpolation and smoothing splines
# * linalg - Linear algebra
# * odr - Orthogonal distance regression
# * spatial - Spatial data structures and algorithms (GIS)
# * stats - Statistical distributions and functions
# <br></br>
# 
# <a id="pandas"></a>
# # Dataset
# ---
# But, before we get into analysis we need to get our wind data into our notebook environment. Pandas has a very comprehensive .csv importer which makes ingesting data from an NRG logger or a visualization tool like Windographer very easy. EDF Renwables has provided some anonymized, altered, otherwise unusable data from a meteorological mast in North America along with some nearby reference station data. We'll use these data throughout the rest of the tutorial.
# 
# <a id="importing"></a>
# ## Importing
# The primary means by which to get data into a notebook is the read_csv method. Here you can find the [documentation and some working examples](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html). We'll use this in the next couple of cells to import our mast and reference data.

# In[ ]:


import pandas as pd
import numpy as np

import os
print(os.listdir("../input"))


# In[ ]:


mast_data = pd.read_csv('../input/demo_mast.csv', index_col=0, parse_dates=True, infer_datetime_format=True)
mast_data.head()


# In[ ]:


ref_data = pd.read_csv('../input/demo_refs.csv', index_col=0, parse_dates=True, infer_datetime_format=True)
ref_data.head()


# <a id="dataframe"></a>
# # DataFrame
# ---

# In[ ]:


# what is this df object?
type(mast_data)


# In[ ]:


# get num rows and columns# get n 
mast_data.shape


# In[ ]:


# shape is an attribute, not a function
mast_data.shape()


# In[ ]:


mast_data.info()


# <a id="headtail"></a>
# ## Head and tail

# In[ ]:


mast_data.head()


# In[ ]:


mast_data.tail()


# In[ ]:


mast_data.describe()


# <a id="slicing"></a>
# ## Slicing data
# 
# The axis labeling information in pandas objects serves many purposes ([documentation](https://pandas.pydata.org/pandas-docs/stable/indexing.html)):
# 
# * Identifies data (i.e. provides metadata) using known indicators, important for analysis, visualization, and interactive console display.
# * Enables automatic and explicit data alignment.
# * Allows intuitive getting and setting of subsets of the data set.
# 
# In this section, we will focus on the final point: namely, how to slice, dice, and generally get and set subsets of pandas DataFrames of wind data.
# 
# <a id="columns"></a>
# ### Columns

# In[ ]:


# subset a single column with a column label
mast_data['SPD_59_COMB_AVG']


# In[ ]:


mast_data.SPD_59_COMB_AVG


# In[ ]:


mast_data.SPD_59_COMB_AVG.head()


# In[ ]:


# delete columns
# this won't drop in-place unless you use the inplace parameter
mast_data.drop('SPD_59_COMB_AVG', axis=1).head()


# In[ ]:


# Stamp is the index, NOT a column


# In[ ]:


mast_data.Stamp # this will fail


# In[ ]:


mast_data.index


# <a id="rows"></a>
# ### Rows
# 
# [loc](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html) indexes by labels or booleans as opposed to [iloc](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.iloc.html#pandas.DataFrame.iloc) which indexes by integer position. For the purposes of wind analysis I use loc almost exclusively. 
# 

# In[ ]:


# first row
mast_data.loc['2014-12-17 11:00:00']


# In[ ]:


# 5th row
mast_data.loc['2014-12-17 11:40:00']


# In[ ]:


# first row
mast_data.iloc[0]


# In[ ]:


# 5th row
mast_data.iloc[4]


# In[ ]:


# last row
mast_data.iloc[-1]


# <a id="rowscolumns"></a>
# ### Rows and columns

# In[ ]:


# the bracket notation
# row subsetter
# comma
# column subsetter
mast_data.loc['2014-12-17 11:10:00', 'SPD_59_COMB_AVG']


# In[ ]:


mast_data.loc['2014-12-17 11:00:00':'2014-12-17 12:00:00', ['SPD_59_COMB_AVG', 'DIR_80_AVG', 'T_4_AVG']]


# In[ ]:


mast_data.loc[mast_data.index < '2014-12-17 12:00:00', ['SPD_59_COMB_AVG', 'DIR_80_AVG', 'T_4_AVG']]


# In[ ]:


mast_data.loc[mast_data.index.month == 12, ['SPD_59_COMB_AVG', 'DIR_80_AVG', 'T_4_AVG']]


# Finally, look at all the cool things dataframes can do!
# 
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# 
# # Split - Apply - Combine
# ---
# [Groupby](https://pandas.pydata.org/pandas-docs/stable/groupby.html) is a very useful concept adopted from SQL and can be very helpful in wind profile calculations.
# 
# 
# <a id="grouped"></a>
# ## Grouped and aggregated calculations
# 
# Pandas has a lot of grouping and aggregating functionality built in. This is esspecially helpful for time series analysis of wind data. 
# 
# 

# In[ ]:


mast_data.index.year.unique()


# In[ ]:


mast_data.groupby(mast_data.index.year).mean()


# In[ ]:


mast_data.groupby([mast_data.index.year, mast_data.index.month]).mean()


# In[ ]:


annual_profile = mast_data.groupby(mast_data.index.month).mean().SPD_59_COMB_AVG
annual_profile


# <a id="resample"></a>
# ## Resample
# 
# Very helpful method for wind analysis for frequency conversion and resampling of time series. Object must have a datetime-like index. Also refered to as reaveraging by the industry. [Documentation here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.resample.html). Documentation on time periods [here](http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases).

# In[ ]:


mast_data.resample('MS').mean()


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


annual_profile.plot();
# annual_profile.plot(ylim=[0,10], title='Annual wind speed profile');


# In[ ]:


import plotly
import plotly.graph_objs as go
plotly.__version__


# In[ ]:


fig = go.FigureWidget(data=[{'x':annual_profile.index,'y':annual_profile.values}])
fig


# <a id="rolling"></a>
# ## Rolling
# 
# A typical consistency analysis for reference station data includes plotting the 12-month normalized rolling average wind speeds. The rolling method, along with resample and Panda's automatic index alignment, make it possible to do this in six lines of code. [Documentation here](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.rolling.html).

# In[ ]:


ref_data.head()


# In[ ]:


yearly_monthly_means = ref_data.resample('MS').mean()
monthly_means = ref_data.groupby(ref_data.index.month).mean()
monthly_means_for_normal = monthly_means.loc[yearly_monthly_means.index.month]
monthly_means_for_normal.index = yearly_monthly_means.index
yearly_monthly_means_normal = yearly_monthly_means/monthly_means_for_normal
yearly_monthly_means_normal_rolling = yearly_monthly_means_normal.rolling(12, center=True, min_periods=10).mean()
yearly_monthly_means_normal_rolling.head(10)


# In[ ]:


annual_profile.plot(ylim=[0,10])


# In[ ]:


import plotly.graph_objs as go

data = [{'x':yearly_monthly_means_normal_rolling.index,'y':yearly_monthly_means_normal_rolling[ref], 'name':ref} for ref in yearly_monthly_means_normal_rolling.columns]
fig = go.FigureWidget(data=data)
fig


# In[ ]:


fig.layout.title = 'Normalized 12-month rolling average'


# In[ ]:


# remove references 9 and 10
ref_data.loc[:,'1':'8'].head(10)


# In[ ]:


# remove references 9 and 10
ref_data.drop(['9','10'], axis=1).head(10)


# <a id="shear"></a>
# # Shear analysis
# ---
# Applying the [power law](https://en.wikipedia.org/wiki/Wind_profile_power_law) to our measured met mast data.
# 
# ![power law](https://wikimedia.org/api/rest_v1/media/math/render/svg/585548e6acb797708bc9e78580bd9481809b6915)
# 
# We can do this by fitting a [simple least squares regression](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.linregress.html) to the natural log of the measurement heights and mean wind speeds using the scientific computational library [SciPy](https://docs.scipy.org/doc/scipy-0.14.0/reference/tutorial/general.html). To be more techincally accurate we would probably use [ODR](https://docs.scipy.org/doc/scipy-0.14.0/reference/odr.html) but for simplicity we don't here.

# In[ ]:


mast_data.head()


# In[ ]:


anemometers = ['SPD_59_COMB_AVG','SPD_47_COMB_AVG','SPD_32_COMB_AVG','SPD_15_COMB_AVG']
heights = [59,47,32,15]
ano_data = mast_data.loc[:,anemometers]
ano_data.head()


# In[ ]:


ano_data = ano_data.dropna()
ano_data.head()


# In[ ]:


ws = ano_data.mean().values
ano_data.mean()


# In[ ]:


from scipy import stats
alpha, intercept, r_value, p_value, std_err = stats.linregress(np.log(heights),np.log(ws))
alpha, intercept, r_value**2, p_value, std_err


# In[ ]:


print(f'Alpha: {alpha:.3f}; R2: {r_value**2:.4f}; Std error: {std_err*100.0:.2f}%')


# <a id="mcp"></a>
# # MCP - Long-term analysis
# ---
# Similarly, we can apply the typical industry measure-correlate-predict method between our reference and mast data using simple least squares. For demonstration purposes we'll apply a monthly correlation between our site and the reference stations.

# In[ ]:


# select the column you'd like to use from the mast data
# select the column you'd like to use from the reference data
site_corr_data = mast_data.SPD_59_COMB_AVG
ref_corr_data = ref_data.loc[:,'1':'8']


# In[ ]:


# resample to monthly averages
site_corr_data = site_corr_data.resample('MS').mean()
ref_corr_data = ref_corr_data.resample('MS').mean()


# In[ ]:


# concatenate into a single dataframe
corr_data = pd.concat([site_corr_data, ref_corr_data], axis=1)
corr_data = corr_data.dropna()
corr_data.head(10)


# In[ ]:


results = []
for ref in ref_corr_data.columns:
    temp_results = stats.linregress(corr_data[ref],corr_data.SPD_59_COMB_AVG)
    results.append(temp_results)
results


# In[ ]:


results = [stats.linregress(corr_data[ref],corr_data.SPD_59_COMB_AVG) for ref in ref_corr_data.columns]
results


# In[ ]:


results = pd.DataFrame.from_dict(results)
results.index = ref_corr_data.columns
results


# In[ ]:


# add a new column
results['r2'] = results.rvalue**2
results


# 
# <a id="freqdist"></a>
# # Frequency distribution
# ---
# Lastsly for analysis, we can derive the directional wind speed frequency distribution relying heavily on built-in methods. There is a more in-depth blog post about wind roses and frequency distributions using Pandas by [Rob Story](https://github.com/wrobstory) which you can find [here](http://wrobstory.github.io/2013/04/real-world-pandas-2.html). Clearly, Vestas was on the cutting edge back in 2013.
# 
# <a id="wsfreqdist"></a>
# ## Wind speed frequency distribution

# In[ ]:


mast_data.head()


# In[ ]:


ws = mast_data.SPD_59_COMB_AVG.dropna()
ws.head()


# In[ ]:


freq_dist = ws.groupby(ws.round()).count()
freq_dist/ws.size*100.0


# Round works great for simple cases but you can be more sophisticated with [cut](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.cut.html). 

# In[ ]:


bins = pd.cut(ws, bins=np.arange(0,26,0.5))
ws.groupby(bins).count()/ws.size*100.0


# <a id="windrose"></a>
# ## Windrose

# In[ ]:


vane = mast_data.DIR_80_AVG.dropna()
vane.head()


# In[ ]:


vane.describe()


# In[ ]:


vane = vane.replace(360.0,0.0)


# In[ ]:


sectors = 12
bin_width = 360/sectors
dir_bins = np.floor_divide(np.mod(vane + (bin_width/2.0),360.0),bin_width)
print(f'Number of direction sectors: {sectors}; Sector bin width: {bin_width}')
dir_bins.tail(15)
wind_rose = vane.groupby(dir_bins).count()
wind_rose


# In[ ]:


dir_edges = np.append(np.append([0],np.arange(bin_width/2, 360+bin_width/2, bin_width)),360)
dir_labels = np.arange(0,361,bin_width)
# dir_edges

dir_bins = pd.cut(vane, bins=dir_edges)
dir_bins.sort_values().unique()

# dir_bins = pd.cut(vane, bins=dir_edges, right=False, labels=dir_labels) #zero inclusive
# vane.groupby(dir_bins).count()


# <a id="visualization"></a>
# # Visualization - visulalizing data and results
# ---
# This is an entire tutorial unto itself. I will try to cover the basics and introduce you to some popular plotting libraries. If you'd like to know more you can watch Jake VanderPlas' [presentation from PyCon 2017](https://www.youtube.com/watch?v=FytuB8nFHPQ) or just Google 'python visualization'. 
# 
# <a id="matplotlib"></a>
# ## Matplotlib
# 
# While there is plenty of online criticism of this package it remains the most popular and standard plotting library for Python, despite being first released in 2003. If you'd like to know more about the history of this package you can go [here](https://matplotlib.org/users/history.html). The gist is that matplotlib was designed to produce publication-quality charts with a MATLAB-ish interface.  It is very powerful and I find myself using this library often to create effective, stative visuals. The fact that this package is already integrated with Pandas is also very convenient.
# 

# In[ ]:


ws.plot()


# In[ ]:


vane.plot(figsize=[20,5], style='.');


# In[ ]:


# Colors for plotting
EDFGreen = '#509E2F'
EDFOrange = '#FE5815'
EDFBlue = '#001A70'

ws.plot(figsize=[20,5], color=EDFBlue, title='Wind speed')


# In[ ]:


corr_data.head()


# In[ ]:


corr_data.plot(kind='scatter', x='1', y='SPD_59_COMB_AVG', xlim=[0,10], ylim=[0,10], color=EDFBlue, title='Monthly wind speed correlation')


# In[ ]:


freq_dist.plot(kind='bar');


# In[ ]:


fig = plt.figure(figsize=[12,8])
ax = fig.add_subplot(111)
freq_dist.plot(kind='bar', color=EDFBlue, ax=ax)
ax.set_ylabel('Bin count')
ax.set_xlabel('Wind speed bin [m/s]')
ax.set_title('Frequency distribution')
ax.set_xticks(np.arange(0,26,5))
ax.set_xticklabels(np.arange(0,26,5))
plt.show()


# In[ ]:


ref = '3'
fig = plt.figure(figsize=[8,6])
ax = fig.add_subplot(111)
corr_data.plot(kind='scatter', x=ref, y='SPD_59_COMB_AVG', color=EDFBlue, ax=ax)
ax.plot([0,10], np.array([0,10])*results.loc[ref,'slope']+results.loc[ref,'intercept'], color=EDFGreen)
ax.set_xlim([0,10])
ax.set_ylim([0,10])
ax.set_xlabel('Reference [m/s]')
ax.set_ylabel('Site [m/s]')
ax.set_title(f"Monthly wind speed correlation (R2: {results.loc[ref,'r2']:.3f})")
plt.show()


# In[ ]:


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='polar')
dir_bin_width_radians = np.radians(bin_width)
ax.set_theta_direction('clockwise')
ax.set_theta_zero_location('N')
ax.bar(np.radians(wind_rose.index.values*bin_width), wind_rose.values, width=dir_bin_width_radians, color=EDFGreen)
ax.set_title('Wind rose')
ax.set_yticklabels([])
ax.set_xticklabels(['N', '', 'E', '', 'S', '', 'W', ''])
plt.show()


# <a id="plotly"></a>
# ## Plotly
# 
# [Plolty](https://plot.ly/d3-js-for-python-and-pandas-charts/) combines the very powerful and interactive d3 JavaScript library with a Python API. This allows a wind analyst the ability to make interactive graphics with Python code. You can find a good introductory tutorial [here](https://plot.ly/python/ipython-notebook-tutorial/).  Plotly just recently released Version 3, which allows the user to access and update each attribute using Python syntax. You can read more about this in [their Medium post](https://medium.com/@plotlygraphs/introducing-plotly-py-3-0-0-7bb1333f69c6).

# In[ ]:


import plotly
print(plotly.__version__)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode(connected=True)


# In[ ]:


fig = go.FigureWidget(data=[{'x':freq_dist.index, 'y':freq_dist.values, 'type':'bar'}])
fig


# In[ ]:


fig.layout.title = 'Wind speed frequency distribution'


# In[ ]:


fig.layout.margin.t = 25
fig.layout.width = 900
fig.layout.height = 500
fig.layout.xaxis.title = 'Wind speed [m/s]'
fig.layout.yaxis.title = 'Frequency [count]'
fig.layout.yaxis.tickvals = []


# In[ ]:


fig = go.FigureWidget(
    data=[{'x':corr_data[ref], 
           'y':corr_data.SPD_59_COMB_AVG, 
           'type':'scatter', 
           'mode':'markers', 
           'name':'Wind speeds',
           'marker':{'color':EDFBlue,
                     'size':8},
           'text':corr_data.index.strftime('%Y-%m'),
           'hoverinfo':'text'},
          {'x':[0,10], 
           'y':np.array([0,10])*results.loc[ref,'slope']+results.loc[ref,'intercept'], 
           'type':'scatter', 
           'mode':'lines', 
           'name':'Fit',
           'line':{'color':EDFGreen,
                   'width':5}}
         ],
    layout={'title':'Monthly wind speed correlation',
            'width':600,
            'font':{'size':14},
            'margin':{'t':30,'b':35,'r':0,'l':35},
            'xaxis':{'rangemode':'tozero',
                     'title':'Reference [m/s]'},
            'yaxis':{'rangemode':'tozero',
                     'title':'Site [m/s]'}})
fig


# <a id="exporting"></a>
# # Exporting data
# ---
# Getting data and results out of your notebook is just as important as getting data into a notebook. [to_csv](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_csv.html) and [to_parquet](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_parquet.html) are my goto methods to do this.

# In[ ]:


mast_data.to_csv('mast_output.csv')
mast_data.to_parquet('mast_output.parquet')
print(os.listdir("../working"))


# In[ ]:


get_ipython().run_cell_magic('timeit', '', "pd.read_csv('mast_output.csv')")


# In[ ]:


get_ipython().run_cell_magic('timeit', '', "pd.read_parquet('mast_output.parquet')")


# <a id="anemoi"></a>
# # Anemoi
# ---
# EDF has been developing a wind-specific analysis library called [Anemoi](https://github.com/coryjog/anemoi). This package is based on the PyData ecosystem: NumPy, Pandas, SciPy, statsmodels, and Plotly. This is a freely available package, hosted on PiPy, that anyone or any organization can download and use. We would love to recruit contributors. 

# In[ ]:


import anemoi as an
an.__version__


# <a id="naming"></a>
# ## Sensor naming
# 
# If the column labels adhere to the [following sensor naming convention](https://coryjog.github.io/anemoi/docs_data_model.html#sensor-naming-convention) then Anemoi can parse the relevant information needed for a wind analysis. The convention consists of sensor type, height, orientation, and signal type, all delimited with an underscore. 
# 1. Sensor type - [SPD, DIR, T, P, etc.]
# 2. Height [m]
# 3. Orientation - [N, NE, E, SE, S, SW, W, NW]
# 4. Signal - [AVG, SD, MIN, MAX]
# 
# Sensor name examples:
# * SPD_58_N_AVG
# * SPD_58.2_N_AVG
# * DIR_48_N_AVG
# * DIR_48__AVG (**if orientation will not be used in the analysis then you can omit with a double-underscore**)
# * T_3__AVG
# *P_3__AVG
# 
# Custom orientations can be used to denote combined signals such as SEL for selectively averaged or COMB for combined. 
# 
# The example data adhere to this naming convention and the DataFrame can be turned into an Anemoi MetMast object:

# In[ ]:


mast_data.columns.name = 'sensor'
mast_data.head()


# In[ ]:


mast = an.MetMast(data=mast_data, name='Example mast', primary_ano='SPD_59_COMB_AVG', primary_vane='DIR_80_AVG')
mast.data.head()


# <a id="metmast"></a>
# ## MetMast
# The anemoi MetMast object is a pandas DataFrame with a specific format. Namely, with a DateTime Index and column labels following a standardized naming convention. Some other metadata such as latitude, longitude, mast name, elevation, and primary anemometer and wind vane names are also stored. With all this information, the package can infer the required columns needed to perform shear, long-term, and frequency distribution analysis.   

# In[ ]:


mast.metadata


# <a id="modules"></a>
# ## Analysis modules
# 
# While Anemoi is still in the very early stages of development it has a couple modules designed to help with wind analysis. You can explore the modeles in the [documentation](https://coryjog.github.io/anemoi/) or by using tab-complete with an.analysis.*tab-complete*.

# In[ ]:


# an.analysis.


# <a id="shear"></a>
# ### Shear

# In[ ]:


shear = an.analysis.shear.mast_annual(mast)
shear


# <a id="correlate"></a>
# ### Correlate

# In[ ]:


# ?an.analysis.correlate.ws_correlation_orthoginal_distance_model()


# In[ ]:


# ??an.analysis.correlate.ws_correlation_orthoginal_distance_model()


# In[ ]:


corr_data.head()


# In[ ]:


an.analysis.correlate.ws_correlation_orthoginal_distance_model(corr_data, ref_ws_col='1', site_ws_col='SPD_59_COMB_AVG', force_through_origin=False)


# <a id="ploting"></a>
# ## Plotting module

# In[ ]:


# an.plotting.


# In[ ]:


shear_fig = an.plotting.shear.annual_mast_results(shear)
offline.iplot(shear_fig)


# In[ ]:




