#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bokeh
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, Legend, HoverTool
from bokeh.palettes import Spectral10


# In[ ]:


sns.set_style("whitegrid")
bokeh.plotting.output_notebook()


# In[ ]:


data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")


# ### The Progression - Look at South Korea

# In[ ]:


time_country_data = data[(data.ConfirmedCases>100) &                         (data["Country/Region"]!= "China")]                             .groupby(["Country/Region", "Date"])                                  [["ConfirmedCases", "Fatalities"]].sum().reset_index()


# In[ ]:


time_country_data["Date"] = pd.to_datetime(time_country_data.Date, format="%Y/%m/%d")
hover = HoverTool(tooltips = [
    ("Date", "@Date{%F}"),
    ("Country", "@{Country/Region}"),
    ("Fatalities", "@Fatalities"),
    ("ConfirmedCases", "@ConfirmedCases")],
                  formatters={'Date': 'datetime'})


# In[ ]:


p = figure(title = "The Progression",
          x_axis_label="Date",
          y_axis_label="ConfirmedCases", x_axis_type="datetime", tools='pan,box_zoom')
p.add_tools(hover)


legend_lt = []

for i, country in enumerate(time_country_data["Country/Region"].unique().tolist()):
    cds_time_data = ColumnDataSource(time_country_data[time_country_data["Country/Region"] == country])
    c = p.line("Date", "ConfirmedCases",source=cds_time_data,color=Spectral10[i%10])
    legend_lt.append((country, [c]))

legend = Legend(items=legend_lt, location=(0, 0))
p.add_layout(legend, 'right')
show(p)


# ### Comparing Countries from Day 0 - United States seems to be the worse.

# In[ ]:


q = figure(title = "Compare Countries from beginning",
          x_axis_label="Days",
          y_axis_label="ConfirmedCases", tools='pan,box_zoom', tooltips=[('country', '@country')])

legend_lt = []

for i, country in enumerate(time_country_data["Country/Region"].unique().tolist()):
    rolling = time_country_data[time_country_data["Country/Region"] == country].rolling(7, min_periods=1)        ["ConfirmedCases"].mean().tolist()
    
    rolling_cdf = ColumnDataSource({"index": range(len(rolling)), "rolling": rolling, 
                                    "country": [country for c in range(len(rolling))]})

    c = q.line("index", "rolling", source=rolling_cdf, color=Spectral10[i%10])
    legend_lt.append((country, [c]))

legend = Legend(items=legend_lt, location=(0, 0))
q.add_layout(legend, 'right')
show(q)


# In[ ]:




