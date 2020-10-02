#!/usr/bin/env python
# coding: utf-8

# # Formula 1 Grand Prix Analysis
# Data Analysis over Formula 1 dataset (1950-2017)

# Formula 1 is one of most popular and highly enjoyed sport across the globe. Extreme precision, high accuracy and excellent team are the key constituents for a team to participate and win the grand prix. The moment anyone decides to act on their own, the team can either lose position or straight away get out of the competition. All the major teams like Scuderia Ferrari, McLaren, Redbull, Renault and numerous others compete for the world championship. As data science enthusiasts and huge fans of Formula1, we thought of analyzing the Formula1 dataset that we took from Kaggle which provided us with the open dataset that in turn provided us with numerous data attributes like Formula1 drivers, races, lap timings, seasons data, pitstop status and other related attributes.

# TL;DR Here's the EDA analysis over Formula 1 Grand Prix data using Plotly and Tableau

# In[ ]:


from IPython.display import Image

Image("../input/formula-1-grand-prix/f1.png")


# In[ ]:


get_ipython().system('pip install chart-studio')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualization libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import chart_studio.plotly as py
from plotly.graph_objs import *
from IPython.display import Image
pd.set_option('display.max_rows', None)

import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ### Data Preprocessing: 
# Read the contents of the dataframe to understand the data composition

# In[ ]:


seasons_df = pd.read_csv("../input/formula-1-race-data-19502017/seasons.csv", encoding='latin')
circuits_df = pd.read_csv("../input/formula-1-race-data-19502017/circuits.csv", encoding='latin')
drivers_df = pd.read_csv("../input/formula-1-race-data-19502017/drivers.csv", encoding='latin')
races_df = pd.read_csv("../input/formula-1-race-data-19502017/races.csv", encoding='latin')
results_df = pd.read_csv("../input/formula-1-race-data-19502017/results.csv", encoding='latin')
status_df = pd.read_csv("../input/formula-1-race-data-19502017/status.csv", encoding='latin')
constructors_df = pd.read_csv("../input/formula-1-race-data-19502017/seasons.csv", encoding='latin')


# In[ ]:


seasons_df.head()


# In[ ]:


circuits_df.head()


# In[ ]:


drivers_df.head()


# In[ ]:


races_df.head()


# In[ ]:


results_df.head()


# In[ ]:


status_df.head()


# In[ ]:


constructors_df.head()


# ### Visualize the Grand Prix Timeline

# In[ ]:


fig = go.Figure(go.Bar(
    x = races_df['year'],
    y = races_df['name'],
    text=['Bar Chart'],
    name='Grand Prix',
    marker_color=races_df['circuitId']
))

fig.update_layout(
    height=800,
    title_text='Grand Prix Timeline',
    showlegend=True
)

fig.show()


# ### Visualize drivers from different nationalities

# ### Group data

# In[ ]:


# def group_f1(data,country):
#     cases = data.groupby(country).size()
#     cases = np.log(cases)
#     cases = cases.sort_values()
    
#     # Visualize the results
#     fig=plt.figure(figsize=(35,7))
#     plt.yticks(fontsize=8)
#     cases.plot(kind='bar',fontsize=12,color='orange')
#     plt.xlabel('')
#     plt.ylabel('Number of cases',fontsize=10)

# group_f1(covid_19_data,'Country/Region')


# ### Visualize dataset on Map

# In[ ]:


import geopandas as gpd
from shapely.geometry import Point, Polygon

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


geometry = [Point(xy) for xy in zip(circuits_df.lat, circuits_df.lng)]


# In[ ]:


circuits_df['geometry'] = geometry
circuits_df.head()


# # Tableau Results

# ### Gantt chart depicting history of Grand Prix in the period (1950 - 2017):

# In[ ]:


Image("../input/formula-1-grand-prix/tableau-results/sheet-1.png")


# ### Line chart depicting Grand prix distribution across the year from January to December

# In[ ]:


Image("../input/formula-1-grand-prix/tableau-results/sheet-2.png")


# ### Symbol map depicting the distribution of race tracks across the world

# In[ ]:


Image("../input/formula-1-grand-prix/tableau-results/sheet-3.png")


# ### Circle chart describing the nationalities of F1 drivers with various colors

# In[ ]:


Image("../input/formula-1-grand-prix/tableau-results/sheet-4.png")


# ### Bubble chart depicting the number of rounds across the participating drivers in F1

# In[ ]:


Image("../input/formula-1-grand-prix/tableau-results/sheet-5.png")


# ### Stacked Bar chart describing the race status for grand prix for the participating drivers in F1

# In[ ]:


Image("../input/formula-1-grand-prix/tableau-results/sheet-6.png")

