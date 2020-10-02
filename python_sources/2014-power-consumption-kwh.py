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


# In[ ]:


import plotly as py
import plotly.graph_objs as go
import pandas as pd
from plotly.offline import download_plotlyjs, init_notebook_mode,plot,iplot


# In[ ]:


init_notebook_mode(connected=True) # here we initialize notebook mode


# In[ ]:


df=pd.read_csv("../input/World_Power_Consumption.txt")
df.head()


# We need to create a data object and layout object before geographical plotting with plotly

# In[ ]:


init_notebook_mode(connected=True)
data = dict(type = 'choropleth',
            locations = df["Country"],
            locationmode="country names",
            colorscale= "Viridis",
            reversescale=True,
            text= df["Text"],
            z=df["Power Consumption KWH"],
            colorbar = {'title':'KWH'})
layout=dict(title = 'Power Consumption in KWH by State',
              geo = dict(
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )


# In[ ]:


choromap=go.Figure(data,layout)
iplot(choromap)


# choromap=go.Figure(data,layout)
# iplot(choromap)

# we can create different map styles by changing projection style which can be found in plotly documentation

# In[ ]:


init_notebook_mode(connected=True)
data = dict(type = 'choropleth',
            locations = df["Country"],
            locationmode="country names",
            colorscale= "Viridis",
            reversescale=True,
            text= df["Text"],
            z=df["Power Consumption KWH"],
            colorbar = {'title':'KWH'})

layout = dict(title = '2014 Power Consumption KWH',
                geo = dict(showframe = False,projection = {'type':'stereographic'})
             )


# In[ ]:


choromap=go.Figure(data,layout)
iplot(choromap)

