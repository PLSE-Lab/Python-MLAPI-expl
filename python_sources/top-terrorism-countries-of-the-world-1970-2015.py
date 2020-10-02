#!/usr/bin/env python
# coding: utf-8

# 

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:





# In[3]:


# Plot a thematic - Top Terrorism Countries of the World (1970-2015).
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import iplot, init_notebook_mode
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd

init_notebook_mode(connected=True)

df = pd.read_csv("../input/top_terror_countries.csv")
locations = df["country_name_iso2"]

data = [ dict(
        type = 'choropleth',
        locationmode ="country names", 
        locations = df["country_name_iso2"],
        z = df['fatalities'],
        text = df["country_name_iso2"],
        colorscale = 'Electric'  ,         
        autocolorscale = False,
        reversescale = True,
        marker = dict(
            line = dict (
                color = 'rgb(180,180,180)',
                width = 0.5
            ) ),
        colorbar = dict(
            autotick = False,
            #tickprefix = '$',
            title = 'fatalities<br>per 1.000'),
      ) ]

layout = dict(
    title = '1970-2015 Terrorism Attack<br>Source:\
            <a href="http://start.umd.edu/gtd/">\
            GTD Global Terrorism Database</a>',
    geo = dict(
        showframe = False,
        showcoastlines = True,
        showcountries = True,
        projection = dict(
            type = 'Mercator'
    
        )
    )
)

fig = dict( data=data, layout=layout )
iplot( fig, validate=False, filename='d3-world-map' )


# In[4]:


df.head()


# In[ ]:


len(df)

