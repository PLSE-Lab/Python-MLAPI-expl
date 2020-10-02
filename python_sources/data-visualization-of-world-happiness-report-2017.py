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


# Import WHR 2017 Data

# In[ ]:


df = pd.read_csv('/kaggle/input/world-happiness/2017.csv')


# In[ ]:


df.head()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Happiness Distributions

# In[ ]:


sns.distplot(df['Happiness.Score'])


# In[ ]:


sns.pairplot(data=df)


# In[ ]:


sns.heatmap(df.corr(),cmap='coolwarm')


# In[ ]:


import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot,plot
init_notebook_mode(connected=True) 


# In[ ]:


data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
        reversescale = True,
        locations = df['Country'],
        locationmode = "country names",
        z = df['Happiness.Score'],
        text = df['Country'],
        colorbar = {'title' : 'World Happiness Scores'},
      ) 

layout = dict(title = 'World Happiness Scores',
                geo = dict(showframe = False,projection = {'type':'mercator'})
             )


# In[ ]:


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)


# In[ ]:




