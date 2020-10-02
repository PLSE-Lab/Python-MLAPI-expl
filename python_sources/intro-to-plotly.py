#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
reviews.head()


# `plotly` has off and online modes. We need to use the offline modes becase on Kaggle the online mode disables network access in Python.

# In[ ]:


from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)


# A basic scatter plot

# In[ ]:


import plotly.graph_objs as go

iplot([go.Scatter(x=reviews.head(1000)['points'], y=reviews.head(1000)['price'], mode='markers')])


# Whats niffty about this is that we can hover over the points and see how many samples are in that given point. For example there are 50 sample points that have 90 points and are priced between $100 - $200!
# 
# However, the bad part is that we had to limit the amount pf samples to just 1000. 

# A KDE plot and Scatter plot.

# In[ ]:


iplot([go.Histogram2dContour(x=reviews.head(500)['points'],
                            y=reviews.head(500)['price'],
                            contours=go.Contours(coloring='heatmap')),
      go.Scatter(x=reviews.head(1000)['points'], y=reviews.head(1000)['price'], mode='markers'
)])


# 

# 

# Here is a surface plot

# In[ ]:


df = reviews.assign(n=0).groupby(['points','price'])['n'].count().reset_index()
df = df[df["price"] < 100]
v = df.pivot(index='price', columns='points', values='n').fillna(0).values.tolist()


# In[ ]:


iplot([go.Surface(z=v)])


# 

# We will plot a choroplepths. It is a map whereby it's entities are coloured by some variable in the dataset. 

# In[ ]:


df = reviews['country'].replace("US", "United States").value_counts()

iplot([go.Choropleth(
   locationmode='country names',
   locations=df.index.values,
    text=df.index,
    z=df.values
)])


# 

# # Exercise Time!

# In[ ]:


pokemon = pd.read_csv("../input/pokemon/Pokemon.csv")
pokemon.head(3)


# In[ ]:


import plotly.graph_objs as go

iplot([go.Scatter(x=pokemon.head(1000)['Attack'], y= pokemon.head(1000)['Defense'], mode='markers')])


# 

# 

# 

# 
