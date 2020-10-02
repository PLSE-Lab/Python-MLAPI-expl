#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing libraries
import numpy as np
import pandas as pd

# plotly packages
import plotly
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
cf.go_offline()


# In[ ]:


# Let's create a random data with numpy
df = pd.DataFrame(np.random.randn(200, 4), columns='A B C D'.split())

# check the data once
df.head()


# In[ ]:


# Let's create another data frame
df2 = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Values': [32, 50, 42]})


# In[ ]:


# Plotting randomly with matplotlib
df.plot()


# See this matplotlib gives graphical plot, but we can make it this plot as interactive plot....Let's how!

# In[ ]:


df.iplot()


# Wooooooh....plotly plots an interactive graph...we check any particular point with this plot.

# In[ ]:


# Scatter plot
df.iplot(kind='scatter', x='A', y='B')


# yeah see this is an irregular plot...let's go modify littlebit in code

# In[ ]:


df.iplot(kind = 'scatter', x='A', y='B', mode='markers')


# This looks good

# In[ ]:


# Bar plots
# for a bar plots the data should be bivariant(categorical, numeerical)
df2.iplot(kind='bar', x='Category', y='Values')


# In[ ]:


# we can also do some interesting things with plotlt
# let's see
print(df.iplot(kind='bar'))


# In[ ]:


# count each column in df dataframe and plot bar graphs
print(df.count().iplot(kind='bar'))


# In[ ]:


# Now take sum of each column in df and plot bar graph
print(df.sum().iplot(kind='bar'))


# In[ ]:


# Box plots
df.iplot(kind='box')


# In[ ]:


# 3D surface plot
# Let's create a another random datframe
df3 = pd.DataFrame({'x':[1,2,3,4,5], 'y':[10,20,30,20,10], 'z':[500,400,300,200,100]})
df4 = pd.DataFrame({'x':[1,2,3,4,5], 'y':[10,20,30,20,10], 'z':[1,2,3,4,5]})


# In[ ]:


# This is for 3 Dimensional values
df3


# In[ ]:


df3.iplot(kind='surface')


# In[ ]:


df4.iplot(kind='surface', colorscale='rdylbu') #rd- red, yl-yellow, bu-blue


# In[ ]:


# Histogram plot
df['A'].iplot(kind='hist', bins=25)


# In[ ]:


# We can plot all the columns in df in hist
# On the right side labels, we can on and off any partiucal column
df.iplot(kind='hist')


# In[ ]:


# Spread type visualization
# stocks type
df[['A', 'B']].iplot(kind='spread')


# In[ ]:


# Bubble plots
# which is similar to scatter plot
df.iplot(kind='bubble', x='A', y='B', size='C')


# In[ ]:


# Scatter matrix plot
# Which is similar to pairplots
df.scatter_matrix()


# You can also zoom into the point and look at it.

# In[ ]:




