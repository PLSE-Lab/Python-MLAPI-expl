#!/usr/bin/env python
# coding: utf-8

# # Brief EDA: Happiness Report 2019
# Before I begin, I have to say that this dataset is not the best to work with. This dataset consists of rankings and not actual values. For example, instead of a value for GDP per capita, this dataset will provide a rank to a country (1 being the best). Still though, the dataset is fun to visualize and play around with, just not ideal. 
# 
# To begin, we will write all of our necessary imports.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import os
print(os.listdir("../input"))


# We will now load in our data and print the shape.

# In[ ]:


data = pd.read_csv('../input/world-happiness-report-2019.csv')
print('Rows in Data: ', data.shape[0])
print('Columns in Data: ', data.shape[1])


# We can view the top five ranking countries and the bottom five ranking countries.

# In[ ]:


data.head(5)


# In[ ]:


data.tail(5)


# I wanted to rename the column name "Country (region)". I thought it just looked ugly. I did not bother with renaming some other columns just yet.

# In[ ]:


data = data.rename({'Country (region)':'Country'}, axis=1)
data.dtypes


# We can make a heatmap of the data.

# There is something to be learned from this heatmap. We can see that social support, GDP per capita, and health are all positively correlated with the countries ranking. This means that countries who ranked highest in the happiness category typically ranked higher in the categories I just mentioned.

# In[ ]:


plt.figure(figsize = (16,5))
sns.heatmap(data.corr(), annot=True, linewidths=.2)


# We can now make a map of the world and to show where the happiest and unhappiest countries are. For this visualization, the darker red a country is, the less happy or satisfied they are. The lighter colored countries are the happier ones.

# In[ ]:


map_data = [go.Choropleth( 
           locations = data['Country'],
           locationmode = 'country names',
           z = data["Ladder"], 
           text = data['Country'],
           colorbar = {'title':'Ladder Rank'})]

layout = dict(title = 'Least Satisfied Countries', 
             geo = dict(showframe = False, 
                       projection = dict(type = 'equirectangular')))

world_map = go.Figure(data=map_data, layout=layout)
iplot(world_map)


# To quickly show the top countries with the highest GDP per capita rank, we can sort the data by the GDP column, them print out the top five.

# In[ ]:


GDP = data.sort_values(by='Log of GDP\nper capita')
GDP.head(5)


# Again, we can make a map of the world and this time show the "Freedom" feature. Remember, the darker the red, the lower the ranking. This means that red countries have less freedom.

# In[ ]:


map_data = [go.Choropleth( 
           locations = data['Country'],
           locationmode = 'country names',
           z = data["Freedom"], 
           text = data['Country'],
           colorbar = {'title':'Ladder Rank'})]

layout = dict(title = 'Countries With Least Freedom', 
             geo = dict(showframe = False, 
                       projection = dict(type = 'equirectangular')))

world_map = go.Figure(data=map_data, layout=layout)
iplot(world_map)

