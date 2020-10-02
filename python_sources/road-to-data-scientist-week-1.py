#!/usr/bin/env python
# coding: utf-8

# # Road to data scientist - Week 1 - Wine dataset
# I practice and learn to become a data scientist.
# This week I learned basics of graphing and data visualization.

# ![](https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.virginexperiencedays.co.uk%2Fcontent%2Fimg%2Fproduct%2Flarge%2Ffinest-wine-tasting-with-18130153.jpg&f=1&nofb=1)

# # Importing libraries & chcecking data

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Data Processing
import pandas as pd
import numpy as np

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_palette('husl')

# Special data visualization
import missingno as msno # check missing value
from wordcloud import WordCloud # wordcloud
import plotly.graph_objects as go

# Geographic data visualization
import pycountry
import plotly.express as px

# Check file list
import os
print(os.listdir('../input/wine-reviews'))


# Reading the dataset with pandas

# In[ ]:


wines = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv')
wines.describe(include = 'all')


# In[ ]:


wines.head()


# # Finding and deleting missing data (missingo)

# Using missingo to visualize missing data. Color can be changed as well to rgb values.

# In[ ]:


msno.matrix(wines, color = (255/255, 38/255, 90/255))


# Some rows are missing country and price data. I will delete those rows. It turns out later on, that some rows also dont have variety, so I will delete those as well.

# In[ ]:


wines.dropna(subset = ['country', 'price', 'variety'], inplace = True)

# Visualizing missing values after deleting
msno.matrix(wines, color = (57/255, 194/255, 110/255))


# # Description and wine variety (word cloud)
# Creating 2 word clouds of the most used words in description and variety names. WordCloud makes it really easy.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'fig, ax = plt.subplots(1, 2, figsize=(16, 32))\nwordcloud_description = WordCloud(background_color=\'white\',width=800, height=800).generate(\' \'.join(wines[\'description\']))\nwordcloud_variety = WordCloud(background_color=\'white\',width=800, height=800).generate(\' \'.join(wines[\'variety\']))\nax[1].imshow(wordcloud_variety, interpolation=\'bilinear\')\nax[1].set_title("Wines variety")\nax[1].axis(\'off\')\nax[1].margins(x=0, y=0)\nax[0].imshow(wordcloud_description, interpolation=\'bilinear\')\nax[0].set_title("Wines description")\nax[0].axis(\'off\')\nax[0].margins(x=0, y=0)\nplt.show()')


# # Price and points distribution and relation(seaborn)
# With seaborn I can visualize the wine prices and points distribution. As we can see most of the prices are in the  < 200 \$ range. There are afew wines as expansive as 300$. The spikes in the points distribution diagram are because all points have a 1 point accurace meaning there are no e.g. 88.2 point wines, only whole numbers.

# In[ ]:


fig, axs = plt.subplots(ncols = 2, figsize = (25, 10))
sns.set_style("whitegrid")
sns.kdeplot(wines['price'], shade = True, legend = False, ax = axs[0],  color = 'orange')
axs[0].set_title("Wine prices distribution")
sns.kdeplot(wines['points'], shade = True, legend = False, ax = axs[1], color = 'green')
axs[1].set_title("Wine points distribution")
plt.plot()


# I can also use seaborn to visualize prices distribution in terms of price. With the line we can see, that better wines are on average more expansive. The difference is slight though.

# In[ ]:


fig = plt.figure(figsize = (20, 10), dpi = 72)
sns.regplot(x = wines['points'], y = wines['price'], color = 'red')


# # Choropleth graph by country and by state (express and go)
# Express and go make it really easy to visualize geographic data

# Here I Create a dataframe with points ant total wines for each country. I also convert the countries names to iso_alpha_3 ( using pycountry dictionary ), to make it possible to graph them with express.

# In[ ]:


wines_countries_counts = wines['country'].value_counts()
wines_countries_points = pd.Series(index = wines_countries_counts.index)
for country in wines['country'].unique():
    country_name = wines[wines['country'] == country]
    wines_countries_points.loc[country] = country_name['points'].mean()
    
wines_countries = pd.concat({'total': wines_countries_counts,'points': wines_countries_points}, axis = 1)

# Converting country names to iso_alpha_3
countries = {'US': 'USA', 'England': 'GBR', 'Moldova': 'MDA', 'Macedonia': 'MKD', 'Czech Republic': 'CZE'}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3
    
for country, row in wines_countries.iterrows():
    wines_countries.loc[country, 'iso_alpha'] = countries[country]
wines_countries.head()


# I graph them and we can see that most of the wines are from US, France or Italy

# In[ ]:


fig = px.choropleth(wines_countries, locations="iso_alpha",
                    color="total",
                    hover_name="iso_alpha", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.update_layout(title='Amount of wines by country')
fig.show()


# Here I graph average points. We can see that on average the best wines are in Great Britain.

# In[ ]:


fig = px.choropleth(wines_countries, locations="iso_alpha",
                    color="points",
                    hover_name="iso_alpha", # column to add to hover information
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.update_layout(title='Average wine points by country')
fig.show()


# Here is a dictionary to convert states names to states codes, to be able to graph them.

# In[ ]:


us_iso = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Palau': 'PW',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}


# Creating a dataframe with points and total wines for each states and converting state codes.

# In[ ]:


us_wines_counts = wines.loc[wines['country'] == 'US', 'province'].value_counts()
us_wines_points = pd.Series(index = us_wines_counts.index)
states = wines.loc[wines['country'] == 'US', 'province'].unique()
for state in states:
    state_name = wines[wines['province'] == state]
    us_wines_points.loc[state] = state_name['points'].mean()

us = pd.concat({'total': us_wines_counts,'points': us_wines_points}, axis = 1)
us.drop(['America', 'Washington-Oregon'], inplace = True)

# Converting state codes
states = {}
for state in us_iso:
    states[state] = us_iso[state]
    
for state, row in us.iterrows():
    us.loc[state, 'iso_alpha'] = states[state]
    
us.head()


# I graph amount of wines by state with go

# In[ ]:


fig = go.Figure(data=go.Choropleth(
    locations=us['iso_alpha'], # Spatial coordinates
    z = us['total'], # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Blues',
    colorbar_title = "Total",
))

fig.update_layout(
    title_text = 'US total wines by state',
    geo_scope='usa', # limite map scope to USA
)

fig.show()


# I graph average amount of points by state with go. Best wines are on average in Rhode Island.

# In[ ]:


fig = go.Figure(data=go.Choropleth(
    locations=us['iso_alpha'], # Spatial coordinates
    z = us['points'], # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Points",
))

fig.update_layout(
    title_text = 'US Average wine points by State',
    geo_scope='usa', # limite map scope to USA
)

fig.show()


# # Pie charts (go)
# We can use go to also create pie charts

# In[ ]:


labels = wines_countries.index
values = wines_countries['total']

fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.show()


# We can make the hole in a chart bigger. We can see that the most wines weret tested by Roger Voss.

# In[ ]:


tasters = wines['taster_name'].value_counts()
fig = go.Figure(data=[go.Pie(labels=tasters.index, values=tasters, hole=.5)])
fig.show()


# # Coclusion
# **What I learned:**
# - Visualizing missing data with missingo
# - Creating a word clous with WordCloud
# - Showing basic plots with seaborn
# - Visualizing geographical data with express and go
# - Creating pie charts with go
# 
# I learned a lot this week. Those are only the basics. Next week I will try to improve on these skills and practice customizing graphs.
