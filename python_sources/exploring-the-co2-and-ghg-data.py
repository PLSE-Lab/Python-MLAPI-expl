#!/usr/bin/env python
# coding: utf-8

# ## SUMMARY
# In this notebook we'll explore the data to get an idea what you're working with. I'll be using plotly to visualize the data and get a feel for it. Feel free to copy this notebook and continue the work at any time.

# ## IMPORTS
# I'll be using plotly for interactive plots, so you can click around if needed.

# In[ ]:


# native libs
import os

# Data handling
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import chart_studio.plotly as py

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# Only one file to work with, great.

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## FIRST STEPS

# Interesting data set. It seems like each columns represents a year and each row represents a country. Looking at the magnitude of the data the unit to the values in the columns is probably tonnes of CO2 per year (1000 kilograms or ~2200 pounds of CO2 or ). Usually world emissions are expressed in megatonnes or billions of tonnes, which is somewhat of a roundabout way of saying **a whol' bunch of CO2 y'all**. Needless to say we will be encountering some exponential growth from here on out.

# In[ ]:


df = pd.read_csv('/kaggle/input/emission data.csv', delimiter=',')
df.head()


# 1751 seems pretty far back for CO2 measurements, lets see which countries have data that far back.

# In[ ]:


df[df['1751'] != 0]


# Interestingly, it seems the UK is responsible far all emissions at this point `/s`. At these first data points it seems that only the UK makes up all emissions for EU-28 and World. This indicates that there are at least three different classifications in the country column. Actual countries, the entire world, and... uh `googles "EU-28"` regions such as the European union. We should check when the other countries catch up to the UK and actually have some estimates on CO2.

# In[ ]:


# only select the UK and the world
only_uk_world = df[(df["Country"] == "United Kingdom") | (df["Country"] == "World")]
# get the difference, drop the Country column since it contains strings
diff_uk_world = only_uk_world.drop(columns ="Country").diff(axis=0)


# Now that we have differences, lets make a plot to find out when the other countries start upping their game.

# In[ ]:


import plotly.express as px

fig = go.Figure(data=go.Scatter(x=df.columns, y=df.iloc[1]))
fig.show()


# Seems that other countries don't even have a single record until 1883. I am also a little worried about a lack of uncertainty, since I am assuming that measurements become more accurate over time.

# TO BE CONTINUED...

# ## WORLD PLOT
# <br>
# Plotly makes it fairly easy to make a world color map, so let's give that a try. This type of data seems pretty much ideal for such a plot. One problem is that we have country names, and we will need ISO codes, e.g. FRA for France. `googles "python country name to code"` alright, seems like the Python community has our backs. Lets first see what kind of data it returns.

# In[ ]:


import pycountry
pycountry.countries.search_fuzzy(df.Country.iloc[0])[0].alpha_3


# Pretty good! Thanks Christian Theune for the wonderful library. Time to apply it to the data. I'll make a function that we can apply to our dataframe (not the fastest way, but we have so few rows that is doesn't matter so much).

# In[ ]:


some_countries = ["England", "HerpaDerpaland", "Engla"]

def do_fuzzy_search(country):
    try:
        result = pycountry.countries.search_fuzzy(country)
        return result[0].alpha_3
    except:
        return np.nan

for country in some_countries:
    print(do_fuzzy_search(country))


# Seems like the converter is doing its job. Time to apply the function to the dataframe.

# In[ ]:


df['country_code'] = df["Country"].apply(lambda country: do_fuzzy_search(country))
df.head()


# We can now pass our dataframe to the convenient `px.choropleth` function.

# In[ ]:


plot_df = df.dropna()
fig = px.choropleth(plot_df, locations="country_code",
                    color="2017",
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()


# Two things to notice here. First off, WHOA America, that's a lot of CO2. We can expect a very skewed distribution of CO2 over the world. Secondly, it seems that we are missing some countries such as (the democratic repulic of) Congo.

# In[ ]:


missing_countries = ["Congo", "Democratic Republic of Congo", "Niger", "South Korea"]
correct_codes = {"Congo": "COD", "Democratic Republic of Congo": "COG", "Niger": "NER", "South Korea": "KOR"}
df[df["Country"].isin(missing_countries)]

Turns out, that for these kind of Choropleth maps, we didn't have the right country code, so let's fix that...
# In[ ]:


def update_wrong_country_codes(row):
    if row['Country'] in correct_codes.keys():
        row['country_code'] = correct_codes[row['Country']]
    return row

df = df.apply(lambda x: update_wrong_country_codes(x), axis=1)


# In[ ]:


plot_df = df.dropna()
fig = px.choropleth(plot_df, locations="country_code",
                    color="2017",
                    hover_name="Country",
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()


# One cool thing that plotly allows us to do is making an interactive plot. We will add a slider bar that will allow us to choose a year to analyse.
# 
# To make this happen, first make a data object that plotly can understand. We'll need a list of data objects where each element represents one 'year' column in the data set.

# In[ ]:


import plotly

# constants
first_year = 1900
last_year = 2017
number_of_steps = int(2017 - 1900)

# data is a list that will have one element for now, the first element is the value from the first year column we are interested in.
data = [dict(type='choropleth',
             locations = plot_df['country_code'].astype(str),
             z=plot_df[str(first_year)].astype(float))]

# next, we copy the data from the first cell, append it to the data list and set the data to the value contained in the year column of the dataframe.
for i in range(number_of_steps):
    data.append(data[0].copy())
    index = str(first_year + i + 1)
    data[-1]['z'] = plot_df[index]


# Last step is to define a slider bar.

# In[ ]:


# for each entry in data, we add one step to the slider bar and define a label e.g. Year 1900
steps = []
for i in range(len(data)):
    step = dict(method='restyle',
                args=['visible', [False] * len(data)],
                label='Year {}'.format(i + first_year))
    step['args'][1][i] = True
    steps.append(step)

sliders = [dict(active=number_of_steps,
                pad={"t": 1},
                steps=steps)]    
layout = dict(sliders=sliders)


# We pass the layout and the data info to the plotly map, et voila. An interactive map.
# 
# Note that the scale changes for each year, that is something we could fix, but consider that we are going back to the year 1900. This means that if we scale the graph to the emissions in 2017 that the figure will be 100% blue for pretty much 90% of the time.

# In[ ]:


fig = dict(data=data, 
           layout=layout)
plotly.offline.iplot(fig)


# Work in PROGRESS
