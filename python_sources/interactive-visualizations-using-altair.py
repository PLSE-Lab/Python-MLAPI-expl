#!/usr/bin/env python
# coding: utf-8

# **Altair**
# 
# This kernel uses a Python visualization library called Altair. Altair is quite a new library and I suspect it will become quite popular in upcoming years. It was developed by Jake Vanderplas (the author of Python for Data Science book) and Brian Granger (contributor to IPython Notebook and the leader of Project Jupyter Notebook). It is quite beautiful and very easy and intuitive to code. I highly recommend it!

# 

# In[ ]:


# Importing required libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

import altair as alt
alt.data_transformers.enable('default', max_rows=None)

pd.set_option('display.max_columns', 30)
# pd.options.display.max_rows = 1050


# In[ ]:


import json  # need it for json.dumps
from IPython.display import HTML

# Create the correct URLs for require.js to find the Javascript libraries
vega_url = 'https://cdn.jsdelivr.net/npm/vega@' + alt.SCHEMA_VERSION
vega_lib_url = 'https://cdn.jsdelivr.net/npm/vega-lib'
vega_lite_url = 'https://cdn.jsdelivr.net/npm/vega-lite@' + alt.SCHEMA_VERSION
vega_embed_url = 'https://cdn.jsdelivr.net/npm/vega-embed@3'
noext = "?noext"

altair_paths = {
    'vega': vega_url + noext,
    'vega-lib': vega_lib_url + noext,
    'vega-lite': vega_lite_url + noext,
    'vega-embed': vega_embed_url + noext
}

workaround = """
requirejs.config({{
    baseUrl: 'https://cdn.jsdelivr.net/npm/',
    paths: {paths}
}});
"""

# Define the function for rendering
def add_autoincrement(render_func):
    # Keep track of unique <div/> IDs
    cache = {}
    def wrapped(chart, id="vega-chart", autoincrement=True):
        """Render an altair chart directly via javascript.
        
        This is a workaround for functioning export to HTML.
        (It probably messes up other ways to export.) It will
        cache and autoincrement the ID suffixed with a
        number (e.g. vega-chart-1) so you don't have to deal
        with that.
        """
        if autoincrement:
            if id in cache:
                counter = 1 + cache[id]
                cache[id] = counter
            else:
                cache[id] = 0
            actual_id = id if cache[id] == 0 else id + '-' + str(cache[id])
        else:
            if id not in cache:
                cache[id] = 0
            actual_id = id
        return render_func(chart, id=actual_id)
    # Cache will stay defined and keep track of the unique div Ids
    return wrapped


@add_autoincrement
def render_alt(chart, id="vega-chart"):
    # This below is the javascript to make the chart directly using vegaEmbed
    chart_str = """
    <div id="{id}"></div><script>
    require(["vega-embed"], function(vegaEmbed) {{
        const spec = {chart};     
        vegaEmbed("#{id}", spec, {{defaultStyle: true}}).catch(console.warn);
    }});
    </script>
    """
    return HTML(
        chart_str.format(
            id=id,
            chart=json.dumps(chart) if isinstance(chart, dict) else chart.to_json(indent=None)
        )
    )

HTML("".join((
    "<script>",
    workaround.format(paths=json.dumps(altair_paths)),
    "</script>"
)))


# Load Data

# In[ ]:


df1 = pd.read_csv("../input/Cars93.csv")
df1 = df1.drop(['Unnamed: 0'], axis = 1)


# 

# In[ ]:


df1.rename(columns={'MPG.city':'MPGC'}, inplace=True)
df1.rename(columns={'MPG.highway':'MPGH'}, inplace=True)
df1.rename(columns={'Luggage.room':'Luggroom'}, inplace=True)
df1.rename(columns={'Fuel.tank.capacity':'Fuelcapacity'}, inplace=True)
df1.rename(columns={'Man.trans.avail':'Gear'}, inplace=True)
df1.rename(columns={'Rev.per.mile':'RevMile'}, inplace=True)
df1.rename(columns={'Turn.circle':'Turn.circle'}, inplace=True)
df1.rename(columns={'Rear.seat.room':'RearseatRoom'}, inplace=True)
df1.head()


# In this interactive visualization we are comparing horsepower with the engine size of different type of vehicles so that the potential buyer can select a usa or a non-usa origin vehicle inrespect with higher or lower horsepower/engine size and get to see the number of Cars available in different categories. 

# In[ ]:


interval = alt.selection_interval()

points = alt.Chart(df1).mark_point().encode(
  x='Horsepower',
  y='EngineSize',
  color=alt.condition(interval, 'Origin', alt.value('lightgray'))
).properties(
  selection=interval
)

histogram = alt.Chart(df1).mark_bar().encode(
  x='count()',
  y='Type',
  color='Type'
).transform_filter(interval)

render_alt(points & histogram)


# In this interactive graph the customer can select the desired engine size and weight of cars
# which will in turn shows the miles per gallon on the second graph and vice-versa, as the
# customer select the engine size and mileage it shows the weight of that particular car. This is
# really helpful for the customer who wants to prioritise their car features according to their
# own wish.
# 

# In[ ]:


interval = alt.selection_interval()

base = alt.Chart(df1).mark_point().encode(
  y='EngineSize',
  color=alt.condition(interval, 'Origin', alt.value('lightgray'))
).properties(
  selection=interval
)

render_alt(base.encode(x='Weight') | base.encode(x='MPGC'))


# This graph shows the different prices of distinguished car types with respect to different
# manufactures and the mean price is also given so that customers will get the idea of what all
# cars are available below the mean price value and vice-versa. It allows the customer to easily
# segregate the car types and manufacturers according to different variety of prices.
#  

# In[ ]:


points = alt.Chart(df1).mark_point().encode(
  x='Manufacturer',
  y='Price',
  color='Type'
).properties(
  width=800
)

lines = alt.Chart(df1).mark_line().encode(
  x='Manufacturer',
  y='mean(Price)',

).properties(
  width=800
).interactive()
              
render_alt(points + lines)


# The graph gives us information on the number of vehicles available in each manufacturer
# with respect to different types of cars. The legend explains the type of variety of cars for each
# company. Easy for customer to sort out different companies in the decision making process
# of prioritising manufacturers.

# In[ ]:


chart6 = alt.Chart(df1).mark_bar().encode(
    x='Manufacturer',
    y='count()',
    color="Type",
)
render_alt(chart6)


# The graph shows the miles per gallon in the city in respect to horsepower, engine size &
# weight of the car. This gives an idea to the customer of how the engine size and horsepower
# affects the mileage which in turns shows the fuel efficiency versus the power of the car so
# that they can choose whether they want a faster car or a car with more fuel efficiency

# In[ ]:


chart4 = alt.Chart(df1).mark_circle().encode(
    alt.X('Horsepower', scale=alt.Scale(zero=False)),
    alt.Y('MPGH', scale=alt.Scale(zero=False, padding=1)),
    color='EngineSize',
    size='Weight'
).interactive()
render_alt(chart4)


# The graph put forth the information regarding the horsepower of different cars manufactured
# by different companies. The mean line will help the customer to categorise vehicles above a
# certain horsepower and below it making it easier to sort cars according to their desire.

# In[ ]:


chart5 = alt.Chart(df1).mark_line().encode(
    x='Manufacturer',
    y=alt.Y('mean(Horsepower)')
).interactive()
render_alt(chart5)


# This graph gives a brief idea about safety features of a car and the availability of that feature
# across various manufacturers. Here the graph shows the different manufacturers and their
# different types of cars with number of passengers having airbags for driver &passenger,driver only & no airbags. This gives the customer information regarding the safety features and selects the car according to their needs.
# 

# In[ ]:


interval = alt.selection_interval()

base = alt.Chart(df1).mark_point().encode(
  y='Passengers',
  color=alt.condition(interval, 'Type', alt.value('lightgray'))
).properties(
  selection=interval
)

render_alt(base.encode(x='AirBags') | base.encode(x='Manufacturer'))


# This box plot graph gives the potential buyer about the idea of average mileage of the total number of cars present in the company. 

# In[ ]:


import seaborn as sns
sns.set(style="whitegrid")
ax = sns.boxplot(x=df1["MPGC"])


# The graph shows the different manufacturers and the number of cars equipped with manual
# transmission. Gives customer a brief idea on which of the companies provide car with manual
# gear system, so that they can prioritise their needs to buy one.

# In[ ]:


chart3 = alt.Chart(df1).mark_bar().encode(
    x='Gear',
    y='count()',
    color='Gear',
    column='Manufacturer'
)

render_alt(chart3)


# Here the graph shows the miles per gallon of the cars available and their number in the store
# differentiating with the variety in type. This shows the customer that, for example if we take
# vans with fuel efficiency the graph will show how many vans are left in the company with
# certain mileage to be sold to the customers. 

# In[ ]:


chart2 = alt.Chart(df1).mark_area(
    opacity=0.3,
    interpolate='step'
).encode(
    alt.X('MPGC', bin=alt.Bin(maxbins=100)),
    alt.Y('count()', stack=None),
    alt.Color(
        'Type'
    )
).interactive()

render_alt(chart2)


# This is a prioritised graph, given the criteria that a customer is looking for a vehicle with
# larger size (width wise) and a larger fuel capacity can see the number of cars available in the
# company.

# In[ ]:


chart1 = alt.Chart(df1).mark_rect().encode(
    alt.X('Width', bin=True),
    alt.Y('Fuelcapacity', bin=True),
    alt.Color('count()',
        scale=alt.Scale(scheme='greenblue'),
        legend=alt.Legend(title='Total Records')
    )
).interactive()

render_alt(chart1)


# In[ ]:




