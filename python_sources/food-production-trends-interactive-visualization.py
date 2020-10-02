#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Libraries
import numpy as np
import pandas as pd

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff


# ## Teleport

# 1. [Top 10 items produced](#top-10)<br>
#     1.1. [Year 2013](#year-2013)<br>
#     1.2. [Production development (years 1961 - 2013)](#prod-dev)<br>
# 2. [The most produced item - development by country](#most-prod)<br>
# 3. [The second most produced item - development by country](#sec-prod)<br>
# 4. [Countries' changing production amounts](#count-change)<br>
#     4.1. [Largest positive trend by country and item](#pos-trend)<br>
#     4.2. [Largest negative trend by country and item](#neg-trend)<br>

# Note! We are only focusing food production data (excluding feed).

# In[3]:


food = pd.read_csv('../input/FAO.csv', encoding = "ISO-8859-1")

# Drop columns we do not need
food = food.drop(['Area Abbreviation', 'Area Code', 'Item Code', 'Element Code', 'Unit', 'latitude', 'longitude'], axis=1)
food = food[food['Element'] == 'Food']


# In[4]:


years_str = []
years = []
for i in range(1961, 2014):
    years.append(i)
    years_str.append('Y' + str(i))


# <a id="top-10"></a>
# ### 1. Top 10 items produced

# <a id="year-2013"></a>
# #### 1.1. Year 2013

# In[5]:


item_10 = food.groupby('Item').sum()['Y2013'].sort_values(ascending=False)[:10]

texts = []
for amount in item_10:
    text = '{0} megatons'.format(amount)
    texts.append(text)

data = [
    go.Bar(x=item_10.index, y=item_10,
           textfont=dict(size=16, color='#333'),
           text = texts,
           hoverinfo='text',
           marker=dict(
               color=item_10,
               colorscale='Viridis',
               reversescale=False,
               line=dict(width=1, color='rgb(100, 0, 0 )')
           ))
]

layout = go.Layout(
    autosize=False,
    width=800,
    height=400,
    title='Top 10 produced items'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# <a id="prod_dev"></a>
# #### 1.2. Production development (years 1961 - 2013)

# In[6]:


item_10_dev = food[food['Item'].isin(item_10.index)].groupby('Item').sum().sort_values('Y2013', ascending=False)

data = []
for item_name in item_10_dev.index:
    data.append(
        go.Scatter(x=years, 
                   y=item_10_dev.loc[item_name, years_str],
                   name=item_name,
                   textfont=dict(size=16, color='#333'),
                  )
    )

layout = go.Layout(
    autosize=False,
    width=800,
    height=400,
    title='Production development from 1961 to 2013'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# <a id="most-prod"></a>
# ### 2. The most produced item - development by country

# In[7]:


top_1 = food[(food['Item'] == item_10.index[0])]
years_by_10 = []
for i in range(1961, 2014):
    if i % 10 == 3:
        years_by_10.append('Y' + str(i))
data_bal = []

for year in years_by_10:
    texts = []
    for index, row in top_1.iterrows():
        text = '{0} megatonnes<br>'.format(row[year])
        texts.append(text)
    data = [dict(
        type='choropleth',
        locations=top_1['Area'],
        locationmode='country names',
        z=top_1[year],
        text=texts,
        hoverinfo='text+location',
        colorscale='YlOrRd',
        zmin=0,
        zmax=100e3,
        colorbar = dict(title = "1000 tonns", tickvals=[0, 25e3, 50e3, 75e3, 100e3], tickmode='array'),
        reversescale=True,
        marker=dict(line=dict(color='rgb(120,120,120)', width=0.5)),
    )]
    
    data_bal.extend(data)
    
steps = []
yr = 0
for i in range(0,len(data_bal)):
    step = dict(method = "restyle",
                args = ["visible", [False]*len(data_bal)],
                label = years_by_10[yr][1:]) 
    step['args'][1][i] = True
    steps.append(step)
    yr += 1

sliders = [dict(active = len(years_by_10) - 1,
                currentvalue = {"prefix": "Year: "},
                pad = {"t": 10},
                steps = steps)]

# Set the layout
layout = dict(title = 'Production of {0} between years {1} and {2}'.format(item_10.index[0], years_by_10[0][1:], 
                                                                           years_by_10[-1][1:]),
              geo = dict(showframe=False, showcoastlines=True, showocean=True, oceancolor='rgb(173,216,230)', 
              projection=dict(type='Mercator')),
              sliders = sliders)

fig = dict(data=data_bal, layout=layout)
py.iplot(fig)


# <a id="sec-prod"></a>
# ### 3. The second most produced item - development by country

# In[8]:


top_2 = food[(food['Item'] == item_10.index[1])]
years_by_10 = []
for i in range(1961, 2014):
    if i % 10 == 3:
        years_by_10.append('Y' + str(i))
data_bal = []

for year in years_by_10:
    texts = []
    for index, row in top_2.iterrows():
        text = '{0} megatonnes<br>'.format(row[year])
        texts.append(text)
    data = [dict(
        type='choropleth',
        locations=top_2['Area'],
        locationmode='country names',
        z=top_2[year],
        text=texts,
        hoverinfo='text+location',
        colorscale='YlOrRd',
        zmin=0,
        zmax=200e3,
        colorbar = dict(title = "1000 tonns", tickvals=[0, 50e3, 100e3, 150e3, 200e3], tickmode='array'),
        reversescale=True,
        marker=dict(line=dict(color='rgb(120,120,120)', width=0.5)),
    )]
    
    data_bal.extend(data)
    
steps = []
yr = 0
for i in range(0,len(data_bal)):
    step = dict(method = "restyle",
                args = ["visible", [False]*len(data_bal)],
                label = years_by_10[yr][1:]) 
    step['args'][1][i] = True
    steps.append(step)
    yr += 1

sliders = [dict(active = len(years_by_10) - 1,
                currentvalue = {"prefix": "Year: "},
                pad = {"t": 10},
                steps = steps)]

# Set the layout
layout = dict(title = 'Production of {0} between years {1} and {2}'.format(item_10.index[1], years_by_10[0][1:], 
                                                                           years_by_10[-1][1:]),
              geo = dict(showframe=False, showcoastlines=True, showocean=True, oceancolor='rgb(173,216,230)', 
              projection=dict(type='Mercator')),
              sliders = sliders)

fig = dict(data=data_bal, layout=layout)
py.iplot(fig)


# <a id="count-change"></a>
# ### 4. Countries' changing production amounts

# We are considering average absolute change in production.

# In[9]:


absolute_cols = []

for i in range(1961, 2013):
    abs_col = str(i+1) + '-' + str(i)
    absolute_cols.append(abs_col)
    food[abs_col] = food['Y' + str(i+1)] - food['Y' + str(i)]
    
food['abs_mean'] = food[absolute_cols].mean(axis=1)


# <a id="pos-trend"></a>
# #### 4.1. Largest positive trend

# In[12]:


pos_country_data = food.sort_values('abs_mean', ascending=False)[:5]

data = []
for index, row in pos_country_data.iterrows():
    data.append(
        go.Scatter(x=years, 
                   y=np.array(row[years_str]),
                   name=row['Area'] + ' - ' + row['Item'],
                   textfont=dict(size=16, color='#333'),
                  )
    )

layout = go.Layout(
    autosize=False,
    width=800,
    height=400,
    title='Top 5 largest positive trends by country and item'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# <a id="neg-trend"></a>
# #### 4.2. Largest negative trend

# In[13]:


neg_country_data = food.sort_values('abs_mean', ascending=True)[:5]

data = []
for index, row in neg_country_data.iterrows():
    data.append(
        go.Scatter(x=years, 
                   y=np.array(row[years_str]),
                   name=row['Area'] + ' - ' + row['Item'],
                   textfont=dict(size=16, color='#333'),
                  )
    )

layout = go.Layout(
    autosize=False,
    width=800,
    height=400,
    title='Top 5 largest negative trends by country and item'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)

