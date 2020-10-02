#!/usr/bin/env python
# coding: utf-8

# This competition aims to understand the spread of the COVID-19 Virus across the globe.
# The main additional dataset I will be using is the novel corona virus 2019 datset by SRK. Although it isn't purely 2019 data.
# 
# Here, I will plot the confirmed cases and deaths over time on a map, and then on a scatter graph (for 4 countries). I think that for most countries, a high order polynomial regression can be sucessfull (taking inspiration from the idea behind the Taylor Series, where non-polynomial functions can be approximated with polynomial functions). Obviously, we will have to pre-process the data a lot for this, and I have given ideas for the 3 countries below. I hope my insights can help in fitting a good model, which I plan to do in another notebook. 
# 
# 
# Please upvote if you find the visualizations helpful, as it means a lot to me. - Thanks

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-5/train.csv')
full_data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
full_deaths = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')


# In[ ]:


containment = pd.read_csv('/kaggle/input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv')


# In[ ]:


train.tail()


# In[ ]:


full_data.head()


# In[ ]:


full_deaths.head()


# # Confirmed Cases - World Map
# 
# The below plots are interactive. Howevering over a square will show you the number of confirmed cases in that region. Some countries have only 1 square, while others (look China and US) have it split across.

# In[ ]:


import plotly.graph_objects as go

full_data = full_data.rename(columns= {"Country/Region" : "Country", "Province/State": "Province"})

full_data['text'] = full_data['Country'] + " " + full_data["4/13/20"].astype(str)
fig = go.Figure(data = go.Scattergeo(
    lon = full_data["Long"],
    lat = full_data["Lat"],
    text = full_data["text"],
    mode = "markers",
    marker = dict(
        size = 12,
        opacity = 0.8,
        reversescale = True,
        autocolorscale = True,
        symbol = 'square',
        line = dict(
            width = 1,
            color = 'rgba(102, 102, 102)'
        ),
        cmin = 0,
        color = full_data['4/13/20'],
        cmax = full_data['4/13/20'].max(),
        colorbar_title = "COVID 19 Reported Cases"
    )
))

fig.update_layout(
    title = "COVID19 Confirmed Cases Around the World on 13th April",
    geo = dict(
        scope = "world",
        showland = True,
    )
)


fig.show()


# In[ ]:


full_data = full_data.rename(columns= {"Country/Region" : "Country", "Province/State": "Province"})

full_data['text'] = full_data['Country'] + " " + full_data["4/30/20"].astype(str)
fig = go.Figure(data = go.Scattergeo(
    lon = full_data["Long"],
    lat = full_data["Lat"],
    text = full_data["text"],
    mode = "markers",
    marker = dict(
        size = 12,
        opacity = 0.8,
        reversescale = True,
        autocolorscale = True,
        symbol = 'square',
        line = dict(
            width = 1,
            color = 'rgba(102, 102, 102)'
        ),
        cmin = 0,
        color = full_data['4/30/20'],
        cmax = full_data['4/30/20'].max(),
        colorbar_title = "COVID 19 Reported Cases"
    )
))

fig.update_layout(
    title = "COVID19 Confirmed Cases Around the World on 30th April",
    geo = dict(
        scope = "world",
        showland = True,
    )
)


fig.show()


# In[ ]:


full_data = full_data.rename(columns= {"Country/Region" : "Country", "Province/State": "Province"})

full_data['text'] = full_data['Country'] + " " + full_data["5/4/20"].astype(str)
fig = go.Figure(data = go.Scattergeo(
    lon = full_data["Long"],
    lat = full_data["Lat"],
    text = full_data["text"],
    mode = "markers",
    marker = dict(
        size = 12,
        opacity = 0.8,
        reversescale = True,
        autocolorscale = True,
        symbol = 'square',
        line = dict(
            width = 1,
            color = 'rgba(102, 102, 102)'
        ),
        cmin = 0,
        color = full_data['5/4/20'],
        cmax = full_data['5/4/20'].max(),
        colorbar_title = "COVID 19 Reported Cases"
    )
))

fig.update_layout(
    title = "COVID19 Confirmed Cases Around the World on 4th May",
    geo = dict(
        scope = "world",
        showland = True,
    )
)


fig.show()


# So, we can see that the situation is getting progressively worse all across the world (except for China). All of the squares are getting darker in color - look at India, it has a noticable orange tinge as compared to earlier.
# 
# # Deaths - World Map
# 
# Now, let us move on to see how the number of deaths is progressing. Again, the plots are interactive, and hovering over a square gives you the number of deaths in that region.

# In[ ]:


full_deaths = full_deaths.rename(columns= {"Country/Region" : "Country", "Province/State": "Province"})

full_deaths['text'] = full_deaths['Country'] + " " + full_deaths["4/13/20"].astype(str)
fig = go.Figure(data = go.Scattergeo(
    lon = full_deaths["Long"],
    lat = full_deaths["Lat"],
    text = full_deaths["text"],
    mode = "markers",
    marker = dict(
        size = 12,
        opacity = 0.8,
        reversescale = True,
        autocolorscale = True,
        symbol = 'square',
        line = dict(
            width = 1,
            color = 'rgba(102, 102, 102)'
        ),
        cmin = 0,
        color = full_deaths['4/13/20'],
        cmax = full_deaths['4/13/20'].max(),
        colorbar_title = "COVID 19 Deaths"
    )
))

fig.update_layout(
    title = "COVID19 Deaths Around the World on 13th April",
    geo = dict(
        scope = "world",
        showland = True,
    )
)


fig.show()


# In[ ]:


full_deaths = full_deaths.rename(columns= {"Country/Region" : "Country", "Province/State": "Province"})

full_deaths['text'] = full_deaths['Country'] + " " + full_deaths["4/30/20"].astype(str)
fig = go.Figure(data = go.Scattergeo(
    lon = full_deaths["Long"],
    lat = full_deaths["Lat"],
    text = full_deaths["text"],
    mode = "markers",
    marker = dict(
        size = 12,
        opacity = 0.8,
        reversescale = True,
        autocolorscale = True,
        symbol = 'square',
        line = dict(
            width = 1,
            color = 'rgba(102, 102, 102)'
        ),
        cmin = 0,
        color = full_deaths['4/30/20'],
        cmax = full_deaths['4/30/20'].max(),
        colorbar_title = "COVID 19 Deaths"
    )
))

fig.update_layout(
    title = "COVID19 Deaths Around the World on 30th April",
    geo = dict(
        scope = "world",
        showland = True,
    )
)


fig.show()


# In[ ]:


full_deaths = full_deaths.rename(columns= {"Country/Region" : "Country", "Province/State": "Province"})

full_deaths['text'] = full_deaths['Country'] + " " + full_deaths["5/4/20"].astype(str)
fig = go.Figure(data = go.Scattergeo(
    lon = full_deaths["Long"],
    lat = full_deaths["Lat"],
    text = full_deaths["text"],
    mode = "markers",
    marker = dict(
        size = 12,
        opacity = 0.8,
        reversescale = True,
        autocolorscale = True,
        symbol = 'square',
        line = dict(
            width = 1,
            color = 'rgba(102, 102, 102)'
        ),
        cmin = 0,
        color = full_deaths['5/4/20'],
        cmax = full_deaths['5/4/20'].max(),
        colorbar_title = "COVID 19 Deaths"
    )
))

fig.update_layout(
    title = "COVID19 Deaths Around the World on 4th May",
    geo = dict(
        scope = "world",
        showland = True,
    )
)


fig.show()


# Now, let us try to look at individual countries to understand what is going on.

# # Scatter plots + Regression Ideas

# We are only going to consider China, US, India (my country, so I am biased :), and South Korea.

# In[ ]:


US = full_data[full_data['Country'] == 'US']
India = full_data[full_data['Country'] == 'India']
Italy = full_data[full_data['Country'] == 'Italy']
SK = full_data[full_data['Country'] == 'Korea, South']


# In[ ]:


US


# However, First we need to covert the data into a column, instead of keeping it in a single row. We will also get rid of the province, country, lat, long and text columns, since we know the country is fixed.

# In[ ]:


US.drop(['text', 'Country', 'Province', 'Lat', 'Long'], axis=1, inplace=True)
columns = US.columns

US_data = pd.DataFrame(columns=['day', 'cases'])
index = 1
for col in columns:
    US_data.loc[len(US_data)] = [index, US[col].values[0]]
    index += 1

# ------- -------- ------- -------- ------- -------- ------- -------- ------- --------
India.drop(['text', 'Country', 'Province', 'Lat', 'Long'], axis=1, inplace=True)
columns = India.columns

India_data = pd.DataFrame(columns=['day', 'cases'])
index = 1
for col in columns:
    India_data.loc[len(India_data)] = [index, India[col].values[0]]
    index += 1
    
# ------- -------- ------- -------- ------- -------- ------- -------- ------- --------
Italy.drop(['text', 'Country', 'Province', 'Lat', 'Long'], axis=1, inplace=True)
columns = Italy.columns

Italy_data = pd.DataFrame(columns=['day', 'cases'])
index = 1
for col in columns:
    Italy_data.loc[len(Italy_data)] = [index, Italy[col].values[0]]
    index += 1
    
# ------- -------- ------- -------- ------- -------- ------- -------- ------- --------
SK.drop(['text', 'Country', 'Province', 'Lat', 'Long'], axis=1, inplace=True)
columns = SK.columns

SK_data = pd.DataFrame(columns=['day', 'cases'])
index = 1
for col in columns:
    SK_data.loc[len(SK_data)] = [index, SK[col].values[0]]
    index += 1


# In[ ]:


plt.scatter(US_data['day'], US_data['cases'])


# So, US has managed to slow down the growth. It doesn't look exponential anymore. My guess is that the R (reproductive rate) is roughly 1, so a linear regression might be able to extrapolate accurate values (after cutting off data for the first 70 days).

# In[ ]:


plt.scatter(India_data['day'], India_data['cases'])


# India is still early on in its journey, and is still exponential. Since every non-polynomial function can be approximated by a polynomial function, maybe a polynomial regression (n = 6) can give accurate results. 

# In[ ]:


plt.scatter(Italy_data['day'], Italy_data['cases'])


# Italy's graph is slowing down, so here, I think we can cut of the data from about 70 days, and then try a logarithmic regression. (One possible way of acheiving logarithmic regression is by exponentiating the values, and then fitting a linear model. After that, log the predictions.)

# In[ ]:


plt.scatter(SK_data['day'], SK_data['cases'])


# South Koreas looks like it is reaching the end of the COVID-19 rollercoster, like China. However, the data points have too much variance to use an effective polynomial regression. Hence, a more complicated time series model may be required.

# Thank you for reading this notebook. I will create another notebook to try and create polynomial regression models for at least US, India and Italy. So, stay Tuned.
# 
# Please upvote, as it helps me understand that my work is helpful.

# In[ ]:




