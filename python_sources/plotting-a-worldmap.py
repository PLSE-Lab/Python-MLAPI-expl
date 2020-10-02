#!/usr/bin/env python
# coding: utf-8

# First I read the series and the indicators.

# In[ ]:


import pandas as pd
import geopandas as gpd

df_series = pd.read_csv("../input/world-development-indicators/Series.csv")
df_indicators = pd.read_csv("../input/world-development-indicators/Indicators.csv")

df_series.head()


# I am only interested in the code the topic and the name of a series.

# In[ ]:


df_series = df_series.filter(items=["SeriesCode", "Topic", "IndicatorName"])
df_series.Topic.unique()


# I choose the topic "Health: Population: Dynamics"

# In[ ]:


df_series[df_series.Topic == "Health: Population: Dynamics"]


# In[ ]:


df_indicators.head()


# I select the series "Population Growth" and the year 2014

# In[ ]:


df_pop = df_indicators[df_indicators.IndicatorCode == 'SP.POP.GROW']
df_pop.Year.unique()


# I read the world map and add the column "Population growth" to it.

# In[ ]:


pop_2014 = df_pop[df_pop.Year == 2014]
pop_2014 = pop_2014.filter(["CountryCode", "Value"])
pop_2014.rename(columns={"Value": "Population growth"}, inplace=True)

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.head()


# The country with the highest population growth is surprisingly Oman

# In[ ]:


world = pd.merge(world, pop_2014, left_on="iso_a3", right_on="CountryCode")
world = world.sort_values(by=["Population growth"], ascending=False)
world.head(5)


# In[ ]:


world.plot(column="Population growth", legend=True, figsize=(24, 12), cmap="Reds")


# In[ ]:




