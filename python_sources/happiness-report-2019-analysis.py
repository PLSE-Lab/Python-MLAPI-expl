#!/usr/bin/env python
# coding: utf-8

# # Happiness Report 2019
# 
# Is it really possible to measure **happiness**? 
# 
# For me it is interesting how can you measure an emotion or at least try to get an idea. Before anything it might be important to ask yourself **what is "happiness"**.For me can be from eating ice cream to watching a movie with my family. Maybe for you is buying a new car, traveling to an exotic country or playing with your dog. Think how many different choices someone has to respond this question. Just to get an idea there are more than **7 billion** people to ask. So this question is way harder than it looks.  
# 
# The available information is always a headache. Getting good data just for this question seems imposible. Yet amazing organisations manage to collect and get the best information from all around the world and give it for **free** (How cool is that?).
# 
# In this case here we are with "**The World Happiness Report**" made by the **United Nations**. (For the curious ones: they made reports like this one since 2012, if you want to look around).  
# 
# In this report, the **quality of peoples lives** (by country) is the best approach for this emotion. In this report they present the links between **goverment and happiness**, and **generosity** and the **pro-social** behaviour. The data that we have is a ranking, being 1 the best among all the sample. 
# 
# Lets review for this year which are the happiest and unhappiest countries on earth. And not just that. Lets see **why** are those countries like that.  
# 
# **Lets get to work!**
# 
# Here below we start with the usual imports (i selected them from other Kernels, the most used ones, to compare and get something myself). 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import os
print(os.listdir("../input"))


# We need to know what exactly do we have on the database, so we load it and see its shape. 

# In[ ]:


data = pd.read_csv('../input/world-happiness-report-2019.csv')
print('Rows in Data: ', data.shape[0])
print('Columns in Data: ', data.shape[1])


# In this dataset, the happiness has been approached by **economic and social variables** like the income (GDP per capita), the health conditions (Healthy life expectancy), social support and so on (as you can see in the output). If you think about it, this variables really define some of the answers you might give to the main question. Generally speaking, people focuses on money and health on this matter. So this is a good approach, an a comparable too.
# 
# For the ranking it is important to notice that **the measure of happiness** in each country is based on "**individuals own assessments of their lives**" = Cantril ladder. Basicaly it asks for you to think of a ladder. With the best possible life for you, you will give a '10' and for the worst possible life a '0'. Now you are asked to *rate your own current life* based on that 0 to 10 scale. Then for a country, its rating is the average ladder score. 
# Quite easy!
# 
# The rankings are from nationally representative samples, for **2016-2018**. But what about the variables then, if they are not in the rate-process. Good question!, the six variables: 
# 
# * GDP per capita 
# * Social support 
# * healthy life expectancy
# * freedom 
# * generosity 
# * absence of corruption
# 
# Are used to **explain the variation across countries**. 
# 
# Below we can view the **top ten** ranking countries and the **bottom ten** ranking countries of the sample (156 countries). Another important thing to notice, is that every variable has **ranking** and NOT AN ACTUAL VALUE. This means that 1 is the best and so on..

# HAPPIEST

# In[ ]:


data.head(10)


# ..and the UNHAPPIEST

# In[ ]:


data.tail(10)


# **Finland** according to the ranking is the best place to live!, with the best position in the ladder (1). **South Sudan** on the other hand...well lets say it is not. Just by looking at this tables, we get an idea. Finland, Denmark, Norway are countries with great healthcare and amazing education levels. Afghanistan, Syria and South Sudan are wel-known for being involved in certain conflicts and wars. Lets keep getting more insights. 

# (Just for simplification, the column name "Country (region)" will be renamed as "Country". )

# In[ ]:


data = data.rename({'Country (region)':'Country'}, axis=1)
data.dtypes


# Here we make a heatmap of the data to see the relationship with the data we have. 

# The most important variables here to see (ladder column, the darkest on the graph) are **social support**, **GDP per capita**, and **healthy life expectancy**. 
# 
# This is the same result as the report shows in Figure 2.7 (page 24 in the report, again for the curious ones). 
# 
# There is a strong correlation with these variables. This means, in average, countries with the best social support, the best GDP per capita and better heatlhy life expectancy have **highest rankings**. 
# 
# This, on average, **explains why** the people in these countries are in better life-conditions, and therefore, might be **happier**. 

# In[ ]:


mask = np.zeros_like(data.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.figure(figsize = (19,15))
sns.heatmap(data.corr(), mask = mask, annot=True, cmap="YlGnBu", linewidths=.2, square=True)


# As shown before, the "happiest" 10 countries are very close, *geographically-speaking*. This happens with the "unhappiest" countries too, so it would be interesting to see how is the **distribution of "happiness" in the world**.
# 
# As curiosity, i didn't know you can actually made a global-map with the data you have in here...And i think that is so cool!. So useful in our case (the idea is from other kernels i saw before, which are great too!).
# 
# Lets see how happy is the world. The visualization speaks for itself! 
# 
# *FOR THE VISUALIZATION*: The darker **blue** a country is, **the happier the country is**. 
# 
# **scandinavian countries** are clearly the more "happy" according to the report. Europe in general is quite happy too. 
# 
# In my case, i am from **Colombia** so im quite curious for the ranking in latin america. For Colombia, it has the **43th position**, behind countries like Uruguay, Brazil, Panama and Chile. It has a better position than Argentina, Ecuador, Honduras and Bolivia though. Well were not exactly the happiest country in the world, but we are getting there!
# 
# My neighbour **Venezuela** on the other hand has the **108th position**, which is very far from the latin america average. It can be explain, since the country has a policital controversy with Maduro, has extremely high inflation rates and increasing poverty rates as well. 

# In[ ]:


map_data = [go.Choropleth(
           colorscale =  'YlGnBu',
           locations = data['Country'],
           locationmode = 'country names',
           z = data["Ladder"], 
           text = data['Country'],
           colorbar = {'title':'Ladder Rank'})]

layout = dict(title = 'Happiness distribution', 
             geo = dict(showframe = False, 
                       projection = dict(type = 'equirectangular')))

world_map = go.Figure(data=map_data, layout=layout)
iplot(world_map)


# Lets compare the same map but with freedom. 
# 
# *FOR THE VISUALIZATION*: The darker the **red**, the less freedom the country has. 
# 
# It seems like the countries with less freedom are unhappier according to the data. **Africa** in general has something in particular. These countries have the lowest GDP per capita, the worst life expectancy. Just at first sight it is clear that it has suffered from many years, by wars, poverty and health problems. As a result, it is clearly the region with the lowest ranking and has the **unhappiest countries** of the sample. 

# In[ ]:


map_data = [go.Choropleth( 
           locations = data['Country'],
           locationmode = 'country names',
           z = data["Freedom"], 
           text = data['Country'],
           colorbar = {'title':'Ladder Rank'})]

layout = dict(title = 'Least Freedom', 
             geo = dict(showframe = False, 
                       projection = dict(type = 'equirectangular')))

world_map = go.Figure(data=map_data, layout=layout)
iplot(world_map)


# Lets sort by the income measured as the **GDP per capita**, to see how much does it really change. 
# 
# Lets see the first ten countries with the highest GDP per capita. Here we have **Qatar** as the country with the highest GDP per capita, yet it is not even in the first 20 happiest countries of the sample (29th position). So, the variables themselves do not explain completely why a country is happier than other

# In[ ]:


GDP = data.sort_values(by='Log of GDP\nper capita')
GDP.head(10)


# Here we do the same for **Social support**, to see how it changes. 
# 
# If we think in terms of happiness, as in the ranking, then social support **explains better the ranking**. **Iceland** is the country with the best Social support in the sample. Finland follows, with Norway and Denmark. Very close to the first table we had. It explains better than the actual GDP per capita of the country. Interesting! 

# In[ ]:


SS = data.sort_values(by='Social support')
SS.head(10)


# What about the **Healthy life expectancy**?
# 
# **Singapore** has the best ranking in the sample! followed by Japan and Spain. It has a certain variation of countries, like the one we had with the GDP per capita. Again, it is not the best variable to explain (alone) the ranking of happiness.

# In[ ]:


HLE = data.sort_values(by='Healthy life\nexpectancy')
HLE.head(10)


# As a conclution, for the data, in average, 
# * the variable Social support explains better why a country is happier than other, than the GDP per capita and the Healthy life expectancy. 
# * This variables (Social support, GDP per capita, Healthy life expectancy) in general are the ones that explains better certain level of happiness that a country has. 
# * Countries tend to be as happy as the average in the region; example: the european countries have similar levels of happiness. American countries have similar levels as well. 
# 
# Now, for the first question. *Is it possible to measure happiness?* It is **not imposible**, just really hard. This is just an approach, but it is not perfect. As i said before it is not easy to measure an emotion and generalize with that measure. With this said, thanks to the information in the report we were **closer**, or at least **we got an idea** of the happiness in the world. 
# 
# And not just that, this gives perspective as well, now you know better on which world you are living in! keep making the world a **happy place**. 
