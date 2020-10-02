#!/usr/bin/env python
# coding: utf-8

# Hello!!
# 
# Lets work analyzing the restaurant and bar chain. To do that, I have though in the following objectives:
# 
# 1. To plot and analyze in a visual way data we have
# 2. To plot the distribution of restaurants in the country
# 
# Lets see if we can

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/zomato.csv")
df.shape


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.info()


# As first observation we can see that there are 17 columns and 51717 rows. Now, what i am gonna do is to see the meaning of each column and according the objectives that I have written at the beginning, I will see which columns are interesting or not.
# 
# 1. url - url of the bar - not interesting for my purposes 
# 2. address - Not interesting for my purposes at the beginning. With the location is enough
# 3. Name - Lets keep it
# 4. Online order - Interesting
# 5. book table - Interesting
# 6. rate - very interesting
# 7. votes - it is interesting due to with this information I could interpret how many people is attending the bar
# 8. phone - nah
# 9. location - It seems to be interesting
# 10. rest_type - interesting
# 11. dish_liked - mmmmm I think not
# 12. Cuisines - yeeees
# 12. approx_cost - jejejejej
# 13. reviews_list - Maybe we could try something with NLP.
# 14. menu_item - NLP? I dont know
# 15. listed_in (typ3) - Yes
# 16. listed_in (city) - YEEEESSSS
# 
# If we check the info about location and we compare it with listed_in, we can see than in location there are some null values. Then, I think I could remove location and let just listed_in(city). What do you think?
# 
# So, we will remove next columns: url, address, phone, location, dish_liked
# 
# lets work!

# In[ ]:


semiclean = ["url", "address", "phone", "location", "dish_liked"]
df_semiclean = df.drop(semiclean, axis = 1)
df_semiclean.info()


# Jmmmm some null values in rest, what should we do? I can think in two different options:
# 
#  - Easy: remove null values
#  - Not so easy: to predict these values.
#  
# I will start removing these values.

# In[ ]:


df_clean = df_semiclean.dropna()
df_clean.info()


# In[ ]:


rates = list()
for fila in df_clean.rate:
    aux = fila[:3]
    rates.append(aux)
#df_clean = df_clean.drop(["rate"], axis = 1)
df_clean = df_clean.assign(rate = rates)


# In[ ]:


df_clean.head(20)


# Now, we start analyzing in a visual way the data. How? we will start comparing the following fields:
#  
#  - Rates Vs. Prices (Maybe restaurant boss has increased prices due to good rates)
#  - Votes Vs. cuisines (Maybe we can get some food trend, who knows?)
#  - online_order Vs Prices
#  - historic of prices
#  - Distribution of prices per city
#  
# But before lets to analyze data trying to find some patterns and studying how these are distributed
# Lets start with that to see if we can get some conclusions.

# In[ ]:


# Distribution of rates and prices

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as ply
ply.init_notebook_mode(connected=True)


rate = go.Histogram(x=df_clean["rate"],
                   opacity=0.75,
                   name = "Rate Values",
                   )

price = go.Histogram(x=df_clean["approx_cost(for two people)"],
                   opacity=0.75,
                   name = "Price Values",
                   )

layout = go.Layout(barmode='overlay',
                   title='Histogram',
                   xaxis=dict(title='Values'),
                   yaxis=dict(
                        title='yaxis title',
                    ),
                  )

data = [rate]
fig = go.Figure(data=data, layout=layout)                   
ply.iplot(fig)

data = [price]
fig = go.Figure(data=data, layout=layout)                   
ply.iplot(fig)


# In[ ]:


trace = [go.Scattergl(
                    x = df_clean["rate"],
                    y = df_clean["approx_cost(for two people)"],
                    mode = "markers"
                    )]
ply.iplot(trace)


# **Votes Vs. Cusines**

# In[ ]:


cuisines_ = np.unique(df_clean.cuisines)
counter = list()
for i in cuisines_:
    aux = len(df_clean.index[df_clean["cuisines"]==i])
    counter.append(aux)


# In[ ]:


df_count = pd.DataFrame({"Q": counter}, index = cuisines_)
df_count.head()


# In[ ]:


cuisin = [go.Histogram(x=df_count["Q"],
                   opacity=0.75,
                   name = "Cusine counter",
                   text = df_count.index,
                   )]

fig = go.Figure(data=cuisin)                   
ply.iplot(fig)


# We can see in the image below that here is accomplished the pareto principle: https://en.wikipedia.org/wiki/Pareto_principle
# Thus, next cuisines are the most repeated along india:
# 
#  - African, Burguer
#  - African, Burger, Desserts, Beverages, Fast food
#  - American
#  
# Next cuisines finally refer to allmost every kind of dishes. Now, thinking (a bit, not to much) it is not efficient to find a relation between kind of cuisines and prices. So next action will be deploy how many restaurantes there are in India per city.
#  

# In[ ]:


cities_ = np.unique(df_clean["listed_in(city)"])
counter = list()
for i in cities_:
    aux = len(df_clean.index[df_clean["listed_in(city)"]==i])
    counter.append(aux)

df_city = pd.DataFrame({"Q": counter}, index = cities_)
df_city.head()


# In[ ]:


trace = [go.Pie(labels=df_city.index, values=df_city["Q"])]

ply.iplot(trace)

