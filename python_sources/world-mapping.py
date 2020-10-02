#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install pycountry')


# In[ ]:


import pandas as pd
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")


# In[ ]:


country_col = "Country"
state_col = "Province/State"
target = "Confirmed"


# In[ ]:


import pycountry
countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3
df["iso_alpha"] = df[country_col].map(countries.get)


# In[ ]:


df.head()


# In[ ]:


data = df.groupby("iso_alpha")[target].sum().reset_index()


# In[ ]:


import plotly.express as px
# df2 = px.data.gapminder().query("year == 2007")
fig = px.choropleth(data, locations="iso_alpha",
                     color=target, # which column to use to set the color of markers
#                      hover_name=country_col, # column added to hover information
                     projection="natural earth")
fig.show()


# In[ ]:




