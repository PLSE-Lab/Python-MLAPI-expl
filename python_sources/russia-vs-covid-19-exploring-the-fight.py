#!/usr/bin/env python
# coding: utf-8

# # Russia vs. COVID-19: Exploring the fight
# 
# **Author: Nikhil Praveen(@nxrprime)**

# As you all know, the coronavirus is a global pandemic and needs to be dealt with quickly. Let us see how Russia has handled the coronavirus.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px
import seaborn as sns
sns.set_style("darkgrid")
import plotly.io as pio
pio.templates.default = "plotly_dark"


# In[ ]:


data = pd.read_csv("../input/covid19-in-russia/RussiaCorona.csv")
data.tail()


# In[ ]:


df_confirmed = pd.DataFrame({
    'Date': data.Date,
    'Confirmed': data.Confirmed
})


# In[ ]:


fig = px.line(df_confirmed, x="Date", y="Confirmed", 
              title="Russia's Confirmed Cases Over Time")
fig.show()


# In[ ]:


fig = px.line(df_confirmed, x="Date", y="Confirmed", 
              title="Russia's Confirmed Cases Over Time (Logarithmic Scale)", log_y=True)
fig.show()


# So it seems like a problem in Russia, with the cases only rising.

# In[ ]:


fig = px.bar(data, 
             x='Date', y='Confirmed', color_discrete_sequence=['#D63230'],
             title='Confirmed Cases Daily', text='Confirmed')
fig.show()


# It seems like everything is just rising in Russia. What about deaths?

# In[ ]:


fig = px.line(data, 
             x='Date', y='Deaths', color_discrete_sequence=['#D63230'],
             title='Deaths in Russia (linear scale)', text='Deaths')
fig.show()


# So it seems like deaths are also rising in Russia. What about the increase/decrease in deaths?

# In[ ]:


fig = px.line(data, 
             x='Date', y='DeathsDaily', color_discrete_sequence=['#D63230'],
             title='Deaths Increase Per Day in Russia (linear scale)', text='DeathsDaily')
fig.show()


# ### Network Plot

# In[ ]:


import networkx as nx
df1 = pd.DataFrame(data['Confirmed']).groupby(['Confirmed']).size().reset_index()

G = nx.from_pandas_edgelist(df1, 'Confirmed', 'Confirmed', [0])
colors = []
for node in G:
    if node in data["ConfirmedDaily"].unique():
        colors.append("red")
    else:
        colors.append("lightgreen")
        
nx.draw(nx.from_pandas_edgelist(df1, 'Confirmed', 'Confirmed', [0]), with_labels=True, node_color=colors)


# ## This notebook will be updated every morning 10:30 AM IST. If you like it, please consider upvoting.
