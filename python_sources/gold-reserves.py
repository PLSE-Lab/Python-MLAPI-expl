#!/usr/bin/env python
# coding: utf-8

# # (STILL IN PROGRESS. THERE ARE MORE PLOTS, CONCLUSIONS AND IDEAS COMMING. I WOULD APPRECIATE ANY KIND OF COLABORATION TO EXTEND THIS ANALYSIS)

# # Countries' Declared Gold Reserves from 1950 to 2018

# ### Gold is an important asset and considered by many to be the only real money. This dataset contains quarterly reserves by country since 1950.

# ### Provided and updated by the IMF data portal (https://data.imf.org/?sk=388DFA60-1D26-4ADE-B505-A05A558D9A42).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt                   # For graphics
from matplotlib import animation
import matplotlib.animation as animation
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


g = '/kaggle/input/gold-reserves-by-country-quarterly/gold_quarterly_reserves_ounces.csv'
gold = pd.read_csv(g, index_col='Country Name')


# In[ ]:


gold.info()


# In[ ]:


gold.head()


# In[ ]:


time = gold['Time Period'].str.split('Q', 1).str
gold['Year'] = time[0]
gold['Quarter'] = time[1]
gold = gold.drop('Time Period',axis=1)

gold.head()


# ### Dividing the value by 32000, we switch values from ounces to tonnes

# In[ ]:


gold['Value'] = gold['Value']/32000


# In[ ]:


gold['Year'] = gold['Year'].astype(int)
gold['Value'] = round(gold['Value'],0)


# In[ ]:


gold['Year']


# ### Removing "Country Names" that refers to more than one real country or specific regions

# In[ ]:


gold2 = gold.drop(['Advanced Economies','Sub-Saharan Africa','Central African Economic and Monetary Community','CIS','Emerging and Developing Asia','Emerging and Developing Europe','Emerging and Developing Countries', 'Europe', 'Euro Area','Middle East, North Africa, Afghanistan, and Pakistan','World','Western Hemisphere','West African Economic and Monetary Union (WAEMU)'])


# In[ ]:


gold2.index.unique()


# In[ ]:


gold2 = gold2.loc[gold2['Quarter']=='4']


# In[ ]:


gold2['Country Name'] = gold2.index
gold2.reset_index(drop=True, inplace=True)


# In[ ]:


country = gold2['Country Name'].unique()


# In[ ]:


temp = {i: j for j, i in enumerate(set(gold2['Country Name']))} 
gold2['Color'] = [temp[i] for i in gold2['Country Name']] 


# In[ ]:


gold2.sort_values(['Year','Value'], ascending=[True,False], inplace=True)


# In[ ]:


gold2['Country Name'] = gold2['Country Name'].replace(['Venezuela, Republica Bolivariana de', 'China, P.R.: Mainland', 'Taiwan Province of China', 'Russian Federation', 'Iran, Islamic Republic of'],['Venezuela', 'China', 'Taiwan', 'Russia', 'Iran'])


# In[ ]:


gold3 = pd.concat(gold2[gold2['Year']==i][:20] for i in gold2['Year'].unique())


# In[ ]:


gold3.sort_values(['Year', 'Value'], ascending=True, inplace=True)


# ### Animation to see the evolution of top20 countries' gold reserves from 1950 to 2018

# ### (Click PLAY button)

# In[ ]:


fig = px.bar(gold3, x='Value', y='Country Name', title='Gold', animation_frame='Year', orientation='h', text='Value', width=1100, height=800, color='Color', color_continuous_scale=px.colors.qualitative.Alphabet)

fig.update_layout(xaxis=dict(title='Tonnes of gold', showgrid=True, gridcolor='grey'), yaxis=dict(title=''), paper_bgcolor='white', plot_bgcolor='white', coloraxis=dict(showscale=False))
fig.show()


# Notice the big US gold reserves decrease until 1972 and the increase of Russia and China in the last 20 years.

# ### Evolution of top20 countries' gold reserves from 1950 to 2018

# ### (Click on the Country Names at the right side to filter them)

# In[ ]:


topgold2 = gold2.loc[gold2['Country Name'].isin(['United States','United Kingdom', 'Switzerland', 'Germany', 'France', 'Italy', 'Russian Federation', 'China, P.R.: Mainland', 'Taiwan province of China'])].sort_values('Year')


# In[ ]:


fig = px.line(topgold2, x='Year', y='Value', title='Top Gold Reserves', color='Country Name')

fig.update_layout(xaxis=dict(title='Year', showgrid=False), yaxis=dict(title='Tonnes of Gold', showgrid=True, gridcolor='grey'), paper_bgcolor='white', plot_bgcolor='white', legend=dict(xanchor='right', yanchor='top'))
fig.show()


# It's clear after 1972 the gold reserves fluctuation had become more quiet than the previous 20 years. Especially for US, who felt a big decrease of more than 50% of the initial gold reserves in 1950.

# ## Variation from previous year

# ### We want to visualize total variation from one year to the next one for each country

# ### We analyse the top20 countries of 2018

# In[ ]:


names2018 = gold2.loc[gold2['Year']==2018][:20]


# In[ ]:


names2018 = names2018['Country Name'].values


# In[ ]:


names2018


# In[ ]:


top2018 = gold2.loc[gold2['Country Name'].isin(names2018)]


# In[ ]:


top2018_x = top2018.sort_values(['Country Name', 'Year'], ascending=True)
top2018_x['Variation'] = top2018_x['Value'] - top2018_x['Value'].shift(1)
top2018_x = top2018_x[top2018_x['Country Name'].duplicated(keep='first')]


# In[ ]:


top2018_x['Variation']


# ### Countries' gold reserve variation year by year

# ### (Click on the Country Names at the right side to filter them)

# In[ ]:


fig = px.bar(top2018_x, x='Year', y='Variation', title='Top Gold Reserves variations', color='Country Name')

fig.update_layout(xaxis=dict(title='Year', showgrid=False), yaxis=dict(title='Tonnes of Gold', showgrid=True, gridcolor='grey'), paper_bgcolor='white', plot_bgcolor='white', legend=dict(orientation='h'))
fig.show()


# UNDERSTANDING THE GRAPH

# We can highlight 4 top activity timeframes regarding this graph.

# First: 1950 to 1972. It seems there was a lot of variations during those years, most of this activity came from occidental countries (EU and US)

# Second: 1979. A big drop down for many of top countries (France, Italy, Germany, Netherlands, UK and US)

# Third: 1998. Big increase for some EU countries (Spain, UK, Portugal, Germany, France and Italy)

# Fourth: 2000 to 2018. We notice a increase from Russia and China, but also a decrease in Switzerland. Also notice for top countries (US, Germany, France and Italy) there was no variation in their gold reserves.

# ### Variation between 1950-1972 sum

# In[ ]:


var_til1972 = top2018_x.loc[(top2018_x['Year']<1972)].groupby('Country Name').sum().sort_values('Variation', ascending=False)


# In[ ]:


var_til1972['Country Name'] = var_til1972.index
var_til1972.reset_index(drop=True, inplace=True)


# ### Visualize the variation's sum between 1950 and 1972 for each country

# In[ ]:


fig = px.bar(var_til1972, x='Country Name', y='Variation', title='Total Gold reserves variation 1950-1972', color='Variation')

fig.update_layout(xaxis=dict(title='Country', showgrid=False), yaxis=dict(title='Tonnes of Gold', showgrid=True, gridcolor='grey'), paper_bgcolor='white', plot_bgcolor='white')
fig.show()


# We can see that UK and US where the countries that lost big amounts of gold, since some european countries, Japan, Lebanon and Saudi Arabia, obtained significant gold increases between 1950 and 1972.

# ### Difference between increasing and decreasing countries from 1950 to 1972

# In[ ]:


diffplus = var_til1972.loc[var_til1972['Country Name'].isin(['Germany', 'France', 'Italy', 'Netherlands', 'Switzerland', 'Portugal', 'Austria', 'Japan', 'Spain', 'Lebanon', 'Saudi Arabia', 'Taiwan Province of China'])]
diffplus = diffplus.sum()


# In[ ]:


diffUKUS = var_til1972.loc[var_til1972['Country Name'].isin(['United Kingdom', 'United States'])]
diffUKUS = diffUKUS.sum()
diffUKUS = diffUKUS * (-1) #multiply by -1 to convert value to positive to have a better visualization in the plot


# In[ ]:


diffcomp = pd.DataFrame([diffplus.Variation, diffUKUS.Variation])


# In[ ]:


diffcomp = diffcomp.rename({0:'Variation'}, axis='columns')
diffcomp['Groups'] = ['Rest','UK US']


# In[ ]:


diffcomp


# In[ ]:


fig = px.bar(diffcomp, x='Groups', y='Variation', title='Comparation of increasing and decreasing countries from 1950 to 1972', color='Groups')

fig.update_layout(xaxis=dict(title='Group of countries', showgrid=False), yaxis=dict(title='Tonnes of Gold', showgrid=True, gridcolor='grey'), paper_bgcolor='white', plot_bgcolor='white')
fig.show()


# ### (Notice that the 'UK US' value was converted to positive to have an easier overview in the plot)

# The sum of the increasing countries (13873 tonnes) and the UK US decrease (13125 tonnes) is pretty similar between 1950 and 1972. Could that be related?
