#!/usr/bin/env python
# coding: utf-8

# **BAR CHART RACE with MATPLOTLIB LIBRARY**
# 
# PRELIMINARY INFO
# This notebook creates a bar chart race with Matplotlib library. 
# The idea of bar chart race emerged with [the tweet of Matt Navarra](https://twitter.com/MattNavarra/status/1098580810062618627). 
# 
# Then, John Burn-Murdoch released [his reproducible notebook](https://observablehq.com/@johnburnmurdoch/bar-chart-race-the-most-populous-cities-in-the-world) using d3.js
# 
# Pratap Vardhan released [his codes](https://pratapvardhan.com/blog/bar-chart-race-python-matplotlib/) for visualising a bar chart race using Python's Matplotlib library. 
# 
# In this notebook, I adopted Vardhan's understanding; however, I edited a lot in order to visualise my data better. 
# 
# DATA
# Added CSV file contains the population of 81 Turkish provinces between 2007 and 2018. This file is extracted from Turkish Institute of Statistics databases.
# 
# INTERPRETATION
# The visualisation of a bar chart race is not as charming as Murdoch's code and [Flourish Team's template](https://app.flourish.studio/@flourish/bar-chart-race), since Matplotlib generates less dynamic sequences.
# 
# CONTRIBUTIONS and COMMENTS
# Any contribution is welcomed in order to make this and similar visualisations better. 
# 

# In[ ]:


# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML


# READING CSV FILE

# In[ ]:


df= pd.read_csv("../input/provinces.csv",encoding='latin1')
df.head()



# Let's set the current year as 2018 and list the 15 most populous provinces in descending order. 

# In[ ]:


current_year = 2018
dff = df[df['year'].eq(current_year)].sort_values(by='population', ascending=False).head(15)
dff


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))
ax.barh(dff['province'], dff['population'])
colors = dict(zip(
    ['Marmara Region','Aegean Region','Mediterranean Region','Central Anatolia Region','Black Sea Region','Eastern Anatolia Region','Southeast Anatolia Region'],
    ['#adb0ff', '#ffb3ff', '#90d595', '#e48381', '#aafbff', '#f7bb5f', '#eafb50']
))
group_lk = df.set_index('province')['region'].to_dict()


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))
dff = dff[::-1]   # flip values from top to bottom
# pass colors values to `color=`
ax.barh(dff['province'], dff['population'], color=[colors[group_lk[x]] for x in dff['province']])
# iterate over the values to plot labels and values (Tokyo, Asia, 38194.2)
for i, (value, name) in enumerate(zip(dff['population'], dff['province'])):
    ax.text(value, i,     name,            ha='right')  # Tokyo: name
    ax.text(value, i-.25, group_lk[name],  ha='right')  # Asia: group name
    ax.text(value, i,     value,           ha='left')   # 38194.2: value
# Add year right middle portion of canvas
ax.text(1, 0.4, current_year, transform=ax.transAxes, size=46, ha='right')


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 16))

def draw_barchart(year):
    dff = df[df['year'].eq(year)].sort_values(by='population', ascending=True).tail(20)
    ax.clear()
    ax.barh(dff['province'], dff['population'], color=[colors[group_lk[x]] for x in dff['province']])
    dx = dff['population'].max() / 200
    for i, (value, name) in enumerate(zip(dff['population'], dff['province'])):
        ax.text(value-dx, i,     name,           size=14, weight=800, ha='right', va='bottom')
        ax.text(value-dx, i-.25, group_lk[name], size=8, color='#444444', ha='right', va='baseline')
        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')
    # ... polished styles
    ax.text(1, 0.4, year, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=500)
    ax.text(0, 1.06, 'Population', transform=ax.transAxes, size=12, color='#777777')
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.12, 'The most populous provinces in Turkey from 2007 to 2018',
            transform=ax.transAxes, size=24, weight=600, ha='left')
    ax.text(1, 0, 'by Can Iban', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))
    plt.box(False)
    
draw_barchart(2018)


# In[ ]:


import matplotlib.animation as animation
from IPython.display import HTML

fig, ax = plt.subplots(figsize=(15, 16))
animator = animation.FuncAnimation(fig, draw_barchart, frames=range(2007, 2018))
HTML(animator.to_jshtml()) 

