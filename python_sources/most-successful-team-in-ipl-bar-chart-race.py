#!/usr/bin/env python
# coding: utf-8

# # Bar Chart Race in Python with Matplotlib
# 
# Inspiration & Source : https://colab.research.google.com/github/pratapvardhan/notebooks/blob/master/barchart-race-matplotlib.ipynb#scrollTo=sLZ1E6X7xLQg
# 
# ![gif](https://pratapvardhan.com/img/b/bar-chart-race-2.gif)

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from IPython.display import HTML


# ### Reading & Transforming Data

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


matches = pd.read_csv("../input/ipldata/matches.csv")
matches.head()


# In[ ]:


usecols=['date', 'winner']
winners = matches[usecols]
winners.head()


# In[ ]:


winners['date'] =pd.to_datetime(winners['date'])
winners = winners.sort_values(by='date')
winners.head()


# In[ ]:


winners = winners.reset_index(drop=True)
winners.head()


# In[ ]:


# Get the cumulative counts.
counts = pd.get_dummies(winners['winner']).cumsum()

# Rename the count columns as appropriate.
#counts = counts.rename(columns=lambda col: col+'_count')

# Join the counts to the original df.
winners = winners.join(counts)

winners.head(15)


# In[ ]:


del winners['winner']
winners.head()


# In[ ]:


winners["id"] = winners.index
df = pd.melt(winners, id_vars=['id','date'], var_name='name', value_name='value')
df.head(20)


# In[ ]:


df = df.sort_values(by='date')
df.head(20)


# Run below cell `draw_barchart(2018)` draws barchart for `year=2018`

# In[ ]:


df.date = df.date.astype(str)
df.head()


# In[ ]:


df.tail()


# In[ ]:


dates = df.date.unique().tolist()
dates[0:5]


# In[ ]:


dates[-5:-1]


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))

def draw_barchart(current_date):
    dff = df[df['date'].eq(current_date)].sort_values(by='value', ascending=True).tail(5)
    ax.clear()
    ax.barh(dff['name'], dff['value'])
    dx = dff['value'].max() / 200
    for i, (value, name) in enumerate(zip(dff['value'], dff['name'])):
        ax.text(value-dx, i,     name,           size=14, weight=600, ha='right', va='bottom')
        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')
    ax.text(1, 0.4, current_date, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_ticks_position('top')
    ax.tick_params(axis='x', colors='#777777', labelsize=12)
    ax.set_yticks([])
    ax.margins(0, 0.01)
    ax.grid(which='major', axis='x', linestyle='-')
    ax.set_axisbelow(True)
    ax.text(0, 1.15, 'The most successful teams in IPL from 2008 to 2019',
            transform=ax.transAxes, size=24, weight=600, ha='left', va='top')
    plt.box(False)
    
draw_barchart('2008-04-18')


# In[ ]:


fig, ax = plt.subplots(figsize=(15, 8))
animator = animation.FuncAnimation(fig, draw_barchart, frames = dates )
HTML(animator.to_jshtml())
# or use animator.to_html5_video() or animator.save() 

