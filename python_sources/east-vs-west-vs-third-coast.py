#!/usr/bin/env python
# coding: utf-8

# # East- vs. West- vs. Third Coast
# 
# I'm wondering about the timeline of hip hop production across the three major geographic regions: when does each coast emerge, and which coasts are more productive than others? 
# 
# First, let's just load up some data.

# In[ ]:


import sqlite3
import pandas as pd
from plotly import tools
from plotly.offline import init_notebook_mode, iplot
from plotly.graph_objs import Bar, Scatter, Figure, Layout
init_notebook_mode()

# get the data...
con = sqlite3.connect('../input/database.sqlite')
tags = pd.read_sql_query('SELECT * from tags;', con)
torrents = pd.read_sql_query('SELECT * from torrents;', con)
con.close()


# To measure the broader output of each coast, we'll need to aggregate across cities, states, etc. Luckily, users of What.CD are pretty methodical about tagging, and tags like "east.coast", "west.coast", and "dirty.south" are commonly used. In my experience "west.coast" is by far the most commonly used (I think because the west coast hip hop identity is much stronger than the others), so we'll want to use other tags. 
# 
# I had to use a bunch of city tags for the dirty south since hip hop is found in little pockets across the south (rather than in one or two big cities like on the east and west coasts). Here are the tags:

# In[ ]:


# A list of tags associated with each region. These were handpicked but I think they are fair!
coast_tags = {
    'East Coast': ['new.york', 'east.coast','east.coast.rap'],
    'West Coast': ['bay.area', 'los.angeles', 'west.coast', 'california'],
    'Dirty South': ['dirty.south', 'southern', 'southern.rap','southern.hip.hop', 'new.orleans', 'houston', 'memphis', 'atlanta'],
    }

# Count number of torrents in each tag group
yearly_counts = pd.DataFrame(data = None, columns = ['Year'] + list(coast_tags.keys()))
for year in range(1985, 2017):
    ids = torrents.id.loc[torrents.groupYear==year]
    yeartags = tags.loc[tags.id.isin(ids)]
    
    # create row for dataframe
    row = dict(Year = year)
    for k,v in coast_tags.items():
        releases = yeartags.loc[yeartags.tag.isin(v), 'id']
        row[k] = pd.unique(releases).shape[0]

    # add row
    yearly_counts = yearly_counts.append(row, ignore_index = True)
    


# In[ ]:


linespecs = {
    'East Coast':  dict(color = 'blue', width = 2),
    'West Coast':  dict(color = 'orange', width = 2),
    'Dirty South': dict(color = 'red', width = 2),
    }

handles = []
for k in coast_tags.keys():
    handles.append(
        Scatter(x = yearly_counts.Year, 
                y = yearly_counts[k], 
                name = k, 
                hoverinfo = 'x+name',
                line = linespecs[k]
               )
        )
    
layout = Layout(
    xaxis = dict(tickmode = 'auto', 
                 nticks = 20, 
                 tickangle = -60, 
                 showgrid = False),
    yaxis = dict(title = 'What.CD Tag Frequency'),
    hovermode = 'closest',
    legend = dict(x = 0.05, y = 0.9),
)

fh = Figure(data=handles, layout=layout)
iplot(fh)


# ## Results
# 
# What's most interesting to me is the giant bump from 1994-1996. The emergence of the Dirty South is particularly profound: it has virtually zero tags prior to 1991, but clearly overcomes the East and West coasts as of 1995 (and holds a lead until 1998). This is especially striking because the East-vs-West coast rivalry was most heated from 1994 to 1997, which is when you'd expect releases to be most strongly associated with their respective coasts. 
# 
# We do see a bump along both the East and West coast during that period, but this is also when the Dirty South identity also comes into play:  [*Southernplayalisticadillacmuzik*](https://en.wikipedia.org/wiki/Southernplayalisticadillacmuzik), which finally gave the Dirty South some credibility, was released in 1994. So, really, you can view 1994-1996 as the period in which hip hop was most strongly associated with its geographic origins. 
