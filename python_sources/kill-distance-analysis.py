#!/usr/bin/env python
# coding: utf-8

# # PUBG Kill Distance Analysis

# In this notebook, we explore the distribution of PUBG kill distances, anomalies in recorded kills and best areas to snipe / worst areas to be sniped.

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
from scipy.misc.pilutil import imread
plt.rcParams['figure.figsize'] = (10, 6)
style.use('ggplot')


# In[2]:


from bokeh.plotting import figure, show, output_file, output_notebook
from bokeh.palettes import brewer
from bokeh.models.tools import HoverTool
from bokeh.models.sources import ColumnDataSource

from collections import Counter


# We first work on a smaller sample, loading only a fraction of the PUBG kill data:

# In[3]:


sample = pd.read_csv("../input/pubg-match-deaths/deaths/kill_match_stats_final_0.csv")


# As we see, even in this sample, there are a huge number of recorded kills.

# In[4]:


sample.shape


# We want to calculate the kill distance of all kills and examine the distance distribution. Let us focus on one map at a time. We start with Miramar and only take an even smaller sample first.

# In[5]:


# small = sample.loc[(sample['map'] == 'MIRAMAR') & (sample['killer_position_x'] != 0)
#                   & (sample['victim_position_x'] != 0)][:10000].copy()
small = sample.loc[(sample['map'] == 'MIRAMAR')][:10000].copy()

x_diffs = small['killer_position_x'] - small['victim_position_x']
y_diffs = small['killer_position_y'] - small['victim_position_y']
sq_diffs = x_diffs ** 2 + y_diffs ** 2
dists = np.sqrt(sq_diffs)
log_dists = np.log10(1 + dists)
small.head()


# Let us look at the distribution of kill distances:

# In[6]:


sns.distplot(log_dists.dropna());


# As we see above, there are two anomalies in the distribution, concentrated at distance=0 and distance > 10^5. As we do not yet know the conversion rate from coordinate distance to on-map distance, so far we cannot verify if these kills are legitimate. Let us inspect them by looking at the weapons used in these kills.

# We want to know what weapons are responsible for the outliers in the graph, so we select the section of the distribution and count the kills by each weapon.

# In[7]:


small.loc[log_dists < 1, :].groupby('killed_by').size().plot.bar();


# As we see, most of the deaths here are caused by game physics or sustained effects. It is understandable that in the game logic, these kills might be treated as "suicides" or "instant deaths without a killer", therefore the kill distance of 0 is understandable.

# In[8]:


small.loc[log_dists > 5, :].groupby('killed_by').size().plot.bar();


# On the other extreme, among the longest distance kills, we found some unreasonable kills. Many of the long distance kills are done by melee attacks such as "Punch" or even "Pan"! How is this possible? Let us look at exactly how far the attackers and victims are for these kills.

# To do this, we will first create a visualisation function for examining individual kills. We load up a high-resolution image of the in-game map as the background.

# In[9]:


bg = imread("../input/high-res-map-miramar/miramar-large.png")
fig, ax = plt.subplots(figsize=(12, 12))
ax.imshow(bg);


# Then we define the actual function, which shows an arrow pointing from the attacker to the victim on the map.

# In[10]:


def kill_viz(killer_pos, victim_pos, bg, bg_width, bg_height, zoom=False, figsize=(15, 15)):
    fig, ax = plt.subplots(figsize=figsize)
    x, y = killer_pos
    tx, ty = victim_pos
    dx, dy = tx - x, ty - y
    x *= bg_width / 800000
    y *= bg_height / 800000
    dx *= bg_width / 800000
    dy *= bg_height / 800000
    ax.imshow(bg)
    arrow_width = min((abs(dx) + abs(dy)), (bg_width + bg_height) * 0.2) * 0.02
    if zoom:
        edge = max(abs(dx), abs(dy))
        ax.set_xlim(max(0, min(x, x + dx) - edge * 5), min(max(x, x + dx) + edge * 5, bg_width))
        ax.set_ylim(min(max(y, y + dy) + edge * 5, bg_width), max(min(y, y + dy) - edge * 5, 0))
    ax.arrow(x, y, dx, dy, width=arrow_width, color='r', length_includes_head=True)
    plt.show()


# Let us visualise some of the kills in the ultra-long-distance kill cluster to see what is going on:

# In[11]:


temp = small.loc[(log_dists > 5)].iloc[0]
kp = (temp['killer_position_x'], temp['killer_position_y'])
vp = (temp['victim_position_x'], temp['victim_position_y'])
print(temp)


# In[12]:


kill_viz(kp, vp, bg, 8192, 8292, zoom=True, figsize=(10, 10))


# In[13]:


temp = small.loc[(log_dists > 5)].iloc[50]
kp = (temp['killer_position_x'], temp['killer_position_y'])
vp = (temp['victim_position_x'], temp['victim_position_y'])
print(temp)


# In[14]:


kill_viz(kp, vp, bg, 8192, 8292, zoom=True, figsize=(10, 10))


# In[15]:


temp = small[small['killer_position_x'] == 0].loc[(log_dists > 5)].iloc[0]
kp = (temp['killer_position_x'], temp['killer_position_y'])
vp = (temp['victim_position_x'], temp['victim_position_y'])
print(temp)


# In[16]:


kill_viz(kp, vp, bg, 8192, 8292, zoom=True, figsize=(10, 10))


# We found that almost all kills in this cluster have a killer or victim at coordinate (0, 0). We do not know if (0, 0) has any significance in the game as of now (maybe players glitched to that location for some reason, for instance), but it is likely that (0, 0) may represent a missing or improperly collected value in the dataset. Let us remove all kills with either the killer or victim at (0, 0) and look at the kill distance distribution again:

# In[17]:


small = sample.loc[(sample['map'] == 'MIRAMAR') & (sample['killer_position_x'] != 0)
                  & (sample['victim_position_x'] != 0)][:10000].copy()

x_diffs = small['killer_position_x'] - small['victim_position_x']
y_diffs = small['killer_position_y'] - small['victim_position_y']
sq_diffs = x_diffs ** 2 + y_diffs ** 2
dists = np.sqrt(sq_diffs)
log_dists = np.log10(1 + dists)
small['log_dist'] = log_dists


# In[18]:


sns.distplot(log_dists.dropna());


# As we see, the ultra-long-distance cluster disappeared. So it is indeed coordinate (0, 0) that is causing all the anomalies. How we verify that the longest distance kills are reasonable:

# In[19]:


small.loc[log_dists > 4.5, :].groupby('killed_by').size().plot.bar();


# As we see here, after the data cleaning, the top kills are indeed ahieved by the typical long-distance weapons. Let us visualise the longest distance kill to verify that it is legitimate:

# In[20]:


temp = small.loc[(log_dists == np.max(log_dists))].iloc[0]
kp = (temp['killer_position_x'], temp['killer_position_y'])
vp = (temp['victim_position_x'], temp['victim_position_y'])
print(temp)


# In[21]:


kill_viz(kp, vp, bg, 8192, 8292, zoom=True, figsize=(12, 12))


# As we see on the map, it is pretty extreme but not what we would consider impossible.

# We now create a stacked area plot of kill distance distribution by weapon type.

# In[22]:


top_weapons = list(small[small['killed_by'] != 'Down and Out'].groupby('killed_by').size()                     .sort_values(ascending=False)[:10].index)
top_weapon_kills = small[np.in1d(small['killed_by'], top_weapons)].copy()
top_weapon_kills['bin'] = pd.cut(top_weapon_kills['log_dist'], np.arange(0, 6.2, 0.2), include_lowest=True, labels=False)
top_weapon_kills_wide = top_weapon_kills.groupby(['killed_by', 'bin']).size().unstack(fill_value=0).transpose()


# In[24]:


def  stacked(df):
    df_top = df.cumsum(axis=1)
    df_bottom = df_top.shift(axis=1).fillna(0)[::-1]
    df_stack = pd.concat([df_bottom, df_top], ignore_index=True)
    return df_stack

hover = HoverTool(
    tooltips=[
            ("index", "$index"),
            ("weapon", "@weapon"),
            ("(x,y)", "($x, $y)")
        ],
    point_policy='follow_mouse'
    )

areas = stacked(top_weapon_kills_wide)

colors = brewer['Spectral'][areas.shape[1]]
x2 = np.hstack((top_weapon_kills_wide.index[::-1],
                top_weapon_kills_wide.index)) / 5

TOOLS="pan,wheel_zoom,box_zoom,reset,previewsave"
output_notebook()
p = figure(x_range=(1, 5), y_range=(0, 800), tools=[TOOLS, hover], plot_width=800)
p.grid.minor_grid_line_color = '#eeeeee'

source = ColumnDataSource(data={
    'x': [x2] * areas.shape[1],
    'y': [areas[c].values for c in areas],
    'weapon': list(top_weapon_kills_wide.columns),
    'color': colors
})

p.patches('x', 'y', source=source, legend="weapon",
          color='color', alpha=0.8, line_color=None)
p.title.text = "Distribution of Kill Distance per Weapon"
p.xaxis.axis_label = "log10 of kill distance"
p.yaxis.axis_label = "number of kills"

show(p)


# In[ ]:




