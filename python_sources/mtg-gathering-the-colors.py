#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# In this notebook, we give a high-level look into the 5 core mana colors.
# 
# - Settings
# - Read Data
# - Remove Outliers
# - Preprocess Colors
# - Co-occurrences
# - Correlation Heatmap
# - Interactive Time Series of Releases
# - Cumulative Color Proportion Over Time *(New in version 13)*
# - Mana distribution
# - Power distribution
# - Toughness distribution
# - Interactive 2D Histogram of Power vs Toughness
# - Radar charts

# <a id='settings'></a>
# ### Settings
# We define a mapping between the colorIdentity and color columns so that we can combine the information from both when we categorize cards by color.
# 
# And to reduce data size, we limit the data to columns specified in keeps.

# In[ ]:


colorIdentity_map = {'B': 'Black', 'G': 'Green', 'R': 'Red', 'U': 'Blue', 'W': 'White'}
keeps = ['name', 'colorIdentity', 'colors', 'type', 'types', 'cmc', 'power', 'toughness', 'legalities']


# In[ ]:


import pandas as pd
import numpy as np
from numpy.random import random
from math import ceil

import ggplot as gg
from ggplot import aes

import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.spines import Spine
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from plotly import tools

import seaborn as sns
init_notebook_mode()


# <a id='read'></a>
# ### Read Data

# In[ ]:


raw = pd.read_json('../input/AllSets-x.json')
raw.shape


# In[ ]:


# Some data-fu to get all cards into a single table along with a couple of release columns.
mtg = []
for col in raw.columns.values:
    release = pd.DataFrame(raw[col]['cards'])
    release = release.loc[:, keeps]
    release['releaseName'] = raw[col]['name']
    release['releaseDate'] = raw[col]['releaseDate']
    mtg.append(release)
mtg = pd.concat(mtg)
del release, raw   
mtg.shape


# <a id='outliers'></a>
# ### Remove Outliers

# In[ ]:


# remove promo cards that aren't used in normal play
# Edit 2016-09-30: Could be null because it's too new to have a ruling on it.
mtg_nulls = mtg.loc[mtg.legalities.isnull()]
mtg = mtg.loc[~mtg.legalities.isnull()]

# remove cards that are banned in any game type
mtg = mtg.loc[mtg.legalities.apply(lambda x: sum(['Banned' in i.values() for i in x])) == 0]
mtg = pd.concat([mtg, mtg_nulls])
mtg.drop('legalities', axis=1, inplace=True)
del mtg_nulls

# remove tokens without types
mtg = mtg.loc[~mtg.types.apply(lambda x: isinstance(x, float))]

# Power and toughness that depends on board state or mana cannot be resolved
mtg[['power', 'toughness']] = mtg[['power', 'toughness']].apply(lambda x: pd.to_numeric(x, errors='coerce'))

mtg.shape


# <a id='preprocess'></a>
# ### Preprocess Color
# A core part of Magic: The Gathering (MTG) is how different rules interact with each other.
# For simple visualizations, we have to make some simplifications.
# We'll assign colors to exhibits A and B below but not to C.3.
# 
# - Exhibit A: the edge case when colors is missing but colorIdentity is not:
#     1. Eldrazi creatures that need a specific color mana to cast but is Deviod of a color type.
#     2. Artifacts that do not need colored mana to cast but activations associated with colored mana.
# 
# 
# - Exhibit B: the edge case when colorIdentity is missing but colors is not:
#     1. I think this is dirty data.  I'm not a MTG expert so maybe there is some rule I'm not aware of.
#     2. For example, Serra Angel needs white mana to cast and should count as white for Protection From White.
#     3. For example, Counterspell seems to me like the typical blue spell.
# 
# 
# - Exhibit C: the edge case when both colors and colorIdentity are missing:
#     1. Truly colorless artifacts like Leonin Scimitar.
#     2. Truly colorless artifact creatures like Juggernaut.
#     3. Eldrazi creatures that need a specific color mana to cast but is Deviod of a color type.

# In[ ]:


# Combine colorIdentity and colors
mtg.loc[(mtg.colors.isnull()) & (mtg.colorIdentity.notnull()), 'colors'] = mtg.loc[(mtg.colors.isnull()) & (mtg.colorIdentity.notnull()), 'colorIdentity'].apply(lambda x: [colorIdentity_map[i] for i in x])
mtg['colorsCount'] = 0
mtg.loc[mtg.colors.notnull(), 'colorsCount'] = mtg.colors[mtg.colors.notnull()].apply(len)
mtg.loc[mtg.colors.isnull(), 'colors'] = ['Colorless']
mtg['colorsStr'] = mtg.colors.apply(lambda x: ''.join(x))

# Include colorless and multi-color.
mtg['manaColors'] = mtg['colorsStr']
mtg.loc[mtg.colorsCount>1, 'manaColors'] = 'Multi'


# <a id='co_occurrences'></a>
# ### Co-occurrences

# In[ ]:


mono_colors = np.sort(mtg.colorsStr[mtg.colorsCount==1].unique()).tolist()

for color in mono_colors:
    mtg[color] = mtg.colors.apply(lambda x: color in x)

corr = [mtg.loc[(mtg[color] == True), mono_colors].apply(np.sum) for color in mono_colors]
corr = pd.DataFrame(corr, index=mono_colors)
corr


# <a id='heatmap'></a>
# ### Correlation Heatmap
# Easier to see that Blue and Green combos are relatively rare.
# 
# And Black-Blue and Green-White are more frequent.

# In[ ]:


corr = [mtg.loc[(mtg[color] == True), mono_colors].apply(np.mean) for color in mono_colors]
corr = pd.DataFrame(corr, index=mono_colors)

vmax = ceil(max(corr[corr<1].max()) * 100) / 100
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    ax = sns.heatmap(corr, mask=mask, vmax=vmax, square=True)


# <a id='ts'></a>
# ### Interactive Time Series of Releases
# We can look at the color distribution over time (releases).  In the middle of 2007, Wizards of the Coast changed its strategy.  It started to release more frequently with less cards per release.
# 
# Recommened use:
# - zoom in for a better resolution
# - exit zoom mode into pan mode
# - hover over the markers for counts

# In[ ]:


colors = np.sort(mtg.manaColors.unique()).tolist()
plotly_colors = ['rgb(100,100,100)', 'rgb(70,160,240)', 'rgb(175, 175, 175)', 'rgb(100,200,25)', 'rgb(150, 100, 150)', 'rgb(250,70,25)', 'rgb(225,225,175)']

# get counts
piv = pd.pivot_table(mtg, values='name', index='manaColors', columns='releaseDate', aggfunc=len)
piv.fillna(0, inplace=True)

traces = [go.Scatter(
    x = piv.columns.tolist(),
    y = piv[piv.index==color].iloc[0].tolist(),
    mode = 'lines',
    line = dict(color=plotly_colors[i]),
    connectgaps = True,
    name = color
) for i, color in enumerate(colors)]

layout = go.Layout(
    xaxis=dict(showline=True, showgrid=False),
    yaxis=dict(showgrid=True)
)

fig = go.Figure(data=traces, layout=layout)
iplot(fig, filename='Colors Over Time')


# <a id='color_proportion'></a>
# ### Cumulative Color Proportion Over Time
# We can also look at the cumulative color proportions over time (releases).  I used cumulative proportions to smooth over the variations that could be present in individual releases.
# 
# Notice how there were less multi-colored cards in the beginning compared to colorless cards.  In 2008, multi-colored decks became better established with clearly defined play styles and decks.  Since then, the color distribution has been fairly stable.

# In[ ]:


plt_colors = ['k', 'b', '0.5', 'g', 'm', 'r', 'w']
piv = piv.cumsum(axis=1)
piv = piv.div(piv.sum(axis=0), axis=1)
piv.transpose().plot(kind='area', stacked=True, figsize=(12, 12), title='Cumulative Color Proportion Over Time', color=plt_colors)


# ### Creatures

# In[ ]:


creatures = mtg.loc[
    (mtg.type.str.contains('Creature', case=False))    
    & (mtg.cmc.notnull())
    & (mtg.power.notnull())
    & (mtg.toughness.notnull())
    & (0 <= mtg.cmc) & (mtg.cmc < 16)
    & (0 <= mtg.power) & (mtg.power < 16)
    & (0 <= mtg.toughness) & (mtg.toughness < 16)    
    , ['name', 'cmc', 'power', 'toughness', 'releaseName', 'releaseDate', 'manaColors']
]

creatures['cmc'] = round(creatures['cmc'])
creatures['power'] = round(creatures['power'])
creatures['toughness'] = round(creatures['toughness'])
creatures.shape


# <a id='hist_mana'></a>
# ### Mana distribution
# X-axis is a little weird.  Most start with 1 mana.  Exception is Colorless, which start at 0.  There should not be a gap at 0.5. 
# 
# Blue has relatively little mana 1 creatures.
# 
# Red has relatively low number of creatures with 2 mana but more with 3 or 4 mana.
# 
# Green's creatures are spread across the mana costs
# 
# This is consistent with its reliance on creatures in its strategy.
# 
# Multi-color creatures tend to cost more to cast since they usually have better abilities.

# In[ ]:


gg.ggplot(gg.aes(x='cmc', color='manaColors'), data=creatures) + gg.geom_bar(size=1) + gg.facet_wrap('manaColors') 


# <a id='hist_power'></a>
# ### Power distribution
# X-axis starts at power 0.
# 
# White has lots of power == 2, which are probably variations of humans.
# 
# In addition, since it relies more on enchantments, the inherent power of the creatures are not as high.
# 
# Green has more power >= 4, which is expected since it relies more on creatures to win.
# 
# Red has relatively thicker tail, which is expected since it emphasizes attack.
# 
# Unlike other colors, blue peaks earlier at power = 1.
# 
# This is further evidence of its reliance on instants and sorceries instead of creatures.
# 
# There are also relatively more colorless artifacts with power 0.

# In[ ]:


gg.ggplot(gg.aes(x='power', color='manaColors'), data=creatures) + gg.geom_bar(size=1) + gg.facet_wrap('manaColors')


# <a id='hist_toughness'></a>
# ### Toughness distribution
# X-axis starts at toughness 0.
# 
# Green has more toughness >= 5, which is expected since it relies more on creatures for protection.
# 
# Black and red have more toughness <= 2.
# 
# Black has more expendable creatures, congruent with its abilities to interact with the graveyard.
# 
# Red emphasizes more on speed (mana cost) and attack instead of sustainability.
# Notice how multi-colored creatures tend to survive longer.

# In[ ]:


gg.ggplot(gg.aes(x='toughness', color='manaColors'), data=creatures) + gg.geom_bar(size=1) + gg.facet_wrap('manaColors') 


# <a id='hist_2d'></a>
# ### Interactive 2D Histogram of Power vs Toughness
# I could not get the colors to work correctly but the hover text is alphabetical by color.

# In[ ]:


data = []

for i, color in enumerate(colors):
    trace = go.Histogram2dContour(
        x = creatures.loc[creatures.manaColors == color, 'power'],    
        y = creatures.loc[creatures.manaColors == color, 'toughness'],  
        autobinx = False,
        xbins = dict(start=0, end=16, size=1),
        autobiny = False,
        ybins = dict(start=0, end=16, size=1),        
        #colorscale=[[i, 'rgb({}, {}, {})'.format(250-i*200, 250-i*200, 250-i*200)] for i in np.linspace(0, 1, 10)],
        name = color,
        hoverinfo = 'z',        
        showscale=False
    )
    data.append(trace)
	
layout = go.Layout(
    title='Creatures Attack and Defense by Color',
    xaxis=dict(
        showgrid=True,
        gridcolor='rgb(99, 99, 99)',
        gridwidth=5,
        autotick=False,
        tick0=0,
        dtick=1
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor='rgb(99, 99, 99)',
        gridwidth=5,
        autotick=False,
        tick0=0,
        dtick=1
    ),
)

del creatures
fig = go.Figure(data=data, layout=layout)
iplot(fig, filename='Creatures Attack and Defense by Color')


# ### Radar chart factory
# Code from http://matplotlib.org/examples/api/radar_chart.html

# In[ ]:


def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        # unit circle centered on (0.5, 0.5)
        return plt.Circle((0.5, 0.5), 0.5)

    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1
        # define draw_frame method
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
            """Override fill so that line is closed by default"""
            closed = kwargs.pop('closed', True)
            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts


# <a id='radar_chart'></a>
# ### Radar Chart
# We will look at how the distribution of card types varies by mana color.  Each radar shows the distribution of cards into 5 types with every circle denoting 10%.

# In[ ]:


spoke_labels = ['Artifact', 'Creature', 'Enchantment', 'Instant', 'Sorcery']
for col in spoke_labels:
    mtg[col] = mtg.types.apply(lambda x: col in x)

# Do not double count creatures
# Some double counting of artifacts and enchantments exist but it's rare.
mtg['Creature'] = mtg['Creature'] & ~(mtg['Artifact'] | mtg['Enchantment'])

# Is there a better way of grouping?
grouped = mtg.loc[(mtg.types.apply(lambda x: len(set(x).intersection(set(spoke_labels))))>0)].groupby('colorsStr')
data = grouped[spoke_labels].agg(np.mean)

theta = radar_factory(len(spoke_labels), frame='polygon')
cols = 3


# #### Radar Charts - Mono color decks
# Colorless cards are more or less artifacts.
# 
# Green focuses on creatures.
# 
# Blue has less creatures and more instant spells to control the board state.
# White has relatively more enchantments to buff the creatures.

# In[ ]:


colors = np.sort(mtg.colorsStr[mtg.colorsCount<=1].unique()).tolist()
rows = ceil(len(colors) / cols)
fig = plt.figure(figsize=(10, rows*5))
fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.5, bottom=0.05)

for n, color in enumerate(colors):
    ax = fig.add_subplot(rows, cols, n + 1, projection='radar')
    plt.rgrids(np.linspace(.1, 1, 10).tolist())
    ax.set_title(color, weight='bold', size='medium', position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')

    dat = data.loc[data.index==color, :].iloc[0].tolist()
    if color != 'Colorless':        
        ax.plot(theta, dat, color=color.lower())
        ax.fill(theta, dat, facecolor=color.lower(), alpha=0.5)
    else:
        ax.plot(theta, dat, color='grey')
        ax.fill(theta, dat, facecolor='grey', alpha=0.5)
        
    ax.set_varlabels(spoke_labels)


# #### Radar Charts - Dual-color decks
# Values are percentages so 100% artifacts does not mean there are lots of artifacts of that color.

# In[ ]:


colors = np.sort(mtg.colorsStr[mtg.colorsCount==2].unique()).tolist()
colors = [i for i in colors if i in data.index.values]

rows = ceil(len(colors) / cols)
fig = plt.figure(figsize=(10, rows*5))
fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.5, bottom=0.05)

for n, color in enumerate(colors):
    ax = fig.add_subplot(rows, cols, n + 1, projection='radar')
    plt.rgrids(np.linspace(.1, 1, 10).tolist())
    ax.set_title(color, weight='bold', size='medium', position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')

    dat = data.loc[data.index==color, :].iloc[0].tolist()
    ax.plot(theta, dat, color='y')
    ax.fill(theta, dat, facecolor='y', alpha=0.5)
    ax.set_varlabels(spoke_labels)


# #### Radar Charts - Tri-color decks
# Values are percentages so 100% artifacts does not mean there are lots of artifacts of that color.

# In[ ]:


colors = np.sort(mtg.colorsStr[mtg.colorsCount==3].unique()).tolist()
colors = [i for i in colors if i in data.index.values]

rows = ceil(len(colors) / cols)
fig = plt.figure(figsize=(10, rows*5))
fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.5, bottom=0.05)

for n, color in enumerate(colors):
    ax = fig.add_subplot(rows, cols, n + 1, projection='radar')
    plt.rgrids(np.linspace(.1, 1, 10).tolist())
    ax.set_title(color, weight='bold', size='medium', position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')

    dat = data.loc[data.index==color, :].iloc[0].tolist()
    ax.plot(theta, dat, color='y')
    ax.fill(theta, dat, facecolor='y', alpha=0.5)
    ax.set_varlabels(spoke_labels)


# #### Radar Charts - Multi-color decks
# Values are percentages so 100% artifacts does not mean there are lots of artifacts of that color.

# In[ ]:


colors = np.sort(mtg.colorsStr[mtg.colorsCount>3].unique()).tolist()
colors = [i for i in colors if i in data.index.values]

rows = ceil(len(colors) / cols)
fig = plt.figure(figsize=(10, rows*5))
fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.5, bottom=0.05)

for n, color in enumerate(colors):
    ax = fig.add_subplot(rows, cols, n + 1, projection='radar')
    plt.rgrids(np.linspace(.1, 1, 10).tolist())
    ax.set_title(color, weight='bold', size='medium', position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')

    dat = data.loc[data.index==color, :].iloc[0].tolist()
    ax.plot(theta, dat, color='y')
    ax.fill(theta, dat, facecolor='y', alpha=0.5)
    ax.set_varlabels(spoke_labels)

