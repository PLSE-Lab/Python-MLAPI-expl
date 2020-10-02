#!/usr/bin/env python
# coding: utf-8

# **INTRODUCTION**
# ----------------
# 
# *Magic: The Gathering* (MTG; also known as Magic) is one of the most popular trading card games in the world. First published in 1993 by Wizards of the Coast, it has become a benchmark of the modern fantasy games, getting more than 20 million players around the world (data from 2015).
# 
# In this brief overview, we'll take a look into the engine that drives this game.

# **INDEX OF CONTENTS**
# ---------------------
# 
#  - [Settings][1]
#  - [Importing the data][2]
#  - [Formatting the data][3]
#  - [Data cleaning][4]
#  - [Number of cards][5]
# * [Unique cards][6]
# * [Cards by color][7]
# * [Cards by type][8]
#  - [Playability][9]
# * [Converted Mana Cost (CMC)][10]
#  - [Battle potential][11]
# * [Power][12]
# * [Toughtness][13]
# * Battle strategy: Power vs Toughness
# 
# 
#   [1]: https://www.kaggle.io/svf/409761/4ef5f75ac36b1041b5f0305c514de23c/__results__.html#SETTINGS
#   [2]: https://www.kaggle.io/svf/409761/4ef5f75ac36b1041b5f0305c514de23c/__results__.html#DATA-IMPORTATION
#   [3]: https://www.kaggle.io/svf/409761/4ef5f75ac36b1041b5f0305c514de23c/__results__.html#FORMATING-THE-DATA
#   [4]: https://www.kaggle.io/svf/409761/4ef5f75ac36b1041b5f0305c514de23c/__results__.html#DATA-CLEANING
#   [5]: https://www.kaggle.io/svf/412587/caa961ce745c1f6eea5b66f6ffad771d/__results__.html#NUMBER-OF-CARDS
#   [6]: https://www.kaggle.io/svf/419209/540e9ed055fc5fe86e3c9ed4c34dc6cb/__results__.html#Unique-cards
#   [7]: https://www.kaggle.io/svf/419209/540e9ed055fc5fe86e3c9ed4c34dc6cb/__results__.html#Cards-by-colors
#   [8]: https://www.kaggle.io/svf/419209/540e9ed055fc5fe86e3c9ed4c34dc6cb/__results__.html#Cards-by-type
#   [9]: https://www.kaggle.io/svf/419209/540e9ed055fc5fe86e3c9ed4c34dc6cb/__results__.html#Playability
#   [10]: https://www.kaggle.io/svf/419209/540e9ed055fc5fe86e3c9ed4c34dc6cb/__results__.html#Converted-Mana-Cost-(CMC)
#   [11]: https://www.kaggle.io/svf/419209/540e9ed055fc5fe86e3c9ed4c34dc6cb/__results__.html#Battle-potential
#   [12]: https://www.kaggle.io/svf/419209/540e9ed055fc5fe86e3c9ed4c34dc6cb/__results__.html#Power
#   [13]: https://www.kaggle.io/svf/419209/540e9ed055fc5fe86e3c9ed4c34dc6cb/__results__.html#Toughness

# SETTINGS
# --------
# 
# I'm going to need Pandas, Numpy and Math. I've taken the idea of define a mapping between the colorIdentity and color columns from the work of Willie Liao.

# In[ ]:


#Packages
import pandas as pd
import numpy as np
from numpy.random import random
from math import ceil
from pandas.compat import StringIO
from pandas.io.common import urlopen
from IPython.display import display, display_pretty, Javascript, HTML
from matplotlib.path import Path
from matplotlib.spines import Spine
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
from plotly import tools
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import seaborn as sns


init_notebook_mode()

# Aux variables
colorIdentity_map = {'B': 'Black', 'G': 'Green', 'R': 'Red', 'U': 'Blue', 'W': 'White'}
keeps = ['name', 'colorIdentity', 'colors', 'type', 'types', 'cmc', 'power', 'toughness', 'legalities']


# DATA IMPORTATION
# ----------------

# In[ ]:


raw = pd.read_json('../input/AllSets-x.json')


# FORMATING THE DATA
# ------------------

# In[ ]:


# Data fusion
mtg = []
for col in raw.columns.values:
    release = pd.DataFrame(raw[col]['cards'])
    release = release.loc[:, keeps]
    release['releaseName'] = raw[col]['name']
    release['releaseDate'] = raw[col]['releaseDate']
    mtg.append(release)
mtg = pd.concat(mtg)

del release, raw

# Combine colorIdentity and colors
mtg.loc[(mtg.colors.isnull()) & (mtg.colorIdentity.notnull()), 'colors'] = mtg.loc[(mtg.colors.isnull()) & (mtg.colorIdentity.notnull()), 'colorIdentity'].apply(lambda x: [colorIdentity_map[i] for i in x])
mtg['colorsCount'] = 0
mtg.loc[mtg.colors.notnull(), 'colorsCount'] = mtg.colors[mtg.colors.notnull()].apply(len)
mtg.loc[mtg.colors.isnull(), 'colors'] = ['Colorless']
mtg['colorsStr'] = mtg.colors.apply(lambda x: ''.join(x))

# Include colorless and multi-color.
mtg['manaColors'] = mtg['colorsStr']
mtg.loc[mtg.colorsCount>1, 'manaColors'] = 'Multi'

# Set Date type for the releaseDate
format = '%Y-%m-%d'
mtg['releaseDate'] = pd.to_datetime(mtg['releaseDate'], format=format)


# DATA CLEANING
# -------------
# I think this is a tricky dataset. There are a lot of "peculiar" cases that we have to take in consideration in order to get a clear analysis. I've started from the data tidying of Willie Liao and Donyoe, and I've added some transformations to make the cleaning as complete as possible.

# In[ ]:


# Remove promo cards that aren't used in normal play
mtg_nulls = mtg.loc[mtg.legalities.isnull()]
mtg = mtg.loc[~mtg.legalities.isnull()]

# Remove cards that are banned in any game type
mtg = mtg.loc[mtg.legalities.apply(lambda x: sum(['Banned' in i.values() for i in x])) == 0]
mtg = pd.concat([mtg, mtg_nulls])
mtg.drop('legalities', axis=1, inplace=True)
del mtg_nulls

# Remove tokens without types
mtg = mtg.loc[~mtg.types.apply(lambda x: isinstance(x, float))]

# Transform types to str
mtg.loc[mtg.types.notnull(), 'typesCount'] = mtg.types[mtg.types.notnull()].apply(len)
mtg['typesStr'] = mtg.types.apply(lambda x: ''.join(x))

# Power and toughness that depends on board state or mana cannot be resolved
mtg[['power', 'toughness']] = mtg[['power', 'toughness']].apply(lambda x: pd.to_numeric(x, errors='coerce'))

# Include colorless and multi-color.
mtg['manaColors'] = mtg['colorsStr']
mtg.loc[mtg.colorsCount>1, 'manaColors'] = 'Multi'

# Remove 'Gleemax' and other cards with more than 90 cmc 
mtg = mtg[(mtg.cmc < 90) | (mtg.cmc.isnull())]

# Remove 'Big Furry Monster' and other cards with more than 90 of power and toughness
mtg = mtg[(mtg.power < 90) | (mtg.typesStr != 'Creature')]
mtg = mtg[(mtg.toughness < 90) | (mtg.typesStr != 'Creature')]

# Remove 'Spinal Parasite' and other cards whose power and toughness depends on the number of lands used to cast it
mtg = mtg[(mtg.power > 0) | (mtg.typesStr != 'Creature')]
mtg = mtg[(mtg.toughness > 0) | (mtg.typesStr != 'Creature')]
          
# Remove the duplicated cards
duplicated = mtg[mtg.duplicated(['name'])]
mtg = mtg.drop_duplicates(['name'], keep='first')

# Recode the card type 'Eaturecray' (Atinlay Igpay), which means 'Creature' on Pig-latin
mtg['typesStr'] = mtg['typesStr'].replace('Eaturecray', 'Creature')

cards_recoded_absolutes=(len(mtg[mtg.typesStr=='Vanguard']) + len(mtg[mtg.typesStr=='Scheme']) + len(mtg[mtg.typesStr=='Plane']) + len(mtg[mtg.typesStr=='Phenomenon']) + len(mtg[mtg.typesStr=='Conspiracy']))
cards_recoded_relatives=str(round((((float(len(mtg[mtg.typesStr=='Vanguard']) + len(mtg[mtg.typesStr=='Scheme']) + len(mtg[mtg.typesStr=='Plane']) + len(mtg[mtg.typesStr=='Phenomenon']) + len(mtg[mtg.typesStr=='Conspiracy']))) / float(len(mtg))) * 100), 2))+'%'

# Recode some special card types to 'Other types'
mtg = mtg.replace(['Vanguard', 'Scheme', 'Plane', 'Phenomenon', 'Conspiracy'], 'Other types')

# Transform the multi-choice variable 'types' to a 7-item dichotomized variable
mtg.loc[mtg.types.notnull(), 'typesCount'] = mtg.types[mtg.types.notnull()].apply(len)
mono_types = mtg[mtg.typesCount==1]
mono_types = np.sort(mono_types.typesStr.unique()).tolist()
for types in mono_types:
    mtg[types] = mtg.types.apply(lambda x: types in x)
    
#Transform the multi-choice variable 'colors' to a 5-item dichotomized variable
mono_colors = np.sort(mtg.colorsStr[mtg.colorsCount==1].unique()).tolist()
for color in mono_colors:
    mtg[color] = mtg.colors.apply(lambda x: color in x)


# From the original 31.705 cards, we've removed a total of 1.454 cards (approximatelly the 4,6% of the total) for being too extreme cases, or illegal cards. 
# 
# There's a particular point of this dataset that we need to remark on: in order to keep the dynamic nature of the game, Wizards of the Coast launches new expansions of the game cards every 4-5 months. Each one of these expansions can have between 350 and 143 cards. If all the cards released on each new launch were new (unique) cards, this model of development would be untenable. The solution of Wizards of the Coast to avoid this problem is to re-launch existing charts with a redesign of the illustration of the cards (sometimes this affects the level of rarity of the card itself). This is a great idea to keep the dynamic of the game, but in this dataset the re-designed cards are treated as different (unique) cards, so we have some duplicated cards. I've cleaned this duplicated cards, which represent nearly the 46,9% of the total amount of legal cards.
# 
# Aditionally, we have some particular card types that are very unusual or only available for a particular modallity of game (for example, the card types "Conspiracy", "Vanguard" or "Plane"). These cards represent, approximatelly, the 1,53% of the total amount of unique legal cards, so I've recoded their card type into "Other types". 

# NUMBER OF CARDS
# ===============
# 
# **Unique cards**
# ----------------
# 
# Actually (at date: 23/10/2016) the number of unique, legal cards for the normal game mode is aproximatelly 16.000 cards. We can see this tendency over time:

# In[ ]:


# Get the data
cards_over_time = pd.pivot_table(mtg, values='name',index='releaseDate', aggfunc=len)
cards_over_time.fillna(0, inplace=True)
cards_over_time = cards_over_time.sort_index()

#Create a trace
trace = go.Scatter(x=cards_over_time.index,
                   y=cards_over_time.values)

# Create the range slider
data = [trace]
layout = dict(
    title="Number of new (unique) cards over time",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                    label='YTD',
                    step='year',
                    stepmode='todate'),
                dict(count=1,
                    label='1y',
                    step='year',
                    stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

# Plot the data
fig = dict(data=data, layout=layout)
plotly.offline.iplot(fig)


# How to explore the chart:
# 
#  - Hover over the line to see the counts.
#  - Use the range selector to see the data from a particular date range.
#  - You can also click on the series and slide to the left, or to the right, to create a date range.
#  - Exit the zoom mode by clicking on the "Autoscale" button

# ----------

# **Cards by colors**
# -------------------
# 
# The card colors (and the interaction between them) are one of the core parts of *Magic: The Gathering*. Almost all of the cards can be classificated in one on more colors that represent the *mana* that is used to cast those cards. Usually the cards from the same color have similar characteristics, although some cards from different colors can be combined to create synergies. The decks can have one or more color. That color usually determines the strategies, strengths and weaknesses that deck have.
# 
# The proportion of the cards colors is accurately balanced, as we can see:

# In[ ]:


# Create the list
total_color_freqs=[]

# Get the data for the colors
for i in mono_colors:
    total_color_freqs.append(str(round(float(len(mtg[mtg.colorsStr==i])/len(mtg)*100), 2))+'%')

# Get the data for the colorless cards
total_color_freqs.append(str(round(float(len(mtg[mtg.colors=='Colorless'])/len(mtg)*100), 2))+'%')

# Get the data for the multicolor cards
total_color_freqs.append(str(round(float(len(mtg[mtg.manaColors=='Multi'])/len(mtg)*100), 2))+'%')

#Tidy the data
total_color_freqs = pd.DataFrame(total_color_freqs)
total_color_freqs=total_color_freqs.transpose()
total_color_freqs.columns=['Black', 'Blue', 'Green', 'Red', 'White', 'Colorless', 'Multicolor']

total_color_freqs


#  - There are a few less cards without color, or with more than 1 color.
#  - The proportions between the 5 colors are accurately balanced.
# 
# We can explore this proportions thought the multicolor cards too:

# In[ ]:


#Create a dataframe with one column for each number of colors
freqs_mono=['monocolor',str(len(mtg[mtg.colorsCount==1]))]
freqs_bi=['bicolor',str(len(mtg[mtg.colorsCount==2]))]
freqs_tri=['tricolor',str(len(mtg[mtg.colorsCount==3]))]
freqs_cuatri=['tetracolor',str(len(mtg[mtg.colorsCount==4]))]
freqs=[freqs_mono, freqs_bi, freqs_tri, freqs_cuatri]

#Get the data for each number of colors
count=0
for a in range(1,5):   
    for i in mono_colors:      
        freqs_raw = pd.value_counts(mtg[i][mtg.colorsCount==a].values.flatten())
        freqs_True=freqs_raw[True]
        freqs[count].append(str(round(float(freqs_True*100)/sum(freqs_raw),2))+'%')
    count=count+1

#Create the dataframe
color_freqs = pd.DataFrame(freqs)
color_freqs.columns=['How many colors?', 'Base size', 'Black', 'Blue', 'Green', 'Red', 'White']
color_freqs=color_freqs.set_index('How many colors?')
del color_freqs.index.name

color_freqs


#  - As in the previous case, the proportions between the colors are highly balanced (independently of the number of different colors that the cards have).
#  - The amount of multicolor cards decreases as the number of multiple colors increases.
# 
# We can also explore if this distribution has remained constant over the time:

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

layout = dict(
    title='Number of new (unique) cards by type over time',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                    label='YTD',
                    step='year',
                    stepmode='todate'),
                dict(count=1,
                    label='1y',
                    step='year',
                    stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(),
        type='date'
    )
)

fig = go.Figure(data=traces, layout=layout)
iplot(fig, filename='Colors Over Time')


# I've extracted the code for this chart from the analysis of Willie Liao. The recommended use is the same as in the previous charts. Given the big amount of information present on this chart, I recommend to select the option "1y" and see the data of periods of 1 year. 
# 
#  - In the middle of 1997, the colorless cards experimented its biggest peak of all the history of MTG. The multicolor cards did the same on 2009.
#  - On the last year (Nov 2015 - Sept - 2016) the proportions between the colors have remained steady. On the month of November there were released a few more multicolor new cards, but since February of 2016, Wizzards of the Coast have reduced the production of this kind of cards to balance this effect.

# ----------

# **Cards by type**
# -----------------
# 
# Another important dimension of *Magic: The Gathering* are the types of the cards. Almost all the decks need to have two or more different card types, as each type perform a different role in the game.
# 
# Lets see how the card types interacts with the color dimension:

# In[ ]:


mtg['typesCount'] = 1
mtg.loc[mtg.types.notnull(), 'typesCount'] = mtg.types[mtg.types.notnull()].apply(len)
mtg.loc[mtg.typesCount>1, 'typesStr'] = 'Multi'
types_piv = pd.pivot_table(mtg, values='name',index='manaColors', columns='typesStr', aggfunc=len)
with sns.axes_style("white"):
    ax = sns.heatmap(types_piv)


#  - The creatures are the fundamental card types for all the colors, except for the colorless cards.
#  - The artifacts are more present between the colorless cards group.
#  - The second most important type is more changeful: for the Red cards it is the sorcery, but for the Blue cards, it is the instant.

# Playability
# ===========
# 
# Converted Mana Cost (CMC)
# -------------------------
# In MTG, the players have a limited amount of resources to use their cards. These rescources are called "*mana*", which is the magic power they have to spend in order to invoke creatures and cast spells. This provides the game with an important factor of strategy, as the players have to choose how to spend their mana. A card with a high cost of mana can drain all of your mana, and let you without resources to do anything against your opponents. So this factor makes the cards more "expensive" or "cheaper" to cast. 
# 
# It's known that usually the cards with simillar characteristics (like color, or type) follow a simillar pattern in their mana costs. We can check this fact by visualizing the Converted Mana Cost of the different cards filtering by their colors:

# In[ ]:


plotly_colors=['rgb(225,225,175)', 'rgb(70,160,240)', 'rgb(100,100,100)', 'rgb(250,70,25)', 'rgb(100,200,25)', 'rgb(175, 175, 175)', 'rgb(150, 100, 150)']

traces=[]

for i in range(0,len(mtg['manaColors'].unique())):
    traces.append(go.Box(
        name=mtg['manaColors'].unique()[i],
        y=mtg.cmc[mtg['manaColors']==mtg.manaColors.unique()[i]],
        x=mtg['manaColors'].unique()[i],
        fillcolor=plotly_colors[i],
        boxmean=True,
        marker=dict(
            size=2,
            color='black',
        ),
        line=dict(width=1),
    ))

layout = go.Layout(
    yaxis=dict(
        title='Converted Mana Cost',
        zeroline=False
    ),
    showlegend=False
    )

fig = go.Figure(data=traces, layout=layout)
plotly.offline.iplot(fig)


# To explore the chart, hover over the boxes to see the values of the minimum and maximum (the "whiskers" of the boxes), the mean (which is represented as the dotted line in the box), the quartiles (each line in the box), and the outliers (the dots over the boxes). 
# 
#  - The mean CMC is located between 3-4 in all the colors. However, the white cards are the ones with the lowest mean CMC (3.16) while the multicolor cards are the ones with the highest mean CMC (3.84). This fact makes the white cards one of the "fastest" colors on the game.
#  - The interquartile range in all the cards is 2 manas, with the exception of the multicolor cards. This indicates that the distribution of the CMC is very centraliced.
#  - Althought all the colors have a simillar CMC distribution in the majority of their cards, the outliers are the key difference between the "fast" colors (red and white) and the rest.
#  - The white and the red cards are the colors with the less number of outliers. Probably this is the cause of their relative low mean CMC. 
# 
# We can also see the mana curve of the cards filtering by its card type:

# In[ ]:


traces=[]

for i in range(0,len(mtg['typesStr'].unique())):
    traces.append(go.Box(
        name=mtg['typesStr'].unique()[i],
        y=mtg.cmc[mtg['typesStr']==mtg.typesStr.unique()[i]],
        x=mtg['typesStr'].unique()[i],
        boxmean=True,
        fillcolor='rgb(70,160,240)',
        marker=dict(
            size=2,
            color='black',
        ),
        line=dict(width=1),
    ))

layout = go.Layout(
    yaxis=dict(
        title='Converted Mana Cost',
        zeroline=False
    ),
    showlegend=False
    )

fig = go.Figure(data=traces, layout=layout)
plotly.offline.iplot(fig)


#  - Notice that the land cards can't have a mana cost, as they're itselves the mana's main source.
#  - The creature and sorcery cards have the higher CMC, althought the last one has a relatively lower 2nd quartile.
#  - The instant cards have the lowest interquartile range, and highest number of outliers, which makes them the most versatile card type.

# Battle potential
# ================
# 
# Power
# -----
# 
# This aspect of the creatures represents the potential damage they can make to the enemy player or his/her creatures. The power of the creatures is slightly different across the different colors:
# 
# 

# In[ ]:


#First I'm going to isolate the creature cards
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

#Count the frequencies of the power points
from collections import Counter
c = Counter(creatures.power)

#set the colors
colors = np.sort(mtg.manaColors.unique()).tolist()
plotly_colors = ['rgb(100,100,100)', 'rgb(70,160,240)', 'rgb(175, 175, 175)', 'rgb(100,200,25)', 'rgb(150, 100, 150)', 'rgb(250,70,25)', 'rgb(225,225,175)']

#create the traces
traces=[go.Scatter(
    x = list(c.keys()),
    y = list(Counter(creatures.power[creatures.manaColors == color]).values()),
    mode = 'lines',
    line = dict(color=plotly_colors[i]),
    connectgaps = True,
    name = color) for i, color in enumerate(colors)]

layout = go.Layout(
    xaxis=dict(
        title='Power',
    ))

#plot the results
fig = go.Figure(data=traces, layout=layout)
plotly.offline.iplot(fig)


#  - Each line represents one color. The white creatures are primarily cards with low attack points, the most of them with 2-3 power points.
#  - In contrast, the green creatures are more numerous than the other colors in the range above 4 power points.
# 
# Toughness
# ---------
# 
# On the other hand, the creatures can also block the attack of other creatures using their toughness points. This represents the defense potential of the cards on the battle.
# 
# 

# In[ ]:


from collections import Counter
c = Counter(creatures.toughness)

colors = np.sort(mtg.manaColors.unique()).tolist()
plotly_colors = ['rgb(100,100,100)', 'rgb(70,160,240)', 'rgb(175, 175, 175)', 'rgb(100,200,25)', 'rgb(150, 100, 150)', 'rgb(250,70,25)', 'rgb(225,225,175)']

traces=[go.Scatter(
    x = list(c.keys()),
    y = list(Counter(creatures.toughness[creatures.manaColors == color]).values()),
    mode = 'lines',
    line = dict(color=plotly_colors[i]),
    connectgaps = True,
    name = color) for i, color in enumerate(colors)]

layout = go.Layout(
    xaxis=dict(
        title='Toughness',
    ))

fig = go.Figure(data=traces, layout=layout)
plotly.offline.iplot(fig)


#  - As in the previous chart, the white creatures are mainly located in the range of 2-3 points. 
#  - The black and the blue creatures are the most defensive. They start to rise in the range above 8 toughness points.
# 
# Battle strategy: Power vs Toughness
# -----------------------------------
# 
# As we've seen, the power and the toughness represent the potential of the creatures in the battle. It's also important to know if a creature is more favorable to be used in a defensive way, or in a offensive way. This is equivalent to ask: does this creature have more power points than toughness points, or vice versa? 
# 
# On the next chart we can see the relationship between the power & toughness of the creatures from different colors:

# In[ ]:


colors = mtg.manaColors.unique().tolist()
plotly_colors = ['rgb(225,225,175)', 'rgb(70,160,240)', 'rgb(100,100,100)', 'rgb(250,70,25)', 'rgb(100,200,25)', 'rgb(175, 175, 175)' , 'rgb(150, 100, 150)']

traces=[]

for i in range(0,len(colors)):
    cards=creatures[creatures.manaColors==colors[i]]
    traces.append(go.Scatter(
        x=cards.power,
        y=cards.toughness,
        mode='markers',
        name = colors[i],
        marker = dict(
            size = 10,
            opacity= 0.3,
            color = plotly_colors[i],
            )
        )
    )
    
traces.append(go.Scatter(
    x=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
    y=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],
    mode='lines',
    name='Perfect proportion',
    line = dict(
        color = ('rgb(102, 102, 102)'),
        width = 2,
        dash = 'dot'),
    ))

layout= go.Layout(
    title= 'Battle Potential',
    hovermode= 'closest',
    xaxis= dict(
        title= 'Power',
        ticklen= 5,
        zeroline= False,
        gridwidth= 1,
    ),
    yaxis=dict(
        title= 'Toughness',
        ticklen= 5,
        gridwidth= 1,
    ),
    annotations=[
        dict(
            x=14,
            y=4,
            xref='x',
            yref='y',
            showarrow=False,
            text='Attacking creatures'),
        dict(
            x=4,
            y=14,
            xref='x',
            yref='y',
            showarrow=False,
            text='Defending creatures',
        ),
    ],
    updatemenus=list([
        dict(
            x=-0.05,
            y=1,
            yanchor='top',
            buttons=list([
                dict(
                    args=['visible', [True, True, True, True, True, True, True, True]],
                    label='All',
                    method='restyle'
                ),
                dict(
                    args=['visible', [True, False, False, False, False, False, False, True]],
                    label='White',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False, True, False, False, False, False, False, True]],
                    label='Blue',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False, False, True, False, False, False, False, True]],
                    label='Black',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False, False, False, True, False, False, False, True]],
                    label='Red',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False, False, False, False, True, False, False, True]],
                    label='Green',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False, False, False, False, False, True, False, True]],
                    label='Colorless',
                    method='restyle'
                ),
                dict(
                    args=['visible', [False, False, False, False, False, False, True, True]],
                    label='Multicolor',
                    method='restyle'
                ),
            ]),
        ),
    ]),
)

fig= go.Figure(data=traces, layout=layout)
plotly.offline.iplot(fig)


# Recommendations to use this chart: 
# 
#  - Each dot represent one creature. The color of the dots indicates the color of the cards.
#  - The most of the dots share the same position on the chart, so they overlap. As more intense is the color of the dot, more creatures are overlapping on the same point. As lighter is the dot, less dots are overlapping.
#  - In order to see a clear result, use the dropdown menu located on the top-left corner of the chart to filter the data by a particular color.
#  - The central line represents the perfect proportion between attack power and toughness. The dots passing through that line have exactly the same amount of attack and defense power. The dots that are situated above that line have more toughness than power, so they're mainly defending creatures. Conversely, the dots that are situated below the central line have more power than toughness, so they're mainly attacking creatures.
#  - You can see which colors have more attacker or defensive creatures by filtering with the dropdown menu and watching which dots are more lighter or darker.
#  - For example, if we filter the data by the white cards, we can see that the dots located on the "defensive zone" of the chart are darker than the dots located on the "attacker zone" of the chart. This indicates that the white creatures are more defenders than attackers.
#  - If we filter by the red cards on the dropdown menu, we can see that the dots located on the "attacker zone" are more intensive than the dots located on the "defender zone". This indicates that the red creatures are more attackers than defenders.
