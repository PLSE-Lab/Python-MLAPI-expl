#!/usr/bin/env python
# coding: utf-8

# <center><h1><font color="black">Who is the Strongest?</font></h1></center>

# Hey everybody! In this kernal we will look at what may be the strongest non-legendary pokemon according to total stats and an analysis comparing stats of non-legendary pokemon by their 1st type.
# 
# All feedback is absolutely welcome, thanks in advance!

# <center><b>Giving Credit Where Credit is Due</b></center>

# 
# 
# 1. https://www.kaggle.com/balcosandreea/pokemons-a-story-of-eda-visualizing This analysis gave me the idea to look at stats using the mean and taught me how to create multiple charts in an output, thanks again!
# 
# 2. https://www.kaggle.com/wenxuanchen/pokemon-visualization-radar-chart-t-sne I took the radar chart for this analysis directly from here.

# <center><b>Import Libraries</b></center>

# In[ ]:



import numpy as np 
import pandas as pd 
import seaborn as sns 
from matplotlib import pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# <center><b>Read and Process the Data</b></center>

# In[ ]:


pokemon = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv') #First we read our csv


# In[ ]:


pokemon.head() #Check the data


# In[ ]:


pokemon.dtypes #Data types look good


# In[ ]:


pokemon.isnull().sum() #Check for nulls


# The reason for nullls in Type 2 is because these pokemon do not have a second type, so we are just going to replace them with "None"

# In[ ]:


pokemon.fillna(value = 'None', inplace=True) #Fill them in


# <center><b>Looking for the Strongest Pokemon</b></center>

# Here we check to see right off the bat based on which pokemon has the top 40 higghest "total".
# 

# In[ ]:


most_powerful = pokemon[pokemon['Legendary']!= True].sort_values('Total' ,ascending = False) #Sort the data and exclude legendaries


# We have a problem, the table shows that most of the strongest are "mega" evolutions of the real pokemon. Now I am a old skool pokemon player and stopped after generation 3, so I am going to keep these ones out of the analysis. I also did some research and it appears that these mega evolutions are only temporary during a battle further motivating me to exclude them. If I am wrong please feel free to correct me!

# In[ ]:


most_powerful.nlargest(40,'Total')


# In[ ]:


pokemon_final = df_without_mega = most_powerful[~most_powerful.Name.str.contains("Mega")] #Dropping rows with "Mega"


# In[ ]:


pokemon_final[pokemon_final.Name.str.contains("Mega")].sum() #Check to make sure it worked


# Now all thats left is to prepare the data for plotting!

# In[ ]:


top_10 = pokemon_final.nlargest(10,'Total') #Define the data set


# In[ ]:


import seaborn as sns #Happy plotting!
from matplotlib import pyplot as plt

plt.figure(figsize=(20,8))

sns.set(style='whitegrid')

sns.barplot(x="Name", y="Total", data=top_10)


# <Center><b>Looking for Strongest Type of Pokemon</b></center>

# Following https://www.kaggle.com/balcosandreea/pokemons-a-story-of-eda-visualizing, I wanted to compare the stats of pokemon by their type. I really liked her idea of using the means of all the stats of pokemon grouped by type in order to do this. I also wanted to try out this analysis using a radar graph I found on https://www.kaggle.com/wenxuanchen/pokemon-visualization-radar-chart-t-sne. I decided to use Type 1 only for this analysis because not all pokemon have a type 2 and Type 1 is usally the main type of a pokemon that has 2 types.
# 
# Lets get to it!

# In[ ]:


stats_df = pokemon_final.groupby('Type 1').mean() #Create the data set to put into the graphs


# In[ ]:


stats_df


# First we look at the total stats grouped by type.

# In[ ]:


plt.figure(figsize= (17,8))

sns.barplot(x = 'Total', y='Type 1', data=stats_df.reset_index(),palette='dark')


# Now its time to look at individual stats!

# In[ ]:


stats = stats_df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']] #Get stats without 'Total' column
list_types = stats.index.unique().tolist()#Get types into a list
i= 1 #Set variable to distribute subplots
c=0 #set variable to distribute palettes
palette=['BuGn_r','OrRd','copper','YlOrRd','Blues_d','winter']
plt.figure(figsize=(17,17))


for stat in stats: #function to make a chart for each stat
    plt.subplot(3,2,i)
    i=i+1
    sns.barplot(x =stats[stat], y=list_types,palette = palette[c])
    c=c+1
    
    plt.title(str('Mean of ' + stat))


# For this graph I decided to compare dragon types to steel types because they are the top 2 strongest according to total stats and they both were very sought after types in the games (at least up to generation 3 to my knowledge). It is commonly the elite four's hardest pokemon to defeat, so lets see how this goes!

# In[ ]:


TYPE_LIST = ['Grass','Fire','Water','Bug','Normal','Poison',
            'Electric','Ground','Fairy','Fighting','Psychic',
            'Rock','Ghost','Ice','Dragon','Dark','Steel','Flying']

COLOR_LIST = ['#8ED752', '#F95643', '#53AFFE', '#C3D221', '#BBBDAF', '#AD5CA2', 
              '#F8E64E', '#F0CA42', '#F9AEFE', '#A35449', '#FB61B4', '#CDBD72', 
              '#7673DA', '#66EBFF', '#8B76FF', '#8E6856', '#C3C1D7', '#75A4F9']

# The colors are copied from this script: https://www.kaggle.com/ndrewgele/d/abcsds/pokemon/visualizing-pok-mon-stats-with-seaborn
# The colors look reasonable in this map: For example, Green for Grass, Red for Fire, Blue for Water...
COLOR_MAP = dict(zip(TYPE_LIST, COLOR_LIST))


# A radar chart example: http://datascience.stackexchange.com/questions/6084/how-do-i-create-a-complex-radar-chart
def _scale_data(data, ranges):
    (x1, x2), d = ranges[0], data[0]
    return [(d - y1) / (y2 - y1) * (x2 - x1) + x1 for d, (y1, y2) in zip(data, ranges)]

class RaderChart():
    def __init__(self, fig, variables, ranges, n_ordinate_levels = 6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.8,0.8],polar = True, label = "axes{}".format(i)) for i in range(len(variables))]
        _, text = axes[0].set_thetagrids(angles, labels = variables)
        
        for txt, angle in zip(text, angles):
            txt.set_rotation(angle - 90)
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.grid("off")
        
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], num = n_ordinate_levels)
            grid_label = [""]+[str(int(x)) for x in grid[1:]]
            ax.set_rgrids(grid, labels = grid_label, angle = angles[i])
            ax.set_ylim(*ranges[i])
        
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]

    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)

    def legend(self, *args, **kw):
        self.ax.legend(*args, **kw)
        
# select display colors according to Pokemon's Type 1
def select_color(types):
    colors = [None] * len(types)
    used_colors = set()
    for i, t in enumerate(types):
        curr = COLOR_MAP[t]
        if curr not in used_colors:
            colors[i] = curr
            used_colors.add(curr)
    unused_colors = set(COLOR_LIST) - used_colors
    for i, c in enumerate(colors):
        if not c:
            try:
                colors[i] = unused_colors.pop()
            except:
                raise Exception('Attempt to visualize too many pokemons. No more colors available.')
    return colors



df = stats
df = df.reset_index()
# In this order, 
# HP, Defense and Sp. Def will show on left; They represent defense abilities
# Speed, Attack and Sp. Atk will show on right; They represent attack abilities
# Attack and Defense, Sp. Atk and Sp. Def will show on opposite positions
use_attributes = ['Speed', 'Sp. Atk', 'Defense', 'HP', 'Sp. Def', 'Attack']
# choose the pokemons you like
use_pokemons = ['Steel','Dragon']

df_plot = df[df['Type 1'].map(lambda x:x in use_pokemons)==True] #df[df['Name']
use_pokemons = df_plot['Type 1'].values
datas = df_plot[use_attributes].values 
ranges = [[2**-20, df_plot[attr].max()] for attr in use_attributes]
colors = select_color(df_plot['Type 1']) # select colors based on pokemon Type 1 #'Type 1'

fig = plt.figure(figsize=(10, 10))
radar = RaderChart(fig, use_attributes, ranges)
for data, color, pokemon in zip(datas, colors, use_pokemons):
    radar.plot(data, color = color, label = pokemon)
    radar.fill(data, alpha = 0.1, color = color)
    radar.legend(loc = 1, fontsize = 'small')
plt.title('Mean Stats of '+(', '.join(use_pokemons[:-1])+' and '+use_pokemons[-1] if len(use_pokemons)>1 else use_pokemons[0]))
plt.show() 
      


# <center><b>Conclusion</b></center>

# According to the graph, slaking has the highest total stats than any other non-legendary pokemon in the game and he is a normal type pokemon. Further, the results show that dragon overall has the highest total stats with Steel comming in a close second. Steel has quite the normal defense advantage, however dragon types outperform at everything else.
# 
# Thank you for reading everybody, please upvote if you found this interesting. If you have any questions or feedback please feel free to let me know!
