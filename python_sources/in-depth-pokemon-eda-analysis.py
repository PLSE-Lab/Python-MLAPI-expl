#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns


# # Load Data

# In[ ]:


raw = pd.read_csv('/kaggle/input/pokemon/Pokemon.csv')


# In[ ]:


df = raw.copy()
df.head()


# How many generations are included in this dataset?

# In[ ]:


df.Generation.unique()


# ## How many pokemon are in each generation?

# In[ ]:


pd.crosstab(df['Generation'], columns='count', colnames=[''])


# Wow, there's only 82 pokemon in Generation 6! Actually, this seems suspicious, luckily we can easily check the number of pokemon per generations with a quick google search.
# 
# <table style="width:15%">
#   <tr>
#     <th>Generation 1:</th>
#     <th>151</th> 
#   </tr>
#   <tr>
#    <th>Generation 2:</th>
#     <th>100</th> 
#   </tr>
#    <tr>
#     <th>Generation 3:</th>
#     <th>135</th> 
#   </tr>
#    <tr>
#     <th>Generation 4:</th>
#     <th>107</th> 
#   </tr>
#    <tr>
#     <th>Generation 5:</th>
#     <th>156</th> 
#   </tr>
#    <tr>
#     <th>Generation 6:</th>
#     <th>72</th> 
#   </tr>
# </table>
# 
# Wow, there were significantly fewer pokemon in Generation 6 than any other generation Interesting!

# ## Why are there more pokemon in each generation than the offical count?

# Now there's another issue that arises. There are only 151 pokemon according to [serebii.net](https://serebii.net), lets examine the first generation.

# In[ ]:


df[df.Generation == 1].head(10)


# It's appears that there are alternate forms of pokemon with different stats than the originals which is leading to the increased count of pokemon for each generation. 

# ## How many legendary pokemon are there in each generation?

# In[ ]:


df.groupby('Generation')['Legendary'].sum()


# Wow, that's a lot in the later generations. I can only assume that they have mega evolutions. Lets check.

# In[ ]:


df[(df.Generation == 3) & (df.Legendary == True)]


# Bingo!

# ## What are the most popular primary types (Type 1) for each generation? The least?

# In[ ]:


plt.figure(figsize=(20,8))
crosstab = pd.crosstab(df['Generation'], df['Type 1'])
sns.heatmap(crosstab, annot=True, cmap=sns.color_palette("Purples"), cbar=False, linewidths=.5)
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plt.show()


# There are a lot of water types in generations 1 and 3, almost double of any of the other generations. We can that Bug ,Grass, Normal and Watar are the most populated types. However, there's an issue with drawing suchs conclusions about type proportion from this heatmap and that's because of different sample sizes. Let's find the ratio of each type for each generation.

# In[ ]:


plt.figure(figsize=(20,8))
crosstab = pd.crosstab(df['Generation'], df['Type 1']).apply(lambda x: round(x/x.sum(),3), axis=1)
sns.heatmap(crosstab, annot=True, cmap=sns.color_palette("Purples"), cbar=False, linewidths=.5)
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plt.show()


# Much better. Some quick insights we can glean are that generations 1, 2, and 3 have a significant amount of waters and bug types while flying types are not quite as popular.

# ## What are the most popular secondary types (Type 2) for each generation? The least?

# In[ ]:


plt.figure(figsize=(20,8))
crosstab = pd.crosstab(df['Generation'], df['Type 2']).apply(lambda x: round(x/x.sum(),3), axis=1)
sns.heatmap(crosstab, annot=True, cmap=sns.color_palette("Purples"), cbar=False, linewidths=.5)
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plt.show()


# I guess flying types are quite popular after all, just as type 2. We can see flying is a very popular Type 2 trait. Ground seems somewhat popular as well, albiet not as popular.

# ## What are the most popular combinations of Type 1 and Type 2? The least?

# In[ ]:


plt.figure(figsize=(20,8))
crosstab = pd.crosstab(df['Type 1'], df['Type 2'])
sns.heatmap(crosstab, annot=True, cmap=sns.color_palette("Purples"), cbar=False, linewidths=.5)
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
plt.show()


# As we kind of gleaned from the previous heatmap, flying is very popular. The most popular types are normal/flying, bug/flying, and grass/position in that order.

# ## What pokemon types have the most hp? The least?

# In[ ]:


pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['HP']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)


# ## What pokemon types have the most attack? The least?

# In[ ]:


pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['Attack']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)


# ## What pokemon types have the most defense? The least?

# In[ ]:


pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['Defense']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)


# ## What pokemon types have the most special attack? The least?

# In[ ]:


pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['Sp. Atk']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)


# ## What pokemon types have the most special defense? The least?

# In[ ]:


pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['Sp. Def']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)


# ## What pokemon types have the most speed? The least?

# In[ ]:


pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['Speed']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)


# ## What pokemon types have the most total stats? The least?

# In[ ]:


pd.melt(df, id_vars=['Type 1', 'Type 2'], value_vars=['Total']).groupby(['Type 1', 'Type 2']).mean().sort_values(by='value', ascending=False)


# ## What type has the strongest physical attack and defense? The least?

# In[ ]:


df.groupby(['Type 1', 'Type 2'])[['Attack', 'Defense']].mean().sort_values(['Attack', 'Defense'], ascending=False)


# ## What type has the strongest special attack and defense? The least?

# In[ ]:


df.groupby(['Type 1', 'Type 2'])[['Sp. Atk', 'Sp. Def']].mean().sort_values(['Sp. Atk', 'Sp. Def'], ascending=False)


# ## What type has the strongest phyiscal attack and special defense? The least?

# In[ ]:


df.groupby(['Type 1', 'Type 2'])[['Attack', 'Sp. Def']].mean().sort_values(['Attack', 'Sp. Def'], ascending=False)


# ## What type has the strongest special attack and strongest phyiscal defense? The least?

# In[ ]:


df.groupby(['Type 1', 'Type 2'])[['Sp. Atk', 'Defense']].mean().sort_values(['Sp. Atk', 'Defense'], ascending=False)


# In[ ]:


def plot_pokemon_generation_stats(stat):
    fig, ax = plt.subplots(6,1,figsize=(10,12), constrained_layout=True)
    sns.kdeplot(df[df.Generation == 1][stat], color='#F3370C', shade=True, ax=ax[0])
    sns.kdeplot(df[df.Generation == 2][stat], color='#0C87F2', shade=True, ax=ax[1])
    sns.kdeplot(df[df.Generation == 3][stat], color='#F8CD0D',shade=True, ax=ax[2])
    sns.kdeplot(df[df.Generation == 4][stat], color='#59D10A',shade=True, ax=ax[3])
    sns.kdeplot(df[df.Generation == 5][stat], color='#BB0BE2',shade=True, ax=ax[4])
    sns.kdeplot(df[df.Generation == 6][stat], color='#0AD1D1',shade=True, ax=ax[5])

    ylim = max([ax[0].get_ylim()[1], ax[1].get_ylim()[1], ax[2].get_ylim()[1], ax[3].get_ylim()[1], ax[4].get_ylim()[1], ax[5].get_ylim()[1]])
    xmax = max([ax[0].get_xlim()[1], ax[1].get_xlim()[1], ax[2].get_xlim()[1], ax[3].get_xlim()[1], ax[4].get_xlim()[1], ax[5].get_xlim()[1]])

    for i in range(0,6):
        ax[i].grid(False)
        ax[i].set_title(f'Generation {i+1}')
        ax[i].set_ylim(0, ylim)
        ax[i].set_xlim(-25, xmax)

    fig.suptitle(f'Pokemon {stat.capitalize()} Distribution for Each Generation', fontsize=24)
    fig.text(-0.02, 0.5, "Density", ha="center", va="center", fontsize=22, rotation=90)
    fig.text(0.5,0-.02, f"{stat.capitalize()}", ha="center", va="center", fontsize=22)

    plt.show()


# In[ ]:


plot_pokemon_generation_stats('HP')


# In[ ]:


plot_pokemon_generation_stats('Attack')


# In[ ]:


plot_pokemon_generation_stats('Defense')


# In[ ]:


plot_pokemon_generation_stats('Sp. Atk')


# In[ ]:


plot_pokemon_generation_stats('Sp. Def')


# In[ ]:


plot_pokemon_generation_stats('Speed')


# In[ ]:


plot_pokemon_generation_stats('Total')

