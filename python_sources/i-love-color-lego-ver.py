#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
print(os.listdir('../input/lego-database'))

import matplotlib.image as mpimg
img=mpimg.imread('../input/lego-database/downloads_schema.png')
plt.figure(figsize= (12,12))
imgplot = plt.imshow(img)
plt.show()


# The dataset is quite annoying than I thought.
# It's a graph that reminds me of my previous DB class.

# In[ ]:


# Read All dataset first
colors = pd.read_csv('../input/lego-database/colors.csv')
sets = pd.read_csv('../input/lego-database/sets.csv')
themes = pd.read_csv('../input/lego-database/themes.csv')
parts =pd.read_csv('../input/lego-database/parts.csv')
inventories = pd.read_csv('../input/lego-database/inventories.csv')
inventory_sets = pd.read_csv('../input/lego-database/inventory_sets.csv')
part_categories =  pd.read_csv('../input/lego-database/part_categories.csv')
inventory_parts = pd.read_csv('../input/lego-database/inventory_parts.csv')

colors.head()


# ## 1. Color set : Colors.csv

# In[ ]:


#colors = colors.drop(['id', 'is_trans'], axis=1)
colors['rgb'] = colors['rgb'].apply(lambda x : '#'+x)
colors_set = dict(zip(colors.name, colors.rgb))
colors.head()


# In[ ]:


colors.describe(include='all')


# Let's see what color is in Lego

# In[ ]:


# source code from https://matplotlib.org/3.1.1/gallery/color/named_colors.html
import matplotlib.colors as mcolors


def plot_colortable(colors, title, sort_colors=True, emptycols=0):

    cell_width = 212
    cell_height = 22
    swatch_width = 48
    margin = 12
    topmargin = 40

    # Sort colors by hue, saturation, value and name.
    if sort_colors is True:
        by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                         name)
                        for name, color in colors.items())
        names = [name for hsv, name in by_hsv]
    else:
        names = list(colors)

    n = len(names)
    ncols = 4 - emptycols
    nrows = n // ncols + int(n % ncols > 0)

    width = cell_width * 4 + 2 * margin
    height = cell_height * nrows + margin + topmargin
    dpi = 64

    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)
    fig.subplots_adjust(margin/width, margin/height,
                        (width-margin)/width, (height-topmargin)/height)
    ax.set_xlim(0, cell_width * 4)
    ax.set_ylim(cell_height * (nrows-0.5), -cell_height/2.)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_axis_off()
    ax.set_title(title, fontsize=24, loc="left", pad=10)

    for i, name in enumerate(names):
        row = i % nrows
        col = i // nrows
        y = row * cell_height

        swatch_start_x = cell_width * col
        swatch_end_x = cell_width * col + swatch_width
        text_pos_x = cell_width * col + swatch_width + 7

        ax.text(text_pos_x, y, name, fontsize=14,
                horizontalalignment='left',
                verticalalignment='center')

        ax.hlines(y, swatch_start_x, swatch_end_x,
                  color=colors[name], linewidth=18)

    return fig

plot_colortable(colors_set, "Lego Colors")
plt.show()


# ## 2. themes

# What is themes?

# In[ ]:


themes.head(5)


# In[ ]:


themes['name'].value_counts()


# ## 3. Sets

# In[ ]:


sets.head()


# In[ ]:


sets['name'].value_counts()


# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(25, 8))
sns.countplot(sets['year'],)
plt.xticks(rotation=90)
plt.title('History of Lego Sets')
plt.show()


# ## 4. Inventories & Inventory Sets

# In[ ]:


inventories.head()


# In[ ]:


inventories.describe()


# In[ ]:


inventory_sets.head(5)


# ## 5. Parts & Part_categories

# In[ ]:


parts.head()


# In[ ]:


parts.info()


# In[ ]:


part_categories.head()


# In[ ]:


part_categories.info()


# # 6. Inventory Parts

# In[ ]:


inventory_parts.head()


# In[ ]:


inventory_parts['color_id'].value_counts()


# In[ ]:


color_count = inventory_parts['color_id'].value_counts()


# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(25, 5))
sns.barplot(color_count.index, color_count, ax=ax)
ax.set_xticklabels(sorted(color_count.index), rotation=90)
plt.show()


# Now let's color the graph (origin color)

# In[ ]:


colors_id = dict(zip(colors.id, colors.rgb))
#print(colors_id)
inventory_parts['color'] = inventory_parts['color_id'].apply(lambda id : colors_id[id])


# In[ ]:


import squarify
y = inventory_parts['color'].value_counts()[:30]
    
plt.rcParams['figure.figsize'] = (30, 10)
plt.style.use('fivethirtyeight')

color_this_graph = y.index
squarify.plot(sizes = y.values, label = y.index, color = color_this_graph)
plt.title('Top 30 Color : inventory parts', fontsize = 30)
plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:




