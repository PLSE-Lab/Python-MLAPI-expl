#!/usr/bin/env python
# coding: utf-8

# I'm going to submit 2 kernels on this dataset:
# 1. Pokemon Image Dataset exploration with pandas, matplotlib and seaborn
# 2. DCGAN implementation to create even more pokemons
# 
# This kernel is the frist one.

# In[ ]:


# importing required packages
import os
import numpy as np
import random as rnd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image


# **Dataset Overview**
# 
# Let's try to explore our dataset.

# In[ ]:


os.listdir('../input/pokemon-images-and-types')


# In[ ]:


os.listdir('../input/pokemon-images-and-types/images/images')[:20]


# In[ ]:


len(os.listdir('../input/pokemon-images-and-types/images/images'))


# Pokemon Image Dataset contains images of 809 pokemons and one CSV-file describing them. 
# 
# First, I'm going to work with CSV-file: fill NaNs, add new column and count unique values.

# In[ ]:


pokemons = pd.read_csv('../input/pokemon-images-and-types/pokemon.csv')
pokemons.head(10)


# In[ ]:


pokemons.nunique()


# In[ ]:


def createType(row):
    if row['Type2']=='None':
        return row['Type1']
    return '-'.join([row['Type1'], row['Type2'] ])


# In[ ]:


pokemons['Type2'].fillna('None', inplace=True)
pokemons['Type'] = pokemons.apply(lambda row: createType(row), axis=1)
pokemons.head(10)


# In[ ]:


pokemons.nunique()


# What do we have here:
# * 809 pokemons in total
# * 18 types of pokemons
# * 18 subtypes of pokemons + extra "None" tag for pokemons without subtype
# * 159 combinations of types and subtypes
# 
# Not bad! 
# 
# Let's answer the next question: how many pokemons with subtype are in this dataset?

# In[ ]:


labels = ['One type pokemons', 'Two types pokemons']
sizes = [pokemons['Type2'].value_counts()['None'], 
         pokemons['Type2'].count() - pokemons['Type2'].value_counts()['None']]
colors = ['lightskyblue', 'lightcoral']

patches, texts, _ = plt.pie(sizes, colors=colors, startangle=90, autopct='%1.1f%%')
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()


# About 50% of pokemons in this dataset do have subtype. So I can assume that our data represents quite good diversity of pokemons because subtype of the pokemon can influence its appearance. However, each pokemon has some evolutions in this dataset and this fact makes my assumption weaker.

# Now let's plot distributions of different types and subtypes.

# In[ ]:


def createBarChart(data, name=''):
    colors = {'Water': 'blue', 'Normal': 'orange', 'Grass': 'green', 'Bug': 'pink', 'Fire': 'red',
              'Psychic': 'purple', 'Rock': 'gray', 'Electric': 'yellow', 'Poison': 'lightgreen', 'Ground': 'brown',
              'Dark':  'darkblue', 'Fighting': 'crimson', 'Dragon': 'salmon', 'Ghost': 'orchid', 
              'Steel': 'silver', 'Ice': 'lightblue', 'Fairy': 'darkgreen', 'Flying': 'orangered', 'None': 'black'}
    labels = [name for name in data.keys()]
    values = [data[name] for name in data.keys()]
    bar_colors = [colors[t.split('-')[0]] for t in labels]
    
    plt.bar(labels, values, color=bar_colors)
    plt.xticks(rotation = 90)
    plt.ylabel('Counts')
    plt.title(name)
    
    plt.tight_layout()


# In[ ]:


plt.figure(2, figsize=(13, 6), edgecolor = 'k')

plt.subplot(121)
createBarChart(pokemons['Type1'].value_counts(), name='First type of pokemons')

plt.subplot(122)
createBarChart(pokemons['Type2'].value_counts().drop(['None']), name='Second type of pokemons')

plt.show()


# In[ ]:


plt.figure(18, figsize=(18, 36))

for i, key in enumerate(pokemons['Type1'].value_counts().keys()):
    subtypes = pokemons.loc[pokemons['Type1']==key]['Type'].value_counts()
    plt.subplot(6, 3, i + 1)
    createBarChart(subtypes, name='{} pokemon\'s subtypes distribution'.format(key))

plt.tight_layout()
plt.show()


# With a quick look at these bars I found out at least 2 facts:
# * Two types are represented with more than 100 pokemons: Water and Normal pokemons
# * Flying subtypes can be found in each type
# 
# Heatmap of types would also be useful for our exploration.

# In[ ]:


counts = pokemons['Type'].value_counts()
pokemons['Counts'] = [counts[x] for x in pokemons['Type']]
data = pd.pivot_table(data=pokemons, index='Type1', columns='Type2', values='Counts')

sns.set(rc={'figure.figsize':(8,14)})
sns.heatmap(data, cmap='coolwarm', annot=True, cbar=False, square=True, linewidths=.5)


# Heatmap lets us see the whole picture of pokemons' types and subtypes distribution.
# 
# So now we can finish with CSV-file exploration and go straight to pictures of the pokemons.

# In[ ]:


fig = plt.figure(16, figsize=(18, 18))

for i, pic in enumerate(rnd.sample(os.listdir('../input/pokemon-images-and-types/images/images'), 16)):
    a = fig.add_subplot(4, 4, i + 1)
    img = plt.imshow(mpimg.imread('../input/pokemon-images-and-types/images/images/{}'.format(pic)))
    a.set_title(pic)
    plt.grid(None)

plt.show()


# Whoops, it seems like there are PNG and JPG formats in the dataset. Before implementing DCGAN in the next kernel we shoud convert PNG-images to JPG-images because JPG-images have fewer channels.

# In[ ]:


img = mpimg.imread('../input/pokemon-images-and-types/images/images/psyduck.png')
print(img.shape)


# In[ ]:


img = mpimg.imread('../input/pokemon-images-and-types/images/images/lurantis.jpg')
print(img.shape)


# In[ ]:


images = []

fill_color = (255,255,255)

for img in os.listdir('../input/pokemon-images-and-types/images/images'):
    im = Image.open('../input/pokemon-images-and-types/images/images/{}'.format(img))
    if img.split('.')[1] == 'png':
        im = im.convert("RGBA")
        if im.mode in ('RGBA', 'LA'):
            bg = Image.new(im.mode[:-1], im.size, fill_color)
            bg.paste(im, im.split()[-1]) # omit transparency
            im = bg 
    images.append(np.array(im))


# Now we should check if conversion was correct (while expiremnting, I made lots of random samples to check it).

# In[ ]:


fig = plt.figure(16, figsize=(18, 18))

for i, pic in enumerate(rnd.sample(images, 16)):
    a = fig.add_subplot(4, 4, i + 1)
    img = plt.imshow(Image.fromarray(pic))
    plt.grid(None)

plt.show()


# Thank you for your attention, I hope you liked this little data analysis!
# 
# Next, I'm going to implement DCGAN for this dataset though I still have no idea about how they really work C:
# 
# Upvote if you liked this kernel. See you soon!
# 
