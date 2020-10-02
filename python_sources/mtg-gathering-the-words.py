#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# Just some fun visualizations of the text used in Magic cards.  No particular goal defined.
# 
# Table of Contents:
# 
# - Introduction
# - Settings
# - Read Data
# - Remove Outliers
# - Preprocess Colors
# - Word Cloud
# - Natural Language Processing Definitions
# - Count Vectorize Flavors
# - Frequent Flavors by Color & Heatmap
# - Frequent Texts by Color & Heatmap

# ### Settings
# We define a mapping between the colorIdentity and color columns so that we can combine the information from both when we categorize cards by color.
# 
# And to reduce data size, we limit the data to columns specified in keeps.

# In[ ]:


keeps = ['name', 'colorIdentity', 'colors', 'type', 'types', 'subtypes', 'supertypes', 'cmc', 'power', 'toughness', 'flavor', 'text', 'legalities']
colorIdentity_map = {'B': 'Black', 'G': 'Green', 'R': 'Red', 'U': 'Blue', 'W': 'White'}
plt_colors = ['k', 'b', '0.5', 'g', 'r', 'w', 'm']


# In[ ]:


import pandas as pd
import numpy as np
from numpy.random import random
from math import ceil

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem.porter import PorterStemmer
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


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


# ### Remove outliers

# In[ ]:


# remove promo cards that aren't used in normal play
# Edit 2016-09-30: Could be null because it's too new to have a ruling on it.
mtg_nulls = mtg.loc[mtg.legalities.isnull()]
mtg = mtg.loc[~mtg.legalities.isnull()]

# remove tokens without types
mtg = mtg.loc[~mtg.types.apply(lambda x: isinstance(x, float))]

# Power and toughness that depends on board state or mana cannot be resolved
mtg[['power', 'toughness']] = mtg[['power', 'toughness']].apply(lambda x: pd.to_numeric(x, errors='coerce'))

# Fill
mtg.flavor.fillna('', inplace=True)
mtg.text.fillna('', inplace=True)
mtg.shape


# ### Preprocess Color
# See https://www.kaggle.com/willieliao/d/mylesoneill/magic-the-gathering-cards/mtg-gathering-the-colors for reasoning.

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

# Materialize color columns
mono_colors = np.sort(mtg.colorsStr[mtg.colorsCount<=1].unique()).tolist()

for color in mono_colors:
    mtg[color] = mtg.colors.apply(lambda x: color in x)


# ## Word Cloud
# The text is basically where the rules are written, so there are lots of references to the battlefield, creature, player, etc.
# 
# The biggest surprise for me is the flavor.  I looked at some cards and there are a lot of "no one", "only one", or "one knows" phrases.  I've never noticed that before.  The pervasiveness of "like" is not surprising since the flavors rely heavily on similies and metaphors .
# 
# The prevalence of "goblin" in the card name is a good metaphor for how fast they reproduce and overrun everything.  But it's probably just a linguistic thing.  A lot of the human cards are either named or really a job description, like "Black Knight".  And the elf cards have various parts of speech ("elven", "elvish", "elves", 'elf") so it gets diluted, whereas a "goblin" is a "goblin" is a "goblin".  We'll try to use a stemmer in the next phase to deal with this.

# In[ ]:


wc = WordCloud(width=1000, height=800, max_words=200, relative_scaling=0.5)
cols = ['text', 'flavor', 'type', 'name']
f, axs = plt.subplots(len(cols), figsize=(80, 36))

for i, col in enumerate(cols):
    text = mtg[col].str.cat(sep=' ')    
    wc.generate(text)
    axs[i].imshow(wc)
    axs[i].axis("off")    
    axs[i].set_title(col.upper(), fontsize=24)

del wc, cols, f, axs


# ### Natural Language Processing
# We will now look at flavor and text in more detail.  First, we lower, stem, and remove stop words to reduce sparsity.

# In[ ]:


stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'\w+')

def my_tokenizer(s):
    return [stemmer.stem(t.lower()) for t in tokenizer.tokenize(s) if t.lower() not in stopwords.words('english')]    


# ### Count Vectorize Flavors
# We will only visualize the most frequent 100 words.  The top words seem real philosophical.

# In[ ]:


count_vect = CountVectorizer(tokenizer=my_tokenizer)
flavors = count_vect.fit_transform(mtg.flavor)

flavors_tf = [(w, i, flavors.getcol(i).sum()) for w, i in count_vect.vocabulary_.items()]
flavors_tf = sorted(flavors_tf, key=lambda x: -x[2])[0:99]
flavors.shape


# In[ ]:


flavors_tf[:9]


# ### Frequent Flavors by Color

# In[ ]:


flavors_words = [i for (i, j, k) in flavors_tf]
flavors_indices = [j for (i, j, k) in flavors_tf]

flavors_pivot = []
for color in mono_colors:
    f = flavors[np.where(mtg.manaColors==color)[0], :].tocsc()[:, flavors_indices]
    flavors_pivot.append(f.sum(axis=2).getA1())

f = flavors[np.where(mtg.colorsCount>1)[0], :].tocsc()[:, flavors_indices]
flavors_pivot.append(f.sum(axis=2).getA1())

flavors_pivot = pd.DataFrame(flavors_pivot, index=mono_colors + ['Multi'], columns=flavors_words)
del flavors, flavors_tf, flavors_words, flavors_indices, f            


# ### Flavors by Color Heatmap
# As expected, death + dead = black, goblins + dragons = red, and nature + forest + tree = green.
# 
# Blue's terms are not as concentrated since its strategy is much more multi-dimensional.  Common terms that are uncommon for other colors are know, mind, and master.
# 
# Similarly, no terms are exclusively white's domain.  Terms associated with it but not other colors are war, fight, light, and sword.

# In[ ]:


plt.figure(figsize=(8, 24))
sns.heatmap(flavors_pivot.transpose())


# ### Flavors by Color Stacked Barchart
# Slightly different perspective from the heatmap.  It shows the frequency of words a lot better as opposed to the relative percentages in the heatmap.  The really color-specific words still jump out, especially the ones that capture half of the usage.
# 
# - Black: death, da=ead
# - Green: nature, forest, tree
# - Red: goblin, fire
# - White: light 

# In[ ]:


flavors_pivot.transpose().plot(kind='barh', figsize=(8, 24), title='Flavors Stacked Barchart', stacked=True, color=plt_colors)


# ### Count Vectorize Texts
# We will only visualize the most frequent 100 words.

# In[ ]:


### Count Vectorize Texts
count_vect = CountVectorizer(tokenizer=my_tokenizer)
texts = count_vect.fit_transform(mtg.text)

texts_tf = [(w, i, texts.getcol(i).sum()) for w, i in count_vect.vocabulary_.items()]
texts_tf = sorted(texts_tf, key=lambda x: -x[2])[0:99]
texts.shape


# In[ ]:


texts_tf[0:9]


# ### Frequent Texts by Color

# In[ ]:


texts_words = [i for (i, j, k) in texts_tf]
texts_indices = [j for (i, j, k) in texts_tf]

texts_pivot = []
for color in mono_colors:
    t = texts[np.where(mtg.manaColors==color)[0], :].tocsc()[:, texts_indices]
    texts_pivot.append(t.sum(axis=2).getA1())

t = texts[np.where(mtg.colorsCount>1)[0], :].tocsc()[:, texts_indices]
texts_pivot.append(t.sum(axis=2).getA1())

texts_pivot = pd.DataFrame(texts_pivot, index=mono_colors + ['Multi'], columns=texts_words)
del texts, texts_tf, texts_words, texts_indices, t


# ### Texts by Color Heatmap
# Not as interesting as looking at flavors since the concentration is not as obvious.
# 
# For black and blue, we see more phrases dealing with discarding or drawing cards.
# 
# For red, there's more target, damage, and deal.

# In[ ]:


plt.figure(figsize=(8, 24))
sns.heatmap(texts_pivot.transpose())


# ### Texts by Color Stacked Barchart
# With a stacked barchart, it's easier to see that the distribution is more skewed than the flavors, without any specific color retaining an exclusivity on any word.  The exception is actual references to the mana colors in their stemmed form (b, r, g, w, etc).  These were probably used in the context of specifiying mana, tokens, creatures, etc.

# In[ ]:


texts_pivot.transpose().plot(kind='barh', figsize=(8, 24), title='Texts Stacked Barchart', stacked=True, color=plt_colors)

