#!/usr/bin/env python
# coding: utf-8

# Data loading and setup
# ----
# Pretty classic, pandas for data processing, matplotlib+seaborn for visualisation, spacy, nltk and gensim for NLP tasks. I will be using mostly spacy even tho it is a lot slower than NLTK for my task given this setup (kaggle), but the point here is not performance but ease of use and reproductibility (I assume more people might be familiar with spacy nowadays).
# For faster re-runs, I saved the NLP-processed data

# In[ ]:


import pandas as pd
from tqdm import tqdm
from collections import Counter, defaultdict
from gensim.models.word2vec import Word2Vec
from nltk.tokenize import word_tokenize
import spacy
import numpy as np
import pylab as pl
import seaborn as sns
import random
import math

sns.set()
tqdm.pandas()

en = spacy.load('en')

df = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')
print(df.columns)


# Data and text cleaning
# ----
# 1./ **Data correction:** After digging into the data I found thge types that were not what I expected: quick corrections. (ex: list of floats loaded as strings and not as lists)
# 
# 2./ **Text pre-processing:** game name and description: no caps, no stopwords (the, or, with etc...), no punctuation and data cleaning (bad encoding such as '\\n'). This is pretty basic and most of my following uses cases don't need anything more advanced. The main idea here is to reduce the vocabulary size (number of words / tokens) without losing too much semantic information.
# 
# NOTE: *you can see the processing time using spacy on 1cpu (using tqdm/progress_apply)... Pretty slow but you can go faster with a custom setup.*

# In[ ]:


def is_nan(value):
    try:
        return math.isnan(value)
    except TypeError:
        return False

df['In-app Purchases'] = df['In-app Purchases'].apply(lambda r: r if is_nan(r) else [float(i) for i in r.split(', ')])
pd.to_numeric(df['Average User Rating'])

df['name_doc'] = df.Name.progress_apply(lambda r: en(r.replace('\\n', ' ')))
df['name_clean'] = df.name_doc.progress_apply(lambda r: [str(i).lower() for i in r if not (i.is_stop or i.is_punct)])

df['desc_doc'] = df.Description.progress_apply(lambda r: en(r.replace('\\n', ' ')))
df['desc_clean'] = df.desc_doc.progress_apply(lambda r: [str(i).lower() for i in r if not (i.is_stop or i.is_punct)])


print(df[['Name', 'name_clean']].head(10))


# Data aggregation

# In[ ]:


df['price_in_app'] = df.apply(lambda r: 1.0 if r['Price'] and not is_nan(r['In-app Purchases']) else np.nan, axis=1)
# somehow sometimes NaN is loaded as float-NaN and sometimes as np.NaN. math.isnan is consistent
mapping = {
    'Average User Rating': ('rating', None),
    'Price': ('price', lambda x: x <= 0.0),
    'In-app Purchases': ('in_app', None),
    'User Rating Count': ('rate_count', None),
    'price_in_app': ('price_in_app', None),
}


def update_sdict(_d, words, value, key, cutoff_rule=None):
    if is_nan(value):
        return
    if str(value).lower() == 'nan':
        return
    if cutoff_rule is not None and cutoff_rule(value):
        return
    try:
        value = max(value)
    except TypeError:
        pass
    for w in words:
        if value is np.nan:
            print(value)
        _d[w][key].append(value)

def make_aggreg_df(df, mapping, word_source='name_clean'):
    x = defaultdict(lambda: {v[0]: [] for v in mapping.values()})
    c = Counter()
    null = df[word_source].apply(lambda r: c.update(set(r)))
    for k, v in c.items():
        x[k]['nb'] = v
    for key in mapping.keys():
        null = df.apply(lambda r: update_sdict(x, set(r[word_source]), r[key], mapping[key][0], mapping[key][1]), axis=1)
    for k, v in x.items():
        v['word'] = k
    agg_df = pd.DataFrame([r for w, r in x.items()])
    for v in mapping.values():
        agg_df['nb_%s' % v[0]] = agg_df[v[0]].apply(len)
        agg_df['avg_%s' % v[0]] = agg_df[v[0]].apply(np.mean)
        agg_df['std_%s' % v[0]] = agg_df[v[0]].apply(np.std)
    agg_df['nb_price_only'] = agg_df.nb_price - agg_df.nb_price_in_app
    agg_df['nb_in_app_only'] = agg_df.nb_in_app - agg_df.nb_price_in_app
    agg_df['nb_free'] = agg_df.nb - agg_df.nb_price - agg_df.nb_in_app + agg_df.nb_price_in_app
    agg_df['nb_pay'] = agg_df.nb_price + agg_df.nb_in_app - agg_df.nb_price_in_app
    return agg_df

agg_df = make_aggreg_df(df, mapping)
print(agg_df[['word', 'nb', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].head(10))


# 1./ Title words and app pricing relationship
# ----
# The objective here might seem weird but the idea is to look if certain words in a strategy game title cane be an indication of the pricing policy of the game (free, pay, in-app purchases, both).

# In[ ]:


sub = agg_df.sort_values(by='nb', ascending=False)
sub = sub[sub.nb > 50]
sub.set_index('word', inplace=True)
flatui = ["#0974f6", "#f6b709", "#f67809", "#f64009"]
cmap = sns.color_palette(flatui, 4)


# In[ ]:


f, (a0, a1) = pl.subplots(2, 1, gridspec_kw={'height_ratios': [1, 5]})
sub[['nb_free', 'nb_in_app_only', 'nb_price_only', 'nb_price_in_app']].div(sub.nb, axis='index').head(30).plot(
    kind='bar',
    stacked=True,
    color=cmap,
    figsize=(20, 16),
    ax=a1

)
sub['nb'].head(30).plot(
    kind='bar',
    ax=a0
)
a1.set_xlabel('Most commonly found words')
a1.set_ylabel('nb of games, per category (stacked)')
a0.set_ylabel('word occurences')
a1.legend(['100% free', 'in-app purchases only', 'buy only', 'buy AND in-app purchases'])
pl.suptitle('Game monetization for the most common words found in the game title - normalized')


# What do we see? A few interesting things, such as:
# - Pretty much all games containing the word 'free' are free in the sense that you don't have to buy the game to play it, but an important proportion of these contain in-app purchases.
# - The word 'Pro' is a good indicator of a game that you will have to buy, even tho in additionj to that a fair amound of these also include in-app purchases
# - 'heroes' is likely a "free" game including in-app purchases (I don't think this will surprise anyone that is a bit familiar with the gane section on mobile app stores)
# - words indication an improvement on something or a quality (2, HD, 3D) are more frequently buy-to-play (Even tho '3' does not follow that rule)
# 
# Finally the graph for words occurences in all games titles is here to show that:
# - the world 'game' is by far the most frequently found in a game title... original
# - the numbers of titles taken into account is rather high for each word, so I don't think all of this is due to chance

# In[ ]:


ax = sub[['nb_free', 'nb_in_app_only', 'nb_price_only', 'nb_price_in_app']].div(sub.nb, axis='index').sort_values(by='nb_free', ascending=False).head(30).plot(
    kind='bar',
    stacked=True,
    color=cmap,
    figsize=(20, 16),
    title='Frequent title words the most associated with a TRUE free game - normalized',
)
ax.set_xlabel('Frequent title words sorted by FREE ratio')
ax.set_ylabel('monetization type proportion')
ax.legend(['100% free', 'in-app purchases only', 'buy only', 'buy AND in-app purchases'])


# Not sure what to take of this graph... Except 'lite' being mostly free games and 'tic' 'tac' 'toe' as three very similar monetization distributions... when one would have done the job (let's blame the tokenizer)

# In[ ]:


ax = sub[['nb_free', 'nb_in_app_only', 'nb_price_only', 'nb_price_in_app', 'nb_pay']].div(sub.nb, axis='index').sort_values(by='nb_pay', ascending=False).head(50)[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].head(30).plot(
    kind='bar',
    stacked=True,
    color=cmap,
    figsize=(20, 16),
    title='Frequent title words the most associated with a pay for game - normalized',
)
ax.set_xlabel('Frequent title words sorted by all NON-FREE ratio')
ax.set_ylabel('monetization type proportion')
ax.legend(['100% free', 'in-app purchases only', 'buy only', 'buy AND in-app purchases'])


# A few obvious notes again:
# - 'pro' and even more 'premium' are the kings of 'purchase to play' games (with still a fair part of these including in-app pruchases too)
# - once again the top in-app purchase words will probably not surprise many of you: evolution, heroes, rpg, empire, age, kingdom, clicker

# In[ ]:


ax = sub[['nb_free', 'nb_in_app_only', 'nb_price_only', 'nb_price_in_app']].head(30).plot(
    kind='bar',
    stacked=True,
    color=cmap,
    figsize=(20, 16),
    title='Game monetization for the most common words found in the game title',
)
ax.set_xlabel('Most commonly found words')
ax.set_ylabel('nb of games, per category (stacked)')
ax.legend(['100% free', 'in-app purchases only', 'buy only', 'buy AND in-app purchases'])


# 

# In[ ]:


fig, axes = pl.subplots(nrows=3, ncols=3, figsize=(20, 16))
sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].div(sub.nb, axis='index').sort_values(by='nb_free', ascending=False).head(10).plot(kind='bar', stacked=True, color=cmap, ax=axes[0,0])
sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].head(10).plot(kind='bar', stacked=True, color=cmap, ax=axes[0,1])
sub[['nb']].head(10).plot(kind='bar', stacked=True, ax=axes[0,2])
sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].div(sub.nb, axis='index').sort_values(by='nb_price_only', ascending=False).head(10).plot(kind='bar', stacked=True, color=cmap, ax=axes[1,0])
sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].div(sub.nb, axis='index').sort_values(by='nb_in_app_only', ascending=False).head(10).plot(kind='bar', stacked=True, color=cmap, ax=axes[1,1])
sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app']].div(sub.nb, axis='index').sort_values(by='nb_price_in_app', ascending=False).head(10).plot(kind='bar', stacked=True, color=cmap, ax=axes[1,2])
sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app', 'nb_pay']].div(sub.nb, axis='index').sort_values(by='nb_pay', ascending=False).head(10)[['nb_free', 'nb_pay']].plot(kind='bar', stacked=True, color=cmap, ax=axes[2,0])
sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app', 'nb_price']].div(sub.nb, axis='index').sort_values(by='nb_price', ascending=False).head(10)[['nb_free', 'nb_price', 'nb_in_app_only']].plot(kind='bar', stacked=True, color=cmap, ax=axes[2,1])
sub[['nb_free', 'nb_price_only', 'nb_in_app_only', 'nb_price_in_app', 'nb_in_app']].div(sub.nb, axis='index').sort_values(by='nb_in_app', ascending=False).head(10)[['nb_free', 'nb_price_only', 'nb_in_app']].plot(kind='bar', stacked=True, color=cmap, ax=axes[2,2])


# A graph of everything at once (sorted by all possible monetization combinations)

# 2./ Let's generate stuff
# ----
# Basic markow chain game description generator (results not garanted)
# 
# WIP

# In[ ]:


class MarkovTextGenerator:
    def __init__(self):
        self.model = None

    def train(self, texts):
        model = defaultdict(Counter)
        STATE_LEN = 2
        for text in tqdm(texts):
            for i in range(len(text) - STATE_LEN):
                state = text[i:i + STATE_LEN]
                next = text[i + STATE_LEN]
                model[' '.join(state)][next] += 1
        self.model = model

    def run(self, max_len=25):
        state = random.choice(list(self.model)).split()
        out = list(state)
        for i in range(max_len):
            try:
                x = random.choices(list(self.model.get(' '.join(state))), self.model.get(' '.join(state)).values())
            except TypeError:
                break
            out.extend(x)
            state = state[1:]
            state.append(out[-1])
        return ' '.join(out)

    
model = MarkovTextGenerator()
model.train(df.desc_doc.progress_apply(lambda r: [str(i).lower() for i in r]))

for i in range(10):
    print(model.run(50))
    print()


# 3./ Coming next (WIP):
# ----
# - in depth analysis of certain words
# - a word2vec on the titles and descriptions (I tried it and to my surprise it performs rather well)
# - a study of ratings
# 
