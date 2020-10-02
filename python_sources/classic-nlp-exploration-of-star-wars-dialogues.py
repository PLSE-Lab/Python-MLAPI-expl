#!/usr/bin/env python
# coding: utf-8

# # Classic NLP exploration with Star Wars dialogues
# 
# Using Star Wars Episode IV dialogues, let's find out if rebels use different words than imperials.
# 
# ## 1. Loading and first exploration

# In[11]:


# Basic data loading
import pandas as pd

df = pd.read_table('../input/SW_EpisodeIV.txt',
                   delim_whitespace=True, header=0, escapechar='\\')
df.sample(10)


# In[2]:


df.shape


# So we have 1010 lines.
# 
# ## 2. Grouping by characters
# 
# Let's see how many lines each character said...

# In[3]:


df.groupby('character').count().sort_values('dialogue', ascending=False)


# There are a lot of secondary characters here! In order to make it easier, let's group them in 3 groups: rebels, imperials and neutrals.

# In[4]:


def character_group(name: str) -> str:
    rebel = ('BASE VOICE', 'CONTROL OFFICER', 'MAN', 'PORKINS', 'REBEL OFFICER', 'RED ELEVEN',
             'RED TEN', 'RED SEVEN', 'RED NINE', 'RED LEADER', 'BIGGS', 'GOLD LEADER',
             'WEDGE', 'GOLD FIVE', 'REBEL', 'DODONNA', 'CHIEF', 'TECHNICIAN', 'WILLARD',
             'GOLD TWO', 'MASSASSI INTERCOM VOICE')
    imperial = ('CAPTAIN', 'CHIEF PILOT', 'TROOPER', 'OFFICER', 'DEATH STAR INTERCOM VOICE',
                'FIRST TROOPER', 'SECOND TROOPER', 'FIRST OFFICER', 'OFFICER CASS', 'TARKIN',
                'INTERCOM VOICE', 'MOTTI', 'TAGGE', 'TROOPER VOICE', 'ASTRO-OFFICER',
                'VOICE OVER DEATH STAR INTERCOM', 'SECOND OFFICER', 'GANTRY OFFICER', 
                'WINGMAN', 'IMPERIAL OFFICER', 'COMMANDER', 'VOICE')
    neutral = ('WOMAN', 'BERU', 'CREATURE', 'DEAK', 'OWEN', 'BARTENDER', 'CAMIE', 'JABBA',
               'AUNT BERU', 'GREEDO', 'NEUTRAL', 'HUMAN', 'FIXER')

    if name in rebel:
        return 'rebels'
    elif name in imperial:
        return 'imperials'
    elif name in neutral:
        return 'neutrals'
    else:
        return name


df['character'] = df['character'].apply(character_group)
df.groupby('character').count().sort_values('dialogue', ascending=False)


# ## 3. Using Td-idf to extract relevant words (features)
# 
# A classic way to extract interesting features in text data is using words: counting the most relevant words actually. If we use Td-idf method, each word will have a value in each line, showing its importance.
# 
# We will get 200 most relevant words, that appear in less than 0.1 * 1010 = 101 lines, so they are scatter enough. We will remove some common stop_words (like "and", "or", etc).

# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(max_df=0.1, max_features=200, stop_words='english')

features = tfidf_vec.fit_transform(df.dialogue)
X = pd.DataFrame(data=features.toarray(), 
                 index=df.character, 
                 columns=tfidf_vec.get_feature_names())
X.sample(10)


# ## 4. Displaying lines in a graph
# 
# Now, let's use PCA to display each line in a 2D graph.

# In[6]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

df_reduced = pd.DataFrame(X_reduced)
df_reduced['character'] = X.index
df_reduced.head(10)


# Let's assign some colors to each character:
# * main rebels (like Luke) will be shown in blue
# * secondary rebels in cyan
# * Vader will be shown in red
# * other imperials in magenta
# * neutrals in black

# In[8]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


def character_to_color(name: str):
    color = {'LUKE': 'b', 'HAN': 'b', 'THREEPIO': 'b', 'BEN': 'b', 'LEIA': 'b',
             'VADER': 'r', 'imperials': 'm', 'rebels': 'c', 'neutrals': 'k'}
    return color[name]


df_reduced['color'] = df_reduced['character'].apply(character_to_color)

plt.figure(figsize=(10, 10))
plt.scatter(x=df_reduced[0], y=df_reduced[1],
            color=df_reduced['color'], alpha=0.5)


# While blue and cyan dots (rebels) are scattered over all plane, it looks like red and magenta (imperials) are limited to that *line* going down from the left. 
# This basically means that **rebels use a wider vocabulary than imperials**.
# 
# Looking closer, there is just one magenta outsider in the upper side, near 0.2-0.6... let's find out the line!

# In[9]:


df_reduced[(df_reduced[0]>0.1) & (df_reduced[1]>0.55) & (df_reduced[1]<0.6)]


# In[10]:


df.loc[714]


# This is not a long line, and probably those words are more common on the rebel side.
# 
# **I hope you liked this quick exploration!**
