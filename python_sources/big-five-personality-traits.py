#!/usr/bin/env python
# coding: utf-8

# ### The direct data source and questions can be found here: https://www.kaggle.com/tunguz/big-five-personality-test#codebook.txt

# #### TODO:
# * Check for normal distribution (+ Validity)
# * Find the questions that vary the most for each sect[](http://)ion (+ Validity)
#  > Does country make a difference in inter-trait variability  (i.e. did a language barrier prevent understanding)?
# * Find the correlates between section (+ Curiosity)
# * Map each trait to a country (+ Curiosity)

# In[ ]:


import pandas as pd
import numpy as np
import os
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import geopandas as gpd
import pycountry


# In[ ]:


os.popen('cd ../input/big-five-personality-test/IPIP-FFM-data-8Nov2018; ls').read()
path = r'../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv'
df_full = pd.read_csv(path, sep='\t')
pd.options.display.max_columns = 999
df_full.head()


# In[ ]:


df = df_full.copy() # reset df


# In[ ]:


# Removing any null values
start_rows = len(df)
df = df.replace(0, np.nan).dropna(axis=0).reset_index(drop=True)
remove_rows = start_rows - len(df)
print(f'Removed {remove_rows} rows that had incomplete pieces of data.')
print(f'This was {round(remove_rows/start_rows * 100,2)}% of the total data.')
print(f'Number of countries: {len(set(df.country.values))}')


# In[ ]:


country_dict = {i.alpha_2: i.alpha_3 for i in pycountry.countries}
countries = pd.DataFrame(df.country.value_counts()).T              .drop('NONE', axis=1)              .rename(columns=country_dict, index={'country': 'count'})
countries


# In[ ]:


countries_rank = countries.T.rename_axis('iso_a3').reset_index()
countries_rank['rank'] = countries_rank['count'].rank()
countries_rank.T


# In[ ]:


sns.set_style("white")

file = gpd.datasets.get_path('naturalearth_lowres')
world = gpd.read_file(file)
world = pd.merge(world, right=countries_rank, how='left', on='iso_a3').fillna(0)

fig, ax = plt.subplots(figsize=(20,10))
# sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=world['rank'].min(), vmax=world['rank'].max()))
# sm.set_array([])
# fig.colorbar(sm)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Countries who completed the assessment (by rank)', size=16)
world.drop(159).plot(column='rank', cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8'); sns.set()
plt.box(on=None)


# In[ ]:


pos_questions = [ # positive questions adding to the trait.
    'EXT1','EXT3','EXT5','EXT7','EXT9',                       # 5 Extroversion
    'EST1','EST3','EST5','EST6','EST7','EST8','EST9','EST10', # 8 Neuroticism
    'AGR2','AGR4','AGR6','AGR8','AGR9','AGR10',               # 6 Agreeableness
    'CSN1','CSN3','CSN5','CSN7','CSN9','CSN10',               # 6 Conscientiousness
    'OPN1','OPN3','OPN5','OPN7','OPN8','OPN9','OPN10',        # 7 Openness
]
neg_questions = [ # negative (negating) questions subtracting from the trait.
    'EXT2','EXT4','EXT6','EXT8','EXT10', # 5 Extroversion
    'EST2','EST4',                       # 2 Neuroticism
    'AGR1','AGR3','AGR5','AGR7',         # 4 Agreeableness
    'CSN2','CSN4','CSN6','CSN8',         # 4 Conscientiousness
    'OPN2','OPN4','OPN6',                # 3 Openness
]

df[pos_questions] = df[pos_questions].replace({1:-2, 2:-1, 3:0, 4:1, 5:2})
df[neg_questions] = df[neg_questions].replace({1:2, 2:1, 3:0, 4:-1, 5:-2})
cols = pos_questions + neg_questions
df = df[sorted(cols)]
df.head()


# In[ ]:


traits = ['EXT', 'EST', 'AGR', 'CSN', 'OPN']
trait_labels = ['Extroversion', 'Neuroticism', 'Agreeableness', 'Conscientiousness', 'Openness']

for trait in traits:
    trait_cols = sorted([col for col in df.columns if trait in col and '_E' not in col])
    df[trait] = df[trait_cols].sum(axis=1)
df[traits].head()


# In[ ]:


fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(18,9))
plt.subplots_adjust(left=None, bottom=None, right=None, top=1.3, wspace=None, hspace=None)
row = -1; col = 2
for i, (trait, label) in enumerate(zip(traits, trait_labels)):
    if not i % 2:
        row += 1
    if not i % 2:
        col -= 2
    i += col
    sns.distplot(df[trait], ax=axs[row][i], axlabel='', kde=False, bins=40).set_title(label, pad=10)
fig.delaxes(axs[2][1])


# In[ ]:


# Correlations
df[traits].rename(columns={k:v for k, v in zip(traits, trait_labels)}).corr()


# In[ ]:


sns.pairplot(df[traits].rename(columns={k:v for k, v in zip(traits, trait_labels)}).sample(250), diag_kind="kde", kind="reg", markers=".");


# In[ ]:


fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(33,5))
for i, (trait, label) in enumerate(zip(traits, trait_labels)):
    trait_cols = sorted([col for col in cols if trait in col and '_E' not in col])
    g = df[trait_cols].apply(lambda col: col.value_counts()).T.plot(kind='bar', stacked=True, ax=axs[i])
    g.set(yticklabels=[], title = label + ' Questions')
    if not i:
        g.legend(loc='center left', bbox_to_anchor=(-.25, 0.5), ncol=1)
    else:
        g.legend_.remove()


# ## Question mappings

# In[ ]:


question_schema = {
    'EXT1':'P', 'EXT2':'N', 'EXT3':'P', 'EXT4':'N', 'EXT5' :'P',
    'EXT6':'N', 'EXT7':'P', 'EXT8':'N', 'EXT9':'P', 'EXT10':'N',
    'EST1':'P', 'EST2':'N', 'EST3':'P', 'EST4':'N', 'EST5' :'P',
    'EST6':'P', 'EST7':'P', 'EST8':'P', 'EST9':'P', 'EST10':'P',
    'AGR1':'N', 'AGR2':'N', 'AGR3':'N', 'AGR4':'P', 'AGR5' :'N',
    'AGR6':'P', 'AGR7':'N', 'AGR8':'P', 'AGR9':'P', 'AGR10':'P',
    'CSN1':'P', 'CSN2':'N', 'CSN3':'P', 'CSN4':'N', 'CSN5' :'P',
    'CSN6':'N', 'CSN7':'P', 'CSN8':'P', 'CSN9':'P', 'CSN10':'P',
    'OPN1':'P', 'OPN2':'N', 'OPN3':'P', 'OPN4':'N', 'OPN5' :'P',
    'OPN6':'N', 'OPN7':'P', 'OPN8':'P', 'OPN9':'P', 'OPN10':'P',
}


# ```
# EXT1	P	I am the life of the party.
# EXT2	N	I don't talk a lot.
# EXT3	P	I feel comfortable around people.
# EXT4	N	I keep in the background.
# EXT5	P	I start conversations.
# EXT6	N	I have little to say.
# EXT7	P	I talk to a lot of different people at parties.
# EXT8	N	I don't like to draw attention to myself.
# EXT9	P	I don't mind being the center of attention.
# EXT10   N	I am quiet around strangers.
# EST1	P	I get stressed out easily.
# EST2	N	I am relaxed most of the time.
# EST3	P	I worry about things.
# EST4	N	I seldom feel blue.
# EST5	P	I am easily disturbed.
# EST6	P	I get upset easily.
# EST7	P	I change my mood a lot.
# EST8	P	I have frequent mood swings.
# EST9	P	I get irritated easily.
# EST10   P	I often feel blue.
# AGR1	N	I feel little concern for others.
# AGR2	P	I am interested in people.
# AGR3	N	I insult people.
# AGR4	P	I sympathize with others' feelings.
# AGR5	N	I am not interested in other people's problems.
# AGR6	P	I have a soft heart.
# AGR7	N	I am not really interested in others.
# AGR8	P	I take time out for others.
# AGR9	P	I feel others' emotions.
# AGR10   P	I make people feel at ease.
# CSN1	P	I am always prepared.
# CSN2	N	I leave my belongings around.
# CSN3	P	I pay attention to details.
# CSN4	N	I make a mess of things.
# CSN5	P	I get chores done right away.
# CSN6	N	I often forget to put things back in their proper place.
# CSN7	P	I like order.
# CSN8	N	I shirk my duties.
# CSN9	P	I follow a schedule.
# CSN10   P	I am exacting in my work.
# OPN1	P	I have a rich vocabulary.
# OPN2	N	I have difficulty understanding abstract ideas.
# OPN3	P	I have a vivid imagination.
# OPN4	N	I am not interested in abstract ideas.
# OPN5	P	I have excellent ideas.
# OPN6	N	I do not have a good imagination.
# OPN7	P	I am quick to understand things.
# OPN8	P	I use difficult words.
# OPN9	P	I spend time reflecting on things.
# OPN10   P	I am full of ideas.```

# In[ ]:




