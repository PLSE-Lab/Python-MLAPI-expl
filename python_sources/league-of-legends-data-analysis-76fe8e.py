#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# LoL analysis code
# import libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import json
sns.set_style('darkgrid')

# Use this in notebook to show plots
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def idToName(ID, dic):
    '''
    ID: champion ID as listed in original dataframe
    dic: champInfo from json, defined above

    used to convert ID's to champion names
    '''
    champ = dic['name'][ID]
    return champ

def getTag(name, data):
    '''
    name: champion name as listed in dataframe
    data: champInfo

    used to get primary tag from champInfo
    '''
    tags = data['tags'][name][0]
    return tags

def numToColor(data):
    '''
    data: main dataframe

    used to get color of team from 0 or 1
    '''
    if data == 0:
        color = 'blue'
    else:
        color = 'red'
    return color


# ## import original data

# In[ ]:


data = pd.read_csv('../input/games.csv')
data.head()


# ## import champion info json and grab the data

# In[ ]:


jDict = pd.read_json('../input/champion_info_2.json')
champInfo = pd.read_json((jDict['data']).to_json(), orient='index')
champInfo.head()


# In[ ]:


spellJson = pd.read_json('../input/summoner_spell_info.json')
spellInfo = pd.read_json((spellJson['data']).to_json(),orient='index')
spellInfo.head()


# ### Temporarily set index to id 

# In[ ]:


champInfo.set_index(['id'], inplace=True)
champInfo.head()


# ## create list of columns of user picks and another list for bans

# In[ ]:


champCols = ['t1_champ1id','t1_champ2id','t1_champ3id','t1_champ4id','t1_champ5id',
             't2_champ1id','t2_champ2id','t2_champ3id','t2_champ4id','t2_champ5id']
banCols = ['t1_ban1','t1_ban2','t1_ban3','t1_ban4','t1_ban5',
             't2_ban1','t2_ban2','t2_ban3','t2_ban4','t2_ban5',]
sumSpellsCols = ['t1_champ1_sum1','t1_champ1_sum2','t1_champ2_sum1','t1_champ2_sum2','t1_champ3_sum1','t1_champ3_sum2',
                 't1_champ4_sum1','t1_champ4_sum2','t1_champ5_sum1','t1_champ5_sum2','t2_champ1_sum1','t2_champ1_sum2',
                 't2_champ2_sum1','t2_champ2_sum2','t2_champ3_sum1','t2_champ3_sum2','t2_champ4_sum1','t2_champ4_sum2',
                 't2_champ5_sum1','t2_champ5_sum2']


# ## apply the idToName function for these columns so we have champion names rather than ID's

# In[ ]:


for c in champCols:
    data[c] = data[c].apply(lambda x: idToName(x, champInfo))

for c in banCols:
    data[c] = data[c].apply(lambda x: idToName(x, champInfo)) 

for c in sumSpellsCols:
    data[c] = data[c].apply(lambda x: idToName(x, spellInfo))


# In[ ]:


data[champCols].head()


# In[ ]:


data[banCols].head()


# In[ ]:


data[sumSpellsCols].head()


# ## Set champInfo dataframe index to champion names

# In[ ]:


champInfo.set_index(['name'],inplace=True)
champInfo.head()


# ## apply the getTag function for these columns so we have new primary champion tags columns

# In[ ]:


for col in champCols:
    data[col + '_tags'] = data[col].apply(lambda x: getTag(x, champInfo))
data.head()


# ## Let's look at champion picks and bans # 

# Create sorted series of the picks and bans, as well as a series of the primary tag for each champion pick

# In[ ]:


sumPicks = pd.concat([data['t1_champ1id'],data['t1_champ2id'],data['t1_champ3id'],data['t1_champ4id'],data['t1_champ5id'],
                      data['t2_champ1id'],data['t2_champ2id'],data['t2_champ3id'],data['t2_champ4id'],data['t2_champ5id']],
                      ignore_index=True)
sortedPicks = sorted(sumPicks)
sumBans = pd.concat([data['t1_ban1'],data['t1_ban2'],data['t1_ban3'],data['t1_ban4'],data['t1_ban5'],
                     data['t2_ban1'],data['t2_ban2'],data['t2_ban3'],data['t2_ban4'],data['t2_ban5']],
                     ignore_index=True)
sortedBans = sorted(sumBans)


# ## Let's make a countplot for total champion picks and bans over the entire dataset #

# In[ ]:


fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=False, figsize=(15,30))
plt.xticks(rotation=90)
sns.countplot(y=sortedPicks, data=data, ax=ax1)
sns.countplot(y=sortedBans, data=data, ax=ax2)
ax1.set_title('Champion Picks')
ax2.set_title('Champion Bans')


# ### Interesting to see how many popular Thresh and Tristana have become, as well as how much people REALLY don't like to play against Yasuo, Zed, Cho'Gath and Darius!

# ## Just for fun, a countplot of types of champions and summoner spells used ##

# In[ ]:


tagsCols = ['t1_champ1id_tags', 't1_champ2id_tags', 't1_champ3id_tags',
       't1_champ4id_tags', 't1_champ5id_tags', 't2_champ1id_tags',
       't2_champ2id_tags', 't2_champ3id_tags', 't2_champ4id_tags',
       't2_champ5id_tags']


# In[ ]:


tagsTotals = data[tagsCols].apply(pd.value_counts)
tagsTotals['count'] = tagsTotals[tagsCols].sum(axis=1)
tagsTotals


# In[ ]:


sns.barplot(x=tagsTotals.index,y=tagsTotals['count'])


# ### Not entirely surprising, but still interesting to see! ###

# In[ ]:


spellsTotals = data[sumSpellsCols].apply(pd.value_counts)
spellsTotals['count'] = spellsTotals[sumSpellsCols].sum(axis=1)
spellsTotals


# In[ ]:


spellColors = ["#6E2C00","#1A5276","#9A7D0A","#F1C40F","#3498DB","#58D68D","#E74C3C","#F39C12","#8E44AD"]
sns.barplot(x=spellsTotals.index,y=spellsTotals['count'],palette=spellColors)


# ## Looks like if you don't have flash you're clearly in the minority! ##

# ## Let's look at who is getting most of the 'first' objectives! ##
# We need to create a new dataframe with the team colors as the values to make plotting easier later on

# In[ ]:


dataClean = data.replace([0,1,2],['neither','blue','red'])
dataClean.head()


# ### Here, we take the columns we want and get a value count on them, then reindex them to make the plot look cleaner

# In[ ]:


firsts = ['firstBlood','firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald']
firstTotals = dataClean[firsts].apply(pd.value_counts)
newIndex = ['blue','red','neither']
firstSort = firstTotals.reindex(index=newIndex)
firstSort


# ### For the plots, we'll create a custom color palette, then create the subplots with a for loop

# In[ ]:


plotColors = ['#3498DB','#E74C3C','#BDC3C7']
firstLabels = ['First Blood','First Tower', 'First Inhibitor', 'First Baron', 'First Dragon', 'First Rift Herald']
nrows, ncols = 2,3
fig = plt.figure(figsize=(15,10))
for i in range(1,7):
    ax = fig.add_subplot(nrows,ncols,i)
    sns.barplot(x=firstSort.index,y=firstSort[firstSort.columns[i-1]],palette=plotColors)
    ax.set_ylabel('Count')
    ax.yaxis.set_ticklabels([])
    ax.set_title(firstLabels[i-1])


# ### Interesting to see. Here are some points I see worth making:
# * Most of the plots are pretty even, but the difference in first baron could be due to the proximity of the baron pit to the red base, and the fact that their positioning makes it easier to steal? 
# * The 'neither' count for first baron shows that over half of the games ended without a baron being taken.
# * First Blood is slightly in favor of the blue team. Perhaps the blue team has an edge early game in terms of jungle control? 
# 
# ### I'd love to hear what you guys think!
# 

# ## I want to see if badass sounding spells actually do more damage
# to do that I'll have to figure out which spells have sentiment that would qualify as "badass". I'm going to use `nltk` and look for the most bad-ass ones then figure out which spells were used in the most victories. 

# In[ ]:


import pandas as pd
import nltk


# In[ ]:


# the spells and their descriptions
# goes: 
# data : { desc... }
spellJson = pd.read_json('../input/summoner_spell_info.json')
spellInfo = pd.read_json((spellJson['data']).to_json(),orient='index')
spellInfo.head()


# In[ ]:


# tokenizing words 
def tokenier(df):
    newdf = df.copy()
    for row, index in df.iterrow():
        sentence = df['description']
        tokens = nltk.word_tokenize(sentence)
        tokens
        


# In[ ]:


tokenizer('spellInfo')


# In[ ]:


# or maybe I should use keras
# it's pretty hot 
from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# tokenize the document
result = text_to_word_sequence(text)
print(result)


# In[ ]:




