#!/usr/bin/env python
# coding: utf-8

# I am really new to kaggle(started 3 days back) and this is my first ever attemp to analyse a dataset by myself.I didn't know where to start to start from so I picked up this ataset and I found it really interesting . The approach that I have taken might not be the most appropriate or optimum one. Please feel free to comment on that. If you find anything wrongor you have some queries please feel free to point out that too.
# 
# ## WTA Matches and Rankings
# This dataset has records for women tennis players mostly after the years of formation of WTA in 1973. Many of the datasets given has only records after the year 2000.In this notebook I have tried to analyse the countries which are most dominant in world of womens tennis. I have also to figure out who were the most dominant players of all time in womens tennis. Some insight has also been given about the players who made the biggest climb in rankings.
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from collections import Counter
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


matches = pd.read_csv('../input/extracted-dataset/matches.csv')
matches.head()


# In[ ]:


matches.columns


# In[ ]:


players = pd.read_csv('../input/extracted-dataset/players.csv',encoding='latin1',index_col=0)
players.head()


# In[ ]:


players.columns


# In[ ]:


q_matches = pd.read_csv('../input/extracted-dataset/qualifying_matches.csv')
q_matches.head()


# In[ ]:


q_matches.columns


# In[ ]:


ranks = pd.read_csv('../input/extracted-dataset/rankings.csv')
ranks.head()


# In[ ]:


ranks.columns


# ### Top 20 Countries with most players in WTA
# I stated by figuring out which countries are biggest powerhouses in womens tennis with most number of players. The conclusion I was able to draw was most players in womens tennis comes predominantly from North American and European Countries with few exceptions like Japan, India, Brazil.But it seems like USA is ahead of other countries by a big margin.

# In[ ]:


colours = list()
for i in range(0,10):
    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))
players['country_code'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title="Countries with most players in WTA")
plt.show()


# ### Top 20 players with most wins after year 2000
# Secondly I tried to figure out which were the most dominant players in womens tennis after the year 2000(the dataset had records only after that).It seems line Williams sister has been most dominant in womens tennis for more than last decade. This is not a suprise as Serena Williams is considered as one of the most dominant female tennis players of all time.

# In[ ]:


colours = list()
for i in range(0,10):
    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))
matches['winner_name'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title="Top 20 players with most wins")
plt.show()


# ### Top 20 players with most loses after year 2000
# I also tried to take a look at players with most losses in womens tennis.

# In[ ]:


colours = list()
for i in range(0,10):
    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))
matches['loser_name'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title="Top 20 players with most losses")
plt.show()


# ### Handedness of players
# I got a bit curious when I came accross the question as to which hand is more preferred by tennis players over the other. I think most players prefer right hand over left by quite a margin.
# #### Can someone explain to me what does U mean?

# In[ ]:


players['hand'].value_counts().plot.bar(color={'#8470ff','#3cb371','#ff4500'},figsize=(10,6),title="Handedness of players")
plt.show()


# ### Top 20 countries with most wins in WTA
# USA might be the player with most players in WTA but does it do well when it comes to performance. It turns out that Russia outperforms USA in womens tennis despite its comaparatively low player base in comaprision to USA.

# In[ ]:


colours = list()
for i in range(0,10):
    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))
matches['winner_ioc'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title=" Top 20 countries with most wins in WTA")
plt.show()


# ### Top 20 countries with most loses in WTA
# It seems USA tops the chart for most losses but it isn't a surprise looking at the fact that it has the biggest payer base in WTA. But it seems like when it comes to womens tennis Russia and US go neck to neck.

# In[ ]:


colours = list()
for i in range(0,10):
    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))
matches['loser_ioc'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title=" Top 20 countries with most loses in WTA")
plt.show()


# ### Countries in top 20 of rankings
# USA is the country with biggest player base in WTA but does it translate to indivudual rankings also. The data doesn't say otherwise. USA has got the most players in Top 20 by a big margin over the second that is Russia.

# In[ ]:


top_20 = ranks[ranks['ranking']<=20]
top_20['country_ioc'] = 0
top_20['player_id'] = top_20['player_id'].astype(int)
country = []
for index,row in top_20.iterrows():
    country.append((players.loc[row['player_id']])['country_code'])
    
top_20['country_ioc'] = country


# In[ ]:


colours = list()
for i in range(0,10):
    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))
top_20['country_ioc'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title=" Top 20 countries with highest rankings in WTA")
plt.show()


# ### Countries in Top 5 of rankings
# If you filter the data for Top 5 ranks USA still tops by a big margin

# In[ ]:


top_5 = ranks[ranks['ranking']<=5]
top_5['country_ioc'] = 0
top_5['player_id'] = top_5['player_id'].astype(int)
country = []
for index,row in top_5.iterrows():
    country.append((players.loc[row['player_id']])['country_code'])
    
top_5['country_ioc'] = country

colours = list()
for i in range(0,10):
    colours.append((random.random(), random.random(), random.random(),random.uniform(0.8,1)))
top_5['country_ioc'].value_counts().head(20).plot.bar(color=colours,figsize=(10,6),title=" Top 5 countries with highest rankings in WTA")
plt.show()


# ### Most Dominant Female Players in Tennis of all time
# Next I tried to figure out who were the most dominant individual female tennis players of all time. The data spoke highly of Steffi Graff, Martina Navratilova and Serena Williams which is not a surprise as all these players are all time best players in womens tennis.

# In[ ]:


names = players
names['name'] = players['first_name'] + ' ' + players['last_name']


# In[ ]:


dom_5 = ranks[ranks['ranking']<=5]
dom_5['name'] = 0
dom_5['player_id'] = dom_5['player_id'].astype(int)
name_list = []
for index,row in dom_5.iterrows():
    name_list.append((names.loc[row['player_id']])['name'])
    
dom_5['name'] = name_list


# In[ ]:


dom_dict = dict()
for index,row in dom_5.iterrows():
    if row['ranking'] in dom_dict:
        dom_dict[row['ranking']] = row['name'] + '  ' + dom_dict[row['ranking']]
    else:
        dom_dict[row['ranking']] = row['name']
        
for key in dom_dict:
    dom_dict[key] = dom_dict[key].split("  ")


# In[ ]:


top5dom = dict()
for key in dom_dict:
    d = Counter(dom_dict[key])
    top5dom[key] = d.most_common(3)
    
for key in top5dom:
    print("Top 3 players for WTA Rank {0} :{1}".format(key,top5dom[key]))


# ### Q&A's
# ### Which player did the most rapid climb through the ranks during these years?
# Next I tried to figure out the players who climbed the most in rankings in WTA.

# In[ ]:


climb = players
comp = ranks
climb['rank_diff'] = 0
comp.dropna(subset=['player_id'], inplace = True)
comp['player_id'] = comp['player_id'].astype(int)


# In[ ]:


rank_diff = list()
for index,row in climb.iterrows():
    temp = comp[comp['player_id'] == index]
    rank_diff.append(temp['ranking'].max()-temp['ranking'].min())


# In[ ]:


climb['rank_diff'] = rank_diff
climb.sort_values(['rank_diff'],ascending=False,inplace=True)
print("These are the players with biggest climbs in womens tennis history:")
print(climb.head())


# ### How does age correlate with winrates?
# Is there a certain age when athletes win the most or does age affect performance. The following analyses answered those questions pretty well. It seems like there is a sweetspot where you get a perfect blend of physical fitness and experience.

# In[ ]:


mod_age = round(matches['winner_age'])
mod_age = mod_age.dropna()
mod_age = mod_age.astype(int)
mod_age.plot.hist(title="Age at which players are winnig the most")
plt.show()


# ### Does the rank correlates with the money earn by the player?
# ### There is some deterministic factor to own the match?
# I found the data to be insufficient to answer the above two questions. Perhaps someone has some idea please comment. I will end my analyses here. Please feel free to comment regarding ideas,queries,suggestions,complaints.
