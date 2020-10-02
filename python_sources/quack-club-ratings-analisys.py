#!/usr/bin/env python
# coding: utf-8

# # [Quack Clube de Jogos](https://quack.com.br/)
# 
# I have a group of friends who make a [podcast/youtube channel](https://quack.com.br/) about PC games, they have been making it since 2016 and have more than 200 episodes.
# 
# In one of the courses of [Dataquest](https://www.dataquest.io/), we do some analisys on [Metacritic](https://www.metacritic.com/), [Imdb](https://www.imdb.com/) and some others 'movie reviews' site, and that got me thinking about them, do they give a similar score then Metacritic? Can I find what every host favorite genre is? Do one of the host always give a lower or higher score than the rest of the guys?
# 
# Let's find out.

# ## Table of Contents
# - 1. Importing and preprocessing
#     - 1.1. Importing data
#     - 1.2. Data cleaning
# - 2. Web Scraping
#     - 2.1. API
#     - 2.2. More data cleaning
# - 3. Data Analysis
#     - 3.1. Quack VS Metacritic
#     - 3.2. Quack VS DateTime
#         - 3.2.1. Oldest Game
#         - 3.2.2. Newest Game
#         - 3.2.3. Max Delta Game
#         - 3.2.4. Min Delta Game
#         - 3.2.5. Histogram
#     - 3.3. Quack VS Genre
#         - 3.3.1. Quack Most Reviewed Genre
#         - 3.3.2. Quack and Hosts Favorite Genre
# - 4. Conclusion
#     - 4.1. Export
#     - 4.2. TL;DR
#     - 4.3. Objective & Final Considerations

# ## 1. Importing and preprocessing
# 
# Every week, one of the hosts picks a game to review. 
# They dont have rules for picking a game, normaly is a PC game, but there where instances of mobile games and other plataforms.
# They have played new and old games alike.
# 
# ### 1.1. Importing data
# 
# My friends have a google shet with the number of the episode, data, title of the game, who picked, Metacritic ratings, and each host ratings.
# Since they do it manualy, just on the initial lookup, I can see that a lot of the Metacritic ratings are not present, some scores are missing, we don't have all the episodes, etc.
# 
# Let's import the dataset and then we can run some cleaning.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # beautiful graphs
import matplotlib.pyplot as plt #math stuf
import datetime as dt #for date math

from kaggle_secrets import UserSecretsClient # to hide my API key
user_secrets = UserSecretsClient() # to hide my API key

#Let's import the dataset
df = pd.read_csv('/kaggle/input/quack-club-score-original/quack_club_score.csv')
print(df.describe)
print('\n**************************************************************************\n')
print(df.describe(include='all'))


# ### 1.2. Data cleaning
# 
# Ok, we are going to do a lot of cleaning here, let's go by parts:
# 
# - The columns don't follow the standard python names (all lowercases)
# - Arara is the only one who gave a zero score, since it is only in one datapoint, I went and confirmed the value on the podcast. He gave a zero because he had could not play the game, so we are gonna turn that to a null value (NaN) to not change his mean value.
# - Some Metacritic ratings are missing and marked with '?', let's just remove the '?' characters.
# - Some picks (who picked the game that week) are empty, since there are only 3 unmarked, I checked on the podcast manualy.

# In[ ]:


# Lowercase content
df = df.applymap(lambda s:s.lower() if isinstance(s, str) else s)

# Lowercase columns
df.columns = map(str.lower, df.columns)

# Remove arara zero
df = df.replace(0,np.NaN)

# Remove '?' char
df = df.replace('?',np.NaN)

# Add picks
df.loc[df['ep']==209,'pick'] = 'madz'
df.loc[df['ep']==210,'pick'] = 'storm'
df.loc[df['ep']==211,'pick'] = 'cosmos'


# - From episode 189, Cosmos entered the podcast, but he seen's to be marked as a Partner, and from episode 198 he is marked as Rune.

# In[ ]:


# Fix Cosmos scores
for ep in range(189,198):
    df.loc[df['ep']==ep,'cosmos'] = df.loc[df['ep']==ep,'partner']
    df.loc[df['ep']==ep,'partner'] = np.nan
    df.loc[df['ep']==ep,'partner'] = df.loc[df['ep']==ep,'partner 2']
    df.loc[df['ep']==ep,'partner 2'] = np.nan
    
for ep in range(198,212):
    df.loc[df['ep']==ep,'cosmos'] = df.loc[df['ep']==ep,'rune']
    df.loc[df['ep']==ep,'rune'] = np.nan


# - Some dates are missing, since we are only missing 15 datapoints, we are going to fill this one by hand.

# In[ ]:


dates = {
    198:'03/01/20',
    199:'10/01/20',
    200:'17/01/20',
    201:'24/01/20',
    202:'31/01/20',
    203:'07/02/20',
    204:'14/02/20',
    205:'21/02/20',
    206:'28/02/20',
    207:'06/03/20',
    208:'13/03/20',
    209:'20/03/20',
    210:'27/03/20',
    211:'03/04/20'
}

for ep in dates:
    df.loc[df['ep']==ep,'date'] = dates[ep]


# Ok, that should be it for cleaning.
# Let's ditch the last row, since it's not complete.

# In[ ]:


df = df[:-1]


# ## 2. Web Scraping
# ### 2.1. API
# 
# **Chicken Coop API Documentation**

# In[ ]:


import requests

def get_metacrit(series):
    parameters = {
    'platform': 'pc'
    }
    
    #my_key = user_secrets.get_secret("rapidapi_key")
    
    headers = {
    'x-rapidapi-host': 'chicken-coop.p.rapidapi.com',
    'x-rapidapi-key': '5fc0af0c7amsh9ceb6a2d4914fabp15856fjsn75748452ca35'
    }

    path = 'https://chicken-coop.p.rapidapi.com/games/'
    
    name = str(series['title'])
    response = requests.get(path + name,headers=headers, params=parameters)
    game_info = response.json()
    
    if (game_info['result'] == 'No result'):
        series['error'] = 1
    else:
        game_info = game_info['result']
        series['error'] = 0
        series['title'] = game_info['title']
        series['mc'] = game_info['score']
        series['release_date'] = game_info['releaseDate']
        series['developer'] = game_info['developer']
        series['rating'] = game_info['rating']
        series['genre'] = game_info['genre']
        series['publisher'] = game_info['publisher']
        series['also_available'] = game_info['alsoAvailableOn']
    return series
            
df = df.apply(get_metacrit, axis=1)


# **Whouuuuu, that took a while.**
# 
# Let's see how many games we didn't get info on metacrit:

# In[ ]:


print(df['error'].sum())


# Ok, so we missed info on 16 datapoints, a litle less than 10%.
# 
# If we had a big dataset, we would probably have to think of a better way to check the missing ones. Probably use the api to run a search on the error games or something.
# 
# Since it's only 16 datapoints, let's do it by hand and save some time. 

# In[ ]:


print(df[df['error']==1][['ep','title']])


# So we have some common cases:
# - special characters
# - incomplete names
# - game dosent exist on metacritic
# - the game is mobile (We are only doing the PC games)
# 
# Let's change the names, drop the ones without Metacritic score, and then try again.

# In[ ]:


correct_title={
    45:'odallus the dark call',
    53:np.nan,
    66:'abzu',
    71:'valkyrie drive bhikkhuni',
    107:np.nan,
    115:np.nan,
    154:'ace combat 7 skies unknown',
    157:'toejam earl back in the groove',
    163:'peggle deluxe',
    177:'outer wilds',
    189:'underrail',
    192:'yooka laylee and the impossible lair',
    198:'team fortress 2',
    199:'jamestown legend of the lost colony',
    202:'halo reach remastered',
    204:'nioh complete edition'
}

for ep in correct_title:
    df.loc[df['ep']==ep,'title'] = correct_title[ep]
df = df.dropna(subset=['title'], axis='index')

df[df['error']==1] = df[df['error']==1].apply(get_metacrit, axis=1)


# Nice!
# 
# Unfortunately the realase date and the date of the podcast are not on datetime standart. Let's fix that.
# 
# ### 2.2 More data cleaning
# 
# Let's first check if the date of the podcast is not up to standard.

# In[ ]:


print('date column')
print(type(df['date'][1]))
print(df['date'][1])

print('\nrelease date column')
print(type(df['release_date'][1]))
print(df['release_date'][1])


# Ok, it's not datetime, we could go back and import the .csv better, but since we will have to change the relase date anyway, let's do it all here.

# In[ ]:


df['date'] = pd.to_datetime(df['date'],dayfirst=True,format='%d/%m/%y')


# Ok, the date is now on datetime type. Moving on.
# 
# The release date has some datapoints with a 'TBA - Early Access' mark. Since it's just four datapoints, again I just resolved by hand.
# Another option would be to find another API, maybe [**Steam Web API**](https://developer.valvesoftware.com/wiki/Steam_Web_API). But let's leave that to another day and just move on.

# In[ ]:


correct_release_date = {
    80:'Aug 24, 2017',
    162:'Feb 25, 2016',
    205:'Dec 13, 2019',
    209:'Feb 26, 2020'
}

for ep in correct_release_date:
    df.loc[df['ep']==ep,'release_date'] = correct_release_date[ep]

df['release_date'] = pd.to_datetime(df['release_date'],format='%b %d, %Y')


# Done. I think this is it for data cleaning...
# 
# ## 3. Data Analysis
# 
# ### 3.1. Quack VS Metacritic
# 
# Ok, so....what do we want to know?
# Let's start with the basics and do a simple analysis on the score.
# 
# **Quack** uses a 10 points score (with no half-point). Metacritic uses a 100 points score.
# Since we have 5-7 columns of folks from quack and only one from metacritic, let's just divide the metacritic score by 10.
# 
# Then, let's just do a boxplot to see how each person score is distributed.

# In[ ]:


df['mc'] = df['mc'].astype(float)/10.0


# In[ ]:


df_score = pd.melt(df, value_vars=['mc', 'arara', 'madz','storm','rune','cosmos','partner'])

sns.set_style('darkgrid')
sns.set(font_scale=2)
plt.figure(figsize=(20, 8))

ax = sns.boxplot(x='variable',y='value',orient='v',data=df_score,fliersize=8)
#ax = sns.swarmplot(x='variable', y='value', orient='v',data=df_score, size=3, color=".3", linewidth=0)

ax.set(xlabel='', ylabel='score')


# In[ ]:


df[['mc', 'arara', 'madz','storm','rune','cosmos','partner']].describe()


# Ok, so this graphic and the shows us some interesting things:
# - Metacritic has a bigger standard deviation, this is beacause a lot of the games have a 0 (Non-existing) score. *How the hell 'Cave Story' doesn't have a Metacritic score???*
# - Without all the zeros, metacritic has a smaller standard deviation, wich is not something I want on a reviewer.
# - The guys from Quack give on average a similar score among themselves, with approximately the same mean, standard deviation, and so on. Except Cosmos, but that is probably beacause he is the newest member, with only 22 datapoints.
# - Cosmos and Arara seen to give bigger scores.
# 
# Now let's check metacritic against Quack mean scores.

# In[ ]:


df['quack_mean'] = df[['arara', 'madz','storm','rune','cosmos','partner','partner 2']].mean(skipna=True, axis='columns')

df_nozero = df[df['mc'] != 0]

print('Pearson correlation coefficient: ')
print(df_nozero['quack_mean'].corr(df_nozero['mc']))

sns.set_style('darkgrid')
sns.set(font_scale=2)
plt.figure(figsize=(10, 10))
ax = sns.scatterplot(x='quack_mean', y='mc', data=df_nozero, s=150)

plt.plot([0, 10], [0, 10], linewidth=2, alpha=.5, color='green')

m, b = np.polyfit(df_nozero['quack_mean'], df_nozero['mc'], 1)
plt.plot(df_nozero['quack_mean'],b+m*df_nozero['quack_mean'], linewidth=2, alpha=.5, color='red')

ax.set(xlabel='Quack', ylabel='Metacritic',xlim=(-0.5,10.5),ylim=(-0.5,10.5))


# As we can see, the correlation between the Metacritic and Quack is weak (r=0.44).  
# In the graph we can see that as the score diminishes, Metacritic gives higher scores compared to Quack.  
# Quack seems to be more rigorous for 'bad' games.
# 
# Just out of curiosity, let's see what are these games that have a Quack Score of less than 4 but have a high score on Metacritic.

# In[ ]:


df_nozero[df_nozero['quack_mean'] < 4][['title','mc','quack_mean']].sort_values(by=['mc'],ascending=False)


# ### 3.2. Quack VS DateTime
# 
# We have the release date of the game and the release date of the episode, let's see how far apart those two are, more recent game, oldest game, etc.
# 
# #### 3.2.1. Oldest Game

# In[ ]:


df[df['release_date'] == df['release_date'].min()][['title','release_date','quack_mean','mc']]


# #### 3.2.2. Newest Game

# In[ ]:


df[df['release_date'] == df['release_date'].max()][['title','release_date','quack_mean','mc']]


# #### 3.2.3. Max Delta Game
# 
# Let's check the game with the biggest time between launch and review by Quack.

# In[ ]:


#We removed some rows while cleaning, so lets reset the index
df.reset_index(inplace=True)

print('Delta: ' + str((df['date'] - df['release_date']).max().days) + ' days\n')

date_max = (df['date'] - df['release_date']).idxmax()
print(df.iloc[[date_max]][['title','release_date','date','quack_mean','mc']])


# #### 3.2.4. Min Delta Game
# 
# Let's check the game with the least time between launch and review by Quack.

# In[ ]:


print('Delta: ' + str((df['date'] - df['release_date']).min().days) + ' days\n')

date_min = (df['date'] - df['release_date']).idxmin()
print(df.iloc[[date_min]][['title','release_date','date','quack_mean','mc']])


# ### God dammit! We did it guys! **We invented time travel!**
# 
# Nahh.
# *'One Hour One Life'* is a game lauched first on a website (on 27 February 2018), then the publisher lauched on Steam (on 8 November 2018).
# Metacritic apparently considers the launch date on Steam, and Quack played on the website version.
# 
# #### 3.2.5. Histogram

# In[ ]:


df['delta_date'] = (df['date'] - df['release_date']).dt.days

plt.figure(figsize=(10, 10))
ax = sns.distplot(df['delta_date'], bins=20, kde=False, norm_hist=False)
ax.set(xlabel='Days', ylabel='Frequency')


# As we can see, Quack plays games mostly in the five year span, but has no problem playing some games 20 years older.
# Lets zoom in and see if we have any details on a three year span.

# In[ ]:


plt.figure(figsize=(10, 10))
ax = sns.distplot(df[df['delta_date']<=1000.0]['delta_date'], bins=20, kde=False, norm_hist=False)
ax.set(xlabel='Days', ylabel='Frequency')


# In[ ]:


print('Games reviewed before one year delta: ' + str(df[df['delta_date']<=365.0]['delta_date'].count()))
print('Games reviewed after one year delta: ' + str(df[df['delta_date']>365.0]['delta_date'].count()))


# We can see that Quack does a good mix of recent and old games. While we have 107 datapoints on the one year mark, we have 48 datapoints that are evenly distributed on a 20 year span.

# ### 3.3. Quack VS Genre
# 
# We got a list of genres on the metacritic site, let's see what are the prefered genres for each host,etc
# 
# #### 3.3.1. Quack Most Reviewed Genre
# 
# What is the most reviewed genre of game on Quack?
# 
# This one is complicated to get. We have a series of lists with the genres of the games, since every game can have multiple genres.
# Since lists are not hashable, we cant just do a *value_counts* and be happy. 
# 
# First we flatten the list, then we change it to a Series and use *value_counts*.

# In[ ]:


import operator 
from functools import reduce


genre_list = [item for sublist in df['genre'].dropna() for item in sublist]
pd.Series(genre_list).value_counts(normalize=False).head(10)


# So, Quack reviews mainly Action, 2D, RPG and Plataformers.

# #### 3.3.2. Quack and Hosts Favorite Genre
# Now lets find out if the host have a favorite genre. 
# 
# This one is more complitaced then the last one.
# How can we define a favorite genre for the hosts? The most reviewed genre for quack was simpler, we just wanted to find the genre that appeared the most.
# 
# But what about the *favorite* genre?
# 
# We could sum the rating giving by the host for every game of the genre and then calculate the mean and compare for genres. But what if they reviewed a realyyyyyy bad game of that genre? That would smear the results for the genre. If a host gave a bad rating to a bad game, does it mean that the host do not like that genre? Probably not.
# 
# But if a host gave a better rating than their peers constantly for a genre, than that host probably like that genre more then the other host, right?
# 
# We could use a [z-score](https://en.wikipedia.org/wiki/Standard_score) for that.

# In[ ]:


from scipy import stats

host_list=['arara', 'madz','storm','rune','cosmos']

zs = pd.DataFrame(stats.zscore(df[host_list], nan_policy='omit', axis=0), columns=host_list)
zs['genre'] = df['genre']
zs


# Ok, now we have a *database* **zs** with the z-scores of the hosts and the genre of the games.  
# As we can see on the number below, the z-score for each host does not vary much.

# In[ ]:


for host in host_list:
    print(host+' z-score mean: '+'%.4f' % (zs[host].abs().sum()/len(zs[host].dropna())))


# Now let's see the if we have any significant variation per genre.

# In[ ]:


zs_genre = pd.DataFrame(columns=host_list, index=set(genre_list))

zs_explode = zs.explode(column='genre')
for genre in set(genre_list):
    for host in host_list:
        #zs_genre[host][genre]=zs_explode[zs_explode['genre']==genre][host].dropna().abs().sum()
        zs_genre[host][genre]=zs_explode[zs_explode['genre']==genre][host].dropna().sum()
        if len(zs_explode[zs_explode['genre']==genre][host].dropna()) != 0:
            zs_genre[host][genre]=zs_genre[host][genre]/len(zs_explode[zs_explode['genre']==genre][host].dropna())
zs_genre


# In[ ]:


for host in host_list:
    g_idxmax = zs_genre[host].astype(float).idxmax(skipna=True)
    g_max = zs_genre[host].astype(float).max()
    g_idxmin = zs_genre[host].astype(float).idxmin(skipna=True)
    g_min = zs_genre[host].astype(float).min()
    print(host+' z-score result:')
    print('MAX:  '+'%.4f' % g_max+' '+g_idxmax)
    print('MIN: '+'%.4f' % g_min+' '+g_idxmin+'\n')
    
df_explode = df.explode(column='genre')

df_explode[df_explode['genre']=='Golf'][['title','arara', 'madz','storm','rune','cosmos']]


# What the hell is going on with golf?  
# Well, Quack only played one golf game, with a big spread of the ratings. This show us that the analysis we did suffer a lot from just one value being too diferent.
# One way to fix that would be to get more data. Then, even if we had one Golf game with a big variance, the other games would hopefully clean the noise.
# 
# Since we dont have that possibility, let's say that we want only to analyse the top genres.
# For that, let's define that the top genres appear on at least 10% of the games.
# 
# Then, we will run the analysis again, but only considering the top genres.
# 
# This time, instead of just given a list, let's do a heatmap of the z-scores of the hosts.

# In[ ]:


top_genre_list = df_explode['genre'].value_counts()[df_explode['genre'].value_counts() > len(df['ep'])*0.1]
top_zs_genre = zs_genre.loc[top_genre_list.index].astype(float)

top_zs_genre

f, ax = plt.subplots(figsize=(8, 8))

#sns.heatmap(top_zs_genre, linewidths=4, cmap="YlGnBu", vmin=-0.5, vmax=0.5, annot=True, fmt='.2f')
sns.heatmap(top_zs_genre, linewidths=4, cmap="YlGnBu", vmin=-0.5, vmax=0.5)


# Ok, first some disclaimers...  
# We don't have not nearly enough datapoints to do an analysis of this type. And Cosmos has even less datapoints.
# This analysis is done more to show a cool heatmap and probably does not reflect the taste of the hosts.
# 
# But anyway...Cool insights:
# - Arara
#     - Like: 2D & Plataformers
#     - Dislike: Action RPG
# - Madz
#     - Like: Shooter
#     - Dislike: Adventure & Action RPG
# - Storm
#     - Like: 2D, Plataformers & Shooters
#     - Dislike: Strategy & Adventure
# - Rune
#     - Like: Action Adventure
#     - Dislike: Action RPG
# - Cosmos
#     - Like: Adventure & Shooters
#     - Dislike: Action Adventure & Action RPG

# ## 4. Conclusion
# 
# ### 4.1. Export
# 
# We added a lot of data, with the help of the API.
# Let's do some final export of the data and update the dataset on Kaggle.

# In[ ]:


df.to_csv('/kaggle/working/quack_club_score_v2.csv',index = False)


# ### 4.2. TL;DR
# 
# I have a group of friends who make a podcast/youtube channel about PC games, they have been making it since 2016 and have more than 200 episodes.
# 
# They:
# - Have a week (r=0.44) ratings correlation with Metacritic
# - Are more rigorous with 'bad' games
# - Played a game that was 21 years old
# - Played a game that was -90 days old
# - Play more games that are a couple months old
# 
# The hosts:
# - Cosmos and Arara seen to give bigger scores
# - Give on average a similar score among themselves
# - Have the following genre preferences:
#     - Arara
#         - Like: 2D & Plataformers
#         - Dislike: Action RPG
#     - Madz
#         - Like: Shooter
#         - Dislike: Adventure & Action RPG
#     - Storm
#         - Like: 2D, Plataformers & Shooters
#         - Dislike: Strategy & Adventure
#     - Rune
#         - Like: Action Adventure
#         - Dislike: Action RPG
#     - Cosmos
#         - Like: Adventure & Shooters
#         - Dislike: Action Adventure & Action RPG

# ### 4.3. Objective & Final Considerations
# 
# The initial objective was accomplished, since we trained some analysis tools (Cool graphs on seaborn, use of APIs, some simple statistics analysis), but well, this notebook is still a work in progress.
# 
# I will probably think about some cool question to ask our data in the future, and probably receive some feedback and questions from the host and public of the podcast.
# 
# **Finally, I would like to thank the guys from the podcast for all the good times listening to their rambles**
