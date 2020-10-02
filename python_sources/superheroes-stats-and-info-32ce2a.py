#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from IPython.display import Image
from matplotlib import pyplot as plt


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None  # default='warn'


# In[ ]:


import os
print(os.listdir("../input/marvel-superheroes"))
print(os.listdir("../input/marvel-comics"))

# Any results you write to the current directory are saved as output.


# In[ ]:


charcters_stats = pd.read_csv("../input/marvel-superheroes/charcters_stats.csv")
charcters_stats.head()


# In[ ]:


superheroes_power_matrix = pd.read_csv("../input/marvel-superheroes/superheroes_power_matrix.csv")
superheroes_power_matrix.head()


# In[ ]:


marvel_dc_characters = pd.read_excel("../input/marvel-superheroes/marvel_dc_characters.xlsx")
marvel_dc_characters.head()


# In[ ]:


marvel_characters_info = pd.read_csv("../input/marvel-superheroes/marvel_characters_info.csv")
marvel_characters_info.head()


# In[ ]:


marvel_comics = pd.read_csv("../input/marvel-comics/comics.csv")
marvel_comics.head()


# we want to build the master group to fight evil, kind of an avengers 2.0, but only better, lets select the captain, the one with the most total stats  (obviously his Alignment must be good to fight evil)

# In[ ]:


# level: easy
good_characters = charcters_stats[charcters_stats['Alignment'] == 'good']
good_characters.loc[good_characters['Total'].idxmax()][['Name','Total']]


# A great team needs great diversity, and to be great at everything, get the best hero at each statistical category

# In[ ]:


# level: easy - medium
good_characters = charcters_stats[charcters_stats['Alignment'] == 'good']
stats = ['Intelligence','Strength','Speed','Durability','Power','Combat']
max_stats_rows = []
for stat in stats:
    max_stats_rows.append(good_characters.loc[good_characters[stat].idxmax()][['Name',stat]])
pd.concat(max_stats_rows)


# Is your strngth and intelligence related?. Show a scatter chart where the x axis is stength, and the y axis is intelligence, scatter heros and villans as two different color dots****

# In[ ]:


# level: easy - medium
good = charcters_stats[charcters_stats['Alignment'] == 'good'] #fire contains all fire pokemons
bad = charcters_stats[charcters_stats['Alignment'] == 'bad']  #all water pokemins
plt.scatter(good.Strength.head(100),good.Intelligence.head(100),color='B',label='Good',marker="*",s=50) #scatter plot
plt.scatter(bad.Strength.head(100),bad.Intelligence.head(100),color='R',label="Bad",s=25)
plt.xlabel("Stength")
plt.ylabel("Intelligence")
plt.legend()
plt.plot()
fig=plt.gcf()  #get the current figure using .gcf()
fig.set_size_inches(12,6) #set the size for the figure
plt.show()


# To truly be a great superhero, you can't be a one trick pony, you need to posess multipule abilities. Create a series of every superhero and how many different abilities they posess, in descending order

# In[ ]:


# level: medium
ability_columns = [c for c in superheroes_power_matrix.columns if c != 'Name']
ability_count = [ int(sum([row[c] for c in ability_columns])) for index, row in superheroes_power_matrix.iterrows() ]
superheroes_power_matrix['Ability Count'] = ability_count
superheroes_power_matrix.set_index('Name')['Ability Count'].sort_values(ascending=False).head()


# In[ ]:


marvel_characters = pd.read_csv("../input/marvel-comics/characters.csv")
marvel_characters.head()


# In[ ]:


marvel_characters_to_comics = pd.read_csv("../input/marvel-comics/charactersToComics.csv")
marvel_characters_to_comics.head()


# **All of these questions are based on the marvel_comics datasets, not the Marvel Superheros ones**

# People will pay big money for original vintage comic books, retrive all first issue comic books

# In[ ]:


# level: easy
marvel_comics[marvel_comics['issueNumber'] == 1.0].head()


# On the other hand, long lasting series are great as well :), retrive the comic book with the biggest issue number

# In[ ]:


#level: easy
marvel_comics.loc[marvel_comics['issueNumber'].idxmax()]


# 

# It's the holiday season, and to celebrate marvel usually comes out with holiday special comic books, retrive all  holiday special comic books (the word 'Holiday' will appeer in the title)

# In[ ]:


# level: easy
marvel_comics[marvel_comics.title.str.contains('Holiday')].head() 


# Create a serires that counts the number of comic book appeerences for each hero
# Bonus: show the top 10 heros in a pie chart

# In[ ]:


#level: easy - medium
superhero_comic_performences = pd.merge(marvel_characters_to_comics,marvel_characters,left_on='characterID',right_on='id')['name'].value_counts()

#level: easy - medium
top_10 = superhero_comic_performences.nlargest(10)
labels = top_10.index.tolist()
sizes = top_10.values
colors = ['Y', 'B', '#00ff00', 'C', 'R', 'G', 'silver', 'white', 'M','gray']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.title("Top 10 comic performences")
plt.plot()
fig=plt.gcf()
fig.set_size_inches(7,7)
plt.show()


# Pick any hero from the previous question and list all the comic book titles that he appeared in

# In[ ]:


#level: medium
all_character_comics = pd.merge(marvel_characters_to_comics,marvel_characters,left_on='characterID',right_on='id')
all_ca_comics_id = all_character_comics[all_character_comics['name'] == 'Captain America']['comicID']
marvel_comics[marvel_comics['id'].isin(all_ca_comics_id)]['title'].head()


# It's the holiday season once again, since we already have a list of all holiday comics, retrive all heros who have participated in a holiday comic book

# In[ ]:


# level: easy - medium
marvel_characters_to_comics[marvel_characters_to_comics['comicID'] == 17429]
holiday_comics = marvel_comics[marvel_comics.title.str.contains('Holiday')]
holiday_character_to_comic = marvel_characters_to_comics[marvel_characters_to_comics['comicID'].isin(holiday_comics['id'])]
pd.merge(holiday_character_to_comic,marvel_characters,left_on='characterID',right_on='id')


# Two of the most iconic marvel superheros, Iron Man and Captain America, appeer together quite offten. see if you can get the ammount of comic books they both appear in

# In[ ]:


# level: medium - hard
all_character_comics = pd.merge(marvel_characters_to_comics,marvel_characters,left_on='characterID',right_on='id')

all_ca_comics_id = all_character_comics[all_character_comics['name'] == 'Captain America']['comicID']
all_ca_comics = marvel_comics[marvel_comics['id'].isin(all_ca_comics_id)]

all_im_comics_id = all_character_comics[all_character_comics['name'] == 'Iron Man']['comicID']
all_im_comics = marvel_comics[marvel_comics['id'].isin(all_im_comics_id)]

len(pd.merge(all_im_comics,all_ca_comics,on='id', how='inner'))


# Now that we know how many comic books both of those guys have appeared together at, are they the best power duo in the marvel universe?.
# craete a series with a multi index of 2 superheros(name1,name2) and count for each of them the ammount of comic books they have been in together in, order by that ammount in a descending order

# In[ ]:


# level: really hard :)
comics_to_duos = pd.merge(marvel_characters_to_comics,marvel_characters_to_comics,on='comicID')
comics_to_duos = comics_to_duos[comics_to_duos['characterID_x'] > comics_to_duos['characterID_y']] # remove acuurences where the 2 heros are the same, 
# and removes duplicates, because hero1,hero2 == hero2,hero1
comic_duos = pd.merge(comics_to_duos,marvel_characters,left_on='characterID_x',right_on='id').drop(['id','characterID_x'],axis=1).rename(columns={'name':'name1'})
comic_duos = pd.merge(comic_duos,marvel_characters,left_on='characterID_y',right_on='id').drop(['id','characterID_y'],axis=1).rename(columns={'name':'name2'})
comic_duos.set_index(['name1','name2']).groupby(level=[0,1]).size().sort_values(ascending=False).head()

