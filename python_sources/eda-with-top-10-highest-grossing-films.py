#!/usr/bin/env python
# coding: utf-8

# Hi we will do some basic plotting with this dataset.
# First thing: We need to import the dataset and the required libraries. 

# In[ ]:


# import the required libraries
import matplotlib.pyplot as plt
import pandas as pd

# some editing on the display form 
plt.style.use('seaborn-notebook')


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Second: We will read the data and take a look at it

# In[ ]:


df = pd.read_csv('../input/top-10-highest-grossing-films-19752018/blockbusters.csv')
print(df.head())


# first let's check the missing values 

# In[ ]:


null = pd.DataFrame(df.isnull().sum() / len(df) * 100).transpose()
print(null)


# As you can see only 'Genre_2' and 'Genre_3' are missing.
# This is maybe because the films with these missing features actually don't have more then one or to genres
# so we will fill these missing values with 'None' .

# In[ ]:


df.fillna('None', axis=0, inplace=True)


# Now let's make some analysis!
# 
# **Who are the leading studios(those who most of the films they made were successful)?**
# 
# To know this let's run the following code
# 

# In[ ]:




studio = [stu for stu, df in df.groupby('studio')] # to extrat studios' names

# plot studio by rating
plt.barh(studio, df.groupby('studio').mean()['imdb_rating'], label='mean rating (imdb)')

# plot studio by rank
plt.barh(studio, df.groupby('studio').mean()['rank_in_year'], alpha=0.8, label='mean rank')
plt.legend()
plt.title('Rating of Holywood films between (1975-2018)')
plt.xlabel('Rating')
plt.ylabel('Studio name')
plt.show()


# The **orange color** is the 'mean rank' of the films that were made by the studio during the years(1975-2018).
# 
# The **blue color** is the 'mean rating' of the films that were made by the studio during the years(1975-2018).
# 
# As you can see, the films that were made by 'Embassy Pictures' studio are the highest ranks (10).
# And the films that were made by Pixar studio are the highest rate by imdb.
# 

# Now let's look at the top 10 movies.

# In[ ]:


# First let's add the year that the film was made to every title for better look (:
df["title"] = df["title"] + ' (' + df["year"].astype(str) + ')'

top_10 = df[['title', 'imdb_rating', 'year', 'length', 'worldwide_gross']].sort_values(by='imdb_rating', ascending=False)
print(top_10.head())


# Let's plot it

# In[ ]:


name = [name for name in top_10.iloc[:, 0]] # to extract the names

plt.barh(name[:10], top_10['imdb_rating'][:10], color='lightblue', label='ratings')


# That is cool!
# 
# Now let's plot the length of the TOP 10 films

# In[ ]:


plt.barh(name[:10], top_10['length'][:10], color='darkgray', alpha=0.8, label='length(hours)')
plt.title('Length of the TOP 10 films')
plt.xlabel('length(minutes)')
plt.ylabel('names of the films')


# That's cool!
# But it will make more sense when we turn these minutes to hours:

# In[ ]:


top_10['length'] /= 60 
plt.barh(name[:10], top_10['length'][:10], color='darkgray', alpha=0.8, label='length(hours)')
plt.title('Length of the TOP 10 films')
plt.xlabel('length(hours)')
plt.ylabel('names of the films')


# let's plot the 'imdb_rating' and 'length' together.

# In[ ]:


plt.barh(name[:10], top_10['imdb_rating'][:10], color='lightblue', label='ratings')
plt.barh(name[:10], top_10['length'][:10], color='darkgray', alpha=0.8, label='length(hours)')

plt.legend()
plt.title('Top 10 films')
plt.xlabel('Rating')
plt.ylabel('Film name')


# that's awesome!

# now let's plot the worldwide gross of the top 10 films:

# In[ ]:


# preprocessing the feature 
top_10['worldwide_gross'] = top_10['worldwide_gross'].str.replace('$', '')
top_10['worldwide_gross'] = top_10['worldwide_gross'].str.replace(',', '')
# convert it to numeric
top_10['worldwide_gross'] = top_10['worldwide_gross'].astype(float)

plt.barh(name[:10], top_10['worldwide_gross'][:10], color='lightblue', label='worldwide_gross')
plt.title('The worldwide gross of the top 10 films')
plt.xlabel('Worldwide_gross')
plt.ylabel('Top 10 films')
plt.show()


# Now let's take insight about genres.
# There are three genres,
# let's see what are the most rated genres

# In[ ]:



# plot the main genres
main_genre = df.groupby('Main_Genre').mean()
main_genre_names = [gnr for gnr, df in df.groupby('Main_Genre')]
import seaborn as sns
plt.barh(main_genre_names[:10], main_genre['rank_in_year'][:10],alpha=0.8, label='first genre')
######################
# plot the second genres

genre_2 = df.groupby('Genre_2').mean()
genre_names_2 = [gnr for gnr, df in df.groupby('Genre_2')]

plt.barh(genre_names_2[:10], genre_2['rank_in_year'][:10], alpha=0.7, label='second genre')
######################
# plot the third genres

genre_3 = df.groupby('Genre_3').mean()
genre_names_3 = [gnr for gnr, df in df.groupby('Genre_3')]
plt.barh(genre_names_3[:10], genre_3['rank_in_year'][:10], alpha=0.6, label='third genre')

plt.title('Most rated films by genres')
plt.xlabel('Rate')
plt.ylabel('Genres')
plt.legend()
plt.show()


# As you can see the films with genres that contain 'Action' genre have more chance to be successful.
# And the films contain 'Documentary' genre have less chance  to be successful.

# **I think there is no important analysis to do with this data more then what we did above.
# I hope this notebook was helpful and useful for you.**
# 
# Please upvote this notebook if you fou6nd it useful or important 
