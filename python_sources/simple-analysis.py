#!/usr/bin/env python
# coding: utf-8

# # List of video games considered the best
# 
# ## Content
# 
# This is a list of video games that have consistently been considered the best of all time by video game journalists and critics. The games listed here are included on at least six separate "best/greatest of all time" lists from different publications. While any single publication's list reflects the personal opinions of its writers, when the lists are taken in aggregate, a handful of notable games have achieved something approaching critical consensus by multiple appearances in such lists.
# 
# ## Context
# This notebook is based on a dataset based on data from a wikipedia page of the same name.

# Read the csv file with the necessary content

# In[ ]:


import pandas as pd
df = pd.read_csv('../input/list-of-video-games-considered-the-best/video_games_considered_the_best.csv')


# Lists on columns in the dataset

# In[ ]:


df.columns


# 1. Randomly picks up five games

# In[ ]:


df.sample(5)


# Return a tuple representing the dimensionality

# In[ ]:


df.shape


# ## All games by number of references

# In[ ]:


pd.set_option('max_rows', 99999)
df.sort_values('refs', ascending=False)


# ## Publishers by games

# In[ ]:


publishers = df.groupby('publisher').size()

publishers_sorted = sorted(dict(publishers).items(), key=lambda x: x[1], reverse=True)
df_publishers = pd.DataFrame(publishers_sorted)
df_publishers.columns = [ 'publisher', 'games']
df_publishers


# ## Genre by	games

# In[ ]:


publishers = df.groupby('genre').size()

publishers_sorted = sorted(dict(publishers).items(), key=lambda x: x[1], reverse=True)
df_genre = pd.DataFrame(publishers_sorted)
df_genre.columns = [ 'genre', 'games']
df_genre


# ## Games by a specific genre

# In[ ]:


stealth_games = df[df['genre'] == 'Stealth']
stealth_games


# Show the word frequency 

# In[ ]:


from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

wordcloud = WordCloud(max_font_size=40).generate(" ".join(list(df['title'])))
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# ## What was the best year for games?

# In[ ]:


import matplotlib.pyplot as plt

plt.title('Game references by year')
plt.ylabel('References')
plt.xlabel('Year')
plt.plot(df['year'], df['refs'])
plt.grid(True)

