#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os


# In[ ]:


movies=pd.read_csv ("../input/Movie_Movies.csv")
genres=pd.read_csv ("../input/Movie_Genres.csv")
ratings=pd.read_csv ("../input/Movie_AdditionalRating.csv")


# In[ ]:


movies.info()


# In[ ]:


ratings.info()
len(ratings.imdbID.unique())


# In[ ]:


ratings.drop_duplicates(subset='imdbID',inplace=True)
#Only one rating per film is needed for the filtering


# In[ ]:


#makings sure there are no identicals
movies.drop_duplicates(subset='imdbID',inplace=True)


ratings.drop_duplicates(subset='imdbID',inplace=True)
#Only one rating per film is needed for the filtering

#the following definitely won't be used


movies.drop(['Awards','DVD', 'Poster', 'Plot','Website', 'Production','Released','imdbVotes','Type','Rated'], axis = 1, inplace = True)
genres.Genre= [x.strip()for x in genres.Genre]


# In[ ]:


#without any filtering
rawdirectors=movies["Director"].value_counts()[:10]
pd.DataFrame({'Director':rawdirectors.index, 'NumberOfMoviesProduced':rawdirectors.values})


# The answer of the E2 question is the above. However, I don't think this result is what someone has in mind when asking for the most prolific directors. There are Adult films and short films directors among the top 10, so for my further analysis I will try to answer the same question using a little more of "business sense".

# For the first filtering, only rated movies will be used. Data from the ratings database will be used since it contains ratings for more films than the ones with an imdb rating (just a little but since it's here) in the movies database. Notice that, this is an inner merge (default) so movies that are not contained in the ratings database are dropped.

# In[ ]:


#using rated films only

rated=pd.merge(movies,ratings, on = 'imdbID')
rated.drop_duplicates(subset='imdbID',inplace=True)

rateddirectors=rated['Director'].value_counts()[:10]
pd.DataFrame({'Director':rateddirectors.index, 'NumberOfMoviesProduced':rateddirectors.values})


# Only D.W. Griffith and Kevin Dunn have survived from the 1st attempt on raw data! Indeed this is an improvement, since all of the results are well-known, prolific directors.

# For the next attempt, genre wise filtering will be  used. The genres Adult and Short Films have been selected to be cut. One could argue about Sport, Music and Documentary being cut in order to provide different answers. This can be easily done by adding those in the notwanted list. Notice that the merge now is left, meaning that films without a genre entry will still be used.

# In[ ]:


#using films from specified genres

updated=pd.merge(movies,genres, on = 'imdbID',how="left")

notwanted=["Short", "Adult"]
notwantedids=updated[updated.Genre.isin(notwanted)]["imdbID"]

#dropping the ids that have at least a genre belonging in the notwanted
updated.drop(updated[updated.imdbID.isin(notwantedids)].index, inplace=True)
updated.drop_duplicates(subset='imdbID',inplace=True)
updated = updated.reset_index(drop=True)


# In[ ]:


updateddirectors=updated['Director'].value_counts()[:10].index.tolist()
filmstopten=updated[updated.Director.isin(updateddirectors)]

genredirectors=updated['Director'].value_counts()[:10]
pd.DataFrame({'Director':genredirectors.index, 'NumberOfMoviesProduced':genredirectors.values})


# In[ ]:


#using both


rated.drop(rated[rated.imdbID.isin(notwantedids)].index, inplace=True)


finaldirectors=rated['Director'].value_counts()[:10]
DoubleFilteredDirectors=pd.DataFrame({'Director':finaldirectors.index, 'NumberOfMoviesProduced':finaldirectors.values})
DoubleFilteredDirectors


# Let's see it in a barplot:

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')
# figure size in inches

plt.figure(figsize=(15,10))
boom=sns.barplot(x="Director", y='NumberOfMoviesProduced', palette="ch:.25", data=DoubleFilteredDirectors)
plt.xticks(rotation= 45)
plt.xlabel('Directors')
plt.ylabel('Number of Films')
plt.title('Films per Director')
plt.show()


# One does not simply filter Kevin Dunn out...
