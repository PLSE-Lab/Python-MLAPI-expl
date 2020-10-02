#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#I'm start in data analisys and I'm applying here some skills that I learned

import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

df = pd.read_csv('../input/top50spotify2019/top50.csv', encoding = 'latin-1', index_col=0)

#Renaming
df.columns = ['Track', 'Artist', 'Genre', 'BPM', 'Energy', 'Danceability', 'Loudness', 
               'Liveness', 'Valence', 'Length', 'Acousticness', 'Speechiness', 'Popularity']

print('setup completed')



# In[ ]:


df.head()


# In[ ]:


#Shape of dataset
print(df.shape)


# In[ ]:


#Identifying types
df.dtypes
#Turning "Popularity" into objeto
df.Popularity.astype('object')



# In[ ]:


#Most popular artist
art_pop = df.groupby('Artist')['Popularity'].mean().sort_values(ascending = False)
print(art_pop)


# In[ ]:


#Most popular genre
pop_gen = df.groupby('Genre')['Popularity'].mean().sort_values(ascending = False)
print(pop_gen)


# In[ ]:


#Music per artist

df.groupby('Artist').size()


# In[ ]:


#Artists and their most played genre 
df.groupby('Artist')['Genre'].max()


# In[ ]:


#Most popular music and its duration

mus_pop = df.groupby('Track')['Popularity','Length'].max().sort_values(by= 'Popularity', ascending= False)
print(mus_pop)

#https://docs.python.org/3/library/functions.html


# In[ ]:


#Pop artits popularity
pop_an = df.loc[df.Genre == 'dance pop']
group_pop = pop_an.groupby('Artist')['Popularity'].mean().sort_values(ascending = False)
print(group_pop)

plt.figure(figsize = (12,6))
sns.lineplot(data = group_pop)
plt.title('POP Artists Popularity')
plt.show()


# In[ ]:




