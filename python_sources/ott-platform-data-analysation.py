#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('/kaggle/input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv', index_col=0)
df.set_index('ID', inplace=True)


# In[ ]:


df.head(2)


# **search movies in OTT with certain year and imdb rating as well as genres**

# In[ ]:


prime = df[(df["Prime Video"] == 1) & 
           (df["IMDb"] > 7) & 
           (df.Year >= 2016) & 
           (df.Genres.str.contains('Thriller')) &
           (df.Language.str.contains('English'))].sort_values('Year', ascending=False)


# In[ ]:


netflix = df[(df["Netflix"] == 1) & 
           (df["IMDb"] > 7) & 
           (df.Year >= 2016) & 
           (df.Genres.str.contains('Thriller')) &
           (df.Language.str.contains('English'))].sort_values('Year', ascending=False)


# In[ ]:


Disney = df[(df["Disney+"] == 1) & 
           (df["IMDb"] > 7) & 
           (df.Year >= 2010) & 
           (df.Genres.str.contains('Thriller')) &
           (df.Language.str.contains('English'))].sort_values('Year', ascending=False)


# In[ ]:


prime[["Title","IMDb","Year","Genres"]].sort_values('IMDb', ascending=False).head() #primevideo movies after 2016


# In[ ]:


netflix[["Title","IMDb","Genres"]].sort_values('IMDb', ascending=False).head() #primevideo movies after 2016


# In[ ]:


top_genre = df['Genres'].str.get_dummies(',').sum().reset_index().rename(columns = {'index':'genre',0 : "count"})
top_genre.sort_values(by = 'count',ascending =False,inplace =True)


# In[ ]:


fig = px.bar(top_genre, y='count', x= 'genre',color='count',title='TOP GENRES')
fig.show()


# In[ ]:


df1 = (df[['Netflix', 'Hulu', 'Prime Video', 'Disney+']].sum()/df.shape[0]).reset_index().rename(columns = 
                                                                                                     {'index':'platform', 
                                                                                                      0 : "content"})


# In[ ]:


fig = px.pie(df1, values='content', names= 'platform',title='Content across Platforms')
fig.show()


# In[ ]:


country_produce = df['Country'].str.get_dummies(',').sum().reset_index().rename(columns = {'index':'country', 0 : "movies"})


# In[ ]:


fig = px.bar(country_produce.sort_values(by='movies', ascending=False).head(10), 
             y='movies', 
             x= 'country',
             color = 'country',
             title = 'MOST CONTENT PRODUCERS WORLDWIDE')
fig.show()


# In[ ]:


fig = go.Figure(
    data = go.Choropleth(locations=country_produce['country'],
    z = country_produce['movies'].astype(float),
    locationmode = 'country names',
    colorscale = 'blues',
    colorbar_title = "movies"))

fig.update_layout(title_text = ' TOP CONTENT PRODUCERS WORLDWIDE')

fig.show()


# In[ ]:


top_movies = df.sort_values('IMDb',ascending = False).head(10)
fig = px.bar(top_movies, x='Title', y='IMDb', color='Title', height=600)
fig.show()

