#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import datetime
import re


# In[ ]:



PATH = "../input/movies-on-netflix-prime-video-hulu-and-disney/MoviesOnStreamingPlatforms_updated.csv"

df = pd.read_csv(PATH)


# ## Shape, Structure

# In[ ]:


print("Dataset has {} rows and {} columns.".format(df.shape[0], df.shape[1]))


# In[ ]:


df.info()


# ## Platform Distribution

# In[ ]:


platform = df[['Netflix', 'Hulu', 'Prime Video', 'Disney+']]
platform['Prime Video'].value_counts()


# ## Common movies

# In[ ]:


df.query('Netflix == 1 and Hulu == 1 and `Prime Video` == 1')['Title']


# ## Age restrictions

# In[ ]:


px.bar(
       x=df['Age'].value_counts().index,
       y=df['Age'].value_counts().values,
       labels = {'x': "Age Restrictions", 'y':"Count"}
    )


# ## Highest Rated (IMDb)

# In[ ]:


fig = px.bar(
       x=df.nlargest(columns='IMDb', n=10)['IMDb'].values,
       y=df.nlargest(columns='IMDb', n=10)['Title'].values,
       orientation='h',
       labels = {'x': "Rating", 'y':"Movie Titles"},
       color=df.nlargest(columns='IMDb', n=10)['Title'].values,
       color_discrete_sequence=plotly.colors.sequential.algae
    )
fig.update_layout(showlegend=False)
fig.show()


# In[ ]:




