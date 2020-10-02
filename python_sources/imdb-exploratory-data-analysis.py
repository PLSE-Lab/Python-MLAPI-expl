#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis on IMDB Dataset

# This notebook is dedicated to exploratory data analysis of IMDB dataset. Below you can find the movie data between 2006 and 2016. I've tried to answer the questions that comes across my mind as a cinephile, such as, if higher rating results in higher revenue in the box office, if metacritics and users get along really well, or which director is the most successful.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


imdb = pd.read_csv("../input/imdb-data/IMDB-Movie-Data.csv",sep=',')
imdb.head()


# Fixing column names by removing parenthesis for better use of pandas.

# In[ ]:


imdb.rename(columns={'Revenue (Millions)':"Revenue","Runtime (Minutes)":"Runtime"},inplace=True)
print(imdb.columns)


# In[ ]:


imdb.describe(include='all')


# # Grouped movies by years, trying to see if the rating follows a pattern throughout the years, if the quality of the movies follow a trend. 

# In[ ]:


import plotly.express as px
years=imdb.groupby("Year")["Rating"].mean().reset_index()
px.scatter(years,x="Year", y="Rating").show()


# The users don't seem to be satisfied with the movies nowadays.

# **Let's see which movie has the most success in the box office.**

# In[ ]:


mostearned=imdb[imdb["Revenue"]==imdb["Revenue"].max()]
print(mostearned)


# Let's look at the user-favourite director between 2006-2016.

# In[ ]:


imdbtop=imdb[["Title","Director","Rating"]][imdb["Rating"]==imdb["Rating"].max()]
imdbtop.head()


# I've grouped all directors by the rating they have and took the mean value, then sorted them in descending.

# In[ ]:


directors=imdb.groupby("Director")["Rating"].mean().reset_index()
directors.sort_values("Rating", ascending=False)


# # Let's see if critics and users get along really well.

# In[ ]:


imdb['Rating'].corr(imdb['Metascore'])


# There seems to be a correlation, let's visualize.

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.style.use('ggplot')

plt.scatter(imdb.Metascore, imdb.Rating)
plt.show()


# Let's see if there's a correlation between the revenue movie makes and the rating. Does it mean that most earned is always the most liked?

# In[ ]:


imdb['Rating'].corr(imdb['Revenue'])


# There's a weak correlation between them. Let's plot.

# In[ ]:


plt.scatter(imdb.Rating, imdb.Revenue)
plt.show()


# I want to see if longer duration always results in better rating.

# In[ ]:


imdb['Rating'].corr(imdb['Runtime'])


# Well, meh. Let's plot this.

# In[ ]:


import seaborn as sns
f, ax = plt.subplots(figsize=(10, 6))
ax.set(xscale="log")
sns.scatterplot(imdb.Rating, imdb.Runtime, ax=ax)
plt.show()


# Thank you for looking at the analysis, hit me up if there's anything you'd like me to discover! 
