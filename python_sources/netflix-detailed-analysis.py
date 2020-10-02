#!/usr/bin/env python
# coding: utf-8

# ## Introduction:
# In this kernel, I'll discus the various aspects and trends of the dataset on ***Netflix Shows and Movies***. I'll try to cover a detailed exploratory analysis of the dataset.

# ## Importing the required libraries:

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ### Read the dataset:

# In[ ]:


df=pd.read_csv('/kaggle/input/netflix-shows/netflix_titles_nov_2019.csv')


# ## Let's have Overview of the Data:

# In[ ]:


df.head()


# ### Brief Description:

# In[ ]:


df.describe(include='all')


# ### Brief Information of various variables:

# In[ ]:


df.info()


# ## Country-wise content creation:

# In[ ]:


plt.figure(1, figsize=(15, 7))
plt.title("Country with maximum content creation")
sns.countplot(x = "country", order=df['country'].value_counts().index[0:15] ,data=df,palette='Accent')


# As we can see that **U.S. and India** have maximum content creation. Rapid Development in India since *Netflix* is the new to India.

# ## Types of Rating and their Frequency:

# In[ ]:


plt.figure(1, figsize=(15, 7))
plt.title("Frequency")
sns.countplot(x = "rating", order=df['rating'].value_counts().index[0:15] ,data=df,palette='Accent')


# In[ ]:


df['rating'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(17,7))


# 33% Fall in catogery TV-MA ("TV-MA" is a rating assigned by the TV Parental Guidelines to a television program that was designed for mature audiences only.)
# 
# 23% fall in catigery TV-14 (Programs rated TV-14 contains material that parents or adult guardians may find unsuitable for children under the age of 14.
# 
# 12.5 % fall in category TV-PG (TV-PG: Parental guidance suggested. This program contains material that parents may find unsuitable for younger children)

# ## Year-wise growth

# In[ ]:


plt.figure(1, figsize=(15, 7))
plt.title("Frequency")
sns.countplot(x = "release_year", order=df['release_year'].value_counts().index[0:15] ,data=df,palette='Accent')


# In[ ]:


import plotly.graph_objects as go
d1 = df[df["type"] == "TV Show"]
d2 = df[df["type"] == "Movie"]

col = "release_year"

vc1 = d1[col].value_counts().reset_index()
vc1 = vc1.rename(columns = {col : "count", "index" : col})
vc1['percent'] = vc1['count'].apply(lambda x : 100*x/sum(vc1['count']))
vc1 = vc1.sort_values(col)

vc2 = d2[col].value_counts().reset_index()
vc2 = vc2.rename(columns = {col : "count", "index" : col})
vc2['percent'] = vc2['count'].apply(lambda x : 100*x/sum(vc2['count']))
vc2 = vc2.sort_values(col)

trace1 = go.Scatter(
                    x=vc1[col], 
                    y=vc1["count"], 
                    name="TV Shows", 
                    marker=dict(color = 'rgb(249, 6, 6)',
                             line=dict(color='rgb(0,0,0)',width=1.5)))

trace2 = go.Scatter(
                    x=vc2[col], 
                    y=vc2["count"], 
                    name="Movies", 
                    marker= dict(color = 'rgb(26, 118, 255)',
                              line=dict(color='rgb(0,0,0)',width=1.5)))
layout = go.Layout(hovermode= 'closest', title = 'Content added over the years' , xaxis = dict(title = 'Year'), yaxis = dict(title = 'Count'),template= "plotly_dark")
fig = go.Figure(data = [trace1, trace2], layout=layout)
fig.show()


# #### We can see that 2018 is the year when we see maximum content creation and there has been a rapid development in a content creation.

# ## Main Analysis of TV-series and Movies

# In[ ]:


plt.figure(1, figsize=(4, 4))
plt.title("TV v/s Movies")
sns.countplot(x = "type", order=df['type'].value_counts().index[0:15] ,data=df,palette='Accent')


# We can see that there are more movies than TV series.

# In[ ]:


df['type'].value_counts().plot.pie(autopct='%1.1f%%',shadow=True,figsize=(17,7))


# Of all the content available on Netflix, there is nearly **67.5 percent** Movies and **32.5 percent** TV-Shows.

# In[ ]:


movie=df[df['type']=='Movie']
tv=df[df['type']=='TV Show']


# # Movies

# ## Director with most movies

# In[ ]:


plt.figure(1, figsize=(20, 7))
plt.title("Director with most movies")
sns.countplot(x = "director", order=movie['director'].value_counts().index[0:10] ,data=movie,palette='Accent')


# It can be clearly seen in the plot that ***Raul Campos*** has made most number of movies.

# ## Movies from Genres

# In[ ]:


from collections import Counter
col = "listed_in"
categories = ", ".join(movie['listed_in']).split(", ")
counter_list = Counter(categories).most_common(50)
labels = [_[0] for _ in counter_list][::-1]
values = [_[1] for _ in counter_list][::-1]
trace1 = go.Bar(y=labels, x=values, orientation="h", name="Movie")
data = [trace1]
layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(data, layout=layout)
fig.show()


# #### We can see that *Dramas* and *comedies* are genres on which most the content has been made.

# ## Comparing the length of Movies

# In[ ]:


dur=[]
for i in movie['duration']:
    dur.append(int(i.strip('min')))
plt.figure(1, figsize=(20, 7))
plt.title("Comparing the length of Movies")
sns.distplot(dur,rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"})


# Statistically we can see that length of movies follow a ***normal distribution***.

# In[ ]:


plt.figure(1, figsize=(20, 7))
plt.title("Comparing the length of Movies")
sns.countplot(x = "duration", order=movie['duration'].value_counts().index[0:15] ,data=df,palette='Accent')


# It is observed that most of the movies are of length around **90 minutes**.

# # TV Shows

# ## Director with most TV Shows

# In[ ]:


plt.figure(1, figsize=(20, 7))
plt.title("Director with most movies")
sns.countplot(x = "director", order=tv['director'].value_counts().index[0:10] ,data=tv,palette='Accent')


# **Alastair Fothergill** has made the most (3) shows.

# In[ ]:


from collections import Counter
col = "listed_in"
categories = ", ".join(tv['listed_in']).split(", ")
counter_list = Counter(categories).most_common(50)
labels = [_[0] for _ in counter_list][::-1]
values = [_[1] for _ in counter_list][::-1]
trace1 = go.Bar(y=labels, x=values, orientation="h", name="TV Shows")
data = [trace1]
layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))
fig = go.Figure(data, layout=layout)
fig.show()


# Most of the TV content has been made on **Dramas and Comedies**.

# In[ ]:


dur=[]
for i in tv['duration']:
    if 'Seasons' in i:
        dur.append(int(i.strip('Seasons')))
    else:
        dur.append(int(i.strip('Season')))
plt.figure(1, figsize=(20, 7))
plt.title("Comparing the length of TV")
sns.distplot(dur,rug=True, rug_kws={"color": "g"},
                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                  hist_kws={"histtype": "step", "linewidth": 3,
                            "alpha": 1, "color": "g"})


# *Since the length of TV shows is expressed in terms of Number of Seasons and it Follows discrete distribution, therefore it is not much useful to define the above plot.*

# In[ ]:


tv['dur']=dur
top=tv.nlargest(15,['dur'])
plt.figure(1, figsize=(20, 7))
sns.barplot(x="title", y="dur", data=top, ci="sd")


# **Grey's Anatomy** has most number of Seasons.

# In[ ]:




