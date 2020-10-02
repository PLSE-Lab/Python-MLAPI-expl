#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import missingno as msno


# In[ ]:


df = pd.read_csv('../input/top50spotify2019/top50.csv',encoding = "ISO-8859-1")


# In[ ]:


#checking missing values
check_MV = msno.nullity_sort(df)
msno.matrix(check_MV, color=(0.25, 0.1, 0.6))


# In[ ]:


df.rename(columns={"Unnamed: 0":"ID"}, inplace=True)


# In[ ]:


df.rename(columns=lambda x : str(x).replace(".",""), inplace=True)
df.set_index("ID", inplace=True)


# In[ ]:


df.head()


# In[ ]:


df_numbers = df.drop(columns=["TrackName", "ArtistName"])

pair_grid = px.scatter_matrix(df_numbers, width=1000, height=1000, color="Genre")
pair_grid.show()


# In[ ]:


gb_artistname =df.groupby("ArtistName")


# In[ ]:


artist_popularity = gb_artistname.mean().reset_index()[["ArtistName","Popularity"]].sort_values("Popularity")

artist = artist_popularity.ArtistName
popularity = artist_popularity.Popularity


# In[ ]:


scatter = go.Figure()

scatter.add_trace(go.Scatter(
    x=popularity,
    y=artist,
    marker=dict(
        color='rgba(156, 165, 196, 0.95)',
        line_color='rgba(156, 165, 196, 1.0)')
    
))

scatter.update_traces(mode='markers', marker=dict(line_width=1, symbol='circle', size=16))
scatter.update_layout(
    title = "Popularity Ranking of Artist",
    xaxis=dict(
        showgrid=False,
        showline=True,
        linecolor='rgb(105, 105, 105)',
        tickfont_color='rgb(102, 102, 102)',
        showticklabels=True,
        dtick=10,
        ticks='outside',
        tickcolor='rgb(102, 102, 102)',
    ),
    margin=dict(l=10, r=10, b=10, t=50),
    legend=dict(
        font_size=10,
        yanchor='middle',
        xanchor='right',
    ),
    width=800,
    height=1000,
    paper_bgcolor='white',
    plot_bgcolor='whitesmoke',
    hovermode='closest',
)


# In[ ]:


gb_genre =df.groupby("Genre")

genre_popularity = gb_genre.mean().reset_index()[["Genre","Popularity"]].sort_values("Popularity")

genre = genre_popularity.Genre
popularity_2 = genre_popularity.Popularity


# In[ ]:


scatter_2 = go.Figure()

scatter_2.add_trace(go.Scatter(
    x=popularity_2,
    y=genre,
    marker=dict(
        color='rgba(90, 90, 90, 0.95)',
        line_color='rgba(156, 165, 196, 1.0)')
    
))

scatter_2.update_traces(mode='markers', marker=dict(line_width=1, symbol='star', size=16))
scatter_2.update_layout(
    title = "Popularity Ranking of Genre",
    xaxis=dict(
        showgrid=False,
        showline=True,
        linecolor='rgb(105, 105, 105)',
        tickfont_color='rgb(80, 102, 102)',
        showticklabels=True,
        dtick=10,
        ticks='outside',
        tickcolor='rgb(80, 102, 102)',
    ),
    margin=dict(l=10, r=10, b=10, t=50),
    legend=dict(
        font_size=10,
        yanchor='middle',
        xanchor='right',
    ),
    width=800,
    height=700,
    paper_bgcolor='white',
    plot_bgcolor="lightskyblue",
    hovermode='closest',
)


# In[ ]:


test_fig = px.parallel_coordinates(df,
                               color="Popularity",
                               color_continuous_scale=px.colors.diverging.Tealrose)
test_fig.show()


# In[ ]:




