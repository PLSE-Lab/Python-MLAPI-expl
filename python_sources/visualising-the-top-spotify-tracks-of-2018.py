#!/usr/bin/env python
# coding: utf-8

# ## Visualising the Top Spotify Tracks of 2018

# In[ ]:


import pandas as pd 
import igraph as ig
import matplotlib.pyplot as plt

# load data
data = pd.read_csv("../input/top-spotify-tracks-of-2018/top2018.csv")
data.head()


# In the dataset, we have 100 songs (identified by id or name) performed by artists and the songs' respective characteristics such as danceability, energy, key, etc. To help us understand the data better, we begin by making some visualisation. Let us start with a network graph.
# 
# Represent each song and each artist as a vertex, and let a directed edge from a song to an artist denote the relation between the song and the artist who performed it. 

# In[ ]:


G = ig.Graph()
typeseq = []
for index,row in data.iterrows():
    G.add_vertices(row['name'])
    typeseq.append("song")
    # create a node for artist only if it doesn't exist yet
    if row['artists'] not in G.vs['name']:
        G.add_vertices(row['artists']) 
        typeseq.append("artist")
    G.add_edge(row['name'],row['artists'])

# define danceability attribute
G.es["danceability"] = data["danceability"]
ig.summary(G)


# The output of the summary command tells us that we have 170 nodes and 100 edges, meaning, out of the 100 song-artist relation given, there are songs that were performed by the same artist.
# 
# Next, let us customise the nodes' sizes such that artists with more songs in the Top 100 are visually bigger. Let us also show artists in green, and songs in pink.

# In[ ]:


G.vs["type"] = typeseq
nodesize = [i.degree()*5 for i in G.vs]     # vary node size based on degree
nodecolor = ["#fbb4ae" if t=="song" else "#ccebc5" for t in G.vs["type"]]     # different node colors for artists and songs
nodelabel = [i["name"] if i.degree() >= 3 else '' for i in G.vs]     # put labels for artists with 3 or more songs
layout = G.layout("fr")

# color edges by danceability
edgecolor = []
for i in G.es["danceability"]:
    pal = ig.RainbowPalette(n=8, alpha=i)
    edgecolor.append(pal.get(0))

ig.plot(G, layout = layout, 
        vertex_size=nodesize, 
        vertex_color=nodecolor,
        vertex_label = nodelabel,
        vertex_label_color="#1f78b4",
        edge_color = edgecolor,
        bbox=(450,450))


# We used Fruchterman-Reingold layout algorithm as this force-directed layout gives a better view of the graph.
# 
# The node sizes show that XXXTENTACION, Post Malone and Drake had the highest number of songs that made it in the Top 100 Spotify Tracks of 2018. Majority of the edge colors are red, indicating that most of the songs have high danceability.
