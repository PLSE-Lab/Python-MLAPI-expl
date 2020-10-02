#!/usr/bin/env python
# coding: utf-8

# <font size="5">The network we chose to explore is the **"Astro Physics Collaboration" network**.</font>
# http://snap.stanford.edu/data/ca-AstroPh.html
#     

# * Arxiv ASTRO-PH (Astro Physics) collaboration network covers scientific collaborations between authors papers submitted to Astro Physics category.
# * In this network, nodes represent scientists & edges represent collaborations.
# * If an author i co-authored a paper with author j, the graph contains an undirected edge from i to j.
# * If the paper is co-authored by k authors this generates a completely connected (sub)graph on k nodes.

# In[1]:


import pandas as pd # for data processing

#Reading data & showing samples
data=pd.DataFrame(pd.read_csv("../input/Data_Astroph.csv"))
data.sample(3)


# <font size="5">We would like to find out:</font>

# Who is the most influencial astrophiscis in the data base. We will measure effectiveness of a certain astrophisicis by the number of sub graphs related to him.
