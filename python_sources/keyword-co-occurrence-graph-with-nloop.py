#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from pprint import pprint 
import  matplotlib.pyplot as plt

import networkx as nx

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # [NLOOP](https://github.com/syasini/NLOOP)

# NLOOP is a python package that provides a convenient interface for exploring and analyzing text data. 
# Behind the scene, NLOOP uses spaCy and gensim to take care of cleaning, tokenization, dependency parsing, keyword extraction and much more in one fell swoop. Here I will use it to build a keyword co-occurrence graph from the titles and abstracts of the research papers in the provided dataset. 
# 
# You can install NLOOP from the following address. Checkout the github repository for more examples. 

# In[ ]:


get_ipython().system('pip install git+https://github.com/syasini/NLOOP.git@master')


# In[ ]:


from nloop import Text


# # Covid-19 Keyword Co-occurrence Graph

# In[ ]:


data_fname = os.path.join("..","input","CORD-19-research-challenge", "metadata.csv")
data = pd.read_csv(data_fname, index_col=0,)


data = data.sample(5000) # let's look at a small sample of the data 
data.reset_index(inplace=True)


# In[ ]:


# drop all the nan values in titles and abstracts
data.dropna(subset = ["abstract", "title"], inplace=True)

# combine title and abstract into new column called 'text'
data["text"] = data["title"].combine(data["abstract"], lambda s1, s2: ". ".join([s1, s2]))


# In[ ]:


# process text with nloop
text = Text(data["text"], fast=False)

# This will take a while for the entire corpus
# use fast=True if you're only interested in clean tokens
# and don't need dependencies, named entities, and keywords


# In[ ]:


# show word cloud
text.show_wordcloud()


# In[ ]:


#show the most common tokens
text.token_counter.most_common(20)


# In[ ]:


# run LDA 
text.lda.run(num_topics=10)


# In[ ]:


# show all the topics
pprint(text.lda.model.show_topics(10))


# In[ ]:


# extract keywords and their ranks (NLOOP uses the pytextrank pipeline from spacy)
keywords_list = [x for item in text.keywords.texts for x in item]
ranks_list = [x for item in text.keywords.ranks for x in item]


# In[ ]:


# only keep top keywords that pass the rank_cutoff threshold
rank_cutoff = 0.1 

top_keywords = []
for keywords, ranks in zip(text.keywords.texts , text.keywords.ranks ):
    top_keywords.append((np.array(keywords)[np.array(ranks)>rank_cutoff]).tolist())
    


# In[ ]:


from itertools import combinations, chain
from collections import Counter

# form pairs of keywords from each document
keyword_pair = list(chain(
                    *[list(combinations(kw_list,2)) for kw_list in top_keywords])
                   )


# In[ ]:


# count all the unique pairs
pair_counter = Counter(keyword_pair).items()


# In[ ]:


print(len(pair_counter))


# In[ ]:


def get_pair_graph(pair_counter, weight_times=1, degree_cutoff=50):
    """construct the co-ocurrence graph from the keyword pairs
    
    Parameters
    ----------
    pair_counter: Counter dictionary
        consists of keyword pairs as keys and their counts as values
    weight_times: int or float (scalar)
        multplicative factor for weights of edges in the graph
    degree_cutoff: int
        nodes with degrees below this number will be ignored
        
    Return
    ------
    G: networkx graph instance
    
    """
    
    G = nx.Graph()

    #construct the graph from the edges
    for pair, weight in pair_counter:
    
        G.add_edge(*pair, weight=weight_times*(weight))
    
    # remove nodes with degrees smaller than the cutoff
    node_list = []
    for node in np.copy(G.nodes):
        if G.degree(node)<degree_cutoff:
            
            G.remove_node(node) 
    
    return G


# In[ ]:


# get the keyword pair graph
G = get_pair_graph(pair_counter, degree_cutoff=50)


# In[ ]:


# calculate the node sizes using arbitrary transformation 
node_sizes= [20*G.degree[node]**2+100 for node in G.nodes]

# construct the label dictionary
labels = {i:i for i in list(G.nodes)}


# In[ ]:


print(len(G.nodes))


# In[ ]:


# draw the graph
plt.figure(figsize=(10,10),dpi=100)

pos = nx.spring_layout(G, k=3, 
                       fixed=["viruses"], pos={"viruses":(0,0)}, 
                       dim=2, iterations=50)


nx.draw_networkx_nodes(G, pos, 
                       #with_labels=True, 
                       node_color="tab:orange",
                       node_size=node_sizes, 
                       node_shape="8", 
                       edgecolors="tab:red",
                      )

nx.draw_networkx_edges(G, pos, 
                       #with_labels=True, 
                       edgecolors="grey",
                       alpha=0.1,
                      )

_= nx.draw_networkx_labels(G, pos, 
                        labels=labels, 
                        )


# **Challenge**: Find the bats!

# In[ ]:




