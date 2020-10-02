#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# We are a group of engineers at Atos/Bull.
# 
# ### Goals
# 
# * Generate quickly and simply a citation network of the dataset corpus to provide a insight of the relation between papers
# * Generate a score for each paper depending on their importance in the corpus.  
# * Provide a kaggle dataset so people can reuse this citation graph and the pagerank score of the papers
# 
# ### What's cool
# * easy & fast
# * reusable since provided as a dataset
# * it is pertinent for any task (to know if a paper is meaningful in the corpus)
# 
# ### What's less cool
# * to keep thing fast and easy, the matching is made on title lowercase. If a paper is cited with a typo in reference, the match won't be made
# 
# ### Improvement to come
# * using some fuzzy matching on title strinsg to avoid fails in matches
# 

# # Quickly and simply create a citation network using networkx

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import random

kaggle_data_path=os.path.join(os.sep, "kaggle", "input", "CORD-19-research-challenge")


# In[ ]:


df = pd.read_csv(kaggle_data_path+"/metadata.csv")


# In[ ]:


datafiles = []
for dirname, _, filenames in os.walk(kaggle_data_path):
    for filename in filenames:
        ifile = os.path.join(dirname, filename)
        if ifile.split(".")[-1] == "json":
            datafiles.append(ifile)


# We iterate over the coprus and build up a list of link: article -> citation

# In[ ]:


citations = []
filter_words = ['proc.', 'magazine'] # Words to filter 
remove_names=["publisher's note", "world health organization", "fields virology", "united states census", "geneva: world health organization"] # full name to remove
for file in datafiles:
    with open(file,'r')as f:
        doc = json.load(f)
    reftitle = doc['metadata']['title'].lower()
    
    '''Get citations'''
    for key,value in doc['bib_entries'].items():
        value['title'] = value['title'].lower().split('proc.')[0]
        # ignore remove_names
        if value['title'] in remove_names: 
            continue
        # add citation if not containing filter_words
        if (len(set(value['title'].lower().split(' ')).intersection(set(filter_words))) == 0) and len(value['title'].lower()) > 0:
            citations.append({"title": reftitle, "citation": value['title'].lower()})


# Create the article->citation dataframe

# In[ ]:


dfc = pd.DataFrame(citations)


# Thanks to this dataframe contaninig all the edges(links) of the corpus, it is now extremly easy to build up a graph from it

# In[ ]:


G = nx.from_pandas_edgelist(dfc,source='title',target='citation',create_using=nx.DiGraph)


# In[ ]:


print(f"Reduced citation graph loaded is having {len(list(G.nodes))} nodes and {len(list(G.edges))} edges.")


# simple example usage

# In[ ]:


# get number of time a paper was cite by others and number of its own citations
title = "interferon-stimulated gene 15 conjugation stimulates hepatitis b virus production independent of type i interferon signaling pathway in vitro"
print(f"The paper '{title}' \n is cited {G.in_degree[title]} times. Its annexe contain {G.out_degree[title]} references.")


# In[ ]:


nx.write_gpickle(G, "citation_network.gpickle")


# **Now to reuse this dataset you may simply run : G = nx.read_gpickle("test.gpickle")**

# # Display citation graph of a single paper

# In[ ]:


H= nx.ego_graph(G, title, radius=1)
size = 15
plt.figure(figsize = [size,size]) 
pos = nx.spring_layout(H) 
nx.draw(H, with_labels=True, node_size = 20 , node_color = 'lightblue')
plt.title(f'Paper <<{title}>> ego graph')
plt.savefig('cite.png')


# # Subgraph display

# We use a subset of the full graph for display , too avoid unreadable spaghettis plots
# 
# 

# In[ ]:


graph_size = 20000
Gsub = G.subgraph(list(G.nodes)[0:graph_size])
print(f"Graph loaded is having {len(list(Gsub.nodes))} nodes and {len(list(Gsub.edges))} edges")


# In[ ]:


max_nodes=2000
H= Gsub.subgraph(random.sample(list(Gsub.nodes), max_nodes))
plt.figure(figsize = [15,15]) 
pos = nx.spring_layout(H) 
nx.draw(H, with_labels=False, node_size = 10 , node_color = 'lightblue')
plt.title(f'Random Subgraph of citation network of a subset {max_nodes} papers')
plt.savefig('cite.png')


# # Most cited papers graph

# We will show here some papers seems much more central than some others. Therefore might be more relevant to return to researchers when they want to answer a specific task. That something our team is using is this [notebook](http://https://www.kaggle.com/mrmimic/opinions-extraction-tool-chloroquine-case-study) where we find the closest papers to a question, clustering them by opinion and then give the most relevant papers for each opinion. 

# In[ ]:


def get_paper_cited_K_times_graph(G , M = 500) -> nx.DiGraph:
    """
    Return a network of paper cited at least M times
    """
    Gs = nx.DiGraph()
    for node in G.nodes():
        if G.in_degree[node] > M:
            # We look for adjacent nodes
            for adj_node in G.in_edges(
                    node):  # create link for each paper point to current paper
                Gs.add_node(adj_node)
                Gs.add_node(node)
                Gs.add_edge(adj_node,node)
    return Gs


# In[ ]:


N = 40
G_most_cited = get_paper_cited_K_times_graph(Gsub, N)


# In[ ]:


# This illustrate some article are gathering a lot of attention by the community.
size = 15
plt.figure(figsize = [size,size]) 
pos = nx.spring_layout(H) 
nx.draw(G_most_cited, with_labels=False, node_size = 20 , node_color = 'lightblue')
plt.title(f'Most cited papers')
plt.savefig('most_cited.png')


# # Pagerank to score our papers in corpus

# NetworkX also a us quickly build a pagerank score for each paper. The score provide an insight of how important/relevant is a paper in the network. See [pagerank](https://en.wikipedia.org/wiki/PageRank) for more details.

# In[ ]:


get_ipython().run_line_magic('time', 'pr = nx.pagerank(G)')


# In[ ]:


pagerank = pd.DataFrame(pr.items(), columns=["title", "pagerank"]).sort_values(by="pagerank", ascending=False)


# In[ ]:


title = pagerank.iloc[0]["title"]
print(f"The paper <<{title}>> \n is cited {G.in_degree[title]} times. Its annexe contain {G.out_degree[title]} references.")


# We output it so someone can reused it if needed

# In[ ]:


pagerank.to_csv('pagerank.csv')


# Hope that might be useful to someone. All the best and stay safe

# In[ ]:




