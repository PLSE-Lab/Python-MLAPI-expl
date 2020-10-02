#!/usr/bin/env python
# coding: utf-8

# * In this kernel, I create a citation graph based on the research papers in the json file. 
# * Most papers have bibref section which captures the citations in the paper.
# * Most of the papers in the citation are not present in the cord-19 dataset and details are also in the kernel.
# * I create a citation graph and then apply DeepWalk algorithm to create embeddings of the papers such that related papers are closer.
# * Due to memory constraint of the kernel, I have pruned the graph such that each paper can have at most 25 directed edges going to its references (fan-out). They are selected based on the freqency. 
# * However, a paper can have many edges (fan-in).
# 
# Also, reference to my other kernel on creating vocabulary of short-hand notation in the research papers https://www.kaggle.com/midnitekoder/coronavirus-jargon-vocabulary
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


import os
import json
from multiprocessing import Pool
import random
import pickle
import re
from functools import reduce
# Any results you write to the current directory are saved as output.
import networkx as nx

from gensim.models import Word2Vec
import gc


# In[ ]:


filenames_list = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for each_filename in filenames:
        filenames_list.append(os.path.join(dirname, each_filename))


# In[ ]:


len(filenames_list)


# In[ ]:


# for filename in random.sample(filenames_list, 2):
#     if filename.split(".")[-1] == "json":
#         ifp = open(os.path.join(dirname, filename))
#         research_paper = json.load(ifp)
#         title = research_paper["metadata"]["title"]
#         print(title, "\n\n")
#         abstract_text = " ".join([each["text"] for each in research_paper["abstract"]])
#         print(abstract_text, "\n\n")
#         body_text = " ".join([each["text"] for each in research_paper["body_text"]])
#         print(body_text)


# In[ ]:


research_paper_title_list = []
for filename in filenames_list:
    if filename.split(".")[-1] == "json":
        ifp = open(os.path.join(dirname, filename))
        research_paper = json.load(ifp)
        research_paper_title_list.append(research_paper["metadata"]["title"])
        for each_ref in research_paper["bib_entries"]:
            research_paper_title_list.append(research_paper["bib_entries"][each_ref]["title"])

        
        
        
        
        


# In[ ]:


len(research_paper_title_list)


# In[ ]:


paper_id_dict= dict(zip(research_paper_title_list, list(map(lambda x: str(x), range(len(research_paper_title_list))))))


# In[ ]:


id_paper_dict = dict(zip(paper_id_dict.values(), paper_id_dict.keys()))


# In[ ]:


paper_undirected_degree_dict= dict(zip(research_paper_title_list, [0]*len(research_paper_title_list)))


# In[ ]:


adj_mat = {}
for filename in filenames_list:
    if filename.split(".")[-1] == "json":
        ifp = open(os.path.join(dirname, filename))
        research_paper = json.load(ifp)
        adj_mat[paper_id_dict[research_paper["metadata"]["title"]]] = [paper_id_dict[research_paper["bib_entries"][each_key]["title"]] for each_key in research_paper["bib_entries"]]
        paper_undirected_degree_dict[research_paper["metadata"]["title"]] += len(adj_mat[paper_id_dict[research_paper["metadata"]["title"]]])
        for each_key in research_paper["bib_entries"]:
            paper_undirected_degree_dict[research_paper["bib_entries"][each_key]["title"]] += 1
        


# In[ ]:


pruned_adj_mat = {}
for each_key in adj_mat:
    freq_ref = [paper_undirected_degree_dict[id_paper_dict[each_id]] for each_id in adj_mat[each_key]]
    ref_freq_dict = dict(zip(adj_mat[each_key], freq_ref))
    pruned_adj_mat[each_key] = sorted(adj_mat[each_key], key=lambda x: ref_freq_dict[x], reverse=True)[:25]
            


# In[ ]:


for each_key in random.sample(adj_mat.keys(), 5):
    print(each_key, adj_mat[each_key])


# In[ ]:


degrees = [len(adj_mat[each_key]) for each_key in adj_mat]


# In[ ]:


np.mean(degrees)


# In[ ]:


np.median(degrees)


# In[ ]:


np.max(degrees)


# In[ ]:


np.min(degrees)


# In[ ]:


nodes_in_pruned_graph = list(reduce(lambda x, y: x + y, pruned_adj_mat.values())) + list(pruned_adj_mat.keys())


# In[ ]:


len(nodes_in_pruned_graph)


# In[ ]:


len(set(nodes_in_pruned_graph))


# In[ ]:


citation_graph = nx.from_dict_of_lists(pruned_adj_mat)


# In[ ]:


adj_mat = None
gc.collect()


# In[ ]:


pruned_degrees = list(dict(citation_graph.degree).values())
print(len(pruned_degrees), np.mean(pruned_degrees), np.median(pruned_degrees), np.min(pruned_degrees), np.max(pruned_degrees))


# In[ ]:


def random_walk(arg):
    root_node, walk_length = arg
    walk = [root_node]

    for i in range(1, walk_length):
        cur = walk[i-1]
#         try:
        neighbours = list(citation_graph.neighbors(cur))
        if len(neighbours) > 0:
            walk.append(random.choice(neighbours))
        else:
            walk = walk[:-1]
            break
#         if type(walk[-1]) == str:
#             print(walk[-1])
#             walk = walk[:-1]
#             break
#         except:
#             break
    return walk


# In[ ]:


def deepwalk_random_walks(num_walks, walk_length):
    nodes = list(citation_graph.nodes())
    walks = []
    for i in range(num_walks):
        print("walk no. ", i)
        random.shuffle(nodes)
        with Pool(processes=32) as pool:
            walks = walks + pool.map(random_walk, zip(nodes,[walk_length]*len(nodes)))
    return walks


# In[ ]:


random_walks = deepwalk_random_walks(20, 10)


# In[ ]:


len(random_walks)


# In[ ]:


random_walks[2]


# In[ ]:


model = Word2Vec(random_walks, size=32, window=4, alpha=0.005, min_count=0, sg=1, workers=16, iter=5, negative=5)


# In[ ]:


def most_similar_papers(title, topn=20):
    return [(id_paper_dict[each[0]], each[1]) for each in model.wv.most_similar(paper_id_dict[title], topn=topn)]


# ## Some examples

# In[ ]:


most_similar_papers('Discovery and Characterization of Novel Bat Coronavirus Lineages from Kazakhstan. Viruses')


# In[ ]:


most_similar_papers('Ebola virus enters host cells by macropinocytosis and clathrin-mediated endocytosis')


# In[ ]:


most_similar_papers('Enhanced growth of a murine coronavirus in transformed mouse cells')


# In[ ]:


model.save("/kaggle/working/node2vec_citation_graph_covid19.wv")


# In[ ]:


paper_id_ = list(zip(paper_id_dict.keys(), paper_id_dict.values()))
paper_id_df = pd.DataFrame(paper_id_, columns=["paper_title", "paper_id"])
paper_id_df.to_csv("/kaggle/working/paper_id_map.csv")
ofp = open("/kaggle/working/id_paper_map.pickle", "wb")
pickle.dump(id_paper_dict, ofp)
ofp = open("/kaggle/working/paper_id_map.pickle", "wb")
pickle.dump(paper_id_dict, ofp)

