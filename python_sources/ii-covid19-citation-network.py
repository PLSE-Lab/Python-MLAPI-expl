#!/usr/bin/env python
# coding: utf-8

# ## Objective: 
# In this notebook we want to provide an intriductory idea to create citation network from research publication -COVID19 corpus. 
# We want request you to see **I - COVID19-NLP-Data-Parsing** notebook for better understanding of data accessing and cleaning. In this notebook, we directly focus to gather Bib-entries for each of the documents and try to provide a demo for network analysis.

# ### 1. Getting Data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

datafiles = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        ifile = os.path.join(dirname, filename)
        if ifile.split(".")[-1] == "json":
            datafiles.append(ifile)
        #print(ifile)

# Any results you write to the current directory are saved as output.


# In[ ]:


len(datafiles)


# ### 2. Creation of Citation Network
# 
# Here we want to collect all references for each document and create a network data using NetworkX.

# ### 2.1 Preparing network node and edge data
# 
# The main idea behind network of citation is that, we can put a publication information (e.g., ID, title, year published) in node data and connect that node with other reference nodes chich are cited in the publication. In the code below, 'doc' ia single document where all reference node data are being collected in 'bibEntries' list.
# 
# - For every document, we collect 'title' and 'paper-id'. 
# - For every bib-entries, we are taking 'DOI' as the reference id, title,year,and journal(venue).

# In[ ]:


id2bib = []
for file in datafiles:
    '''id and title of a single document'''
    with open(file,'r')as f:
        doc = json.load(f)
    id = doc['paper_id']
    title = doc['metadata']['title']
    
    '''collect bib-entries of a single document'''
    bibEntries = []
    for key,value in doc['bib_entries'].items():
        refid = key
        title = value['title']
        year = value['year']
        venue = value['venue']
        try:
            DOI = value['other_ids']['DOI'][0]
        except:
            DOI = 'NA'
        
        bibEntries.append({"refid": refid,                      "title":title,                      "year": year,                      "venue":venue,                      "DOI": DOI})
    id2bib.append({"id": id, "bib": bibEntries,"title": title})


# In[ ]:


id2bib[0]


# ### 2.2 Create Network with NetworkX
# 
# After preparing citation network data, we expect that some references are also cited in other documents. This offers a beautiful scenario in the network like structure of the citation among several documents.
# 
#  - First, we iterate over each documents and create node for each document and references. 
#  - Next, we create edge between document and references.
#  
#  ***Note: Are same nodes for references repeated over many documents? No, once a reference with 'DOI' as node id is created, it becomes unique. Rather, one reference could have many document connected.***

# In[ ]:


import networkx as nx

G = nx.Graph()
for item in id2bib:
    '''iterate over each doc'''
    G.add_node(item['id'],title = item['title'])
    for ref in item['bib']:
        '''iterate over each reference'''
        G.add_node(ref['DOI'], title = ref['title'], year = ref['year'], venue = ref['venue'])
        G.add_edge(item['id'], ref['DOI'], value = ref['refid'])  


# In[ ]:


'''How many nodes are there in my network?'''
#155339 nodes
len(G.nodes())


# ### Sample Network Visualization
# 
# We can not create a network with ~0.15 million nodes for visualization. For visualization purpose, we are selecting 200 publications and restricting the references which has 'virus' term in the title.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[ ]:


Gs = nx.Graph()
for item in id2bib[0:200]:
    for ref in item['bib']:
        '''select the title which includes -virus- term'''
        if ref['title'].find('virus'):
            Gs.add_node(item['id'])
            Gs.add_node(ref['DOI'])
            Gs.add_edge(item['id'], ref['DOI'])  


# In[ ]:


import seaborn as sns
sns.set()


# In[ ]:


plt.figure(figsize = [12,10]) 
pos = nx.spring_layout(Gs) 
nx.draw(Gs, with_labels=False, node_size = 1, node_color = 'lightblue') 
plt.savefig('cite.png')


# In[ ]:


len(Gs.nodes())


# ### 3.1 Network Analysis
# 
# What are the benefits of performing network analysis over citation network? Well, just by viewing a network visualization may not be interesting. But, if we consider a specific objective for example targeting 'virus' related pubication, we can get interesting insight.

# In[ ]:


Ga = nx.Graph()
for item in id2bib[0:10]:
    Ga.add_node(item['id'],title = item['title'])
    for ref in item['bib']:
        Ga.add_node(ref['DOI'], title = ref['title'], year = ref['year'], venue = ref['venue'])
        Ga.add_edge(item['id'], ref['DOI'], value = ref['refid'])  


# #### Q: What is the data content in each network node? Let's look at this:

# In[ ]:


i = 0
for item in Ga.nodes().data():
    i += 1
    print(item)
    if i>10:
        break


# #### Q: Can we find the document which cites maximum references for 'virus'? Yes, we can measure the degree centrality and select the node which has maximum centrality.

# In[ ]:


DegreeCentrality = nx.degree_centrality(Ga)
ID = max(DegreeCentrality)
ID, DegreeCentrality[ID]


# In[ ]:


'''the node title which has maximum paper cited on virus'''
Ga.nodes[ID]


# In[ ]:


'''the titles of the cited papers'''
for item in Ga.neighbors(ID):
    print(Ga.nodes[item])


# We are bringing more update soon....
# 

# In[ ]:




