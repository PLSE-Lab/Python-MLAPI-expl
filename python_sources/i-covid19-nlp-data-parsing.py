#!/usr/bin/env python
# coding: utf-8

# ### 1. Objective
# 
# Here we want to understand the raw data and try to isolate the data for different purpose (e.g., text mining, indexing, network-data for citation network). This notebook will walk you through collecting all data file-names in a python list and provide you a internal structure of the document. We will go through isolation of text data and citation mapping table.
# 
# Getting hard to see this notebook? Go here: https://nbviewer.jupyter.org/github/Vasuji/COVID19/blob/master/I%20-%20COVID19-NLP-Data-Parsing.ipynb
# 

# ### 2. Accessing Data
# 
# In this section we have collected all data file address which are 'json' type into a python list called datafiles. It will be easy to handle these files later.

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


# #### How many data files are there?
# Total 13,202 files

# In[ ]:


len(datafiles)


# Let's see some sample data file address.

# In[ ]:


datafiles[0:2]


# ### 3. Sample Data Exploration
# 
# Let's see contents of a single sample file one by one. We can use jeson package to read a sample file.

# In[ ]:


with open(datafiles[0], 'r')as f1:
    sample = json.load(f1)


# In[ ]:


for key,value in sample.items():
    print(key)


# In[ ]:


print(sample['metadata'].keys())
print('abstract: ',sample['abstract'][0].keys())
print('body_text: ',sample['body_text'][0].keys())
print('bib_entries: ',sample['bib_entries'].keys())
print('ref_entries: ', sample['ref_entries'].keys())
print('back_matter: ',sample['back_matter'][0].keys())


# ### 3.1. Collecting all titles
# 
# What if you want to analyse all title involved in the dataset? Here is one method to colect all title alongwith doc id.

# In[ ]:


id2title = []
for file in datafiles:
    with open(file,'r')as f:
        doc = json.load(f)
    id = doc['paper_id'] 
    title = doc['metadata']['title']
    id2title.append({id:title})


# In[ ]:


id2title[0:3]


# One can save this data in a file as json file

# In[ ]:


#with open('id2title.json','w')as f2:
#    json.dump(id2title,f2)


# ### 3.2 Collecting All Abstracts
# 
# You need to iterate over all files one by one and extract abstract section.

# In[ ]:


id2abstract = []
for file in datafiles:
    with open(file,'r')as f:
        doc = json.load(f)
    id = doc['paper_id'] 
    abstract = ''
    for item in doc['abstract']:
        abstract = abstract + item['text']
        
    id2abstract.append({id:abstract})


# In[ ]:


id2abstract[0]


# In[ ]:


#with open('id2abstract.json','w')as f3:
#   json.dump(id2abstract,f3)


# ### 3.3 Collecting all Body text
# 
# You need to iterate over all files one by one and extract body text section. You could have done this together with title and abstract, right? :)
# 

# In[ ]:


id2bodytext = []
for file in datafiles:
    with open(file,'r')as f:
        doc = json.load(f)
    id = doc['paper_id'] 
    bodytext = ''
    for item in doc['body_text']:
        bodytext = bodytext + item['text']
        
    id2bodytext.append({id:bodytext})


# In[ ]:


#id2bodytext[0]


# In[ ]:


#with open('id2bodytext.json','w')as f4:
#    json.dump(id2bodytext,f4)


# ### 4. Citations and References data
# 
# Let's try to understand structure of bib-entries from a sample data.
# 
# 

# In[ ]:


bibEntries = []
for key,value in sample['bib_entries'].items():
    refid = key
    title = value['title']
    year = value['year']
    venue = value['venue']
    try:
        DOI = value['other_ids']['DOI'][0]
    except:
        DOI = 'NA'
        
    bibEntries.append({"refid": refid,                      "title":title,                      "year": year,                      "venue":venue,                      "DOI": DOI})


# In[ ]:


bibEntries[0:2]


# ### 4.1 Creating Network data for sample data

# In[ ]:


import networkx as nx


# Let's create a sample referance network.

# In[ ]:


G = nx.Graph()
G.add_node(sample['paper_id'])
for item in bibEntries:
    G.add_node(item["refid"], title = item['title'], year = item['year'], venue = item['venue'])
    G.add_edge(sample['paper_id'], item["refid"])  


# In[ ]:


len(G.nodes())


# #### Network visualization

# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize = [10,8])
pos = nx.spring_layout(G)
nx.draw(G,with_labels=True, node_size =1500, node_color = 'lightblue')
plt.savefig('ref.png')


# #### 4.2 Playing around References:
# 

# In[ ]:


for item in list(G.nodes().data('venue')):
    print(item)


# In[ ]:


for item in list(G.nodes().data('title')):
    print(item)


# In[ ]:


for item in list(G.nodes().data('year')):
    print(item)


# **See you soon at the next Notebook!**
