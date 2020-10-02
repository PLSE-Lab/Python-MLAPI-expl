#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# When it comes to education backgrounds, Kate and Alex from TechCrunch's [Equity podcast](https://techcrunch.com/tag/equity-podcast/) often simply ask the guests of their show: "Stanford or Harvard?". This succinct question illustrates well the pervasive idea that most people involved in startups and venture capital come from few select backgrounds.
# 
# This notebook is dedicated to exploring the education of the individuals involved in the startup ecosystem.

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import holoviews as hv
from holoviews import opts
from holoviews.operation.datashader import datashade, bundle_graph

import networkx as nx


# In[ ]:


hv.extension('bokeh')

defaults = dict(width=600, height=600, padding=0.1, yaxis=None, xaxis=None, show_frame=False)
hv.opts.defaults(
    opts.EdgePaths(**defaults), opts.Graph(**defaults), opts.Nodes(**defaults))


# In[ ]:


people = pd.read_csv('../input/startup-investments/people.csv', index_col=0)
degrees = pd.read_csv('../input/startup-investments/degrees.csv', index_col=0)

df = people.merge(degrees, on='object_id')


# In[ ]:


df.info()


# ## Data Exploration

# In[ ]:


df['full_name'] = df['first_name'].str.cat(df['last_name'],sep=" ")

df['institution'] = df['institution'].replace('Harvard Business School' ,'Harvard University')
df['institution'] = df['institution'].replace('Stanford University Graduate School of Business' ,'Stanford University')


# In[ ]:


df = df[df['affiliation_name'] != 'Unaffiliated']


# In[ ]:


df = df[['object_id', 'full_name', 'birthplace', 'institution', 'degree_type', 'subject', 'graduated_at', 'affiliation_name']]


# In[ ]:


def count_plots(df, col_count):
    for i, col in enumerate(df.columns):
        plt.figure(i, figsize=(10,5))
        sns.countplot(x=col, data=df, order=pd.value_counts(df[col]).iloc[:col_count].index)
        plt.xticks(rotation=70)
        
count_columns = df[['institution', 'degree_type', 'subject', 'affiliation_name']]

count_plots(count_columns, 10)


# In[ ]:


def dual_degree_flag_generator(df):
    group = df.groupby(['object_id', 'institution', 'graduated_at'], as_index=False)['full_name'].count()
    group = group[group['full_name'] > 1]
    object_ids = group['object_id']
    
    df['dual_degree_flag'] = np.where(df['object_id'].isin(object_ids), 1, 0)
    
    return df

df = dual_degree_flag_generator(df)


# In[ ]:


institution_occurance_count = df['institution'].value_counts()
important_institutions = institution_occurance_count[institution_occurance_count >= 5].index.values
df = df[df['institution'].isin(important_institutions)]


# In[ ]:


df = df.dropna()


# In[ ]:


df = df[:5000]


# ## Graph Analysis and Visualisation

# In[ ]:


# Create the graph object
G = nx.Graph()


# In[ ]:


G = nx.from_pandas_edgelist(df, source='full_name', target='institution', 
                            edge_attr=['degree_type', 'subject'])


# In[ ]:


nx.set_node_attributes(G, pd.Series(df['affiliation_name'].values, index=df['full_name']).to_dict(), 'company')
nx.set_node_attributes(G, pd.Series(np.nan, index=df['institution']).to_dict(), 'company')


# In[ ]:


list(G.edges(data=True))[:5]


# In[ ]:


list(G.nodes(data=True))[:5]


# In[ ]:


print(nx.info(G))


# In[ ]:


components = nx.connected_components(G)
largest_component = max(components, key=len)
subgraph = G.subgraph(largest_component)
diameter = nx.diameter(subgraph)
print("Network diameter of largest component:", diameter)


# In[ ]:


triadic_closure = nx.transitivity(G)
print("Triadic closure:", triadic_closure)


# In[ ]:


simple_graph = hv.Graph.from_networkx(G, positions=nx.spring_layout(G))
simple_graph.opts(title="Education Network", node_color='company', cmap='set3', edge_color='degree_type', edge_cmap='set3')

