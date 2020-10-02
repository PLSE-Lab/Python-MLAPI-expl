#!/usr/bin/env python
# coding: utf-8

# # Creating a Knowledge Graph of the COVID-19 literature
# 
# The dataset can be downloaded [here](https://www.kaggle.com/group16/covid19-literature-knowledge-graph).
# 
# ## What is a Knowledge Graph?
# 
# This dataset provides us with a large collection of papers that deal with the COVID-19 virus. For each paper, different information is provided such as the journal in which it was published, the content of the paper and information about the authors. Additionally, each paper has a number of references to other works. In order to deal with these links between papers, a graph representation is needed. A knowledge graph is a specific type of graph: it is a directed graph with labeled edges. Knowledge graphs allow to unify heteregenous sources of information into a single, standardized format. 
# 
# A knowledge graph of the COVID literature could look something like:
# <div style="text-align:center"><img src="https://i.imgur.com/MJ7VeUM.png" style="width: 50%"/></div>

# ## Mapping to RDF data with [RML.io](https://rml.io/)
# <div style="text-align:center">
#     <img src="https://i.imgur.com/bsA1VWM.png" style="width: 10%; display: inline-block;"/> 
#     <img src="https://i.imgur.com/mz95P06.png" style="width: 10%; display: inline-block;"/>
#     <img src="https://i.imgur.com/InjNtFe.png" style="width: 10%; display: inline-block;"/>
# </div>
# In order to convert the provided CSV & JSON files to RDF data (the standard for Knowledge Graph representation), we will make use of [RML.io](https://rml.io/). RML allows to easily define a mapping from different data formats to RDF data. For an interactive example of how it works, check out [this resource](https://rml.io/yarrrml/matey/).

# # Working with RDF data in Python
# 
# To work with RDF data in Python, [rdflib](https://github.com/RDFLib/rdflib) can be used.

# In[ ]:


get_ipython().system('pip install rdflib')
get_ipython().system('apt-get -y install python-dev graphviz libgraphviz-dev pkg-config')
get_ipython().system('pip install pygraphviz')


# In[ ]:


import rdflib
import networkx as nx
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# # Load the data

# In[ ]:


# This takes a while...
g = rdflib.Graph()
g.parse('/kaggle/input/covid19-literature-knowledge-graph/kg.nt', format='nt')


# # Get some basic statistics

# In[ ]:


# Number of triples
print(len(list(g.triples((None, None, None)))))

#Ppredicates
print(len(set(g.predicates())))

# Number of subjects
print(len(set(g.subjects())))


# # Iterate over triples of a paper

# In[ ]:


rand_paper = rdflib.URIRef('http://dx.doi.org/10.1016/j.mbs.2013.08.014')
for i, (s, p, o) in enumerate(g.triples((rand_paper, None, None))):
    print(s, p, o)


# # Visualise part of the KG

# In[ ]:


for s, p, o in g.triples((rand_paper, None, None)):
    print(s, p, o)


# In[ ]:


preds = set()
for s, p, o in g.triples((rdflib.URIRef('http://idlab.github.io/covid19#bf20dda99538a594eafc258553634fd9195104cb'), None, None)):
    print(s, p, o)


# In[ ]:


def create_sub_graph(root, depth):
    # Limit number of hasWords relations to not overcrowd the figure
    words_cntr = 0
    
    # Get all the triples that are maximally 2 hops away from our randomly picked paper
    objects = set()
    nx_graph = nx.DiGraph()
    
    rdf_subgraph = rdflib.Graph()
    to_explore = {root}
    for _ in range(depth):
        new_explore = set()
        for node in to_explore:
            for s, p, o in g.triples((node, None, None)):
                if 'words' in str(p).lower():
                    if words_cntr >= 25:
                        continue
                    words_cntr += 1

                s_name = str(s).split('/')[-1][:25]
                p_name = str(p).split('/')[-1][:25]
                o_name = str(o).split('/')[-1][:25]
                nx_graph.add_node(s_name, name=s_name)
                nx_graph.add_node(o_name, name=o_name)
                nx_graph.add_edge(s_name, o_name, name=p_name)
                rdf_subgraph.add((s, p, o))
                
                new_explore.add(o)
        to_explore = new_explore
    return nx_graph, rdf_subgraph
    
nx_graph, rdf_subgraph = create_sub_graph(rand_paper, 3)
        
plt.figure(figsize=(20, 20))
_pos = nx.kamada_kawai_layout(nx_graph)
_ = nx.draw_networkx_nodes(nx_graph, pos=_pos)
_ = nx.draw_networkx_edges(nx_graph, pos=_pos)
_ = nx.draw_networkx_labels(nx_graph, pos=_pos, fontsize=8)
names = nx.get_edge_attributes(nx_graph, 'name')
_ = nx.draw_networkx_edge_labels(nx_graph, pos=_pos, edge_labels=names, fontsize=8)


# In[ ]:


rdf_subgraph.serialize(destination='sub.ttl', format='turtle')


# **We can distinguish some information about the content of the paper in the top of the figure. Some citation information on the left and author information in the bottom**
