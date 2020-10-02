#!/usr/bin/env python
# coding: utf-8

# # Get familiar with a book without reading it
# This is a short and simple notebook which demonstrates gaining insights by creation of a graph which visualizes the connections network between the main places and people in a book, and by using centrality measures.
# 
# I use the following tools:<br>
# 1. spaCy - An NLP (natural language processing) package, to perform NER (named entity recognition).
# 2. NetwrokX - A package for the creation and manipulation of complex networks.
# 3. [Cytoscape](https://cytoscape.org/) - An external software, used to visualize the graph.
# 
# I create the graph of the book ["The Adventures of Huckleberry Finn"](http://www.gutenberg.org/files/76/76-0.txt), by Mark Twain.

# # Imports

# In[ ]:


import re
import spacy
from collections import Counter
import networkx as nx
from networkx import centrality as cent
from IPython.display import Image


# # Extract the data

# In[ ]:


def _entity_normalizer(entity):
    """Returns a clean entity."""
    
    ent = entity.text.lower()
    ent = ent.replace('_','').replace('-','').replace('\'s','').replace('\n',' ').strip()
    if 'sid' in ent: return 'Sid Sawyer'
    if 'garrick' in ent: return 'David Garrick the younger'
    if 'emmeline' in ent: return 'Emmeline Grangerford'
    if 'huck' in ent or 'finn' in ent or 'huckleberry' in ent: return 'Huckleberry Finn'
    if ent == 'tom' or ent == 'sawyer'  or ent == 'tom sawyer': return 'Tom Sawyer'
    if 'orleans' in ent: return 'New Orleans'
    return ent.capitalize()


# In[ ]:


def get_entites_dict_from_paragraph(paragraph, nlp):
    """Returns a dictionary with entities from input paragraph."""
    
    # instantiate output dictionary
    output = {'GPE': set(), 'PERSON': set()}
    
    # process paragraph
    doc = nlp(paragraph)
    
    # extract places and people entities
    for entity in doc.ents:
        if entity.label_ == 'GPE' or entity.label_ == 'PERSON':
            output[entity.label_].add(_entity_normalizer(entity))  # clean the entity
            
    # return the dictionary if contains at least 2 entities
    return output if len(output['GPE']) + len(output['PERSON']) >= 2 else None


# In[ ]:


def get_edges_from_entities(entities):
    """
    Returns a list of edges between entities.
    
    An edge is created if 2 entities appear on the same paragraph.
    """
    
    edges = set()
    
    # links of person and place
    for loc in entities['GPE']:
        for person in entities['PERSON']:
            if loc==person:  # in case spacy identified a person as a location (happens to Jim)
                continue
            edges.add(((loc, 'loc'), (person, 'prs')))
    
    # links of people
    for person_1 in entities['PERSON']:
        for person_2 in entities['PERSON']:
            # skip same person links
            if person_1==person_2:
                continue
            if person_1 > person_2:
                person_1, person_2 = person_2, person_1  # switch order to avoid 2-way edges
            edges.add(((person_1, 'prs'), (person_2, 'prs')))
            
    return list(edges)


# In[ ]:


# open the file
path = '../input/data/Huckleberry_Finn.txt'
with open(path, 'r') as file:
    finn = file.read()


# Depends on the NLP task different pre-processing is needed. <br>
# For this toy example, not much is needed...

# In[ ]:


# remove text which is not part of the story
begin_ind = re.search(r'CHAPTER I', finn).start()
end_ind = re.search(r'THE END', finn).end()
finn = finn[begin_ind : end_ind]

# lowercase
finn = finn.lower()

# replace "t'word" with "the word"
finn = re.sub(r"[t]\'(.*)(?=\s)", r'the \1', finn)

# replace "bekase" with "because"
finn = re.sub(r'bekase', 'because', finn)

finn = re.sub(r"de bes'", 'the best', finn)

# split the book to paragraphs
paragraphs = finn.split("\n\n")


# In[ ]:


# load english NLP module
nlp_processor = spacy.load('en_core_web_lg')


# In[ ]:


edges_list = []

# iterate through paragraphs and extract links between places and people or people and people
for para in paragraphs:
    
    # get entities from paragraph
    entities = get_entites_dict_from_paragraph(paragraph=para, nlp=nlp_processor)
    
    # create edges
    if entities is not None:
        edges_list += get_edges_from_entities(entities=entities)

# count occurrences of each edge (number of paragraphs each edge appears)
counted_edges = dict(Counter(edges_list))


# # Create the network graph

# In[ ]:


def add_edges(g, attribute, min_occurrences):
    """Add to graph g nodes and edges with corresponding attributes."""
    
    for edge, occurrences in counted_edges.items():
        if edge[0][1] == attribute and occurrences >= min_occurrences:
            g.add_node(edge[0][0], typ = attribute)
            g.add_node(edge[1][0], typ = 'prs')
            g.add_edge(edge[0][0], edge[1][0], weight = occurrences)


# In[ ]:


G = nx.Graph()

add_edges(g=G, attribute='loc', min_occurrences=1)  # add edge if location-person appears at least once
add_edges(g=G, attribute='prs', min_occurrences=2)  # add edge if person-person appears at least twice


# In[ ]:


# export the graph as .gml file, to use Cytoscape
nx.write_gml(G, "Huckleberry_Finn_graph.gml")


# # Visualize the network using Cytoscape:

# In[ ]:


Image(filename='../input/data/Huckleberry_Finn_Zoom_Out.png')


# In[ ]:


Image(filename='../input/data/Huckleberry_Finn_Zoom_In.png')


# People nodes are colored green. <br>
# Places nodes are colored orange. <br>
# Node size denotes how many distinct people were mentioned with that person or place. <br>
# Edge width denotes how many times (in how many paragraphs) two entities were mentioned together.
# 
# ** This image was created after using Spacy version 2.1.3 and networkX version 2.2, thus, some inconsistency might be present.

# ### Visualization insights

# Without ever reading "The Adventures of Huckleberry Finn", and only by looking at the graph, we can come to the following conclusions: 
# - Huckleberry Finn, Tom Sawyer, and Jim are the main characters of the book.
# - The story takes place in different parts of the world, including Middle east, Europe and the USA.
# - Jim and Tom Sawyer had a lot of time together, while Huckleberry Finn never met William the Conqueror.

# # Centrality measures

# Centraly measures come to reveal the central nodes, by different aspects.<br>
# We can reveal a lot without the need for visualization (and when visualziation does not help).<br>

# This time, the network is composed of people only, to gain some insights about their centrality.<br>
# As opposed to the people and places network, this time an edge exist if 2 people appear at least once togeher (and not twice).<br>
# In addition, to avoid high centrality scores in small connected components, I analyze only the greatest connected component.<br>

# In[ ]:


def top_central(graph, centrality_measure, k=10):
    """Returns a list of tuples, which are the top k scored people and the scores according to input centrality measure."""
    
    # a map of centrality measures to be determined by an input string
    func = {
        'closeness': cent.closeness_centrality,
        'degree': cent.degree_centrality,
        'betweeness': cent.betweenness_centrality
    }
    
    # create a list of tuples (person, centrality measure score) of input centrality measure
    output = [(person, score) for person, score in func[centrality_measure](graph).items()]
    
    # sort the list by score, top to bottom
    output.sort(key=lambda tup: tup[1], reverse=True)
    
    # return k-top
    return output[:k]


# In[ ]:


PG = nx.Graph()

# add edge if person-person appears at least once
add_edges(g=PG, attribute='prs', min_occurrences=1)

# create a subgraph from greatest connected component
gcc_nodes = max(nx.connected_components(PG), key=len)
PG = PG.subgraph(gcc_nodes)


# In[ ]:


# determine number of top scores
k=6


# In[ ]:


# create 3 lists of scores by different centrality measures
top_k_closeness_scores = top_central(graph=PG, centrality_measure='closeness', k=k)
top_k_betweeness_scores = top_central(graph=PG, centrality_measure='betweeness', k=k)
top_k_degree_scores = top_central(graph=PG, centrality_measure='degree', k=k)


# In[ ]:


# an example
top_k_betweeness_scores


# Centrality measures tend to be highly correlated.<br>
# However, valuable insights could be gained when comparing the results, and looking at the uncorrelated ones.<br>

# In[ ]:


# extract only names from above lists
top_k_closeness_names = list(zip(*top_k_closeness_scores))[0]
top_k_betweeness_names = list(zip(*top_k_betweeness_scores))[0]
top_k_degree_names = list(zip(*top_k_degree_scores))[0]


# In[ ]:


# find names with high closeness-centrality score and low degree-cetrality score
high_closeness_low_degree = [name for name in top_k_closeness_names if name not in top_k_degree_names]
high_closeness_low_degree


# In[ ]:


# find names with high betweenes-centrality score and low degree-cetrality score
high_betweeness_low_degree = [name for name in top_k_betweeness_names if name not in top_k_degree_names]
high_betweeness_low_degree


# ### Centrality measures insights:

# - A node with high closeness-centrality score (how close a node is to all other nodes) and a low degree-centrality score (based on the degree of a node), is a node with a central position in the network, because it succeeded in gaining a central position while not creating a lot of edges with other nodes. <br>
# We can observe that Ben, Phelps and Watson are in central positions in the story.
# 
# - High betweenness-centrality nodes are bottlenecks of the network; i.e., most of the paths between communities in the network go through them.<br>
# We can observe that Mary Jane is such a bottleneck, while not having an high degree-centrality score.
