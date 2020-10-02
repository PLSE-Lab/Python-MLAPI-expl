#!/usr/bin/env python
# coding: utf-8

# # Kensho Derived Wikimedia Dataset (KDWD) - Wikidata Introduction
# 
# This notebook will introduce you to the Wikidata Sample of the Kensho Derived Wikimedia Dataset (KDWD).  We'll explore the files and make some basic "getting to know you" plots.  Lets start off by importing some packages.

# In[ ]:


from collections import Counter
import csv
import gc
import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm


# In[ ]:


pd.set_option('max_colwidth', 160)
sns.set()
sns.set_context('talk')


# Next we'll check the input directory to see what files we have access to. 

# In[ ]:


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# All of the KDWD files have one "thing" per line.  We'll hard code the number of lines in the files we're going to use so we can have nice progress bars when streaming through them.

# In[ ]:


NUM_STATEMENT_LINES = 141_206_854
NUM_ITEM_LINES = 51_450_317
kdwd_path = os.path.join("/kaggle/input", "kensho-derived-wikimedia-data")


# # Properties
# We'll begin by reading the property metadata.

# In[ ]:


file_path = os.path.join(kdwd_path, "property.csv")
property_df = pd.read_csv(file_path, keep_default_na=False, index_col='property_id')
property_df


# The two most important ontological properties in Wikidata are,  
#  * [P31 (instance of)](https://www.wikidata.org/wiki/Property:P31)
#  * [P279 (subclass of)](https://www.wikidata.org/wiki/Property:P279)

# In[ ]:


property_df.loc[31, 'en_description']


# In[ ]:


property_df.loc[279, 'en_description']


# # Ontological Statements 
# Now we will move on to statements. In order to stay within the 16GB RAM limit of this kernel, we will focus on the ontological statements in Wikidata. Remember that statements are triples of the form (source item, property, target item). Lets read just the `P31 (instance of)` and `P279 (subclass of)` statements from `statements.csv`

# In[ ]:


file_path = os.path.join(kdwd_path, "statements.csv")
chunksize = 1_000_000
qpq_df_chunks = pd.read_csv(file_path, chunksize=chunksize)
qpq_p31_df = pd.DataFrame()
qpq_p279_df = pd.DataFrame()
for qpq_df_chunk in tqdm(qpq_df_chunks, total=NUM_STATEMENT_LINES/chunksize, desc='reading ontology statements'):
    qpq_p31_df = pd.concat([
        qpq_p31_df, 
        qpq_df_chunk[qpq_df_chunk['edge_property_id']==31][['source_item_id', 'target_item_id']]
    ])
    qpq_p279_df = pd.concat([
        qpq_p279_df, 
        qpq_df_chunk[qpq_df_chunk['edge_property_id']==279][['source_item_id', 'target_item_id']]
    ])


# In[ ]:


# instance of statements
qpq_p31_df


# In[ ]:


# subclass of statements
qpq_p279_df


# # Engish Labels and Descriptions
# Now lets read the English labels and descriptions for the items that are in our ontological statement DataFrames. 

# In[ ]:


keep_p279_ids = (
    set().
    union(set(qpq_p279_df['source_item_id'].values)).
    union(set(qpq_p279_df['target_item_id'].values))
)

keep_p31_ids = (
    set().
    union(set(qpq_p31_df['source_item_id'].values)).
    union(set(qpq_p31_df['target_item_id'].values))
)

keep_item_ids = keep_p279_ids.union(keep_p31_ids)


# In[ ]:


file_path = os.path.join(kdwd_path, "item.csv")
chunksize = 1_000_000
item_df_chunks = pd.read_csv(
    file_path, chunksize=chunksize, index_col='item_id', keep_default_na=False)
item_df = pd.DataFrame()
for item_df_chunk in tqdm(item_df_chunks, total=NUM_ITEM_LINES/chunksize, desc='reading item labels'):
    item_df = pd.concat([
        item_df, 
        item_df_chunk.loc[set(item_df_chunk.index.values).intersection(keep_item_ids)]
    ])


# In[ ]:


item_df


# Notice that not every item has a label and description in English.

# In[ ]:


item_df[item_df['en_label']=='']


# The first item without an English label or description is Q7868.  At the time this kernel was run (2020-01-31), the live Wikidata page (https://www.wikidata.org/wiki/Q7868) indicated that the label for this item was "cell", the description was "the basic structural and functional unit of all organisms", and the linked English Wikipedia page was https://en.wikipedia.org/wiki/Cell_(biology).  The full edit histoy of every Wikidata item is available for anyone to view, so lets investigate. The edit history for Q7868 can be viewed at https://www.wikidata.org/w/index.php?title=Q7868&action=history.  The Wikidata dump for the KDWD was made on 2019-12-02, and we can see that someone vandalized the page on 2019-12-01 and that it was reverted on 2019-12-03.  This is a good example of the pros and the cons of working with crowd-sourced data!  However, not all empty English descriptions are vandalism.  The next item that is empty is Q44423.  Visiting the Wikidata page (https://www.wikidata.org/wiki/Q44423) indicated that it had labels in Japanese, Korean, and Chinese but not in English. 

# Also, note that the English labels of Wikidata items are not unique. 

# In[ ]:


item_df[item_df['en_label']=='city']


# # instance of what?
# Now lets examine the `P31 (instance of)` statements.  Grouping them by `target_item_id` will show us the most common things in our sample. 

# In[ ]:


is_instance_counts = (
    qpq_p31_df.groupby(['target_item_id']).
    size().
    sort_values(ascending=False).
    to_frame().
    rename(columns={0: 'is_instance_count'})
)


# In[ ]:


is_instance_counts


# That's great, but lets merge that with the `item_df` DataFrame so we can see labels and descriptions.

# In[ ]:


is_instance_df = pd.merge(
is_instance_counts,
item_df,
left_index=True,
right_index=True)


# In[ ]:


is_instance_df


# We can see that the most common use of the [P31 (instance of)](https://www.wikidata.org/wiki/Property:P31) property is to indicate that an item is an instance of [Q5 (human)](https://www.wikidata.org/wiki/Q5).  There are 6,221,695 humans in our Wikidata sample but only 5.3 million pages total in our Wikipedia sample.  This indicates that many of these humans are on the target side of statements as opposed to the source side. The second most common target is [Q16521 (taxon)](https://www.wikidata.org/wiki/Q16521). For example [Q33609 (polar bear)](https://www.wikidata.org/wiki/Q33609) is an instance of taxon and the polar bear item has statements about the [P105 (taxon rank)](https://www.wikidata.org/wiki/Property:P105) being [Q7432 (species)](https://www.wikidata.org/wiki/Q7432), and the [P171 (parent taxon)](https://www.wikidata.org/wiki/Property:P171) being [Q243359 (Ursus)](https://www.wikidata.org/wiki/Q243359). 

# # A Closer Look at One Item
# Lets take a closer look at the instance of statements for one item, [Q61 (Washington, D.C.)](https://www.wikidata.org/wiki/Q61).

# In[ ]:


item_id = 61
p31_for_q61 = qpq_p31_df[qpq_p31_df['source_item_id']==item_id]
p31_for_q61


# In[ ]:


item_df.reindex(p31_for_q61['target_item_id'])


# Here we see that there are four instance of statements with the following target items, 
#  * [Q5119 (capital)](https://www.wikidata.org/wiki/Q5119)
#  * [Q475050 (federal district)](https://www.wikidata.org/wiki/Q475050)
#  * [Q1093829 (city of the United States)](https://www.wikidata.org/wiki/Q1093829)
#  * [Q1549591 (big city)](https://www.wikidata.org/wiki/Q1549591)
#  
# Note that [Q5119 (capital)](https://www.wikidata.org/wiki/Q5119) and [Q475050 (federal district)](https://www.wikidata.org/wiki/Q475050) have associated English Wikipedia pages while [Q1093829 (city of the United States)](https://www.wikidata.org/wiki/Q1093829) and [Q1549591 (big city)](https://www.wikidata.org/wiki/Q1549591) do not.

# # "Subclass of" Subgraph
# Now lets build a networkx directed graph from the [P279 (subclass of)](https://www.wikidata.org/wiki/Property:P279) statements and draw a small part of it around [Q1093829 (city of the United States)](https://www.wikidata.org/wiki/Q1093829).

# In[ ]:


subclass_graph = nx.DiGraph()
subclass_graph.add_edges_from(qpq_p279_df.values)


# Lets examine the `out_edges` and the `in_edges` for [Q1093829 (city of the United States)](https://www.wikidata.org/wiki/Q1093829).  `out_edges` will show us things that `Q1093829 (city of the United States)` is a subclass of while `in_edges` will show us things that are subclass of `Q1093829 (city of the United States)`.

# In[ ]:


item_id = 1093829 # city of the United States
in_edges = subclass_graph.in_edges(item_id)
out_edges = subclass_graph.out_edges(item_id)

print('out edges')
print('-' * 20)
for source_item_id, target_item_id in out_edges:
    print('Q{} (label={}, description={})\n is subclass of\nQ{} (label={}, description={})'.format(
        source_item_id,
        item_df.loc[source_item_id, 'en_label'],
        item_df.loc[source_item_id, 'en_description'],
        target_item_id,
        item_df.loc[target_item_id, 'en_label'],
        item_df.loc[target_item_id, 'en_description']))
    print()

print('in edges')
print('-' * 20)
for source_item_id, target_item_id in in_edges:
    print('Q{} (label={}, description={})\n is subclass of\nQ{} (label={}, description={})'.format(
        source_item_id,
        item_df.loc[source_item_id, 'en_label'],
        item_df.loc[source_item_id, 'en_description'],
        target_item_id,
        item_df.loc[target_item_id, 'en_label'],
        item_df.loc[target_item_id, 'en_description']))
    print()


# Now lets build a subgraph around the neighborhood of `Q1093829 (city of the United States)` and draw it. Remember, we are still only looking at `subclass of` statements.

# In[ ]:


def build_neighborhood(graph, start_qid, k_max):
    subnodes = set([start_qid])
    for k in range(k_max):
        nodes_to_add = set()
        for qid in subnodes:
            nodes_to_add.update(graph.neighbors(qid))
        subnodes.update(nodes_to_add)
    return graph.subgraph(subnodes)


# In[ ]:


def add_attributes_to_graph(graph, item_df):
    for node in graph:
        graph.nodes[node]['qid'] = node
        graph.nodes[node]['label'] = item_df.loc[node, 'en_label']
        graph.nodes[node]['description'] = item_df.loc[node, 'en_description']
    return graph


# In[ ]:


def plot_graph(graph):
    fig, ax = plt.subplots(figsize=(14,14))
    pos = nx.circular_layout(graph, scale=2.0)

    node_labels = nx.get_node_attributes(graph, 'label')
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=18, font_weight=1000)
    nx.draw_networkx_nodes(graph, pos, node_size=800, node_color='red')
    nx.draw_networkx_edges(graph, pos, arrowsize=30, min_target_margin=20)

    xpos = [el[0] for el in pos.values()]
    xmin = min(xpos)
    xmax = max(xpos)
    ypos = [el[1] for el in pos.values()]
    ymin = min(ypos)
    ymax = max(ypos)

    xdif = xmax - xmin
    ydif = ymax - ymin
    fac = 0.3

    ax.set_xlim(xmin-xdif*fac, xmax+xdif*fac)
    ax.set_ylim(ymin-ydif*fac, ymax+ydif*fac)


# In[ ]:


start_qid = 1093829
k_max = 2
sg = build_neighborhood(subclass_graph, start_qid, k_max)
sg = add_attributes_to_graph(sg, item_df)


# In[ ]:


plot_graph(sg)

