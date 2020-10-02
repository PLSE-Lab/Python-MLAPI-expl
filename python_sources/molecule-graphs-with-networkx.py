#!/usr/bin/env python
# coding: utf-8

# After reading this [discussion](https://www.kaggle.com/c/champs-scalar-coupling/discussion/96586#latest-557994), I have decided to explore the [**graph**](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics) (in its mathematical definition) of each molecule and see what I can get from it. For that, I will be using the [networkx](https://networkx.github.io/documentation/stable/) library. 
# 
# Let's go!

# In[ ]:


# Some imports
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pylab as plt


# # Load the data and add L2 distance computation between atoms

# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
structures_df = pd.read_csv('../input/structures.csv')
test_df['scalar_coupling_constant'] = np.nan
df = pd.concat([train_df, test_df])
del train_df
del test_df


# In[ ]:



for atom_index in [0, 1]:
    renamed_columns = {col: col + "_" + str(atom_index) for col in ["x", "y", "z", "atom_index", 
                                                                    "atom"]}
    df = (df.merge(structures_df.rename(columns=renamed_columns),
                   on=['molecule_name', 'atom_index_' + str(atom_index)], how='inner'))
df['distance_l2'] = ((df['x_0'] - df['x_1']) ** 2 + (df['y_0'] - df['y_1'])
                     ** 2 + (df['z_0'] - df['z_1']) ** 2) ** 0.5


# In[ ]:


MOLECULE_NAMES = df['molecule_name'].unique()


# # Construct the graph and plot it

# In[ ]:


def get_molecule_graph(df, molecule_name):
    molecule_df = df.loc[lambda df: df['molecule_name'] == molecule_name]
    labels = molecule_df[['atom_1', 'atom_index_1']].set_index('atom_index_1')['atom_1'].to_dict()
    labels.update(molecule_df[['atom_0', 'atom_index_0']].set_index('atom_index_0')['atom_0'].to_dict())
    graph = nx.from_pandas_edgelist(molecule_df, source='atom_index_0', 
                                    target='atom_index_1', edge_attr='scalar_coupling_constant', 
                                    create_using=nx.Graph())
    return graph, labels

def draw_graph(graph, labels, weight="distance_l2"):
    position = nx.spring_layout(graph, weight=weight)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    nx.draw_networkx_nodes(graph, position, node_color='red', alpha = 0.8, ax=ax)
    nx.draw_networkx_edges(graph, position, edge_color='blue', alpha = 0.6, ax=ax)
    nx.draw_networkx_labels(graph, position, labels, font_size=16, ax=ax)
    return ax


# In[ ]:


# Plotting for only few molecules
for molecule_name in MOLECULE_NAMES[:10]:
    graph, labels = get_molecule_graph(df, molecule_name)
    ax = draw_graph(graph, labels)
    ax.set_title(f"Graph for {molecule_name}")


# That's it for now, I will update this kernel whenever I find new interesting things. :)
