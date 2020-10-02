#!/usr/bin/env python
# coding: utf-8

# This notebook generates the bond-topology of the structures and then calculates the shortest path beween the interaction partners. If the shortest path lenght is 2 bonds i.e. there is only one atom between the interaction partners, the angle is calculated. If there are 3 bonds in between, the dihedral-angle is calculated. The angle improves the model for 2J models, the dihedral-angle improves the 3J models.

# In[ ]:





# In[ ]:


get_ipython().run_line_magic('pylab', 'inline')

import numpy as np
import os
import pandas as pd
import seaborn as sns
from glob import glob
from multiprocessing import Process, Pool, Manager, cpu_count
from ipywidgets import IntProgress as Progress, Layout
import time

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns = 100

import networkx as nx
import numpy as np


# In[ ]:


'''https://chem.libretexts.org/Ancillary_Materials/Reference/Reference_Tables/Atomic_and_Molecular_Properties/A3%3A_Covalent_Radii'''
bond_lengths = {'H': np.array([0.32, np.NaN, np.NaN]),
                'C': np.array([.75, .67, .60]),
                'O': np.array([.63, .57, .54]),
                'N': np.array([.71, .60, .54]),
                'F': np.array([.64, .59, .53])}
bond_lengths


# In[ ]:


# Let's take only the first 100k rows for demonstration
train = pd.read_csv('../input/train.csv', nrows=100000)
test = pd.read_csv('../input/test.csv', nrows=100000)


# In[ ]:


structures = pd.read_csv('../input/structures.csv')
structures['possible_radii'] = [bond_lengths[a] for a in structures.atom]
display(structures.head())


# In[ ]:


def get_graph(structure, cutoff=0.2):
    structure_x = get_structure_x(structure)
    structure_x = structure_x[~((structure_x.atom_x == 'H') & 
                                (structure_x.atom_y == 'H'))]
    g = nx.Graph()
    positions = structure[['x', 'y', 'z']].values
    g.add_nodes_from(structure['atom_index'], pos=positions)
    
    cols = ['atom_index_x', 'atom_index_y', 'distance', 'cutoff']
    X = [ tuple(i) for i in structure_x[cols].values ]
    
    for i_a, i_b, d, c in X:
        if c < cutoff:
            g.add_edge(i_a, i_b, length=d)
        elif nx.number_connected_components(g) == 1:
            break
        else:
            g.add_edge(i_a, i_b, length=d)
    return g

def get_structure(molecule_name):
    return structures[structures['molecule_name'] == molecule_name]

def get_structure_x(structure):
    positions = structure[['x', 'y', 'z']].values
    structure_X = pd.merge(structure, structure, how='outer', on=['molecule_name'])
    structure_X['distance'] = np.linalg.norm(structure_X[['x_x', 'y_x', 'z_x']].values - structure_X[['x_y', 'y_y', 'z_y']].values, axis=1)
    structure_X = structure_X[(structure_X.atom_index_x > structure_X.atom_index_y)]
    structure_X['cutoff'] = (structure_X.distance - (structure_X.possible_radii_x + structure_X.possible_radii_y)).apply(abs).apply(min)
    structure_X.sort_values('cutoff', inplace=True)
    return structure_X


# In[ ]:


def plot_graph(g):
    #labels = {i[0]: f'{i[0]} {i[1]}' for i in structure[['atom_index', 'atom']].values}
    nx.draw_spring(g, with_labels=True)
    show()


# In[ ]:


def get_relations(structure, g, atom_index_0, atom_index_1):

    shortest_path = nx.shortest_path(g, atom_index_0, atom_index_1)
    shortest_path_atoms = ''.join([structure[structure.atom_index == i].atom.values[0] for i in shortest_path[1:-1]])
    shortest_path_n_bonds = len(shortest_path)-1
    
    cos = None
    dihe = None
    
    if shortest_path_n_bonds == 2:
        x0 = structure[structure.atom_index == shortest_path[0]][['x', 'y', 'z']].values[0]
        x1 = structure[structure.atom_index == shortest_path[1]][['x', 'y', 'z']].values[0]
        x2 = structure[structure.atom_index == shortest_path[2]][['x', 'y', 'z']].values[0]
        cos = cosinus(x0, x1, x2)
        
    if shortest_path_n_bonds == 3:
        x0 = structure[structure.atom_index == shortest_path[0]][['x', 'y', 'z']].values[0]
        x1 = structure[structure.atom_index == shortest_path[1]][['x', 'y', 'z']].values[0]
        x2 = structure[structure.atom_index == shortest_path[2]][['x', 'y', 'z']].values[0]
        x3 = structure[structure.atom_index == shortest_path[3]][['x', 'y', 'z']].values[0]
        dihe = dihedral(x0, x1, x2, x3)
        
    results = {
        'molecule_name': structure.molecule_name.values[0],
        'atom_index_0': atom_index_0,
        'atom_index_1': atom_index_1,
        'shortest_path_atoms': shortest_path_atoms,
        'shortest_path_n_bonds': shortest_path_n_bonds,
        'cosinus': cos,
        'dihedral': dihe
               }
    
    return pd.DataFrame(results, index=[0])


def cosinus(x0, x1, x2):
    e0 = (x0-x1)
    e1 = (x2-x1)
    e0 = (e0 / np.linalg.norm(e0))
    e1 = (e1 / np.linalg.norm(e1))
    cosinus = np.dot(e0, e1)
    return np.round(cosinus, 5)


def dihedral(x0, x1, x2, x3):

    b0 = -1.0 * (x1 - x0)
    b1 = x2 - x1
    b2 = x3 - x2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, b1)*(1.0/np.linalg.norm(b1))
    x = np.dot(b0xb1, b1xb2)
    
    grad = np.arctan2(y, x)
    return grad


# In[ ]:


structure = get_structure('dsgdb9nsd_003830')
g = get_graph(structure)
display(get_relations(structure, g, 3, 6))
plot_graph(g)


# In[ ]:


def process(args):
    molecule_name = args['molecule_name']
    structure = get_structure(molecule_name)
    g = get_graph(structure)
    ndxs = zip(args['atom_index_0s'], args['atom_index_1s'])
    return pd.concat([get_relations(structure, g, i, j) for i,j in ndxs])


# In[ ]:


df = pd.concat([train, test])[['molecule_name', 'atom_index_0', 'atom_index_1']]
del train, test


# In[ ]:


import multiprocessing
multiprocessing.cpu_count()


# In[ ]:


pool = Pool(processes=4)
m = Manager()
q = m.Queue()

args = []
start = time.time()

for molecule_name, df in df.groupby('molecule_name'):
    args.append({'molecule_name': molecule_name,
                 'atom_index_0s': df.atom_index_0.values,
                 'atom_index_1s': df.atom_index_1.values})

results = pool.map_async(process, args)

pool.close()
pool.join()
end = time.time()

print(end - start)
print('Run time:', np.round((end - start) / 60 / 60, 2), 'h')

result = pd.concat(results.get())
result.to_csv('angles.csv', index=False)

print(len(result))
display(result)


# In[ ]:


train = pd.read_csv('../input/train.csv')
compare = pd.merge(train, result, on=['molecule_name', 'atom_index_0', 'atom_index_1'])

for t, df_tmp in compare.groupby('type'):
    figure(figsize=(12,4))
    subplot(1,3,1)
    plt.scatter(df_tmp.scalar_coupling_constant, df_tmp.cosinus)
    plt.title('cos')
    subplot(1,3,2)
    plt.scatter(df_tmp.scalar_coupling_constant, df_tmp.dihedral)
    plt.xlabel('scalar_coupling_constant')
    plt.title('dihedral')
    ax = subplot(1,3,3)
    df_tmp.shortest_path_n_bonds.value_counts().sort_index().plot.bar(ax=ax)
    plt.yscale('log')
    plt.title('shortest path n_bonds')
    plt.suptitle(t, x=0.1, size=18)
    plt.tight_layout()
    plt.show()


# In[ ]:


compare['n_type'] = compare.type.apply(lambda x: x[0])
failed = compare[compare.n_type.astype(int) != compare.shortest_path_n_bonds.astype(int)]
failed


# In[ ]:


list(failed.molecule_name.drop_duplicates())


# In[ ]:




