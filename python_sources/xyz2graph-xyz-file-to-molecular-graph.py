#!/usr/bin/env python
# coding: utf-8

# # xyz2graph
# [**xyz2graph**](https://github.com/zotko/xyz2graph) is a Python package for reading of .xyz files and constructing of molecular graphs from atomic coordinates. The molecular graph can be converted into [NetworkX](https://networkx.github.io) graph or [Plotly](https://plot.ly) figure for 3D visualization in a browser window or in a [Jupyter notebook](https://jupyter.org).

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import random

PATH = '../input/champs-scalar-coupling/'


# In[ ]:


get_ipython().system('pip install -i https://test.pypi.org/simple/ xyz2graph')


# In[ ]:


from xyz2graph import MolGraph, to_networkx_graph, to_plotly_figure
from plotly.offline import init_notebook_mode, iplot
import networkx as nx

# initiate the Plotly notebook mode
init_notebook_mode(connected=True)

structures = pd.read_csv(PATH + 'structures.csv')
mol_names = structures.molecule_name.unique()


# In[ ]:


def draw_random_mol():
    """Draws a random molecule"""
    mol_name = random.choice(mol_names)
    # Create the MolGraph object
    mg = MolGraph()
    # Read the data from the .xyz file
    mg.read_xyz(f'{PATH}structures/{mol_name}.xyz')
    # Create the Plotly figure
    fig = to_plotly_figure(mg)
    iplot(fig)

def draw_random_networx_graph():
    """Draws the NetworkX graph of a random molecule"""
    mol_name = random.choice(mol_names)
    # Create the MolGraph object
    mg = MolGraph()
    # Read the data from the .xyz file
    mg.read_xyz(f'{PATH}structures/{mol_name}.xyz')
    # Create the NetworkX figure
    G = to_networkx_graph(mg)
    nodes = list(G.nodes(data=True))
    print('Nodes:')
    print(nodes[:5])
    edges = list(G.edges(data=True))
    print('Edges:')
    print(print(edges[:5]))
    nx.draw(G, with_labels=True)


# In[ ]:


draw_random_mol()


# In[ ]:


draw_random_mol()


# In[ ]:


draw_random_mol()


# In[ ]:


draw_random_mol()


# In[ ]:


draw_random_mol()


# In[ ]:


draw_random_mol()


# In[ ]:


draw_random_mol()


# In[ ]:


draw_random_mol()


# In[ ]:


draw_random_networx_graph()


# In[ ]:


draw_random_networx_graph()

