#!/usr/bin/env python
# coding: utf-8

# # Visualize molecule structure
# 
# The plots are interactive, you can rotate, zoom in/out, etc.

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.offline as py
import plotly.graph_objs as go

from plotly import tools

color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
py.init_notebook_mode(connected=True)

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999


# In[2]:


train_df = pd.read_csv('../input/train.csv')
structure_df = pd.read_csv('../input/structures.csv')


# In[11]:


def show_molecule(mdata, mstruct):
    mdata = mdata.merge(right=mstruct, how='left',
                        left_on=['molecule_name', 'atom_index_0'],
                        right_on=['molecule_name', 'atom_index'])
    mdata.rename(index=str, columns={"x": "x0", "y": "y0", "z": "z0", "atom": "atom0"}, inplace=True)
    mdata.drop(['atom_index'], axis=1, inplace=True)

    mdata = mdata.merge(right=mstruct, how='left',
                  left_on=['molecule_name', 'atom_index_1'],
                  right_on=['molecule_name', 'atom_index']
                 )
    mdata.rename(index=str, columns={"x": "x1", "y": "y1", "z": "z1", "atom": "atom1"}, inplace=True)
    mdata.drop(['atom_index'], axis=1, inplace=True)    
    
    data = []
  
    atoms = mstruct['atom'].unique()
    types = mdata['type'].unique()
    
    atom_cfg = {
        'H': {"name": "Hydrogen", "color": "#757575", "size": 4},
        'C': {"name": "Carbon", "color": "#f44336", "size": 12},
        'O': {"name": "Oxygen", "color": "#03a9f4", "size": 12},
        'N': {"name": "Nitrogen", "color": "#ff9800", "size": 12},
        'F': {"name": "Fluorine", "color": "#673ab7", "size": 12},
    }
    
    type_cfg = {
        '2JHH': {"color": "#757575", "width": 2},
        '3JHH': {"color": "#757575", "width": 3},

        '1JHC': {"color": "#f44336", "width": 1},
        '2JHC': {"color": "#f44336", "width": 2},
        '3JHC': {"color": "#f44336", "width": 3},

        '1JHN': {"color": "#ff9800", "width": 2},
        '2JHN': {"color": "#ff9800", "width": 2},
        '3JHN': {"color": "#ff9800", "width": 3},
    }

    for atom, config in atom_cfg.items(): 
        if atom in atoms:
            data.append(
                go.Scatter3d(
                    x=mstruct[mstruct['atom'] == atom]['x'].values,
                    y=mstruct[mstruct['atom'] == atom]['y'].values,
                    z=mstruct[mstruct['atom'] == atom]['z'].values,
                    mode='markers',
                    marker=dict(
                        color=config['color'],
                        size=config['size'],
                        opacity=0.8
                    ),
                    name=config['name']
                )
            )

    for ctype, config in type_cfg.items():
        if ctype in types:
            eX = []; eY = []; eZ = []
            for row in mdata[mdata['type'] == ctype].iterrows():
                rd = row[1]
                eX += [rd['x0'], rd['x1']]
                eY += [rd['y0'], rd['y1']]
                eZ += [rd['z0'], rd['z1']]            
            
            data.append(
                go.Scatter3d(
                    x=eX,
                    y=eY,
                    z=eZ,
                    mode='lines',
                    line=dict(color=config['color'], width=config['width']),
                    name=ctype
                )
            )            

    axis=dict(showbackground=False, showline=False, zeroline=False, showgrid=False, showticklabels=False, title='')
    layout = go.Layout(
        margin=dict(l=50, r=50, b=50, t=50),
        width=720,
        height=640,
        showlegend=True,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='molecule')


# In[12]:


molecule = 'dsgdb9nsd_000001'
show_molecule(train_df[train_df['molecule_name'] == molecule], structure_df[structure_df['molecule_name'] == molecule])


# In[78]:


molecule = 'dsgdb9nsd_128739'
show_molecule(train_df[train_df['molecule_name'] == molecule], structure_df[structure_df['molecule_name'] == molecule])


# In[13]:


molecule = 'dsgdb9nsd_000037'


# In[14]:


mstructure = structure_df[structure_df['molecule_name'] == molecule]
mstructure.head(20)


# In[15]:


mdata = train_df[train_df['molecule_name'] == molecule]
mdata.head(20)


# In[16]:


show_molecule(mdata, mstructure)


# In[ ]:




