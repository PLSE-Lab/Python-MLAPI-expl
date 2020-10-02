#!/usr/bin/env python
# coding: utf-8

# # 3D Visualization of Molecules with Plotly

# In[ ]:


import numpy as np
import pandas as pd
import random

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot

structures = pd.read_csv('../input/structures.csv')
molecule_names = structures.molecule_name.unique()

# initiate the plotly notebook mode
init_notebook_mode(connected=True)


def plot_molecule(molecule_name, structures_df):
    """Creates a 3D plot of the molecule"""
    
    atomic_radii = dict(C=0.77, F=0.71, H=0.38, N=0.75, O=0.73)  
    cpk_colors = dict(C='black', F='green', H='white', N='blue', O='red')

    molecule = structures_df[structures_df.molecule_name == molecule_name]
    coordinates = molecule[['x', 'y', 'z']].values
    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]
    z_coordinates = coordinates[:, 2]
    elements = molecule.atom.tolist()
    radii = [atomic_radii[element] for element in elements]
    
    def get_bonds():
        """Generates a set of bonds from atomic cartesian coordinates"""
        ids = np.arange(coordinates.shape[0])
        bonds = dict()
        coordinates_compare, radii_compare, ids_compare = coordinates, radii, ids
        
        for _ in range(len(ids)):
            coordinates_compare = np.roll(coordinates_compare, -1, axis=0)
            radii_compare = np.roll(radii_compare, -1, axis=0)
            ids_compare = np.roll(ids_compare, -1, axis=0)
            distances = np.linalg.norm(coordinates - coordinates_compare, axis=1)
            bond_distances = (radii + radii_compare) * 1.3
            mask = np.logical_and(distances > 0.1, distances <  bond_distances)
            distances = distances.round(2)
            new_bonds = {frozenset([i, j]): dist for i, j, dist in zip(ids[mask], ids_compare[mask], distances[mask])}
            bonds.update(new_bonds)
        return bonds            
            
    def atom_trace():
        """Creates an atom trace for the plot"""
        colors = [cpk_colors[element] for element in elements]
        markers = dict(color=colors, line=dict(color='lightgray', width=2), size=5, symbol='circle', opacity=0.8)
        trace = go.Scatter3d(x=x_coordinates, y=y_coordinates, z=z_coordinates, mode='markers', marker=markers,
                             text=elements, name='')
        return trace

    def bond_trace():
        """"Creates a bond trace for the plot"""
        trace = go.Scatter3d(x=[], y=[], z=[], hoverinfo='none', mode='lines',
                             marker=dict(color='grey', size=7, opacity=1))
        for i, j in bonds.keys():
            trace['x'] += (x_coordinates[i], x_coordinates[j], None)
            trace['y'] += (y_coordinates[i], y_coordinates[j], None)
            trace['z'] += (z_coordinates[i], z_coordinates[j], None)
        return trace
    
    bonds = get_bonds()
    
    zipped = zip(range(len(elements)), x_coordinates, y_coordinates, z_coordinates)
    annotations_id = [dict(text=num, x=x, y=y, z=z, showarrow=False, yshift=15, font = dict(color = "blue"))
                   for num, x, y, z in zipped]
    
    annotations_length = []
    for (i, j), dist in bonds.items():
        x_middle, y_middle, z_middle = (coordinates[i] + coordinates[j])/2
        annotation = dict(text=dist, x=x_middle, y=y_middle, z=z_middle, showarrow=False, yshift=15)
        annotations_length.append(annotation)   
    
    updatemenus = list([
        dict(buttons=list([
                 dict(label = 'Atom indices',
                      method = 'relayout',
                      args = [{'scene.annotations': annotations_id}]),
                 dict(label = 'Bond lengths',
                      method = 'relayout',
                      args = [{'scene.annotations': annotations_length}]),
                 dict(label = 'Atom indices & Bond lengths',
                      method = 'relayout',
                      args = [{'scene.annotations': annotations_id + annotations_length}]),
                 dict(label = 'Hide all',
                      method = 'relayout',
                      args = [{'scene.annotations': []}])
                 ]),
                 direction='down',
                 xanchor = 'left',
                 yanchor = 'top'
            ),        
    ])
    
    data = [atom_trace(), bond_trace()]
    axis_params = dict(showgrid=False, showbackground=False, showticklabels=False, zeroline=False, titlefont=dict(color='white'))
    layout = dict(scene=dict(xaxis=axis_params, yaxis=axis_params, zaxis=axis_params, annotations=annotations_id), 
                  margin=dict(r=0, l=0, b=0, t=0), showlegend=False, updatemenus=updatemenus)

    fig = go.Figure(data=data, layout=layout)
    iplot(fig)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)


# In[ ]:


molecule_name = random.choice(molecule_names)
plot_molecule(molecule_name, structures)

