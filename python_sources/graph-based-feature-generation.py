#!/usr/bin/env python
# coding: utf-8

# # Graph Based feature generation - Walk along the atomic bonds, get data
# ## My first public kernal ever - upvote if you like it and surely leave a comment to improve it
# ### Features generated
# * Atoms along the path
# * Atomic bonds for all the atoms along the path
# * Count of C, N, H, F and Os bonded to the atoms along the path
# 
# ### Next steps
# * Very slow at calculations
# * 90k unique molecules will take at least 10 hours for calculation
# * Need to test on a sample of 10-20% data and see about features importances
# * Wonder if anymore parallel processing speed can be improved
# 
# ### Inspirations
# https://www.kaggle.com/mykolazotko/3d-visualization-of-molecules-with-plotly

# In[ ]:


import numpy as np
import pandas as pd

import datetime
import time

import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from sympy.geometry import Point3D

import random
import os

import networkx as nx

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

structures = reduce_mem_usage(pd.read_csv('../input/structures.csv'))
train_df = reduce_mem_usage(pd.read_csv('../input/train.csv'))
train_df.shape


# In[ ]:


molecule_list = [mol for mol in train_df['molecule_name'].drop_duplicates()]
new_list = random.sample(molecule_list, 9000)
len(new_list)


# In[ ]:


structures = structures[structures['molecule_name'].isin(new_list)]
train_df = train_df[train_df['molecule_name'].isin(new_list)]
train_df.shape


# In[ ]:


### Add Fudge factor
atomic_radii = dict(C=0.77, F=0.71, H=0.38, N=0.75, O=0.73)
fudge_factor = 0.05
atomic_radii = {k:v + fudge_factor for k,v in atomic_radii.items()}

def Build_Graph(molecule_name):
    ## Get molecule structure
    molecule = structures[structures.molecule_name == molecule_name]
    
    ## List of atom names
    elements = molecule.atom.tolist()
    
    ## Get all xyz cordinates
    coordinates = molecule[['x', 'y', 'z']].values
    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]
    z_coordinates = coordinates[:, 2]
    
    ## Add radius to each atom to the list of atoms
    radii = [atomic_radii[element] for element in elements]

    ids = np.arange(coordinates.shape[0])
    bonds = pd.DataFrame(columns = ["Atom_0", "Atom_1", "Dist"])
    coordinates_compare, radii_compare, ids_compare = coordinates, radii, ids

    ## For each atom in the atom list, 
        ## calcualte distances to all the other atoms,    
    for _ in range(len(ids)):
        coordinates_compare = np.roll(coordinates_compare, -1, axis=0)
        radii_compare = np.roll(radii_compare, -1, axis=0)
        ids_compare = np.roll(ids_compare, -1, axis=0)
        distances = np.linalg.norm(coordinates - coordinates_compare, axis=1)
        bond_distances = (radii + radii_compare) * 1.3
        mask = np.logical_and(distances > 0.1, distances <  bond_distances)
        distances = distances.round(2)
        
        ## Build tupple for bonds
        for i, j, dist in zip(ids[mask], ids_compare[mask], distances[mask]):
            if i < j:
                bonds = bonds.append(pd.DataFrame([[i, j , dist]], columns = ["Atom_0", "Atom_1", "Dist"]))
            else:
                bonds = bonds.append(pd.DataFrame([[j, i , dist]], columns = ["Atom_0", "Atom_1", "Dist"]))
        bonds = bonds.drop_duplicates()
        subset = bonds[['Atom_0', 'Atom_1']]
        tuples = [tuple(x) for x in subset.values]
    
    ## Use the tupple list of atoms and bonds to build graph
    G=nx.Graph()
    G.add_edges_from(tuples)

    ## Add Node properties - Will be usefull in getting coordinates and atom name for each node
    Node_Properties = (molecule
                           .reset_index(drop=True)
                           .groupby('atom_index')[['atom', 'x', 'y', 'z']]
                           .apply(lambda x: x.to_dict()).to_dict()
                      )
    nx.set_node_attributes(G, Node_Properties)
    return G


# In[ ]:


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

def Crawl_Path(Full_Graph, start, end):
    temp = list()
    for node in nx.shortest_path(Full_Graph, start, end):
        atom = Full_Graph.nodes[node].get("atom").get(node)
        x_Cord = Full_Graph.nodes[node].get("x").get(node)
        y_Cord = Full_Graph.nodes[node].get("y").get(node)
        z_Cord = Full_Graph.nodes[node].get("z").get(node)

        ### Get list of all node connected to the node along the path
        Neighbors = [Full_Graph.nodes[n].get("atom").get(n) for n in Full_Graph.neighbors(node)]

        ### Get counts of C, H, N, O, F connected to the node along the path
        C_Conns = Neighbors.count('C')
        H_Conns = Neighbors.count('H')
        N_Conns = Neighbors.count('N')
        O_Conns = Neighbors.count('O')
        F_Conns = Neighbors.count('F')
        temp.append({'atom': atom, 'x_Coord': x_Cord, 'y_Coord': y_Cord, 'z_Coord': z_Cord, 'F_Conns': F_Conns,
                        'C_Conns': C_Conns, 'H_Conns': H_Conns, 'N_Conns': N_Conns, 'O_Conns': O_Conns})
    return temp
    
def Get_Path_Details(temp_DF):
    molecule_name = str(temp_DF['molecule_name'].iloc[0])
    Full_Graph = Build_Graph(molecule_name)
    temp_DF['Path'] = temp_DF.apply(lambda row: Crawl_Path(Full_Graph, row['atom_index_0'], row['atom_index_1']), axis=1)
    return temp_DF


# In[ ]:


from joblib import Parallel, delayed
import multiprocessing

print(datetime.datetime.now())
def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=4)(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

out = pd.DataFrame()
out = out.append(applyParallel(train_df.groupby('molecule_name'), Get_Path_Details))

print(datetime.datetime.now())


# In[ ]:


def Map_PathInfo(t, Path):
    t = len(Path)
    if t == 2:
        return (Path[0]['atom'], Path[0]['x_Coord'], Path[0]['y_Coord'], Path[0]['z_Coord']
                    , Path[0]['F_Conns'], Path[0]['C_Conns'], Path[0]['H_Conns'], Path[0]['N_Conns'], Path[0]['O_Conns']
                , Path[1]['atom'], Path[1]['x_Coord'], Path[1]['y_Coord'], Path[1]['z_Coord']
                    , Path[1]['F_Conns'], Path[1]['C_Conns'], Path[1]['H_Conns'], Path[1]['N_Conns'], Path[1]['O_Conns']
                , '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    elif t == 3:
        return (Path[0]['atom'], Path[0]['x_Coord'], Path[0]['y_Coord'], Path[0]['z_Coord']
                    , Path[0]['F_Conns'], Path[0]['C_Conns'], Path[0]['H_Conns'], Path[0]['N_Conns'], Path[0]['O_Conns']
                
                , Path[1]['atom'], Path[1]['x_Coord'], Path[1]['y_Coord'], Path[1]['z_Coord']
                    , Path[1]['F_Conns'], Path[1]['C_Conns'], Path[1]['H_Conns'], Path[1]['N_Conns'], Path[1]['O_Conns']
                
                , Path[2]['atom'], Path[2]['x_Coord'], Path[2]['y_Coord'], Path[2]['z_Coord']
                    , Path[2]['F_Conns'], Path[2]['C_Conns'], Path[2]['H_Conns'], Path[2]['N_Conns'], Path[2]['O_Conns']
                
                , '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    elif t == 4:
        return (Path[0]['atom'], Path[0]['x_Coord'], Path[0]['y_Coord'], Path[0]['z_Coord']
                    , Path[0]['F_Conns'], Path[0]['C_Conns'], Path[0]['H_Conns'], Path[0]['N_Conns'], Path[0]['O_Conns']
                    , Path[1]['atom'], Path[1]['x_Coord'], Path[1]['y_Coord'], Path[1]['z_Coord']
                    , Path[1]['F_Conns'], Path[1]['C_Conns'], Path[1]['H_Conns'], Path[1]['N_Conns'], Path[1]['O_Conns']
                    , Path[2]['atom'], Path[2]['x_Coord'], Path[2]['y_Coord'], Path[2]['z_Coord']
                    , Path[2]['F_Conns'], Path[2]['C_Conns'], Path[2]['H_Conns'], Path[2]['N_Conns'], Path[2]['O_Conns']
                    , Path[3]['atom'], Path[3]['x_Coord'], Path[3]['y_Coord'], Path[3]['z_Coord']
                    , Path[3]['F_Conns'], Path[3]['C_Conns'], Path[3]['H_Conns'], Path[3]['N_Conns'], Path[3]['O_Conns'])
    
    else:
        return ('', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                , '', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

print(datetime.datetime.now())

(out['Atom_0'], out['x_Cord_0'], out['y_Cord_0'], out['Z_Cord_0']
        , out['F_Conns_0'], out['C_Conns_0'], out['H_Conns_0']
        , out['N_Conns_0'], out['O_Conns_0']
    , out['Atom_1'], out['x_Cord_1'], out['y_Cord_1'], out['Z_Cord_1']
        , out['F_Conns_1'], out['C_Conns_1'], out['H_Conns_1']
        , out['N_Conns_1'], out['O_Conns_1']
    , out['Atom_2'], out['x_Cord_2'], out['y_Cord_2'], out['Z_Cord_2']
        , out['F_Conns_2'], out['C_Conns_2'], out['H_Conns_2']
        , out['N_Conns_2'], out['O_Conns_2']
    , out['Atom_3'], out['x_Cord_3'], out['y_Cord_3'], out['Z_Cord_3']
        , out['F_Conns_3'], out['C_Conns_3'], out['H_Conns_3']
        , out['N_Conns_3'], out['O_Conns_3']) = (

                zip(*out.apply(lambda row: Map_PathInfo(row['type'], row['Path']), axis = 1))
                                                                                        )

print(datetime.datetime.now())

out.to_pickle(r'../Graph based features')

