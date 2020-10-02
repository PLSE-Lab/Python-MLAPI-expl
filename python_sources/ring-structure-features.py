#!/usr/bin/env python
# coding: utf-8

# This kernel shows the importance of ring structure. This feature is proposed by  
# https://www.kaggle.com/sunhwan/using-rdkit-for-atomic-feature-and-visualization. I've just plotted the histograms. If this kernel is helpful, please not upvote on me, but upvote on him.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} -c rdkit rdkit')


# In[ ]:


get_ipython().run_cell_magic('bash', '-e', 'if ! [[ -f ./xyz2mol.py ]]; then\n  wget https://raw.githubusercontent.com/jensengroup/xyz2mol/master/xyz2mol.py\nfi')


# In[ ]:


# only reading 10% of data for debug
train = pd.read_csv('../input/train.csv')[::10]
test = pd.read_csv('../input/test.csv')[::10]


# In[ ]:


# rdkit & xyz2mol
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions #Only needed if modifying defaults
DrawingOptions.bondLineWidth=1.8
from rdkit.Chem.rdmolops import SanitizeFlags

# https://github.com/jensengroup/xyz2mol
from xyz2mol import xyz2mol, xyz2AC, AC2mol, read_xyz_file
from pathlib import Path
import pickle

CACHEDIR = Path('./')

def chiral_stereo_check(mol):
    # avoid sanitization error e.g., dsgdb9nsd_037900.xyz
    Chem.SanitizeMol(mol, SanitizeFlags.SANITIZE_ALL - SanitizeFlags.SANITIZE_PROPERTIES)
    Chem.DetectBondStereochemistry(mol,-1)
    # ignore stereochemistry for now
    #Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True, force=True)
    #Chem.AssignAtomChiralTagsFromStructure(mol,-1)
    return mol

def xyz2mol(atomicNumList,charge,xyz_coordinates,charged_fragments,quick):
    AC,mol = xyz2AC(atomicNumList,xyz_coordinates)
    new_mol = AC2mol(mol,AC,atomicNumList,charge,charged_fragments,quick)
    new_mol = chiral_stereo_check(new_mol)
    return new_mol

def MolFromXYZ(filename):
    charged_fragments = True
    quick = True
    cache_filename = CACHEDIR/f'{filename.stem}.pkl'
    if cache_filename.exists():
        return pickle.load(open(cache_filename, 'rb'))
    else:
        try:
            atomicNumList, charge, xyz_coordinates = read_xyz_file(filename)
            mol = xyz2mol(atomicNumList, charge, xyz_coordinates, charged_fragments, quick)
            # commenting this out for kernel to work.
            # for some reason kernel runs okay interactively, but fails when it is committed.
            #pickle.dump(mol, open(cache_filename, 'wb'))
        except:
            print(filename)
    return mol

#mol = MolFromXYZ(xyzfiles[1])
#m = Chem.MolFromSmiles(Chem.MolToSmiles(mol, allHsExplicit=True)); m

from multiprocessing import Pool
from tqdm import *
from glob import glob

def MolFromXYZ_(filename):
    return filename.stem, MolFromXYZ(filename)

mols = {}
n_cpu = 4
with Pool(n_cpu) as p:
    molecule_names = np.concatenate([train.molecule_name.unique(), test.molecule_name.unique()])
    xyzfiles = [Path(f'../input/structures/{f}.xyz') for f in molecule_names]
    n = len(xyzfiles)
    with tqdm(total=n) as pbar:
        for res in p.imap_unordered(MolFromXYZ_, xyzfiles):
            mols[res[0]] = res[1]
            pbar.update()


# In[ ]:


# highlight the bond given in train/test
# http://rdkit.blogspot.com/2015/02/new-drawing-code.html
m = mols['dsgdb9nsd_002129']
atom_index_0 = int(train[train.molecule_name == 'dsgdb9nsd_002129'].iloc[0].atom_index_0)
atom_index_1 = int(train[train.molecule_name == 'dsgdb9nsd_002129'].iloc[0].atom_index_1)
highlight = [atom_index_0, atom_index_1]

from IPython.display import SVG
Chem.rdDepictor.Compute2DCoords(m)
drawer = Draw.rdMolDraw2D.MolDraw2DSVG(400, 200)
drawer.DrawMolecule(m,highlightAtoms=highlight)
drawer.FinishDrawing()
svg = drawer.GetDrawingText().replace('svg:','')
#SVG(svg)


# In[ ]:


# 3JHH
row = train[train.type == '3JHH'].iloc[1]
m = mols[row.molecule_name]
atom_index_0 = int(row.atom_index_0)
atom_index_1 = int(row.atom_index_1)
highlight = [atom_index_0, atom_index_1]

from IPython.display import SVG
Chem.rdDepictor.Compute2DCoords(m)
drawer = Draw.rdMolDraw2D.MolDraw2DSVG(400, 200)
drawer.DrawMolecule(m,highlightAtoms=highlight)
drawer.FinishDrawing()
svg = drawer.GetDrawingText().replace('svg:','')
#SVG(svg)


# In[ ]:


from pathlib import Path
from sklearn import *
PATH = Path('../input')

# again, only using 10% of data for the committed kernel to work...
train = pd.read_csv(PATH/'train.csv')[::10]
test = pd.read_csv(PATH/'test.csv')[::10]


# In[ ]:


# https://www.kaggle.com/jazivxt/all-this-over-a-dog
train['atom1'] = train['type'].map(lambda x: str(x)[2])
train['atom2'] = train['type'].map(lambda x: str(x)[3])
test['atom1'] = test['type'].map(lambda x: str(x)[2])
test['atom2'] = test['type'].map(lambda x: str(x)[3])


# In[ ]:


lbl = preprocessing.LabelEncoder()
for i in range(4):
    train['type'+str(i)] = lbl.fit_transform(train['type'].map(lambda x: str(x)[i]))
    test['type'+str(i)] = lbl.transform(test['type'].map(lambda x: str(x)[i]))


# In[ ]:


structures = pd.read_csv(PATH/'structures.csv').rename(columns={'atom_index':'atom_index_0', 'x':'x0', 'y':'y0', 'z':'z0', 'atom':'atom1'})
train = pd.merge(train, structures, how='left', on=['molecule_name', 'atom_index_0', 'atom1'])
test = pd.merge(test, structures, how='left', on=['molecule_name', 'atom_index_0', 'atom1'])
del structures


# In[ ]:


structures = pd.read_csv(PATH/'structures.csv').rename(columns={'atom_index':'atom_index_1', 'x':'x1', 'y':'y1', 'z':'z1', 'atom':'atom2'})
train = pd.merge(train, structures, how='left', on=['molecule_name', 'atom_index_1', 'atom2'])
test = pd.merge(test, structures, how='left', on=['molecule_name', 'atom_index_1', 'atom2'])
del structures


# In[ ]:


def feature_atom(atom):
    prop = {}
    nb = [a.GetSymbol() for a in atom.GetNeighbors()] # neighbor atom type symbols
    nb_h = sum([_ == 'H' for _ in nb]) # number of hydrogen as neighbor
    nb_o = sum([_ == 'O' for _ in nb]) # number of oxygen as neighbor
    nb_c = sum([_ == 'C' for _ in nb]) # number of carbon as neighbor
    nb_n = sum([_ == 'N' for _ in nb]) # number of nitrogen as neighbor
    nb_na = len(nb) - nb_h - nb_o - nb_n - nb_c
    prop['degree'] = atom.GetDegree()
    prop['hybridization'] = int(atom.GetHybridization())
    prop['inring'] = int(atom.IsInRing()) # is the atom in a ring?
    prop['inring3'] = int(atom.IsInRingSize(3)) # is the atom in a ring size of 3?
    prop['inring4'] = int(atom.IsInRingSize(4)) # is the atom in a ring size of 4?
    prop['inring5'] = int(atom.IsInRingSize(5)) # ...
    prop['inring6'] = int(atom.IsInRingSize(6))
    prop['inring7'] = int(atom.IsInRingSize(7))
    prop['inring8'] = int(atom.IsInRingSize(8))
    prop['nb_h'] = nb_h
    prop['nb_o'] = nb_o
    prop['nb_c'] = nb_c
    prop['nb_n'] = nb_n
    prop['nb_na'] = nb_na
    return prop


# In[ ]:


# atom feature of dsgdb9nsd_002129 atom_index_0
molecule_name = 'dsgdb9nsd_002129'
row = train[train.molecule_name == molecule_name].iloc[0]
atom_index_0 = int(row.atom_index_0)
atom_index_1 = int(row.atom_index_1)
m=mols[molecule_name]
a0 = m.GetAtomWithIdx(atom_index_0)
#feature_atom(a0)


# In[ ]:


# extract some simple atomic feature for atom_index_0 and atom_index_1

# use cached rdkit mol object to save memory
if 'mols' in locals(): del mols
import gc
gc.collect()

# fix atom bonds
# dsgdb9nsd_059827: hydrogen has is far apart
nblist = {
    'dsgdb9nsd_059827': {
        13: 3
    }
}

def _features(args):
    idx, row = args
    molecule_name = row.molecule_name
    atom_index_0 = int(row.atom_index_0)
    atom_index_1 = int(row.atom_index_1)
    
    prop = {'molecule_name': molecule_name,
            'atom_index_0': atom_index_0,
            'atom_index_1': atom_index_1}

    # atom_0 is always hydrogen
    m = MolFromXYZ(PATH/f'structures/{molecule_name}.xyz') # less memory intensive in multiprocessing.Pool
    a0 = m.GetAtomWithIdx(atom_index_0)

    a1 = m.GetAtomWithIdx(atom_index_1)
    a1_prop = feature_atom(a1)
    prop.update({'a1_'+k: a1_prop[k] for k in a1_prop.keys()})

    # skipping below for time constraint
    # neighbor of atom_0
    try:
        a0_nb_idx = [a.GetIdx() for a in a0.GetNeighbors() if a.GetIdx() != a0].pop()
    except:
        if molecule_name in nblist and atom_index_0 in nblist[molecule_name]:
            a0_nb_idx = nblist[molecule_name][atom_index_0]
        else:
            print(molecule_name)
            print(row)

    a0_nb = m.GetAtomWithIdx(a0_nb_idx)
    a0_nb_prop = feature_atom(a0_nb)
    for k in a0_nb_prop.keys():
        prop['a0_nb_'+k] = a0_nb_prop[k]
        
    c = m.GetConformer()
    #prop['dist_a0_a0_nb'] = np.linalg.norm(c.GetAtomPosition(atom_index_0) - c.GetAtomPosition(a0_nb_idx))
    prop['x_a0_nb'] = c.GetAtomPosition(a0_nb_idx)[0]
    prop['y_a0_nb'] = c.GetAtomPosition(a0_nb_idx)[1]
    prop['z_a0_nb'] = c.GetAtomPosition(a0_nb_idx)[2]

    # neighbor of atom_1
    try:
        a1_nb_idx = [a.GetIdx() for a in a1.GetNeighbors() if a.GetIdx() != a1].pop()
    except:
        if molecule_name in nblist and atom_index_1 in nblist[molecule_name]:
            a1_nb_idx = nblist[molecule_name][atom_index_1]
        else:
            print(molecule_name)
            print(row)
    a1_nb = m.GetAtomWithIdx(a1_nb_idx)
    a1_nb_prop = feature_atom(a1_nb)
    for k in a1_nb_prop.keys():
        prop['a1_nb_'+k] = a1_nb_prop[k]
    prop['x_a1_nb'] = c.GetAtomPosition(a1_nb_idx)[0]
    prop['y_a1_nb'] = c.GetAtomPosition(a1_nb_idx)[1]
    prop['z_a1_nb'] = c.GetAtomPosition(a1_nb_idx)[2]
    #prop['dist_a1_a1_nb'] = np.linalg.norm(c.GetAtomPosition(a1.GetIdx()) - c.GetAtomPosition(a1_nb.GetIdx()))
    #prop['dist_a0_a1_nb'] = np.linalg.norm(c.GetAtomPosition(a0.GetIdx()) - c.GetAtomPosition(a1_nb.GetIdx()))
    #prop['dist_a1_a0_nb'] = np.linalg.norm(c.GetAtomPosition(a1.GetIdx()) - c.GetAtomPosition(a0_nb.GetIdx()))
    return prop

def features(df):
    prop = []
    n_cpu = 4
    with Pool(n_cpu) as p:
        n = len(df)
        res = _features((0, df.iloc[0]))
        keys = res.keys()
        _df = df[['molecule_name', 'atom_index_0', 'atom_index_1']]
        with tqdm(total=n) as pbar:
            for res in p.imap_unordered(_features, _df.iterrows()):
                # this is faster than using dict
                prop.append([res[_] for _ in keys])
                pbar.update()
        del _df
    
    prop = pd.DataFrame.from_records(prop, columns=keys)
    df = pd.merge(df, prop, how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1'])
    return df

train = features(train)


# In[ ]:


#https://www.kaggle.com/artgor/molecular-properties-eda-and-models
train_p0 = train[['x0', 'y0', 'z0']].values
train_p1 = train[['x1', 'y1', 'z1']].values
train['dist'] = np.linalg.norm(train_p0 - train_p1, axis=1)
train['dist_to_type_mean'] = train['dist'] / train.groupby('type')['dist'].transform('mean')
del train_p0, train_p1


# # Visualization
# Let's focus on the features of the ring size 1st atom belongs to. There are several kinds of cyclic compounds.
# 'none(0)-cyclic', 'tri(3)-cyclic', 'tetra(4)-cyclic', 'penta(5)-cyclic', 'hexa(6)-cyclic', 'hepta(7)-cyclic' and 'opta(8)-cyclic'. I'd like to see the frequency distribution of 'scalar_coupling_constant' of each ring structure features.

# In[ ]:


import matplotlib.pyplot as plt
mol_types=train["type"].unique()

def plot_cyclic_vs_scalar_coupling(mol_type):
    train_=train[train["type"]==mol_type].copy()
    colorlist = ["r", "g", "b", "c", "m", "y", "k", "w"]
    
    fig, ax = plt.subplots(figsize = (12, 6))
    plt.scatter(train_.loc[train_['a0_nb_inring']== 0 , 'scalar_coupling_constant'],train_.loc[train_['a0_nb_inring'] == 0, 'dist'],  label=str(0)+"cyclic",c="black",alpha=0.2);
    for i in [3,4,5,6,7,8]:
        plt.scatter(train_.loc[train_['a0_nb_inring'+str(i)]== 1 , 'scalar_coupling_constant'],train_.loc[train_['a0_nb_inring'+str(i)] == 1, 'dist'],  label=str(i)+"cyclic",c=colorlist[i-3],alpha=0.2);
    lgnd = ax.legend(loc="upper right", numpoints=1, fontsize=15)
    for i in range(7):
        lgnd.legendHandles[i]._sizes = [300]
    plt.ylabel("distance between atoms", fontsize=15)
    plt.xlabel("scalar_coupling_const", fontsize=15)
    plt.title(f'{mol_type} a0_neighbor ring_size vs scalar_coupling_const vs distance', fontsize=15)

    fig, ax = plt.subplots(figsize = (12, 6))
    plt.hist(train_.loc[train_['a0_nb_inring']== 0 , 'scalar_coupling_constant'],bins=30, label="0 cyclic",color="black",alpha=0.2)
    for i in [3,4,5,6,7,8]:
        plt.hist(train_.loc[train_['a0_nb_inring'+str(i)]== 1 , 'scalar_coupling_constant'],bins=30, label=str(i)+" cyclic",color=colorlist[i-3],alpha=0.5)
    lgnd = ax.legend(loc="upper right", numpoints=1, fontsize=15)
    for i in range(7):
        lgnd.legendHandles[i]._sizes = [300]
    plt.title(f'{mol_type} a0_neighbor ring_size vs scalar_coupling_const', fontsize=15)
    plt.xlabel("scalar_coupling_const", fontsize=15)

for mol_type in mol_types:
    plot_cyclic_vs_scalar_coupling(mol_type)


# For some molecule types, the distribution of 'scalar_coupling_constant' looks depending on the ring size the atom belongs to. So this would be effective features to predict scalar coupling constant.
