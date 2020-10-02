#!/usr/bin/env python
# coding: utf-8

# This kernel shows how to use RDKit and PyTorch Geometric (PyG) to prepare dataset for graph model. 
# 
# I am new for Kaggle and graph model, let me know if you see any typo or have any suggestion. 
# 
# Some useful links:
# 
# 1 PyG official tutorial: https://pytorch-geometric.readthedocs.io/en/latest/index.html
# 
# 2 PyG tutorial by Huang: https://towardsdatascience.com/hands-on-graph-neural-networks-with-pytorch-pytorch-geometric-359487e221a8
# 
# 3 GCN tutorial by Tobias: https://towardsdatascience.com/how-to-do-deep-learning-on-graphs-with-graph-convolutional-networks-7d2250723780
# 
# 4 MPNN kernel by fnand: https://www.kaggle.com/fnands/1-mpnn
# 
# In this kernel, I didn't include the target value (scalar coupling), I will update it in the future work. 

# In[ ]:


import numpy as np 
import pandas as pd

import os
import sys

import time
from tqdm import tqdm


# # Install RDKit and XYZ2MOL

# Taken from Jo's kernel with a little bit change. 
# 
# https://www.kaggle.com/sunhwan/using-rdkit-for-atomic-feature-and-visualization

# In[ ]:


get_ipython().system('conda install --yes --prefix {sys.prefix} -c rdkit rdkit')


# In[ ]:


get_ipython().run_cell_magic('bash', '-e', 'if ! [[ -f ./xyz2mol.py ]]; then\n  wget https://raw.githubusercontent.com/jensengroup/xyz2mol/master/xyz2mol.py\nfi')


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
        except:
            print(filename)
    return mol


# # Creating Dataset with PyG

# This part is inspired by:
# 
# 1 AlchemyDataset (https://github.com/tencent-alchemy/Alchemy/blob/master/pyg/Alchemy_dataset.py)
# 
# 2 Heng's discussion and starter kit (https://www.kaggle.com/c/champs-scalar-coupling/discussion/93972)
# 

# In[ ]:


get_ipython().system(' pip install --verbose --no-cache-dir torch-scatter')
get_ipython().system(' pip install --verbose --no-cache-dir torch-sparse')
get_ipython().system(' pip install --verbose --no-cache-dir torch-cluster')
get_ipython().system(' pip install --verbose --no-cache-dir torch-spline-conv')
get_ipython().system(' pip install torch-geometric')


# In[ ]:


import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import networkx as nx
import pathlib
import pandas as pd


DATA_DIR = '../input'

class ChampsDataset(InMemoryDataset):
    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def __init__(self, root, mode='train', transform=None, pre_transform=None, pre_filter=None):
        self.mode = mode
        self.df = pd.read_csv(DATA_DIR + '/%s.csv'%self.mode)
        self.id = self.df.molecule_name.unique()
        
        super(ChampsDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return 'ChampsDataset_%s.pt'%self.mode

    def download(self):
        pass

    def node_features(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            # Atom type (One-hot H, C, N, O F)
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']]
            # Atomic number
            h_t.append(d['a_num'])
            # Acceptor
            h_t.append(d['acceptor'])
            # Donor
            h_t.append(d['donor'])
            # Aromatic
            h_t.append(int(d['aromatic']))
            # Hybradization
            h_t += [int(d['hybridization'] == x)                     for x in (Chem.rdchem.HybridizationType.SP,                         Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr
    

    def edge_features(self, g):
        e={}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                    for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC)]
            e[(n1, n2)] = e_t

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

    # xyz file reader for dataset
    def xyz_graph_reader(self, xyzfiles):
        
        mol = MolFromXYZ(xyzfiles)
        feats = self.chem_feature_factory.GetFeaturesForMol(mol)

        g = nx.DiGraph()

        # Create nodes
        assert len(mol.GetConformers()) == 1
        geom = mol.GetConformers()[0].GetPositions()
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
                    aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                    num_h=atom_i.GetTotalNumHs())

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['acceptor'] = 1
        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j, b_type=e_ij.GetBondType())

        node_attr = self.node_features(g)
        edge_index, edge_attr = self.edge_features(g)
        data = Data(
                x=node_attr,
                pos=torch.FloatTensor(geom),
                edge_index=edge_index,
                edge_attr=edge_attr,
                )
        return data

    def process(self):
        data_list = []
        molecule_names = self.df.molecule_name.unique() 
        xyzfiles = [Path(f'../input/structures/{f}.xyz') for f in molecule_names]
        
        for i in tqdm(range(len(molecule_names))):
            champs_data = self.xyz_graph_reader(xyzfiles[i])
            if champs_data is not None:
                data_list.append(champs_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# In[ ]:


train_data = ChampsDataset(root ='../', mode = 'train')


# In[ ]:


test_data = ChampsDataset(root ='../', mode = 'test')


# Take a look of a random graph data.

# In[ ]:


train_data[50000]


# # Save Dataset

# In[ ]:


torch.save(train_data, 'train.pt')
torch.save(test_data, 'test.pt')

