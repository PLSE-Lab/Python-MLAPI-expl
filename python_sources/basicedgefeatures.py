#!/usr/bin/env python
# coding: utf-8

# This is how you could compute basic edge features as angles between all atom pairs, dihedral angles between all triplets and shortest path (i.e. number of bonds) between all atoms. RDKit is used for everything.

# In[ ]:


import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} -c rdkit rdkit')


# In[ ]:


get_ipython().run_cell_magic('bash', '-e', 'if ! [[ -f ./xyz2mol.py ]]; then\n  wget https://raw.githubusercontent.com/jensengroup/xyz2mol/master/xyz2mol.py\nfi')


# In[ ]:


# Imports

# Standard library
import pickle

# Basic imports
import numpy as np
import pandas as pd

# rdkit & xyz2mol
import rdkit
from rdkit import Chem
from rdkit.Chem.rdmolops import SanitizeFlags
from rdkit.Chem import rdMolTransforms

# https://github.com/jensengroup/xyz2mol
from xyz2mol import xyz2mol, xyz2AC, AC2mol, read_xyz_file
from pathlib import Path
import pickle

from tqdm.auto import tqdm


# In[ ]:


CACHEDIR = Path('./')

def chiral_stereo_check(mol):
    Chem.SanitizeMol(mol, SanitizeFlags.SANITIZE_ALL - SanitizeFlags.SANITIZE_PROPERTIES)
    Chem.DetectBondStereochemistry(mol,-1)
    return mol

def xyz2mol(atomicNumList,charge,xyz_coordinates,charged_fragments,quick):
    AC,mol = xyz2AC(atomicNumList,xyz_coordinates)
    new_mol = AC2mol(mol,AC,atomicNumList,charge,charged_fragments,quick)
    new_mol = chiral_stereo_check(new_mol)
    return new_mol

def MolFromXYZ(filename):
    charged_fragments = True
    quick = True
    try:
        atomicNumList, charge, xyz_coordinates = read_xyz_file(filename)
        mol = xyz2mol(atomicNumList, charge, xyz_coordinates, charged_fragments, quick)
    except:
        print(filename)
    return atomicNumList, mol

def MolFromXYZ_(filename):
    return filename.stem, MolFromXYZ(filename)


# In[ ]:


def symmetrize(matrix):
    return matrix + matrix.T - np.diag(matrix.diagonal())


# In[ ]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
structures = pd.read_csv("../input/structures.csv")
dftr = structures[["molecule_name"]]
dftr = dftr.drop_duplicates()


# In[ ]:


list_of_shortest_paths = list()
list_of_angles = list()
list_of_dihedrals = list()
for i, row in tqdm(dftr.iterrows()):
    name = row["molecule_name"]
    path = Path(f'../input/structures/{name}.xyz')
    ats, molecule = MolFromXYZ(path)
    conformer = molecule.GetConformer(0)
    N_atoms = molecule.GetNumAtoms()
    matrix_of_shortest_paths = np.zeros((N_atoms, N_atoms), dtype="int32")
    matrix_of_angles = np.zeros((N_atoms, N_atoms))
    matrix_of_dihedral_angles = np.zeros((N_atoms, N_atoms))
    for i in range(N_atoms):
        for j in range(i+1, N_atoms):
            shortest_path_indices = Chem.GetShortestPath(molecule, i, j)
            length_of_shortest_path = len(shortest_path_indices) - 1 
            matrix_of_shortest_paths[i, j] = length_of_shortest_path
            if length_of_shortest_path == 2:
                angle = np.abs(rdMolTransforms.GetAngleDeg(conformer, *shortest_path_indices))
                matrix_of_angles[i, j] = rdMolTransforms.GetAngleDeg(conformer, *shortest_path_indices)
            elif length_of_shortest_path == 3:
                matrix_of_dihedral_angles[i, j] = rdMolTransforms.GetDihedralDeg(conformer, *shortest_path_indices)
                if np.isnan(matrix_of_dihedral_angles[i, j]):
                    matrix_of_dihedral_angles[i, j] = 0.0
    list_of_shortest_paths.append(symmetrize(matrix_of_shortest_paths))
    list_of_angles.append(symmetrize(matrix_of_angles))
    list_of_dihedrals.append(symmetrize(matrix_of_dihedral_angles))


# In[ ]:


with open("BasicShortestPathMatrices.pkl", "wb") as f:
    pickle.dump(list_of_shortest_paths, f)
with open("BasicAnglesMatrices.pkl", "wb") as f:
    pickle.dump(list_of_angles, f)
with open("BasicDihedralsMatrices.pkl", "wb") as f:
    pickle.dump(list_of_dihedrals, f)

