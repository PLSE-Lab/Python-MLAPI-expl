#!/usr/bin/env python
# coding: utf-8

# Chemical bonds could be useful in this competition. MatthewMasters hinted in [this thread](https://www.kaggle.com/c/champs-scalar-coupling/discussion/93928#latest-553167) that we could use OpenBabel to extract the bond information automatically. If, like me, you are unfamiliar with OpenBabel, this kernel will show you how to use that library to get the bonds.
# 
# We need to install OpenBabel first. You can install the package directly under "Settings" on the right side of the editor, or use the following command:

# In[ ]:


get_ipython().system('conda install -c openbabel openbabel -y')


# We want to read all the files in ../input/structures and let OpenBabel figure out the chemical bonds for each molecule. We will save the results into a new dataframe similar to the ones we already have.

# In[ ]:


import openbabel
import os
import pandas as pd


# In[ ]:


path = '../input/structures/'

# Initialize the OpenBabel object that we will use later.
obConversion = openbabel.OBConversion()
obConversion.SetInFormat("xyz")

# Define containers
mol_index = [] # This will be our dataframe index
bond_atom_0 = []
bond_atom_1 = []
bond_order = []
bond_length = []


# In[ ]:


for f in os.scandir(path):
    # Initialize an OBMol object
    mol = openbabel.OBMol()
    read_ok = obConversion.ReadFile(mol, f.path)
    if not read_ok:
        # There was an error reading the file
        raise Exception(f'Could not read file {f.path}')
    
    mol_name = f.name[:-4] 
    mol_index.extend([mol_name] * mol.NumBonds()) # We need one entry per bond
    
    # Extract bond information
    mol_bonds = openbabel.OBMolBondIter(mol) # iterate over all the bonds in the molecule
    for bond in mol_bonds:
        bond_atom_0.append(bond.GetBeginAtomIdx() - 1) # Must be 0-indexed
        bond_atom_1.append(bond.GetEndAtomIdx() - 1)
        bond_length.append(bond.GetLength())
        bond_order.append(bond.GetBondOrder())


# In[ ]:


# Put everything into a dataframe
df = pd.DataFrame({'molecule_name': mol_index,
                   'atom_0': bond_atom_0,
                   'atom_1': bond_atom_1,
                   'order': bond_order,
                   'length': bond_length})
    
df = df.sort_values(['molecule_name', 'atom_0', 'atom_1']).reset_index(drop=True)    

print(df.head(10))


# In[ ]:


# My favorite way of storing variables
import shelve
with shelve.open('vars.shelf') as shelf:
    shelf['bonds'] = df


# In[ ]:


# To load the variable later, use:
with shelve.open('vars.shelf') as shelf:
    bonds = shelf['bonds']


# We should use this information with care. OpenBabel might fail to get the bonds right for some of the more exotic molecules. See [here](https://www.kaggle.com/c/champs-scalar-coupling/discussion/94755#546968) and [here](https://www.kaggle.com/c/champs-scalar-coupling/discussion/94117#latest-542356).
# 
# Also note that other users have proposed different ways of extracting bond information:
# * https://www.kaggle.com/aekoch95/bonds-from-structure-data
# * https://www.kaggle.com/asauve/dataset-with-number-of-bonds-between-atoms
