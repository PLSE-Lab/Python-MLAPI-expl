#!/usr/bin/env python
# coding: utf-8

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


# # Intro
# This kernel is a reaction to the ["Predicting Mulliken Charges With ACSF Descriptors"](https://www.kaggle.com/borisdee/predicting-mulliken-charges-with-acsf-descriptors)-kernel from Boris. He introduced the very neat concept of ACSF descriptors and showed how it can be used to predict the Mulliken Charge. IMO unfortunately, he didn't want to share his calculated ACSF values. I'm creating this kernel to fill the gap.
# 
# After some digging, I found the very nice "DScribe" package, including out-of-the-box ACSF functions. Let's go.

# In[ ]:


# import some stuff

import ase as ase
import pandas as pd
import numpy as np
import time, copy
import dscribe as ds
from dscribe import descriptors


# We will 'manually' calculate acsf for our beloved 'dsgdb9nsd_000001' molecule.

# In[ ]:


structure = pd.read_csv('../input/champs-scalar-coupling/structures.csv')


# In[ ]:


# using rcut, g2 and g4 params from Boris

#For all ACSF functions: rcut = 10.0
rcut = 10.0

#G2 - eta/Rs couples:
g2_params = [[1, 2], [0.1, 2], [0.01, 2],
           [1, 6], [0.1, 6], [0.01, 6]]

#G4 - eta/ksi/lambda triplets:
g4_params = [[1, 4,  1], [0.1, 4,  1], [0.01, 4,  1], 
           [1, 4, -1], [0.1, 4, -1], [0.01, 4, -1]]

g3_params = None
g5_params = None


# In[ ]:


# subselect structure
tmp_structure = structure.loc[structure.molecule_name=="dsgdb9nsd_000001", : ].copy()

# define acsf calculator
species = tmp_structure.atom.unique() #array(['C', 'H'], dtype=object)
acsf = ds.descriptors.ACSF(
        species=species,
        rcut=rcut,
        g2_params=g2_params,
        g3_params=g3_params,
        g4_params=g4_params,
        g5_params=g5_params,
    )


# In[ ]:


# create ase.Atoms from structure data
molecule_atoms = tmp_structure.loc[:, 'atom']
molecule_positions = tmp_structure.loc[:, ['x','y','z']]

molecule_system = ase.atoms.Atoms(symbols=molecule_atoms, positions=molecule_positions)

print(molecule_system)
print(molecule_system.get_atomic_numbers())
print(molecule_system.get_positions())


# In[ ]:


# ok ready to create acsf features
acsf_features = acsf.create(molecule_system, n_jobs=1) # structure of return is [[#acsf features] for each position in molecule_system]
acsf_features[0]


# One issue I have with this result is that features don't have any describing labels. Of course, I can't estimate what a g4 with params [x,y,z] would be, but this output alone doesn't describe anything. Unfortunately, I did not find any label_generator in their [git](https://github.com/SINGROUP/dscribe/blob/development/dscribe), so I tried to shadow their implementation and create my own set of labels.

# # Connecting the dots
# Now that you know how to calculate acsf features yourself, we should put the pieces together and create acsf features for all molecules

# In[ ]:


# some functions to help out
def create_feature_labels(species,rcut,
                          g2_params=None,
                          g3_params=None,
                          g4_params=None,
                          g5_params=None,
                          transform_to_symbols=True):

    #sub function to transform from atom numbers to chemical symbols
    def get_atom_id(atom_nr, tranform_to_symbols):

        if transform_to_symbols == True:
            atom_id = nr_to_symbol[atom_nr]
        else:
            atom_id = atom_nr

        return atom_id

    feature_label = []

    g_params={
        'g1': [rcut],
        'g2': g2_params,
        'g3': g3_params,
        'g4': g4_params,
        'g5': g5_params
    }


    # create_atom_numbers -> symbol dict
    tmp_system = ase.Atoms(species, [[0,0,0]]*len(species))

    nr_to_symbol = {number:symbol for symbol, number in
                    zip(tmp_system.get_chemical_symbols(),tmp_system.get_atomic_numbers())
                    }


    atomic_numbers = sorted(tmp_system.get_atomic_numbers())

    for atom_nr in atomic_numbers:

        atom_id = get_atom_id(atom_nr, transform_to_symbols)

        for g in ["g1", "g2","g3"]:

            params = g_params[g]

            if params is not None:

                for para in params:

                    feature_label.append(f'feat_acsf_{g}_{atom_id}_{para}')

    for atom_nr in atomic_numbers:

        atom_id = get_atom_id(atom_nr, transform_to_symbols)

        for i in range(0, atom_nr+1):

            if i in atomic_numbers:

                atom_id_2 = get_atom_id(i, transform_to_symbols)

                for g in ["g4","g5"]:

                    params = g_params[g]

                    if params is not None:

                        for para in params:

                            feature_label.append(f'feat_acsf_{g}_{atom_id}_{atom_id_2}_{para}')

    return feature_label

def calculate_symmetric_functions(df_structure, rcut, g2_params=None,
                                  g3_params=None,
                                  g4_params=None,
                                  g5_params=None):

    species = df_structure.atom.unique()

    acsf = ds.descriptors.ACSF(
        species=species,
        rcut=rcut,
        g2_params=g2_params,
        g3_params=g3_params,
        g4_params=g4_params,
        g5_params=g5_params,
    )

    structure_molecules = df_structure.molecule_name.unique()

    acsf_feature_labels = create_feature_labels(species=species,
                                                rcut=rcut,
                                                g2_params=g2_params,
                                                g3_params=g3_params,
                                                g4_params=g4_params,
                                                g5_params=g5_params,
                                                )

    df_structure= df_structure.reindex(columns = df_structure.columns.tolist() + acsf_feature_labels)

    df_structure = df_structure.sort_values(['molecule_name','atom_index'])

    acsf_structure_chunks = calculate_acsf_in_chunks(structure_molecules, df_structure, acsf, acsf_feature_labels)

    acsf_structure = pd.DataFrame().append(acsf_structure_chunks)

    return acsf_structure

def calculate_acsf_in_chunks(structure_molecules, df_structure, acsf, acsf_feature_labels, step_size=2000):

    mol_counter = 0
    max_counter = len(structure_molecules)
    all_chunks = []
    tic = time.time()
    while mol_counter*step_size < max_counter:

        tmp_molecules = structure_molecules[mol_counter*step_size:(mol_counter+1)*step_size]

        tmp_structure = df_structure.loc[df_structure.molecule_name.isin(tmp_molecules),:].copy()

        tmp_results = calculate_acsf_multiple_molecules(tmp_molecules, tmp_structure, acsf, acsf_feature_labels)

        all_chunks.append(tmp_results.copy())

        print((mol_counter+1)*step_size, time.time()-tic)

        mol_counter += 1

    return all_chunks


def calculate_acsf_multiple_molecules(molecule_names, df_structure, acsf, acsf_feature_labels):

    #acsf_feature_labels = [f'feat_acsf_{nr}' for nr in range(0, acsf.get_number_of_features())]
    #df_molecules = df_structure.loc[df_structure.molecule_name.isin(molecule_names),:].copy()
    counter = 0
    tic = time.time()
    for molecule_name in molecule_names:

        df_molecule = df_structure.loc[df_structure.molecule_name == molecule_name,:]
        acsf_values = calculate_acsf_single_molecule(df_molecule, acsf)


        df_structure.loc[df_structure.molecule_name==molecule_name, acsf_feature_labels] = copy.copy(acsf_values)

        counter += 1

    #print(counter, time.time() - tic)

    return df_structure

def calculate_acsf_single_molecule(df_molecule, acsf):

    molecule_atoms = df_molecule.loc[:, 'atom']
    molecule_positions = df_molecule.loc[:, ['x','y','z']]

    molecule_system = ase.atoms.Atoms(symbols=molecule_atoms, positions=molecule_positions)

    return acsf.create(molecule_system, n_jobs=1)


# In[ ]:


# calculate acsf features with Boris parameter
# this should take ~ 4 hours on kaggle
# there is some issue with ase using 10 cores by default that I couldn't disable. 
# It should be possible to calculate way faster

# I'm only using our beloved molecule to show how the output would look like. Remove the .loc condition if you want to recalc everything.
acsf_structure = calculate_symmetric_functions(structure.loc[structure.molecule_name=='dsgdb9nsd_000001',:].copy(), rcut, 
                                                   g2_params=g2_params,
                                                  g4_params=g4_params)


acsf_structure.head()


# # Check labels
# Let's calculate some g1 and g2s to check the labels

# In[ ]:


# Boris has a better way of doing this, but I'm trying to keep it simple here
def dist(coord_0, coord_1):    
    return np.sqrt(np.sum((coord_0-coord_1)**2))

def fc(dist, rcut):
    return  0.5*(np.cos(np.pi * dist / rcut)+1)


# In[ ]:


# my beloved molecule
test_molecule = structure.loc[structure.molecule_name=='dsgdb9nsd_000001',:]
coord_c = test_molecule.loc[test_molecule.atom == 'C', ['x','y','z']].values[0]

# G1 in regards to atoms of type H
g1_H = 0
for coord_h in test_molecule.loc[test_molecule.atom == 'H', ['x','y','z']].values:
    
    dist_h_c = dist(coord_c, coord_h)
    
    if dist_h_c <= rcut:
        g1_H += fc(dist_h_c, rcut)
        
print(f'g1 value is {g1_H}, using rcut: {rcut}')


for para in g2_params:
    eta= para[0]
    rs = para[1]
    g2_H = 0
    for coord_h in test_molecule.loc[test_molecule.atom == 'H', ['x','y', 'z']].values:

        dist_h_c = dist(coord_c, coord_h)

        g2_H += np.exp(-eta*(dist_h_c-rs)**2) * fc(dist_h_c, rcut)
    
    print(f'g2 value is {g2_H}, using eta: {eta}, rs: {rs}')


# In[ ]:


# Compare the values with the labes above - looks good to me


# # Compare the results
# Now we calculated a lot of numbers but do they have any meaning?

# In[ ]:


# load full dataset

structure_acsf = pd.read_csv('../input/molecules-structure-acsf/structure_with_acsf.csv')


# In[ ]:


# Let's compare feature number.
# Interestingly Boris has twice as many features (!). I honestly have little to no clue about acsf but I'm going to trust the dscribe package on this one.
feature_columns = [col for col in structure_acsf.columns if col.startswith('feat_acsf')]
len_features = len(feature_columns)
print(f"We have {len_features} feautres")
print(f"Boris announced ~ 250")
print("Maybe he is using two sets of rcut (?)")


# In[ ]:


structure_acsf.loc[structure_acsf.molecule_name == 'dsgdb9nsd_000001',:]


# In[ ]:


# Let's check the numbers
# A quick look revealed the following mappings:
#BorisFeatNr -> FeatureLabel
0 -> feat_acsf_g1_H_10.0
1 -> feat_acsf_g2_H_[0.01, 2]

4 -> feat_acsf_g2_H_[1, 2]
5 -> feat_acsf_g2_H_[0.1, 2]
6 -> feat_acsf_g2_H_[0.01, 2]

12 -> feat_acsf_g2_H_[0.01, 6]
18 -> feat_acsf_g2_C_[0.1, 2]
25 -> feat_acsf_g2_C_[0.01, 6]

# I couldn't map all of them but I think this is enough indication to give it a shot


# # Words of warning and encouragement
# This is my first kernel + public dataset so please let me know if I broke some rules or should do things differently.
# I tried to check everything (eg. created labels) but they still might be wrong. If you find inconsistencies or bugs please let me know.
