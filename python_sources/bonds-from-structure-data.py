#!/usr/bin/env python
# coding: utf-8

# **Recovering bonds from structure**
# 
# This kernel presents a method to extract the bonds between atoms in a molecule.  The inputs are the XYZ coordinates of the atoms (as found in the given structure data) and the covalent radius for each element (from wikipedia).  The output is, for each atom, a list of atom_indexes of the other atoms that it is bonded to.  The method is similar to the atomic connectivity step described here: http://proteinsandwavefunctions.blogspot.com/2018/01/xyz2mol-converting-xyz-file-to-rdkit.html.

# In[1]:


import numpy as np
import pandas as pd
import os
from tqdm import tqdm_notebook as tqdm
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
print(os.listdir("../input"))

KAGGLE_DIR = '../input'

# # Atom level properties
# MULLIKEN_CHARGES_CSV = os.path.join(KAGGLE_DIR, 'mulliken_charges.csv')
# SCALAR_COUPLING_CONTRIBUTIONS_CSV = os.path.join(KAGGLE_DIR, 'scalar_coupling_contributions.csv')
# MAGNETIC_SHIELDING_TENSORS_CSV = os.path.join(KAGGLE_DIR, 'magnetic_shielding_tensors.csv')
STRUCTURES_CSV = os.path.join(KAGGLE_DIR, 'structures.csv')

# # Molecule level properties
# POTENTIAL_ENERGY_CSV = os.path.join(KAGGLE_DIR, 'potential_energy.csv')
# DIPOLE_MOMENTS_CSV = os.path.join(KAGGLE_DIR, 'dipole_moments.csv')

# Atom-Atom interactions
TRAIN_CSV = os.path.join(KAGGLE_DIR, 'train.csv')
TEST_CSV = os.path.join(KAGGLE_DIR, 'test.csv')


# **Read and preprocess structure data**
# 
# The most important step here for bond detection is the addition of the atomic radius values.  There are several different definitions of atomic radius, but the most relevant in this situation is the radius of a single covalent bond.  Wikipedia maintains a table with this value at https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page).  I increased the values slightly in order to reduce false negatives.  Atoms that are not bonded repel each other, so it should be rare that this increase will result in false positives.

# In[2]:


atomic_radius = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71} # Without fudge factor

fudge_factor = 0.05
atomic_radius = {k:v + fudge_factor for k,v in atomic_radius.items()}
print(atomic_radius)

electronegativity = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}

structures = pd.read_csv(STRUCTURES_CSV, dtype={'atom_index':np.int8})

atoms = structures['atom'].values
atoms_en = [electronegativity[x] for x in tqdm(atoms)]
atoms_rad = [atomic_radius[x] for x in tqdm(atoms)]

structures['EN'] = atoms_en
structures['rad'] = atoms_rad

display(structures.head())


# **Chemical Bond Calculation**
# 
# 

# In[3]:


i_atom = structures['atom_index'].values
p = structures[['x', 'y', 'z']].values
p_compare = p
m = structures['molecule_name'].values
m_compare = m
r = structures['rad'].values
r_compare = r

source_row = np.arange(len(structures))
max_atoms = 28

bonds = np.zeros((len(structures)+1, max_atoms+1), dtype=np.int8)
bond_dists = np.zeros((len(structures)+1, max_atoms+1), dtype=np.float32)

print('Calculating bonds')

for i in tqdm(range(max_atoms-1)):
    p_compare = np.roll(p_compare, -1, axis=0)
    m_compare = np.roll(m_compare, -1, axis=0)
    r_compare = np.roll(r_compare, -1, axis=0)
    
    mask = np.where(m == m_compare, 1, 0) #Are we still comparing atoms in the same molecule?
    dists = np.linalg.norm(p - p_compare, axis=1) * mask
    r_bond = r + r_compare
    
    bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)
    
    source_row = source_row
    target_row = source_row + i + 1 #Note: Will be out of bounds of bonds array for some values of i
    target_row = np.where(np.logical_or(target_row > len(structures), mask==0), len(structures), target_row) #If invalid target, write to dummy row
    
    source_atom = i_atom
    target_atom = i_atom + i + 1 #Note: Will be out of bounds of bonds array for some values of i
    target_atom = np.where(np.logical_or(target_atom > max_atoms, mask==0), max_atoms, target_atom) #If invalid target, write to dummy col
    
    bonds[(source_row, target_atom)] = bond
    bonds[(target_row, source_atom)] = bond
    bond_dists[(source_row, target_atom)] = dists
    bond_dists[(target_row, source_atom)] = dists

bonds = np.delete(bonds, axis=0, obj=-1) #Delete dummy row
bonds = np.delete(bonds, axis=1, obj=-1) #Delete dummy col
bond_dists = np.delete(bond_dists, axis=0, obj=-1) #Delete dummy row
bond_dists = np.delete(bond_dists, axis=1, obj=-1) #Delete dummy col

print('Counting and condensing bonds')

bonds_numeric = [[i for i,x in enumerate(row) if x] for row in tqdm(bonds)]
bond_lengths = [[dist for i,dist in enumerate(row) if i in bonds_numeric[j]] for j,row in enumerate(tqdm(bond_dists))]
n_bonds = [len(x) for x in bonds_numeric]

#bond_data = {'bond_' + str(i):col for i, col in enumerate(np.transpose(bonds))}
#bond_data.update({'bonds_numeric':bonds_numeric, 'n_bonds':n_bonds})

bond_data = {'bonds':bonds_numeric, 'n_bonds':n_bonds, 'bond_lengths':bond_lengths}
bond_df = pd.DataFrame(bond_data)
structures = structures.join(bond_df)
display(structures.head(20))


# **Validation**

# **The last molecule - 133885**
# 
# Here is a visualization of the xyz file generated by a program called IQmol 
# 
# (Info and download here: http://iqmol.org/)
# 
# ![image.png](attachment:image.png)
# 
# This molecule has 16 atoms.  Below are the generated bonds.  It is fairly easy to verify that the visualization program and the generated bonds agree. (Note: IQmol uses 1-index and I use 0-index)

# In[4]:


structures.tail(16)


# **Bond counts by element**
# 
# Based on the number of electrons in an atom's valence shell, we know how many bonds the atom needs to form to be stable.  Hydrogen needs 1, Flourine needs 1, Oxygen needs 2, Nitrogen needs 3, and Carbon needs 4.  Bonds can be single, double, or triple, but we have not yet calculated the strength of the bonds.  Therefore there is a range of valid bond counts we could get for each atom.
# 
# - Hydrogen         
#     - 1
# - Flourine
#     - 1
# - Oxygen
#     - 1 - 2
# - Nitrogen         
#     - 1 - 3
# - Carbon   
#     - 2 - 4
#     
# When we graph the number of bonds for each element we see that these conditions are met, with the sole exception of several Nitrogen atoms forming 4 bonds.

# In[5]:


elements = structures['atom'].unique()
graphs_per_row = 3
row_count = int(np.ceil(len(elements) / graphs_per_row))
fig, axes = plt.subplots(row_count, graphs_per_row, figsize=(20, row_count * 5))

for i, element in enumerate(elements):
    x = structures[structures['atom'] == element].n_bonds.value_counts().index
    y = structures[structures['atom'] == element].n_bonds.value_counts().values
    ax = axes[i//graphs_per_row, i%graphs_per_row]
    ax.bar(x=x, height=y, tick_label=[str(n) for n in x], label='Bond count')
    ax.set(title=f'Bond count - {element}', xlabel='Bond count', ylabel='frequency')

plt.tight_layout()
plt.show()


# In[6]:


elements = structures['atom'].unique()
graphs_per_row = 3
row_count = int(np.ceil(len(elements) / graphs_per_row))
fig, axes = plt.subplots(row_count, graphs_per_row, figsize=(20, row_count * 5))

for i, element in enumerate(elements):
    y = []
    for l in structures[structures['atom'] == element]['bond_lengths'].values:
        y.extend(l)
    ax = axes[i//graphs_per_row, i%graphs_per_row]
    ax.hist(y, bins=1000)
    ax.set(title=f'Bond lengths - {element}', xlabel='Bond length', ylabel='frequency')

plt.tight_layout()
plt.show()


# In[7]:


structures.head()


# In[ ]:


structures["atom_count"] = structures.groupby("molecule_name")["atom_index"].transform("size")


# In[8]:


structures.atom.value_counts()


# In[11]:


structures.head(15).groupby("molecule_name")["atom_index"].size()


# In[ ]:





# In[ ]:


structures.drop("bonds",axis=1).to_csv("struct_bonds_v1.csv.gz",index=False,compression="gzip")

