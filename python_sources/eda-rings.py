#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports

# Basic imports
import numpy as np
import pandas as pd

# Graphs
# %matplotlib widget
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from tqdm.auto import tqdm


# In[ ]:


# Seaborn advanced settings

sns.set(style='ticks',          # 'ticks', 'darkgrid'
        palette='colorblind',   # 'colorblind', 'pastel', 'muted', 'bright'
        #palette=sns.color_palette('Accent'),   # 'Set1', 'Set2', 'Dark2', 'Accent'
        rc = {
           'figure.autolayout': True,
           'figure.figsize': (14, 8),
           'legend.frameon': True,
           'patch.linewidth': 2.0,
           'lines.markersize': 6,
           'lines.linewidth': 2.0,
           'font.size': 20,
           'legend.fontsize': 20,
           'axes.labelsize': 16,
           'axes.titlesize': 22,
           'axes.grid': True,
           'grid.color': '0.9',
           'grid.linestyle': '-',
           'grid.linewidth': 1.0,
           'xtick.labelsize': 20,
           'ytick.labelsize': 20,
           'xtick.major.size': 8,
           'ytick.major.size': 8,
           'xtick.major.pad': 10.0,
           'ytick.major.pad': 10.0,
           }
       )

plt.rcParams['image.cmap'] = 'viridis'


# In[ ]:


import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} -c rdkit rdkit')


# In[ ]:


get_ipython().run_cell_magic('bash', '-e', 'if ! [[ -f ./xyz2mol.py ]]; then\n  wget https://raw.githubusercontent.com/jensengroup/xyz2mol/master/xyz2mol.py\nfi')


# In[ ]:


from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.rdmolops import SanitizeFlags

from xyz2mol import xyz2mol, xyz2AC, AC2mol, read_xyz_file
from pathlib import Path


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


train_df = pd.read_csv("../input/champs-scalar-coupling/train.csv")
test_df = pd.read_csv("../input/champs-scalar-coupling/test.csv")
structures = pd.read_csv("../input/champs-scalar-coupling/structures.csv")
dftr = structures[["molecule_name"]]
dftr = dftr.drop_duplicates()


# In[ ]:


ring_sizes = {3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
ring_types = dict()
ring_repetitions = dict()
molecules_with_rings = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
angle_minima = dict()
angle_maxima = dict()
angle_means = dict()


# In[ ]:


for i, row in tqdm(dftr.iterrows()):
    path = Path(f'../input/champs-scalar-coupling/structures/{row["molecule_name"]}.xyz')
    ats, molecule = MolFromXYZ(path)
    rings = molecule.GetRingInfo().AtomRings()
    molecules_with_rings[len(rings)] += 1
    for ring in rings:
        ring_sizes[len(ring)] += 1
    ring_list = ["" for ring in rings]
    ring_atom_indices = [[] for ring in rings]
    ring_atom_indices2 = [[] for ring in rings]
    for i, ring in enumerate(rings):
        for atom in ring:
            ring_list[i] += molecule.GetAtomWithIdx(atom).GetSymbol()
            ring_atom_indices[i].append(atom)
            ring_atom_indices2[i].append(molecule.GetAtomWithIdx(atom).GetSymbol())
    tmp_dict = dict()
    sorted_ring_list = list()
    for ring in ring_list:
        sorted_type = "".join(sorted(ring))
        ring_types.setdefault(sorted_type, 0)
        ring_types[sorted_type] += 1
        tmp_dict.setdefault(sorted_type, 0)
        tmp_dict[sorted_type] += 1
        sorted_ring_list.append(sorted_type)
    for key, value in tmp_dict.items():
        if key in ring_repetitions.keys():
            if value > ring_repetitions[key]:
                ring_repetitions[key] = value
        else:
            ring_repetitions[key] = value
    conf = molecule.GetConformer(0)
    for ring_number, ring in enumerate(rings):
        current_ring_angles = list()
        for i in range(-2, len(ring)-2):
            angle = rdMolTransforms.GetAngleDeg(conf,ring[i],ring[i+1],ring[i+2])
            current_ring_angles.append(angle)
        angle_minima.setdefault(sorted_ring_list[ring_number], [])
        angle_maxima.setdefault(sorted_ring_list[ring_number], [])
        angle_means.setdefault(sorted_ring_list[ring_number], [])      
        angle_minima[sorted_ring_list[ring_number]].append(min(current_ring_angles))
        angle_maxima[sorted_ring_list[ring_number]].append(max(current_ring_angles))
        angle_means[sorted_ring_list[ring_number]].append(np.mean(current_ring_angles))


# # Rings in molecules
# 
# Rings (or cycles) significantly change properties of molecules. Molecules with rings are called aromatic. In cycles, electrons are distributed in a different way then in non-aromatic molecules (without cycles), it is said that they are delocalized, as usually in non-aromatic molecules they are localized around their atom. Also, from my physical perspective, rings are similar to coils which has different surrounding electromagnetic field than e.g. wires. It is very important to add information about rings to ML models if they are to achieve good accuracy, because scalar coupling constant is electronic property of a molecule, which depends on how electrons are distributed in a molecule. As there is no way for us to exactly compute electronic wavefunction (where are electrons with highest probability), then adding information about rings which change electronic wavefunction, is the minimum we can do for ML models.

# In[ ]:


for key, value in molecules_with_rings.items():
    print(f"{value} molecules have {key} rings.")


# In[ ]:


plt.figure("MoleculesWithRings")
ax = sns.barplot(x=list(molecules_with_rings.keys()), y=list(molecules_with_rings.values()))
plt.xlabel("Number of rings in molecule")
plt.ylabel("Count of molecules")
plt.title("Molecules with rings")
plt.savefig("MoleculesWithRings.png")
plt.show()


# In[ ]:


for key, value in ring_sizes.items():
    print(f"{value} rings contain {key} atoms.")


# In[ ]:


plt.figure("RingSizes")
ax = sns.barplot(x=list(ring_sizes.keys()), y=list(ring_sizes.values()))
plt.xlabel("Sizes of rings")
plt.ylabel("Count of rings")
plt.title("Ring sizes")
plt.savefig("RingSizes.png")
plt.show()


# # Ring types

# In[ ]:


print(f"There are {len(ring_types.keys())} types of rings.")


# In[ ]:


print("There are:")
for key, value in ring_types.items():
    print(f"{value} {key} rings")


# In[ ]:


ring_types3 = dict()
ring_types4 = dict()
ring_types5 = dict()
ring_types6 = dict()
ring_types7 = dict()
ring_types8 = dict()
ring_types9 = dict()
for key, value in tqdm(ring_types.items()):
    if len(key) == 3:
        ring_types3[key] = value
    elif len(key) == 4:
        ring_types4[key] = value
    elif len(key) == 5:
        ring_types5[key] = value
    elif len(key) == 6:
        ring_types6[key] = value
    elif len(key) == 7:
        ring_types7[key] = value
    elif len(key) == 8:
        ring_types8[key] = value
    elif len(key) == 9:
        ring_types9[key] = value


# In[ ]:


plt.figure("3RingTypes", figsize=(20,10))
ax = sns.barplot(x=list(ring_types3.keys()), y=list(ring_types3.values()))
for i,p in enumerate(ax.patches):
    x = (p.get_x() + p.get_width()/2) - 0.09
    y = p.get_y() + p.get_height() + 0.15
    ax.annotate(list(ring_types3.values())[i], (x, y))
plt.xlabel("Ring types")
plt.ylabel("Count of rings")
plt.title("Types of 3-rings")
plt.savefig("3RingTypes.png")
plt.show()


# In[ ]:


plt.figure("4RingTypes", figsize=(20,10))
ax = sns.barplot(x=list(ring_types4.keys()), y=list(ring_types4.values()))
for i,p in enumerate(ax.patches):
    x = (p.get_x() + p.get_width()/2) - 0.09
    y = p.get_y() + p.get_height() + 0.15
    ax.annotate(list(ring_types4.values())[i], (x, y))
plt.xlabel("Ring types")
plt.ylabel("Count of rings")
plt.title("Types of 4-rings")
plt.savefig("4RingTypes.png")
plt.show()


# In[ ]:


plt.figure("5RingTypes", figsize=(20,10))
ax = sns.barplot(x=list(ring_types5.keys()), y=list(ring_types5.values()))
for i,p in enumerate(ax.patches):
    x = (p.get_x() + p.get_width()/2) - 0.22
    y = p.get_y() + p.get_height() + 0.15
    ax.annotate(list(ring_types5.values())[i], (x, y))
plt.xlabel("Ring types")
plt.ylabel("Count of rings")
plt.title("Types of 5-rings")
plt.savefig("5RingTypes.png")
plt.show()


# In[ ]:


plt.figure("6RingTypes", figsize=(20,10))
ax = sns.barplot(x=list(ring_types6.keys()), y=list(ring_types6.values()))
for i,p in enumerate(ax.patches):
    x = (p.get_x() + p.get_width()/2) - 0.2
    y = p.get_y() + p.get_height() + 0.15
    ax.annotate(list(ring_types6.values())[i], (x, y))
plt.xlabel("Ring types")
plt.ylabel("Count of rings")
plt.title("Types of 6-rings")
plt.savefig("6RingTypes.png")
plt.show()


# In[ ]:


plt.figure("7RingTypes", figsize=(20,10))
x_labels = [label[2:] for label in list(ring_types7.keys())]
ax = sns.barplot(x=x_labels, y=list(ring_types7.values()))
for i,p in enumerate(ax.patches):
    x = (p.get_x() + p.get_width()/2) - 0.15
    y = p.get_y() + p.get_height() + 0.15
    ax.annotate(list(ring_types7.values())[i], (x, y))
plt.xlabel("Ring types [CC+]")
plt.ylabel("Count of rings")
plt.title("Types of 7-rings")
plt.savefig("7RingTypes.png")
plt.show()


# In[ ]:


plt.figure("8RingTypes", figsize=(20,10))
ax = sns.barplot(x=list(ring_types8.keys()), y=list(ring_types8.values()))
for i,p in enumerate(ax.patches):
    x = (p.get_x() + p.get_width()/2) - 0.09
    y = p.get_y() + p.get_height() + 0.15
    ax.annotate(list(ring_types8.values())[i], (x, y))
plt.xlabel("Ring types")
plt.ylabel("Count of rings")
plt.title("Types of 8-rings")
plt.savefig("8RingTypes.png")
plt.show()


# In[ ]:


plt.figure("9RingTypes", figsize=(20,10))
x_labels = [label[3:] for label in list(ring_types9.keys())]
ax = sns.barplot(x=x_labels, y=list(ring_types9.values()))
for i,p in enumerate(ax.patches):
    x = (p.get_x() + p.get_width()/2) - 0.09
    y = p.get_y() + p.get_height() + 0.15
    ax.annotate(list(ring_types9.values())[i], (x, y))
plt.xlabel("Ring types [CCC+]")
plt.ylabel("Count of rings")
plt.title("Types of 9-rings")
plt.savefig("9RingTypes.png")
plt.show()

