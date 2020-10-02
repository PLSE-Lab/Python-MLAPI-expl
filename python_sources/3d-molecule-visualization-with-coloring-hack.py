#!/usr/bin/env python
# coding: utf-8

# ## **Adding Coloring Hack to 3D Visualization**
# ![Before and After Color Hack Example](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F256034%2F8437314cf8ea8b244d91be549a4ad4b3%2F3D_view_color_hack_illustration2.jpg?generation=1563162851977006&alt=media)
# **Warning: **
# 
# **Please don't attempt to convince your chemistry professor that you can randomly replace atoms in molecules "because you saw pictures" :) **
# 
# This kernel is based on Boris D's [How To: Easy Visualization of Molecules](https://www.kaggle.com/borisdee/how-to-easy-visualization-of-molecules)
# for  [Predicting Molecular Properties](https://www.kaggle.com/c/champs-scalar-coupling/overview) challenge.
# 
# It is helpful to see the actual coupled atoms we are predicting. The following quick hack works quite well: replace the J-coupled atoms based on this dictionary: `{"H":"He","C":"Be","N":"B"}`. This tricks the visualizer into substituting differently colored spheres for the atoms in question. There must be a better way for this but this is the best I've got.
# 
# **Don't attempt this with actual molecules in the lab because, at best, it won't work. In the worst case, you may hurt yourself. And wear protective goggles at all times if you do still want to try this despite my warning.**

# In[ ]:


import pandas as pd
import random

struct_file = pd.read_csv('../input/structures.csv')
train = pd.read_csv('../input/train.csv')


# In[ ]:


get_ipython().system('pip install ase  # added !pip install, Internet must be on. ')
                  # Alternatively, use Boris's instructions: click on the *Settings* tab on the right panel, then click on *Install...*, 
                  # right next to *Packages*. In the *pip package name* entry, just write **ase** then hit *Install Package*. 
                  # Kaggle is going to do its things then restart the Kernel.


# In[ ]:


import ase
from ase import Atoms
import ase.visualize


# In[ ]:


# Select molecule based on name. Will use this molecule to demonstrate the coloring scheme.
molecule = struct_file[struct_file['molecule_name'] == 'dsgdb9nsd_000010']
display(molecule)


# Next we need to retrieve the atomic coordinates in a numpy array form:

# In[ ]:


# Get atomic coordinates
atoms = molecule.iloc[:, 3:].values
print(atoms)


# The last thing we need is the atomic symbols:

# In[ ]:


# Get atomic symbols
symbols = molecule.iloc[:, 2].values
print(symbols)


# Finally, let's put everything into something that **ase** can process. You can rotate the molecule with a left click, translate it with a middle click, and zoom in or out using right click. 

# In[ ]:


system = Atoms(positions=atoms, symbols=symbols)
ase.visualize.view(system, viewer="x3d")


# Here comes my quick and dirty hack to color two atoms that are J-coupled. J-coupled atoms are replaced based on the dictionary: `{"H":"He","C":"Be","N":"B"}`. This is chemically absurd but it happens to work for this visualization. Compare this visualization with the above molecule to see the new color scheme for J-coupled atoms in question: 'H' ='light blue', 'C' = 'light green', 'N' = 'pinkish tan'. The replaced atoms are of slightly different sizes: 'H' is slightly undersized, 'C' and 'N' are slightly oversized. This does not pose any obvious problems and, in any case, I did not find better size replacements.

# In[ ]:


my_dict = {"H":"He","C":"Be","N":"B"}
print(my_dict)
atom_index=(1,2,3)

jj_symbols=symbols.copy()
for k in atom_index:
    if jj_symbols[k] in my_dict:
        print (k, jj_symbols[k],my_dict[jj_symbols[k]])
        jj_symbols[k]=my_dict[jj_symbols[k]]
    else:
        print("Atom is not in the dictionary")
        
system = Atoms(positions=atoms, symbols=jj_symbols)
ase.visualize.view(system, viewer="x3d")


# Below is the function that visualizes a molecule and colors the two atoms (optional). It uses row index from train dataframe to get molecule name and the J-coupled atom locations.

# In[ ]:


def view_jj(train_ix, df=train, color =True):
    # Get molecule name and J-J indices
    mol_jj = df.iloc[[train_ix],1:4]
    atom_index=(mol_jj['atom_index_0'].values[0],mol_jj['atom_index_1'].values[0])
    molecule = mol_jj['molecule_name'].values[0]
    
    # Select a molecule
    mol = struct_file[struct_file['molecule_name'] == molecule]
    
    # Get atomic coordinates
    xcart = mol.iloc[:, 3:].values
    
    # Get atomic symbols
    symbols = mol.iloc[:, 2].values
    
    # Actual hack where J-J atoms we are predicted are substituted by approximately sized atoms of different colors:
    jj_symbols=symbols.copy()
    if color == True:
        for k in atom_index:
            if jj_symbols[k] in my_dict:
                print (k, jj_symbols[k],my_dict[jj_symbols[k]])
                jj_symbols[k]=my_dict[jj_symbols[k]]
            else:
                print("Atom is not in the dictionary")
    
    # Display molecule
    system = Atoms(positions=xcart, symbols=jj_symbols)
       
    #system = Atoms(positions=xcart, symbols=symbols)
    print('Molecule Name: %s.' %molecule)
    return ase.visualize.view(system, viewer="x3d")

view_jj(665999)
#view_jj(665999, color=False)  # Uncomment to see the original version without atom replacement


# Let me know if there is a better way to add J-coupled atoms to an interactive 3D viewer. I am new to Python so any code suggestions are highly appreciated. 
# 
# And last but not least: **Vote early and vote often! ;) **
