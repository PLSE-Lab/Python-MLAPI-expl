#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# This kernel is based on a seriously underrated kernel by BigIronSphere: [Molecular Geometry - Feature Engineering & EDA](https://www.kaggle.com/bigironsphere/molecular-geometry-feature-engineering-eda), in which he computed four extremely good features for small molecules, and he used [HOW TO: Easy Visualization of Molecule](https://www.kaggle.com/borisdee/how-to-easy-visualization-of-molecules) to show some feature examples. 
# 
# In BigIronSphere's original kernel, computing the only 4 features for all train and test data takes 13 hours! I profiled his code offline and found that the culprit is `pandas`!!!! Everytime a huge dataframe is queried, it takes hundreds more time than reading an `xyz` from structures folder. I would like to demostrate how to make use of the `xyz` files provided by the competition organizers.
# 
# For example, you can speed up [the bonds calculation](https://www.kaggle.com/scaomath/parallelization-of-coulomb-yukawa-interaction) even more!
# 
# Lastly, multiprocessing is applied the groupby iterator to speed up the computation even more.
# 
# If you like to skip the explanation and directly the use the features, just concat the features with the train and test dataframe you are good to go.

# In[ ]:


get_ipython().system('pip install ase')


# In[ ]:


import numpy as np 
import pandas as pd 
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
import math
from numpy.linalg import svd, norm
from multiprocessing import Pool

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import ase
from ase import Atoms
import ase.visualize
def view(molecule):
    # Select a molecule
    mol = struct[struct['molecule_name'] == molecule]
    
    # Get atomic coordinates
    xcart = mol.iloc[:, 3:].values
    
    # Get atomic symbols
    symbols = mol.iloc[:, 2].values
    
    # Display molecule
    system = Atoms(positions=xcart, symbols=symbols)
    print('Molecule Name: %s.' %molecule)
    return ase.visualize.view(system, viewer="x3d")


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
struct = pd.read_csv('../input/structures.csv')


# In[ ]:


all_molecule_names = struct['molecule_name'].unique()


# In the original kernel, SVD function only returns the first eigenvector, which represents the principle direction of the molecule (the direction with the most variance in terms of coordinates distribution). Here I changed it a little bit so we can have more features generated. 

# In[ ]:


#quick PCA via singular value decomp
def PCA_SVD(a, n_vec = 1):
    a_mean = a.mean(axis=1)
    #normalize
    a = (a.T - a_mean.T).T
    u, s, v = svd(a.T)
    return v[:n_vec], s[:n_vec]

#obtain plane with minimum sum distance from nuclei
def get_plane(a):
    a = np.reshape(a, (a.shape[0], -1))
    centroid = a.mean(axis=1)
    #normalise
    x = a - centroid[:,np.newaxis]
    m = np.dot(x, x.T) 
    return centroid, svd(m)[0][:,-1]


# # New geometric features
# 
# For an $n\times 3$ matrix which represents a point cloud, we can have 3 singular values by doing SVD. Denote these 3 singular values by $s_i (i=1,2,3)$ sorted from the biggest to the smallest. 
# 
# * `thinness_metric` is $(s_2^2+s_3^2)/s_1^2$, if the molecule is thin and long, then this number should be close to 0.
# * `flatness_metric` is now changed to $s_3^2/(s_1^2+s_2^2)$, if the molecule is flat, this number should be close to 0, and there is no $\leq 3$ atoms molecule problem like the old one where `np.nan` has to be filled.
# * `roundness_metric` is $s_2/s_1$, if the molecule is flat or can be inscribed in a ball, this number should be close to 1.
# * `bond_angle_plane` and `bond_angle_axis` are unchanged from BigIronSphere's original kernel, which represents the angles of the bonds with the plane and principle eigenvector respectively.
# 
# The first three features are all relative, which may describe some invariant geometric features no matter a molecule is big or small.

# # Benchmark
# 
# We benchmark the code using first 100 molecule from the structures, which contains about 70 molecules in the train.

# In[ ]:


N_mol = 100
molecule_names = all_molecule_names[:N_mol]
small_idx = train.molecule_name.isin(molecule_names)
small_train = train.loc[small_idx]

print(f"There are {small_train.molecule_name.nunique()} molecules in the train to benchmark.")


# In[ ]:


get_ipython().run_cell_magic('time', '', "size_list = []\nflatness_list = []\nthinness_list = []\nroundness_list = []\nbond_angle_plane = []\nbond_angle_axis = []\n\nfor i in tqdm_notebook(range(len(molecule_names))):\n    \n    mol = molecule_names[i]\n    #obtain structure and bond information for each molecule\n    temp_struct = struct.loc[struct.molecule_name==mol, :]\n    bonds = train.loc[train.molecule_name == mol, :]\n    \n    #number of constituent atoms\n    size = len(temp_struct)\n    size_list.extend(np.full(len(bonds), size))\n    \n    #nuclei coords\n    coords = np.column_stack([temp_struct.x.values,\n                              temp_struct.y.values,\n                              temp_struct.z.values]).T\n    \n    #principal axis of molecular alignment\n    axis_vector_all, singular_val_all = PCA_SVD(coords, n_vec=3)\n    axis_vector = axis_vector_all[0] # major axis vector\n    \n    thinness = (singular_val_all[1]**2 + singular_val_all[2]**2)/singular_val_all[0]**2\n    thinness_list.extend(np.full(len(bonds), thinness))\n    \n    flatness = singular_val_all[2]**2/(singular_val_all[0]**2 + singular_val_all[1]**2)\n    flatness_list.extend(np.full(len(bonds), flatness))\n    \n    roundness = singular_val_all[1]/singular_val_all[0]\n    roundness_list.extend(np.full(len(bonds), roundness))\n    \n    \n    #obtain flatness metric and plane angles for binds if nuclei > 3\n    if size > 3:\n        \n        coords = coords - coords.mean()\n        #calculate best fitting 2D plane\n        ctr, norm = get_plane(coords) \n        \n        #calculate distances of each atom from plane\n        dists = np.zeros(size)\n        for j in range(0, size):\n            dists[j] = np.dot(norm, coords[:, j] - ctr) \n        \n        #obtain angle subtended by each atomic bind and plane\n        #print('There are {} bonds to calculate.'.format(len(bonds)))\n        for j in range(0, len(bonds)):\n            \n            #obtain atom index numbers for this bond\n            atom1 = bonds.atom_index_0.values[j]\n            atom2 = bonds.atom_index_1.values[j]\n            \n            #get 3D coords\n            atom1_coords = coords[:, atom1]\n            atom2_coords = coords[:, atom2]\n            \n            #bond vector\n            atom_vec = np.array([atom1_coords[0] - atom2_coords[0],\n                                 atom1_coords[1] - atom2_coords[1],\n                                 atom1_coords[2] - atom2_coords[2]])\n            \n            #angle subtended by bond vector and molecule plane\n            angle = np.dot(norm, atom_vec)/(np.linalg.norm(norm)*np.linalg.norm(atom_vec))\n            axis_angle = np.dot(axis_vector, atom_vec)/(np.linalg.norm(norm)*np.linalg.norm(atom_vec))\n            \n            #standardise to degrees <= 90\n            angle = np.arccos(angle)\n            axis_angle = np.arccos(axis_angle)\n\n            if angle > np.pi/2:\n                angle = np.pi - angle\n            \n            if axis_angle > np.pi/2:\n                axis_angle = np.pi - axis_angle\n                \n            #not needed but somewhat easier to visualise\n            angle = np.pi/2 - angle\n            axis_angle = np.pi/2 - axis_angle\n            bond_angle_plane.append(angle)\n            bond_angle_axis.append(axis_angle)\n               \n    else:\n        bond_angle_plane.extend(np.full(len(bonds), np.nan))\n        \n        for j in range(0, len(bonds)):\n            \n            #obtain atom index numbers for this bond\n            atom1 = bonds.atom_index_0.values[j]\n            atom2 = bonds.atom_index_1.values[j]\n            \n            #get 3D coords\n            atom1_coords = coords[:, atom1]\n            atom2_coords = coords[:, atom2]\n            \n            #bond vector\n            atom_vec = np.array([atom1_coords[0] - atom2_coords[0],\n                                 atom1_coords[1] - atom2_coords[1],\n                                 atom1_coords[2] - atom2_coords[2]])\n            \n            #angle subtended by bond vector and molecule principal axis\n            axis_angle = np.dot(axis_vector, atom_vec)/(np.linalg.norm(axis_vector)*np.linalg.norm(atom_vec))\n            \n            #standardise to degrees <= 90\n            axis_angle = np.arccos(axis_angle)\n                 \n            if axis_angle > np.pi/2:\n                axis_angle = np.pi - axis_angle\n                \n            #not needed but somewhat easier to visualise\n            axis_angle = np.pi/2 - axis_angle\n            bond_angle_axis.append(axis_angle)\n  ")


# ## Manipulating xyz files
# 
# The code above takes about ~53 seconds to run for about 70 molecules/~1800 rows of the train...But we have 7 million rows which you guys can compute... Now let us optimize the code above by reading `xyz` files instead of reading the position from the `struct` dataframe. Except reading coordinates from the `xyz` files, the main changes I have made are:
# * the bonds features are now extracted using `groupby`, so that the for loop does not query molecule name that is not the dataframe (molecule names that are in structures may not be in the train which further makes the execution slower).

# In[ ]:


folder = '../input/structures/'


# Simple example using 1 molecule `dsgdb9nsd_133885` which has 16 atoms.

# In[ ]:


with open(folder + "/dsgdb9nsd_133885.xyz") as f:
    positions = []
    symbols = []
    for row, line in enumerate(f):
        
        print(row, line.replace('\n', ''))
        
        fields = line.split(' ')
        
        if row < 2: 
            continue
        
        # Then rows of atomic positions and chemical symbols.
        else:
            positions.append(fields[1:4])
            print(f"{fields[0]} -> {symbols}\n")
            symbols.append(fields[0])       

print(f"Number of atoms: {len(symbols)}")


# Now onto the same routine.

# In[ ]:


get_ipython().run_cell_magic('time', '', "flatness_list = []\nthinness_list = []\nroundness_list = []\nbond_angle_plane = []\nbond_angle_axis = []\n\nN = small_train['molecule_name'].nunique()\n\nwith tqdm(total=N) as pbar:\n    \n    for i, (mol, bonds) in enumerate(small_train.groupby(['molecule_name'])):\n        pbar.update(1)\n        ## instead of querying structures dataframe, we read from xyz file\n        filename = folder + mol + '.xyz'\n        positions = []\n        with open(filename) as f:\n            for row, line in enumerate(f):\n                fields = line.split(' ')\n                if row < 2:\n                    continue\n                # Then rows of atomic positions and chemical symbols.\n                else:\n                    positions.append(fields[1:4])\n\n        size = len(positions)\n        n_bonds = len(bonds)\n\n        #nuclei coords\n        coords = np.array(positions, dtype=float).T\n\n        #principal axis of molecular alignment\n        axis_vector_all, singular_val_all = PCA_SVD(coords, n_vec=3)\n        axis_vector = axis_vector_all[0] # major axis vector\n\n        thinness = (singular_val_all[1]**2 + singular_val_all[2]**2)/singular_val_all[0]**2\n        thinness_list.extend(np.full(n_bonds, thinness))\n\n        flatness = singular_val_all[2]**2/(singular_val_all[0]**2 + singular_val_all[1]**2)\n        flatness_list.extend(np.full(n_bonds, flatness))\n\n        roundness = singular_val_all[1]/singular_val_all[0]\n        roundness_list.extend(np.full(n_bonds, roundness))\n\n        \n        #obtain flatness metric and plane angles for binds if nuclei > 3\n        if size > 3:\n\n            coords = coords - coords.mean()\n            #calculate best fitting 2D plane\n            ctr, norm = get_plane(coords) \n\n            #calculate distances of each atom from plane\n            dists = np.zeros(size)\n            for j in range(0, size):\n                dists[j] = np.dot(norm, coords[:, j] - ctr) \n\n            #obtain angle subtended by each atomic bind and plane\n            #print('There are {} bonds to calculate.'.format(len(bonds)))\n            for j in range(0, n_bonds):\n\n                #obtain atom index numbers for this bond\n                atom1 = bonds.atom_index_0.values[j]\n                atom2 = bonds.atom_index_1.values[j]\n\n                #get 3D coords\n                atom1_coords = coords[:, atom1]\n                atom2_coords = coords[:, atom2]\n\n                #bond vector\n                atom_vec = np.array([atom1_coords[0] - atom2_coords[0],\n                                     atom1_coords[1] - atom2_coords[1],\n                                     atom1_coords[2] - atom2_coords[2]])\n\n                #angle subtended by bond vector and molecule plane\n                angle = np.dot(norm, atom_vec)/(np.linalg.norm(norm)*np.linalg.norm(atom_vec))\n                axis_angle = np.dot(axis_vector, atom_vec)/(np.linalg.norm(norm)*np.linalg.norm(atom_vec))\n\n                #standardise to degrees <= 90\n                angle = np.arccos(angle)\n                axis_angle = np.arccos(axis_angle)\n\n                if angle > np.pi/2:\n                    angle = np.pi - angle\n\n                if axis_angle > np.pi/2:\n                    axis_angle = np.pi - axis_angle\n\n                #not needed but somewhat easier to visualise\n                angle = np.pi/2 - angle\n                axis_angle = np.pi/2 - axis_angle\n                bond_angle_plane.append(angle)\n                bond_angle_axis.append(axis_angle)\n\n        else:\n            bond_angle_plane.extend(np.full(n_bonds, np.nan))\n\n            for j in range(0, n_bonds):\n\n                #obtain atom index numbers for this bond\n                atom1 = bonds.atom_index_0.values[j]\n                atom2 = bonds.atom_index_1.values[j]\n\n                #get 3D coords\n                atom1_coords = coords[:, atom1]\n                atom2_coords = coords[:, atom2]\n\n                #bond vector\n                atom_vec = np.array([atom1_coords[0] - atom2_coords[0],\n                                     atom1_coords[1] - atom2_coords[1],\n                                     atom1_coords[2] - atom2_coords[2]])\n\n                #angle subtended by bond vector and molecule principal axis\n                axis_angle = np.dot(axis_vector, atom_vec)/(np.linalg.norm(axis_vector)*np.linalg.norm(atom_vec))\n\n                #standardise to degrees <= 90\n                axis_angle = np.arccos(axis_angle)\n\n                if axis_angle > np.pi/2:\n                    axis_angle = np.pi - axis_angle\n\n                #not needed but somewhat easier to visualise\n                axis_angle = np.pi/2 - axis_angle\n                bond_angle_axis.append(axis_angle)   ")


# In[ ]:


def get_geometric_features(df):
    mol = df.molecule_name.values[0]
    bonds = df
    filename = folder + mol + '.xyz'
    positions = []
    flatness_list = []
    thinness_list = []
    roundness_list = []
    bond_angle_plane = []
    bond_angle_axis = []
    
    with open(filename) as f:
        for row, line in enumerate(f):
            fields = line.split(' ')
            if row < 2:
                continue
            # Then rows of atomic positions and chemical symbols.
            else:
                positions.append(fields[1:4])

    size = len(positions)
    n_bonds = len(df)

    #nuclei coords
    coords = np.array(positions, dtype=float).T

    #principal axis of molecular alignment
    axis_vector_all, singular_val_all = PCA_SVD(coords, n_vec=3)
    axis_vector = axis_vector_all[0] # major axis vector

    thinness = (singular_val_all[1]**2 + singular_val_all[2]**2)/singular_val_all[0]**2
    thinness_list.extend(np.full(n_bonds, thinness))

    flatness = singular_val_all[2]**2/(singular_val_all[0]**2 + singular_val_all[1]**2)
    flatness_list.extend(np.full(n_bonds, flatness))

    roundness = singular_val_all[1]/singular_val_all[0]
    roundness_list.extend(np.full(n_bonds, roundness))


    #obtain flatness metric and plane angles for binds if nuclei > 3
    if size > 3:

        coords = coords - coords.mean()
        #calculate best fitting 2D plane
        ctr, norm = get_plane(coords) 

        #calculate distances of each atom from plane
        dists = np.zeros(size)
        for j in range(0, size):
            dists[j] = np.dot(norm, coords[:, j] - ctr) 

        #obtain angle subtended by each atomic bind and plane
        #print('There are {} bonds to calculate.'.format(len(bonds)))
        for j in range(0, n_bonds):

            #obtain atom index numbers for this bond
            atom1 = bonds.atom_index_0.values[j]
            atom2 = bonds.atom_index_1.values[j]
            
            #get 3D coords
            atom1_coords = coords[:, atom1]
            atom2_coords = coords[:, atom2]

            #bond vector
            atom_vec = np.array([atom1_coords[0] - atom2_coords[0],
                                 atom1_coords[1] - atom2_coords[1],
                                 atom1_coords[2] - atom2_coords[2]])

            #angle subtended by bond vector and molecule plane
            angle = np.dot(norm, atom_vec)/(np.linalg.norm(norm)*np.linalg.norm(atom_vec))
            axis_angle = np.dot(axis_vector, atom_vec)/(np.linalg.norm(norm)*np.linalg.norm(atom_vec))
            
            #standardise to degrees <= 90
            angle = np.arccos(angle)
            axis_angle = np.arccos(axis_angle)
        
            if angle > np.pi/2:
                angle = np.pi - angle

            if axis_angle > np.pi/2:
                axis_angle = np.pi - axis_angle

            #not needed but somewhat easier to visualise
            angle = np.pi/2 - angle
            axis_angle = np.pi/2 - axis_angle
            bond_angle_plane.append(angle)
            bond_angle_axis.append(axis_angle)

    else:
        bond_angle_plane.extend(np.full(n_bonds, np.nan))

        for j in range(0, n_bonds):

            #obtain atom index numbers for this bond
            atom1 = bonds.atom_index_0.values[j]
            atom2 = bonds.atom_index_1.values[j]

            #get 3D coords
            atom1_coords = coords[:, atom1]
            atom2_coords = coords[:, atom2]

            #bond vector
            atom_vec = np.array([atom1_coords[0] - atom2_coords[0],
                                 atom1_coords[1] - atom2_coords[1],
                                 atom1_coords[2] - atom2_coords[2]])

            #angle subtended by bond vector and molecule principal axis
            axis_angle = np.dot(axis_vector, atom_vec)/(np.linalg.norm(axis_vector)*np.linalg.norm(atom_vec))

            #standardise to degrees <= 90
            axis_angle = np.arccos(axis_angle)

            if axis_angle > np.pi/2:
                axis_angle = np.pi - axis_angle

            #not needed but somewhat easier to visualise
            axis_angle = np.pi/2 - axis_angle
            bond_angle_axis.append(axis_angle) 

    features = pd.DataFrame(index=df.index, dtype=np.float32)
    features['flatness_metric'] = np.asarray(flatness_list)
    features['bond_angle_plane'] = np.asarray(bond_angle_plane)
    features['bond_angle_axis'] = np.asarray(bond_angle_axis)
    features['thinness_metric'] = np.asarray(thinness_list)
    features['roundness_metric'] = np.asarray(roundness_list)
    
    return features


# ## Multiprocessing
# 
# The same code takes about ~0.25s, which is 250 times speed up...Yet when applying to the full training dataframe, it takes about 8 minutes to generate all the features.
# 
# Yet here we want to utilize all 4 CPU cores, without considering the setup overhead, the speed is about triple of the original, which should take only about 4 minutes for setting up and processing for all the training data.

# In[ ]:


import multiprocessing as mp


# In[ ]:


chunk_iter = small_train.groupby(['molecule_name'])
pool = mp.Pool(4) # use 4 CPU cores

funclist = []
for df in tqdm_notebook(chunk_iter):
    # process each data frame
    f = pool.apply_async(get_geometric_features,[df[1]])
    funclist.append(f)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'result = []\nfor f in tqdm_notebook(funclist):\n    result.append(f.get()) \n\n# combine chunks with transformed data into a single training set\nfeatures = pd.concat(result, ignore_index=True)')


# # Feature Evaluation
# 
# Now we can examine some of the results. Let's start with a molecule that has a high flatness metric. Recall that the higher this metric, the *less* flat a molecule is.

# In[ ]:


features.shape


# In[ ]:


features.sort_values('flatness_metric', ascending=False).head(5)


# The least flat molecule is location 0. Let's view it... (it is a tetrahedron, so...)

# In[ ]:


view(train.iloc[0]['molecule_name'])


# In[ ]:


features.sort_values('flatness_metric', ascending=True).head(5)


# The most flat (as well as thinnest and longest) molecule is train location 292.

# In[ ]:


view(train.iloc[292]['molecule_name'])


# # Generating the feature files
# 
# Now let us generate all the features for train and test.

# In[ ]:


def get_features(df):
    chunk_iter = df.groupby(['molecule_name'])
    pool = mp.Pool(4) # use 4 CPU cores

    funclist = []
    for df in tqdm_notebook(chunk_iter):
        # process each data frame
        f = pool.apply_async(get_geometric_features,[df[1]])
        funclist.append(f)

    result = []
    for f in tqdm_notebook(funclist):
        result.append(f.get()) 

    # combine chunks with transformed data into a single training set
    features = pd.concat(result, ignore_index=True)
    
    return features


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_features = get_features(train)')


# In[ ]:


train_features.to_csv('train_geometric_features.csv',index=False)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test_features = get_features(test)')


# In[ ]:


test_features.to_csv('test_geometric_features.csv',index=False)

