#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Summary: efficient kernel using the `qml` package to generate eigenvalues of Coulomb matrix/atom-centered Coulomb matrix, and merge eigenvalues into the structures dataframe. 
# 
# The Coulomb potential matrix is a symmetric square matrix $\mathbf{C}_{ij} = (C_{ij})_{n\times n}$ where $n$ is the total number of atoms in a molecule:
# $$
# C_{ij} =
#   \begin{cases}
#      \dfrac{1}{2} Z_{i}^{2.4} & \text{if } i = j 
#      \\
#      \dfrac{Z_{i}Z_{j}}{\| {\bf R}_{i} - {\bf R}_{j}\|}       & \text{if } i \neq j
#   \end{cases}
# $$
# which is essentially the core of the [QM7 dataset](http://quantum-machine.org/datasets/) which are using a different set of molecules with the Champs competition.
# 
# 
# In the formula above, 
# - $i, j$: atom indices within a molecule.
# - $Z$: nuclear charge.
# - $\mathbf{R}$: the coordinate in the Euclidean space.
# 
# The reason it is called "potential" is because its spacial gradient gives the Coulomb force (inverse square law).
# 
# One big challege to extract features from the Coulomb matrix is that its size varies from different molecules. So a simple idea is to use its eigenvalues (and/or eigenvectors).
# 
# To use the final `struct_eig` dataframe, you can `pd.merge` it with the original structure file by `molecule_name` and `atom_index`, or more simply, you can drop duplicated columns first then `pd.concat` the two dataframe along axis 1.
# 
# 
# ### Reference
# * http://www.qmlcode.org/tutorial.html
# * Scirpus's kernel: https://www.kaggle.com/scirpus/coulomb-matrix-representation

# In[ ]:


get_ipython().system('pip install qml')


# In[ ]:


import qml


# In[ ]:


import pandas as pd
import numpy as np
from numpy.linalg import eig, svd
from sklearn.decomposition import PCA
from tqdm import tqdm

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[ ]:


folder = '../input/structures/'


# In[ ]:


structures = pd.read_csv('../input/structures.csv')


# In[ ]:


all_molecule_names = structures['molecule_name'].unique()


# # Coulomb matrix playground
# 
# Here I demostrate how the matrix is like using a medium-sized molecule (17 atoms).

# In[ ]:


filenames = ['dsgdb9nsd_133883.xyz']


# In[ ]:


for filename in tqdm(filenames):
    # Create the compound object mol from the file which happens to be methane
    mol = qml.Compound(xyz=folder+filename)
    mol.generate_coulomb_matrix(size=mol.natoms, sorting="unsorted")
    a = mol.representation


# In[ ]:


## this is the structure of molecule dsgdb9nsd_133883
structures.loc[structures.molecule_name==filename[:-4]]


# In[ ]:


print(mol.coordinates, '\n')
print(mol.atomtypes, '\n')
print(mol.nuclear_charges,'\n')
print(mol.name, '\n')
print(mol.natoms, '\n')


# `a` is a matrix represented using linear index, the follow routine converts it back to matrix. Since the matrix it represented is symmetric, we only need $n(n+1)/2 = 153$ entries to reprensent the Coulomb matrix.

# In[ ]:


c_mat1 = qml.representations.vector_to_matrix(a)


# In[ ]:


# this is 17x17 matrix, and it is symmetric
print(a.shape, c_mat1.shape, np.allclose(c_mat1, c_mat1.T))


# In[ ]:


pca = PCA(n_components = min(7,mol.natoms))
pca.fit(c_mat1) 


# In[ ]:


print(pca.mean_, '\n\n', pca.singular_values_)


# # Atom-centered Coulomb matrix
# 
# The atom-centered Coulomb matrix is similar to Coulomb matrix except it adds cut-off based on a center atom the user chooses. Now the entry of the matrix is:
# 
# $$
# M_{ij}(k) =
#   \begin{cases}
#      \dfrac{1}{2} Z_{i}^{2.4} \cdot f_{ik}^2 & \text{if } i = j \\[5pt]
#      \dfrac{Z_{i}Z_{j}}{\| {\bf R}_{i} - {\bf R}_{j}\|} \cdot f_{ik}f_{jk}f_{ij} 
#      & \text{if } i \neq j,
# \end{cases}
# $$
# 
# where 
# $$
# f_{ij} =
#   \begin{cases}
#      1 & \text{if } \|{\bf R}_{i} - {\bf R}_{j} \| \leq r - \Delta r \\[5pt]
#      \dfrac{1}{2} \left(1 + \cos\Big(\pi \frac{\|{\bf R}_{i} - {\bf R}_{j} \|
#         - r + \Delta r}{\Delta r} \Big)\right)     
#         & \text{if } r - \Delta r < \|{\bf R}_{i} - {\bf R}_{j} \| \leq r - \Delta r 
#         \\[5pt]
#      0 & \text{if } \|{\bf R}_{i} - {\bf R}_{j} \| > r
#   \end{cases}
# $$
# 
# is used as cut-offs similar to Atom-Centered Symmetric Functions (ACSF).

# In[ ]:


for filename in tqdm(filenames):    
    mol = qml.Compound(xyz=folder+filename)
    mol.generate_atomic_coulomb_matrix(size=mol.natoms, 
                                       sorting="distance", 
                                       central_cutoff=6.0, 
                                       central_decay=3, 
                                       interaction_cutoff=3.0, 
                                       interaction_decay=1)
    a = mol.representation


# Now `a` is a 2D array such that its `i`-th row represents `i`-th atom's atom-centered Coulomb matrix. Let us consider the first atom as follows.

# In[ ]:


c_mat2 = qml.representations.vector_to_matrix(a[0])


# In[ ]:


# 17x17 matrix, and it is symmetric
print(a.shape, c_mat2.shape, np.allclose(c_mat2, c_mat2.T))


# In[ ]:


pca = PCA(n_components=min(7,mol.natoms))
pca.fit(c_mat2)  


# In[ ]:


print(pca.mean_, '\n\n', pca.singular_values_)


# # Generation of eigenvalue related features
# 
# The Coulomb potential matrix $C$ can be used as a distance measure to build the graph Laplacian, which is quite simple:
# $$L=D-A,$$
# where where $D$ is the degree matrix and $A$ is the adjacency matrix of the graph. Intuitively speaking, if $C_{ij}$ is big, the interaction between two atoms is strong.
# 
# 

# In[ ]:


filenames = [i + '.xyz' for i in all_molecule_names]


# In[ ]:


def get_laplacian(A, tol=1e-10):
    '''
    input: square Coulomb matrix
    '''
    A = A + tol
    L = np.exp(-1/A)
    L[np.diag_indices_from(L)] = 0
    G = np.diag((L*(L>tol)).sum(axis=1)) - L
    
    return G


# In[ ]:


stats = []
NUM_SINGULAR_VALUES = 5
NUM_PCA_COMPONENTS = 7
CUTOFF = 1e-8 # cut-off for zero interaction
TOL = 1e-8 # tol for zero eigenvalue

for filename in tqdm(filenames):
    
    # Create the compound object mol from the file which happens to be methane
    mol = qml.Compound(xyz=folder+filename)
    natoms = mol.natoms
    mol.generate_atomic_coulomb_matrix(size=natoms, 
                                       sorting='distance', 
                                       central_cutoff=6.0, 
                                       central_decay=3, 
                                       interaction_cutoff=3.0, 
                                       interaction_decay=1)
    ac_c_mat = mol.representation # atom-centered Coulomb matrices collection
    for i in range(natoms): # a loop for every atoms in this molecule
        atomstats = {}
        atomstats['molecule_name'] = filename[:-4]
        atomstats['atom_index'] = i
        
        a = qml.representations.vector_to_matrix(ac_c_mat[i])
        _, eigvals, _ = svd(a)
        atomstats['eigv_min'] = eigvals[np.abs(eigvals)>TOL].min()
        atomstats['eigv_max'] = eigvals.max()
        atomstats['eigv_gap'] = atomstats['eigv_max'] - atomstats['eigv_min']
        
        L = get_laplacian(a, tol=CUTOFF)
        _, eigvals, _ = svd(L)
        
        atomstats['fiedler_eig'] = eigvals[eigvals>TOL][-1]
        atomstats['connectedness'] = (eigvals<TOL).sum()
        
        pca = PCA(n_components = min(NUM_PCA_COMPONENTS, natoms))
        pca.fit(a)
        sv = pca.singular_values_
        atomstats['sv_min'] = sv[sv>TOL][-1]
        atomstats['coulomb_mean'] = pca.mean_[0]
        
        if natoms < NUM_SINGULAR_VALUES: # if there are less than certain atoms/singular values
            sv = np.r_[sv, np.zeros(NUM_SINGULAR_VALUES-natoms)] 
            
        for k in range(NUM_SINGULAR_VALUES):
            atomstats['sv_'+str(k)] = sv[k]
        stats.append(atomstats)


# In[ ]:


struct_eig = pd.DataFrame(stats)


# In[ ]:


struct_eig.tail(10)


# In[ ]:


structures.tail(10)


# In[ ]:


struct_eig.corrwith(struct_eig['eigv_min'])


# In[ ]:


sns.distplot(struct_eig['eigv_min']);


# In[ ]:


sns.distplot(struct_eig['fiedler_eig']);


# In[ ]:


sns.distplot(struct_eig['coulomb_mean']);


# In[ ]:


sns.distplot(struct_eig['sv_1']);


# In[ ]:


struct_eig.to_csv('struct_eigen.csv',index=False)

