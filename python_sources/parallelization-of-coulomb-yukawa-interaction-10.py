#!/usr/bin/env python
# coding: utf-8

# # Summary
# 
# `pandas` only uses 1 CPU core to process data, which is not very economical considering there are 4 cores at our disposal.
# 
# ### Reference:
# 
# * [coulomb_interaction - speed up!](https://www.kaggle.com/rio114/coulomb-interaction/notebook)
# * [coulomb_interaction - Parallelized](https://www.kaggle.com/brandenkmurray/coulomb-interaction-parallelized/notebook)
# * [Coulomb interaction - High perf, no loop (almost)](https://www.kaggle.com/daijin12/coulomb-interaction-high-perf-no-loop-almost)

# In[ ]:


import numpy as np
from tqdm import tqdm_notebook as tqdm
import pandas as pd 
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


structures = pd.read_csv('../input/structures.csv')


# In[ ]:


# nuclear_mass = {'H':1.008, 'C':12.011, 'N':14.007, 'O':15.999, 'F':18.998}
# structures['nuclear_mass'] = [nuclear_mass[x] for x in structures['atom'].values]

nuclear_charge = {'H':1.0, 'C':6, 'N':7, 'O':8, 'F':9}
structures['nuclear_charge'] = [nuclear_charge[x] for x in structures['atom'].values]


# In[ ]:


def compute_all_yukawa(x):   
    #Apply compute_all_dist2 to each atom 
    return x.apply(compute_yukawa_matrix,axis=1,x2=x)

def compute_yukawa_matrix(x,x2):
    # atoms in the molecule which are not the processed one
    notatom = x2[(x2.atom_index != x["atom_index"])].reset_index(drop=True) 
    # processed atom
    atom = x[["x","y","z"]]
    charge = x[['nuclear_charge']]
    
    # compute distance from to processed atom to each other
    notatom['dist'] = ((notatom[["x","y","z"]].values - atom.values)**2).sum(axis=1)
    notatom['dist'] = np.sqrt(notatom['dist'].astype(np.float32))
    notatom['dist'] = charge.values*notatom[['nuclear_charge']].values.reshape(-1)                    *np.exp(-notatom['dist']/2/notatom['dist'].max())/notatom['dist']

    # sort atom per the smallest distance (highest 1/r**2) per group of C/H/N... 
    s = notatom.groupby("atom")["dist"].transform(lambda x : x.sort_values(ascending=False))
    
    # keep only the five nearest atoms per group of C/H/N...
    index0, index1=[],[]
    for i in notatom.atom.unique():
        for j in range(notatom[notatom.atom == i].shape[0]):
            if j < 10:
                index1.append("dist_" + i + "_" + str(j))
            index0.append(j)
    s.index = index0
    s = s[s.index < 10]
    s.index = index1
    
    return s


# ## Benchmark using first 100 molecules

# In[ ]:


small_idx = structures.molecule_name.isin(structures.molecule_name.unique()[:100])
_smallstruct = structures[small_idx]


# Using the current [fastest way](https://www.kaggle.com/daijin12/coulomb-interaction-high-perf-no-loop-almost) to compute takes about 20 seconds.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'smallstruct1 = _smallstruct.groupby("molecule_name").apply(compute_all_yukawa)')


# In[ ]:


smallstruct1.head(10)


# ## Multiprocessing
# 
# The following approach makes use of the `groupby` to get an iterator, so that we can use `multiprocessing` which saves more than 50% of the computation time.
# 
# Reference: [Parallel operations over a Pandas DF](https://www.kaggle.com/gvyshnya/parallel-operations-over-a-pandas-df)

# In[ ]:


get_ipython().run_cell_magic('time', '', "chunk_iter = _smallstruct.groupby(['molecule_name'])\npool = mp.Pool(4) # use 4 processes\n\nfunclist = []\nfor df in tqdm(chunk_iter):\n    # process each data frame\n    f = pool.apply_async(compute_all_yukawa,[df[1]])\n    funclist.append(f)\n\nresult = []\nfor f in tqdm(funclist):\n    result.append(f.get(timeout=120)) # timeout in 120 seconds = 2 mins\n\n# combine chunks with transformed data into a single structure file\nsmallstruct2 = pd.concat(result)")


# In[ ]:


smallstruct2.head(10)


# Just to make sure we are getting the same thing by two methods.

# In[ ]:


np.allclose(smallstruct2.fillna(0), smallstruct1.fillna(0))


# ## Compute Yukawa interaction for all molecules
# 
# Without the parallelization, it takes about 11 hours to run.

# In[ ]:


chunk_iter = structures.groupby(['molecule_name'])
pool = mp.Pool(4) # use 4 CPU cores

funclist = []
for df in tqdm(chunk_iter):
    # process each data frame
    f = pool.apply_async(compute_all_yukawa,[df[1]])
    funclist.append(f)

result = []
for f in tqdm(funclist):
    result.append(f.get(timeout=180)) # timeout in 180 seconds

# combine chunks with transformed data into a single training set
structures_yukawa = pd.concat(result)


# In[ ]:


structures_yukawa.to_csv('structures_yukawa.csv',index=False)

