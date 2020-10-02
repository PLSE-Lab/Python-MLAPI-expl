#!/usr/bin/env python
# coding: utf-8

# Try improving performance of these kernels https://www.kaggle.com/rio114/coulomb-interaction/notebook and https://www.kaggle.com/brandenkmurray/coulomb-interaction-parallelized/notebook

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


# In[ ]:


structure = pd.read_csv('../input/structures.csv')
traindf = pd.read_csv('../input/train.csv')
testdf = pd.read_csv('../input/test.csv')


# In[ ]:


def compute_all_dist(x):   
    #Apply compute_all_dist2 to each atom 
    return x.apply(compute_all_dist2,axis=1,x2=x)


# In[ ]:


def compute_all_dist2(x,x2):
    # atoms in the molecule which are not the processed one
    notatom = x2[(x2.atom_index != x["atom_index"])].reset_index(drop=True) 
    # processed atom
    atom = x[["x","y","z"]]
    
    # compute distance from to processed atom to each other
    notatom["dist"] = 1/((notatom[["x","y","z"]].values - atom.values)**2).sum(axis=1)
    
    # sort atom per the smallest distance (highest 1/r**2) per group of C/H/N... 
    s = notatom.groupby("atom")["dist"].transform(lambda x : x.sort_values(ascending=False))
    
    # keep only the five nearest atoms per group of C/H/N...
    index0, index1=[],[]
    for i in notatom.atom.unique():
        for j in range(notatom[notatom.atom == i].shape[0]):
            if j < 5:
                index1.append("dist_" + i + "_" + str(j))
            index0.append(j)
    s.index = index0
    s = s[s.index < 5]
    s.index = index1
    
    return s


# In[ ]:


def merge_with_struc(df, structure):
    df = df         .merge(structure,
               left_on=["molecule_name",'atom_index_0'],right_on=["molecule_name","atom_index"]) \
        .merge(structure, left_on=["molecule_name",'atom_index_1'],right_on=["molecule_name","atom_index"]) \
        .drop(["atom_index_x","atom_index_y","atom_x","atom_x"],axis=1) \
        .sort_values(["id"]) \
        .reset_index(drop=True)
    return df


# # First 100 molecules

# In[ ]:


get_ipython().run_cell_magic('time', '', '# 10 times faster than the parallelized kernel\nsmallstruct = pd.concat([structure[structure.molecule_name.isin(structure.molecule_name.unique()[:100])][["molecule_name","atom_index","atom"]],\n                         structure[structure.molecule_name.isin(structure.molecule_name.unique()[:100])].groupby("molecule_name").apply(compute_all_dist)],\n                         axis=1).fillna(0)')


# In[ ]:


smallstruct.head()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'merge_with_struc(traindf,smallstruct).head()')


# # All molecules in the structure.csv file

# In[ ]:


get_ipython().run_cell_magic('time', '', 'structure = \\\n    pd.concat([structure[["molecule_name","atom_index","atom"]],\n               structure.groupby("molecule_name",sort=False).apply(compute_all_dist)], axis=1) \\\n    .fillna(0)')


# In[ ]:


structure.head()


# # Merge with train and test

# In[ ]:


get_ipython().run_cell_magic('time', '', 'traindf = merge_with_struc(traindf,structure)\ntestdf = merge_with_struc(testdf,structure)')


# In[ ]:


traindf.head()


# In[ ]:




