#!/usr/bin/env python
# coding: utf-8

# **Local Many-body Tensor Representation**
# 

# In[ ]:


get_ipython().system('pip install dscribe')


# In[ ]:


get_ipython().system('pip install --upgrade --user ase')


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
structures = pd.read_csv(f'../input/structures.csv')


# In[ ]:


from dscribe.descriptors import LMBTR
from ase import Atoms


# In[ ]:


lmbtr = LMBTR(
    species=["C", "H", "N"],
    k2 = {
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": 0.1, "max": 2, "sigma": 0.1, "n": 10},
        "weighting": {"function": "exp", "scale": 0.75, "cutoff": 1e-2}
    },
    k3 = {
        "geometry": {"function": "angle"},
        "grid": {"min": 0, "max": 180, "n": 10, "sigma": 2},
        "weighting": {"function": "exp", "scale": 0.5, "cutoff": 1e-3},
    },
    periodic=False,
    normalization="l2_each",
)


# In[ ]:


# Randomly select a molecule
m1 = structures.loc[structures['molecule_name']=='dsgdb9nsd_133884']
a1 = Atoms(''.join(m1['atom'].values), positions=m1[['x','y','z']].values)
display(a1)
display(m1)


# In[ ]:


# Create LMBTR output for the system
lmbtr_a1 = lmbtr.create(a1, positions=[0,1,2])

print(lmbtr_a1)
print(lmbtr_a1.shape)

