#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In this notebook let's us see how to use biopandas for analyzing pdb files easily

# # BioPandas
# 
# Now computational biologists can analyse the PDB(Protein Data Bank) file format in python. 
# Working with molecular structures of biological macromolecules (from PDB and MOL2 files) in pandas DataFrames is what BioPandas is all about!

# # Install biopandas

# In[ ]:


get_ipython().run_line_magic('pip', 'install biopandas')


# # Analysis of Structural basis for human coronavirus attachment to sialic acid receptors
# 
# data source: [corona virus db](https://www.rcsb.org/structure/6NZK)

# In[ ]:


from biopandas.pdb import PandasPdb

ppdb_df =  PandasPdb().read_pdb('/kaggle/input/6nzk.pdb')


# In[ ]:


type(ppdb_df.df)


# In[ ]:


ppdb_df.df.keys()


# # What is PDB?
# 
# **PDB file format**
# 
# In the PDB data file format for macromolecular models, each atom is designated either ATOM or HETATM (which stands for hetero atom).
# 
# ATOM is reserved for atoms in standard residues of protein, DNA or RNA.
# 
# HETATM is applied to non-standard residues of protein, DNA or RNA, as well as atoms in other kinds of groups, such as carbohydrates, substrates, ligands, solvent, and metal ions.

# In[ ]:


atom_df = ppdb_df.df['ATOM']
atom_df.head()


# In[ ]:


het_df = ppdb_df.df['HETATM']
het_df.head()


# # **Visualize the b_factor**
# 
# what is b_factor?
# 
# The B-factor describes the displacement of the atomic positions from an average (mean) value (mean-square displacement).
# 
# The core of the molecule usually has low B-factors, due to tight packing of the side chains (enzyme active sites are usually located there). The values of the B-factors are **normally between 15 to 30** (sq. Angstroms), but often **much higher than 30 for flexible regions**.

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


atom_df['b_factor'].plot(kind='hist')
plt.title('B-Factor of human coronavirus')
plt.xlabel('B-Factor')


# In[ ]:


atom_df.element_symbol.unique()


# In[ ]:


atom_df['element_symbol'].value_counts().plot(kind='bar')
plt.title('Element symbol distribution')
plt.ylabel('Count')
plt.xlabel('Element symbol')


# In[ ]:


atom_df['atom_name'].value_counts().plot(kind='bar', figsize=(10,8))
plt.title('Atom name distribution')
plt.ylabel('Count')
plt.xlabel('atom_name symbol')


# # Analysis of COVID-19 virus
# 
# dataset source: [COVID-19](https://www.rcsb.org/structure/6lu7)

# In[ ]:


ppdb_df =  PandasPdb().read_pdb('/kaggle/input/6lu7.pdb')


# In[ ]:


catom_df = ppdb_df.df['ATOM']
chtm_df = ppdb_df.df['HETATM']


# In[ ]:


catom_df.head()


# In[ ]:


catom_df['b_factor'].plot(kind='hist')
plt.title('B-Factor of COVID-19')
plt.xlabel('B-Factor')


# In[ ]:


catom_df['element_symbol'].value_counts().plot(kind='bar')
plt.title('Element symbol distribution')
plt.ylabel('Count')
plt.xlabel('Element symbol')


# In[ ]:


catom_df['atom_name'].value_counts().plot(kind='bar', figsize=(10,8))
plt.title('Atom name distribution')
plt.ylabel('Count')
plt.xlabel('atom_name symbol')


# # Insights from the above findings
# 
#  It is observed that **COVID-19 has high b_factor** which means **it is not tightly packed**
