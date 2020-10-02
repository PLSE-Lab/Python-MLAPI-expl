#!/usr/bin/env python
# coding: utf-8

# # Initial EDA - Protein Data Set 
# 
# This data set hase been downloaded from RCSB PDB and kindly acknowledged.
# original data can be down loaded from  http://www.rcsb.org/pdb/
# 
#  The PDB archive is a repository of atomic coordinates and other information describing proteins and other important biological macromolecules. Structural biologists use methods such as X-ray crystallography, NMR spectroscopy, and cryo-electron microscopy to determine the location of each atom relative to each other in the molecule. They then deposit this information, which is then annotated and publicly released into the archive by the wwPDB.
# 
# The constantly-growing PDB is a reflection of the research that is happening in laboratories across the world. This can make it both exciting and challenging to use the database in research and education. Structures are available for many of the proteins and nucleic acids involved in the central processes of life, so you can go to the PDB archive to find structures for ribosomes, oncogenes, drug targets, and even whole viruses. However, it can be a challenge to find the information that you need, since the PDB archives so many different structures. You will often find multiple structures for a given molecule, or partial structures, or structures that have been modified or inactivated from their native form. 
# 
# There are two data sets. Both are arranged on "structureId" of the protein
# 
# 1. pdb_data_no_dups.csv :- Protein data set deatils of classification, extraction methods, etc. Containing 141401 instances and 14 attributes.
# 2. data_seq :- Protein sequence information. Containing 467304 instances and 5 attributes.
# 
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# ## Load Data Sets
# 
# Let us load the data sets...

# In[ ]:


data = pd.read_csv('../input/pdb_data_no_dups.csv')
data_seq = pd.read_csv('../input/pdb_data_seq.csv')


# ## Descriptive Statistics
# 
# We can see how data sets are organized by seeing the initial row of each data sets...

# In[ ]:


print(data.shape) 
data.head()


# In[ ]:


print(data_seq.shape)
data_seq.head()


# In[ ]:


data.describe(include='all').T


# In[ ]:


data_seq.describe(include='all').T


# In[ ]:


data.columns


# In[ ]:


year_df = data.groupby(['publicationYear']).count()['structureId'].reset_index()
year_df = year_df[year_df['publicationYear']!=2017]


# In[ ]:


plt.figure(figsize=(10,7))
plt.plot(year_df['publicationYear'], year_df['structureId'])


plt.xlabel('Proteins entered into PDB')
plt.ylabel('Publication Year')
plt.title('Growth of the PDB over time')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




