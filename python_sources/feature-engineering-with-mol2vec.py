#!/usr/bin/env python
# coding: utf-8

# In this kernel, I extract new features, at molecule level, making use of pre-trained MOL2VEC model.
# 
# The algorith is based on the classical Word2Vec for NLP. I found this beatiful [project on GitHub](https://github.com/samoturk/mol2vec) which easy provides the conversion of molecules into vectors. The pyhton package gives you also all the instruments to train your own vector rapresentation of molecules (here, I make use of a pre-trained model).
# 
# All we need is the SMILE notation for our molecules. I've found a good [kernel](https://www.kaggle.com/roccomeli/easy-xyz-to-smiles-conversion) which has already implemented the creation of SMILE code.
# We need SMILEs because we have to create the Morgan fingerprint. This is just a numeric identifier for a molecule substructure.
# With fingerprints at our disposal we can question MOL2VEC to get the corresponding structure representation of a given molecule.   

# **Install the required packages**

# In[ ]:


get_ipython().system('conda install openbabel -c openbabel -y')


# In[ ]:


import sys
get_ipython().system('conda install --yes --prefix {sys.prefix} -c rdkit rdkit')


# In[ ]:


get_ipython().system('pip install git+https://github.com/samoturk/mol2vec')


# In[ ]:


import numpy as np
import pandas as pd
import os
from tqdm import tqdm_notebook as tqdm

from rdkit import Chem
from mol2vec.features import mol2alt_sentence
import pybel

from gensim.models import Word2Vec


# **Read structure files**

# In[ ]:


file_dir = '../input/champs-scalar-coupling/structures/'
mols_files=os.listdir(file_dir)
mols_index=dict(map(reversed,enumerate(mols_files)))
mol_name = list(mols_index.keys())


# **SMILE creation**

# In[ ]:


# FROM: https://www.kaggle.com/roccomeli/easy-xyz-to-smiles-conversion

def xyz_to_smiles(fname: str) -> str:
    
    mol = next(pybel.readfile("xyz", fname))

    smi = mol.write(format="smi")

    return smi.split()[0].strip()


# In[ ]:


smiles = [xyz_to_smiles(file_dir + i) for i in tqdm(mol_name)]


# In[ ]:


df_smiles = pd.DataFrame({'molecule_name': mol_name, 'smiles': smiles})
df_smiles.head(11)


# **Extract Morgan Fingerprints**

# In[ ]:


sentence = mol2alt_sentence(Chem.MolFromSmiles(df_smiles.smiles[33]), 1)
print('SMILE:', df_smiles.smiles[33])
print(sentence)


# **Load pretrained MOL2VEC**

# In[ ]:


model = Word2Vec.load('../input/mol2vec/model_300dim.pkl')


# In[ ]:


model.wv[sentence[0]]


# **See you on [MEDIUM](https://medium.com/@cerlymarco)**
