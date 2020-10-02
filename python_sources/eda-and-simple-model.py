#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('conda install -y -c conda-forge rdkit')


# In[ ]:


from rdkit import Chem
def canon_smiles(smi):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smi))


# In[ ]:


from pathlib import Path


# In[ ]:


data_path = Path("/kaggle/input/aqsoldb-a-curated-aqueous-solubility-dataset/curated-solubility-dataset.csv")


# In[ ]:


df = pd.read_csv(data_path)


# In[ ]:


df["SMILES"] = df.SMILES.apply(canon_smiles)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.columns


# * IDsource ID of compound (first letter indicates source)
# * NameName of compound
# * InChIThe IUPAC International Chemical Identifier
# * InChIKeyHashed InChI value
# * SMILESSMILES notation of value
# * SolubilityExperimental solubility value (LogS)
# * SDstandard deviation of multiple solubility values (if multiple values exists)
# * Ocurrencesnumber of multiple occurences of compound
# * Groupreliability group see ref paper for details
# * MolWtMolecular weight
# * MolLogPoctonal-water partition coefficient
# * MolMRMolar refractivity
# * HeavyAtomCountNumber of non-H atoms
# * NumHAcceptorsNumber of H acceptors
# * NumHDonorsNumber of H donors
# * NumHeteroatomsNumber of hetero atoms
# * NumRotatableBondsNumber of rotatable bonds
# * NumValenceElectronsNumber of valance electrons
# * NumAromaticRingsNumber of aromatic rings
# * NumSaturatedRingsNumber of saturated rings
# * NumAliphaticRingsNumber of aliphatic rings
# * RingCountNumber of total rings
# * TPSATopological Polar Surface Area
# * LabuteASALabute's Approximate Surface Area
# * BalabanJBalaban's J Index
# * BertzCTA topological complexity index

# In[ ]:


df.info()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.pairplot(df[['Solubility', 'SD',
    'Ocurrences', 'MolWt', 'MolLogP', 'MolMR', 'HeavyAtomCount',
    'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
    'NumValenceElectrons', 'NumAromaticRings', 'NumSaturatedRings',
    'NumAliphaticRings', 'RingCount', 'TPSA', 'LabuteASA', 'BalabanJ',
    'BertzCT']])


# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


y = df.Solubility
x = df[list(k for (k, v) in df.corrwith(df["Solubility"]).items() if (abs(v) >= 0.3 and k != "Solubility")) + ["SMILES"]]


# In[ ]:


y.shape


# In[ ]:


x.shape


# In[ ]:


x.head()


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# # Baseline

# In[ ]:


model = LinearRegression()


# In[ ]:


model.fit(x_train.drop("SMILES", axis=1), y_train)


# In[ ]:


model.score(x_test.drop("SMILES", axis=1), y_test)


# # ECFP

# In[ ]:


from rdkit import Chem
from rdkit.Chem import AllChem


# In[ ]:


def smi2morgan(smi: str, radius: int = 4):
    mol = Chem.MolFromSmiles(smi)
    return list(AllChem.GetMorganFingerprintAsBitVect(mol, radius))


# In[ ]:


features = np.array(list(map(smi2morgan, df.SMILES)))


# In[ ]:


x_mat = np.concatenate((x.drop("SMILES", axis=1).values, features), axis=1)


# In[ ]:


x_mat.shape


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_mat, y, test_size=0.2)


# In[ ]:


model = LinearRegression()
model.fit(x_train[:, :8], y_train)


# In[ ]:


model.score(x_test[:, :8], y_test)


# In[ ]:


model = LinearRegression()
model.fit(x_train, y_train)


# In[ ]:


model.score(x_test, y_test)


# In[ ]:


x.head()


# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


model = RandomForestRegressor()
model.fit(x_train, y_train)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'model.score')


# In[ ]:


model.score(x_test, y_test)


# # Data from doi-103389-fonc202000121
# https://www.frontiersin.org/articles/10.3389/fonc.2020.00121/full#supplementary-material

# In[ ]:


sup_data_path = Path("/kaggle/input/solubilitydoi103389fonc202000121/Supplementary Table S1. 9943 compounds with experimental aqueous solubility values in logarithmic units.xlsx")


# In[ ]:


df_sup = pd.read_excel(sup_data_path)


# In[ ]:


df_sup["SMILES"] = df_sup.SMILES.apply(canon_smiles)


# In[ ]:


df_sup.shape


# In[ ]:


df_sup.head()


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


len(set(df.InChIKey) & set(df_sup.InChIKey))


# Datasets have 1934 equal molecules according to `InChIKey`. Let's check that all molecules are equal.

# In[ ]:


df_merge = df_sup.merge(df, on="InChIKey")


# In[ ]:


df_merge.shape


# In[ ]:


df_merge.SMILES_x


# In[ ]:


df_merge.SMILES_y


# In[ ]:


from rdkit import Chem


# In[ ]:


not_equal_mols = [m for m in zip(df_merge.SMILES_x, df_merge.SMILES_y) if m[0] != m[1]]


# In[ ]:


not_equal_mols


# In[ ]:


len(not_equal_mols)


# Turns out there are 55 not equal molecules
# 
# Let's try substructure match

# In[ ]:


count_not_equal = 0
for m1, m2 in not_equal_mols:
    if Chem.MolFromSmiles(m1) != Chem.MolFromSmiles(m2):
        count_not_equal += 1
        print(m1, m2)


# In[ ]:


count_not_equal


# In[ ]:


m = Chem.MolFromSmiles(not_equal_mols[0][0])
m.GetSubstructMatches(Chem.MolFromSmiles(not_equal_mols[0][1]))
m


# In[ ]:


Chem.MolFromSmiles(not_equal_mols[0][0])


# In[ ]:


Chem.MolFromSmiles(not_equal_mols[0][1])


# In[ ]:


from rdkit.Chem import Draw
img = Draw.MolsToGridImage([Chem.MolFromSmiles(m1).GetSubstructMatches(Chem.MolFromSmiles(m2)) for (m1, m2) in not_equal_mols], molsPerRow=5, subImgSize=(250, 250), maxMols=55,
                           legends=None, useSVG=True)

