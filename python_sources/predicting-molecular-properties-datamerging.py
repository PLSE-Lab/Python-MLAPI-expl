#!/usr/bin/env python
# coding: utf-8

# **Merging data**
# 
# Aim of this this kernel to merge all available data into single dataframe.
# Data from all files are merged to single file for train and test.
# Final **train.csv** and **test.csv** are available under output.
# These files can be directly used further feature engineering.
# 
# >  **File_name-------------------------------Data_frequency-------------------Data_coverage**
# > 1. potential_energy.csv----------------------per molecule---------------------Only train data
# > 1. dipole_moments.csv------------------------per modecule---------------------Only train data
# > 1. mulliken_charges.csv----------------------per atom-------------------------Only train data
# > 1. magnetic_shielding_tensors.csv------------per atom-------------------------Only train data
# > 1. structures.csv----------------------------per atom-------------------------both train + test data
# > 1. scalar_coupling_contributions.csv---------per atom pair--------------------Only train data
# > 1. train.csv---------------------------------per atom pair--------------------
# > 1. test.csv----------------------------------per atom pair--------------------

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings
warnings.filterwarnings('ignore')

import os
print(os.listdir("../input"))


# In[ ]:


train_dataset = pd.read_csv('../input/train.csv')


# In[ ]:


train_dataset.head(10)


# In[ ]:


scalar_coupling_contributions = pd.read_csv('../input/scalar_coupling_contributions.csv')


# In[ ]:


train_dataset = pd.concat([train_dataset, scalar_coupling_contributions[['fc', 'sd', 'pso', 'dso']]], axis=1)


# In[ ]:


train_dataset.head(10)


# In[ ]:


structures = pd.read_csv('../input/structures.csv')


# In[ ]:


train_dataset_0 = train_dataset.merge(structures, 
                                    right_on=['molecule_name', 'atom_index'], 
                                    left_on=['molecule_name', 'atom_index_1'])
train_dataset = train_dataset_0.merge(structures, 
                                    right_on=['molecule_name', 'atom_index'], 
                                    left_on=['molecule_name', 'atom_index_0'], 
                                     suffixes=('_atom_1', '_atom_0'))


# In[ ]:


train_dataset.drop(['atom_index_atom_0',  'atom_index_atom_1'], axis=1, inplace=True)


# In[ ]:


train_arranged_columns = ['id', 'molecule_name', 'type',
           'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0', 
           'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1',
           'fc', 'sd', 'pso', 'dso', 'scalar_coupling_constant']


# In[ ]:


train_dataset = train_dataset[train_arranged_columns]
train_dataset.head(20)


# In[ ]:


test_dataset = pd.read_csv('../input/test.csv')


# In[ ]:


test_dataset_0 = test_dataset.merge(structures, 
                                    right_on=['molecule_name', 'atom_index'], 
                                    left_on=['molecule_name', 'atom_index_1'])
test_dataset = test_dataset_0.merge(structures, 
                                    right_on=['molecule_name', 'atom_index'], 
                                    left_on=['molecule_name', 'atom_index_0'], 
                                     suffixes=('_atom_1', '_atom_0'))

test_dataset.drop(['atom_index_atom_0',  'atom_index_atom_1'], axis=1, inplace=True)


# In[ ]:


test_arranged_columns = ['id', 'molecule_name', 'type',
           'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0', 
           'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1']


# In[ ]:


test_dataset = test_dataset[test_arranged_columns]
test_dataset.head(20)


# In[ ]:


magnetic_shielding_tensors = pd.read_csv('../input/magnetic_shielding_tensors.csv')


# In[ ]:


train_dataset_0 = train_dataset.merge(magnetic_shielding_tensors, 
                                    right_on=['molecule_name', 'atom_index'], 
                                    left_on=['molecule_name', 'atom_index_1'])
train_dataset = train_dataset_0.merge(magnetic_shielding_tensors, 
                                    right_on=['molecule_name', 'atom_index'], 
                                    left_on=['molecule_name', 'atom_index_0'], 
                                     suffixes=('_atom_1', '_atom_0'))
train_dataset.drop(['atom_index_atom_0',  'atom_index_atom_1'], axis=1, inplace=True)


# In[ ]:


train_arranged_columns = ['id', 'molecule_name', 'type', 
                          'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',  
                          'XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                          'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1', 
                          'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1',
                          'fc', 'sd', 'pso', 'dso', 'scalar_coupling_constant']


# In[ ]:


train_dataset = train_dataset[train_arranged_columns]
train_dataset.head(20)


# In[ ]:


mulliken_charges = pd.read_csv('../input/mulliken_charges.csv')


# In[ ]:


train_dataset_0 = train_dataset.merge(mulliken_charges, 
                                    right_on=['molecule_name', 'atom_index'], 
                                    left_on=['molecule_name', 'atom_index_1'])
train_dataset = train_dataset_0.merge(mulliken_charges, 
                                    right_on=['molecule_name', 'atom_index'], 
                                    left_on=['molecule_name', 'atom_index_0'], 
                                     suffixes=('_atom_1', '_atom_0'))
train_dataset.drop(['atom_index_atom_0',  'atom_index_atom_1'], axis=1, inplace=True)


# In[ ]:


train_arranged_columns = ['id', 'molecule_name', 'type', 
                          'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',  
                          'XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                          'mulliken_charge_atom_0',
                          'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1', 
                          'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1',
                          'mulliken_charge_atom_1',
                          'fc', 'sd', 'pso', 'dso', 'scalar_coupling_constant']


# In[ ]:


train_dataset = train_dataset[train_arranged_columns]
train_dataset.head(20)


# In[ ]:


dipole_moments = pd.read_csv('../input/dipole_moments.csv')


# In[ ]:


train_dataset = train_dataset.merge(dipole_moments, 
                                    right_on='molecule_name', 
                                    left_on='molecule_name')


# In[ ]:


train_arranged_columns = ['id', 'molecule_name', 'type', 'X', 'Y', 'Z',
                          'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',  
                          'XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                          'mulliken_charge_atom_0',
                          'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1', 
                          'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1',
                          'mulliken_charge_atom_1',
                          'fc', 'sd', 'pso', 'dso', 'scalar_coupling_constant']


# In[ ]:


train_dataset = train_dataset[train_arranged_columns]
train_dataset.head(20)


# In[ ]:


potential_energy = pd.read_csv('../input/potential_energy.csv')


# In[ ]:


train_dataset = train_dataset.merge(potential_energy, 
                                    right_on='molecule_name', 
                                    left_on='molecule_name')


# In[ ]:


train_arranged_columns = ['id', 'molecule_name', 'type', 'X', 'Y', 'Z', 'potential_energy',
                          'atom_index_0', 'atom_atom_0', 'x_atom_0', 'y_atom_0', 'z_atom_0',  
                          'XX_atom_0', 'YX_atom_0', 'ZX_atom_0', 'XY_atom_0', 'YY_atom_0', 'ZY_atom_0', 'XZ_atom_0', 'YZ_atom_0', 'ZZ_atom_0',
                          'mulliken_charge_atom_0',
                          'atom_index_1', 'atom_atom_1', 'x_atom_1', 'y_atom_1', 'z_atom_1', 
                          'XX_atom_1', 'YX_atom_1', 'ZX_atom_1', 'XY_atom_1', 'YY_atom_1', 'ZY_atom_1', 'XZ_atom_1', 'YZ_atom_1', 'ZZ_atom_1',
                          'mulliken_charge_atom_1',
                          'fc', 'sd', 'pso', 'dso', 'scalar_coupling_constant']


# In[ ]:


train_dataset = train_dataset[train_arranged_columns]
train_dataset.head(20)


# In[ ]:


train_dataset.to_csv('train.csv', index=False)
test_dataset.to_csv('test.csv', index=False)

