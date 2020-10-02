#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook we are going to create 3 different files ready to perform EDA in the **fastest way**:
# 
#  - **train.msg** : Contains the train.csv data + structures + distance metric
#  - **test.msg** : Contains the test.csv data + structures + distance metric
#  - **complete.msg** : Contains the train.csv + all the data from the complementary files + distance metric
#  
# The files are saved in [msgpack format](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_msgpack.html). This allow us to save/load the dataframes very fast (and the dtypes are saved too, so no conversion is needed when load).
# 
# Some people are afraid to use msgpack, because in the pandas docs it is stated that the function is experimental. After one year using mspack, I can ensure that no problem has arrise to me. Anyway you can use csv if you like it more.
# 
# 

# In[ ]:


import numpy as np
import pandas as pd

from IPython.display import display

data_path = '../input'


# In[ ]:


get_ipython().system('ls -lSh $data_path/*.csv')


# In[ ]:


files_names = get_ipython().getoutput('ls $data_path/*.csv')


# In[ ]:


data_dict = {}

for name in files_names:
    data_dict[name.split('/')[-1][:-4]] = pd.read_csv(name)


# # Lets see how the files looks like

# In[ ]:


for k in data_dict.keys():
    display(k)
    display(data_dict[k].head())


# # lets join this data in single dataframe

# In[ ]:


get_ipython().run_cell_magic('time', '', "df_complete = data_dict['train'].copy()\ndf_complete = df_complete.join(data_dict['potential_energy'].set_index('molecule_name'), on='molecule_name')\ndf_complete = df_complete.join(data_dict['dipole_moments'].set_index('molecule_name'), on='molecule_name', lsuffix='dipole_moments_')\ndf_complete = df_complete.join(data_dict['magnetic_shielding_tensors'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_0'], lsuffix='_atom0')\ndf_complete = df_complete.join(data_dict['magnetic_shielding_tensors'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_1'], lsuffix='_atom1')\ndf_complete = df_complete.join(data_dict['mulliken_charges'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_0'], lsuffix='_atom0')\ndf_complete = df_complete.join(data_dict['mulliken_charges'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_1'], lsuffix='_atom1')\ndf_complete = df_complete.join(data_dict['scalar_coupling_contributions'].set_index(['molecule_name', 'atom_index_0', 'atom_index_1']), on=['molecule_name', 'atom_index_0', 'atom_index_1'], rsuffix='_scc')\ndf_complete = df_complete.join(data_dict['structures'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_0'], lsuffix='_atom0_structure')\ndf_complete = df_complete.join(data_dict['structures'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_1'], lsuffix='_atom1_structure')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_train = data_dict['train'].copy()\ndf_train = df_train.join(data_dict['structures'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_0'], lsuffix='_atom0_structure')\ndf_train = df_train.join(data_dict['structures'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_1'], lsuffix='_atom1_structure')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_test = data_dict['test'].copy()\ndf_test = df_test.join(data_dict['structures'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_0'], lsuffix='_atom0_structure')\ndf_test = df_test.join(data_dict['structures'].set_index(['molecule_name', 'atom_index']), on=['molecule_name', 'atom_index_1'], lsuffix='_atom1_structure')")


# # calculate distance

# In[ ]:


get_ipython().run_cell_magic('time', '', "for df in [df_complete, df_train, df_test]:    \n    distance_foo = np.linalg.norm(df[['x_atom1_structure', 'y_atom1_structure', 'z_atom1_structure']].values - df[['x', 'y', 'z']].values, axis=1)\n    df['distance'] = distance_foo")


# # save the data 

# In[ ]:


df_complete.to_msgpack('./complete.msg')
df_train.to_msgpack('./train.msg')
df_test.to_msgpack('./test.msg')


# # Have a nice EDA ;)
