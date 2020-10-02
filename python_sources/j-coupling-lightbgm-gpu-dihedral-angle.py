#!/usr/bin/env python
# coding: utf-8

# #### In this kernel, I mostly use ideas from other kernels. I have added a dihedral angle calculation which gives some boost.
# 
# (I updated the kernel regarding the calculations for the dihedral angle. In previous kernel, even though I was using GPU, the use of pandas apply
# function was removing any advantage of GPU, so it was not fast. Now, I refactored the code and it the calcuation is fast without using any GPU.)

# In[ ]:


get_ipython().system('rm -r /opt/conda/lib/python3.6/site-packages/lightgbm')
get_ipython().system('git clone --recursive https://github.com/Microsoft/LightGBM')


# In[ ]:


get_ipython().system('apt-get install -y -qq libboost-all-dev')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd LightGBM\nrm -r build\nmkdir build\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)')


# In[ ]:


get_ipython().system('cd LightGBM/python-package/;python3 setup.py install --precompile')


# In[ ]:


get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
get_ipython().system('rm -r LightGBM')


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import cupy as cp
from sklearn.model_selection import train_test_split,KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm_notebook as tqdm

sns.set()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train = pd.read_csv('../input/train.csv', header=0)


# In[ ]:


df_train.head()


# In[ ]:


fig, ax = plt.subplots(figsize=(14, 12))
sns.boxplot(x='type', y='scalar_coupling_constant', data=df_train, ax=ax)
plt.show()


# We can see that if we want we can divide by type:
# 
# 1st: 1JHC
# 
# 2nd: 1JHN
# 
# 3rd: 2JHC, 2JHN, 3JHH, 3JHN, 3JHC
# 
# 4th: 2JHH

# In[ ]:


df_test = pd.read_csv('../input/test.csv', header=0)
df_test.head()


# In[ ]:


structures = pd.read_csv('../input/structures.csv')


# In[ ]:


structures.head()


# ### Bond calculations
# 
# (Thanks to https://www.kaggle.com/adrianoavelar/bond-calculation-lb-0-82?scriptVersionId=15911797)

# In[ ]:


atomic_radius = {'H':0.38, 'C':0.77, 'N':0.75, 'O':0.73, 'F':0.71} # Without fudge factor

fudge_factor = 0.05
atomic_radius = {k:v + fudge_factor for k,v in atomic_radius.items()}

electronegativity = {'H':2.2, 'C':2.55, 'N':3.04, 'O':3.44, 'F':3.98}

#structures = pd.read_csv(structures, dtype={'atom_index':np.int8})

atoms = structures['atom'].values
atoms_en = [electronegativity[x] for x in tqdm(atoms)]
atoms_rad = [atomic_radius[x] for x in tqdm(atoms)]

structures['EN'] = atoms_en
structures['rad'] = atoms_rad


# In[ ]:


structures.head()


# In[ ]:


i_atom = structures['atom_index'].values
p = structures[['x', 'y', 'z']].values
p_compare = p
m = structures['molecule_name'].values
m_compare = m
r = structures['rad'].values
r_compare = r

source_row = np.arange(len(structures))
max_atoms = 28

bonds = np.zeros((len(structures)+1, max_atoms+1), dtype=np.int8)
bond_dists = np.zeros((len(structures)+1, max_atoms+1), dtype=np.float32)


for i in tqdm(range(max_atoms-1)):
    p_compare = np.roll(p_compare, -1, axis=0)
    m_compare = np.roll(m_compare, -1, axis=0)
    r_compare = np.roll(r_compare, -1, axis=0)
    
    mask = np.where(m == m_compare, 1, 0) #Are we still comparing atoms in the same molecule?
    dists = np.linalg.norm(p - p_compare, axis=1) * mask
    r_bond = r + r_compare
    
    bond = np.where(np.logical_and(dists > 0.0001, dists < r_bond), 1, 0)
    
    source_row = source_row
    target_row = source_row + i + 1 #Note: Will be out of bounds of bonds array for some values of i
    target_row = np.where(np.logical_or(target_row > len(structures), mask==0), len(structures), target_row) #If invalid target, write to dummy row
    
    source_atom = i_atom
    target_atom = i_atom + i + 1 #Note: Will be out of bounds of bonds array for some values of i
    target_atom = np.where(np.logical_or(target_atom > max_atoms, mask==0), max_atoms, target_atom) #If invalid target, write to dummy col
    
    bonds[(source_row, target_atom)] = bond
    bonds[(target_row, source_atom)] = bond
    bond_dists[(source_row, target_atom)] = dists
    bond_dists[(target_row, source_atom)] = dists

bonds = np.delete(bonds, axis=0, obj=-1) #Delete dummy row
bonds = np.delete(bonds, axis=1, obj=-1) #Delete dummy col
bond_dists = np.delete(bond_dists, axis=0, obj=-1) #Delete dummy row
bond_dists = np.delete(bond_dists, axis=1, obj=-1) #Delete dummy col


bonds_numeric = [[i for i,x in enumerate(row) if x] for row in tqdm(bonds)]
bond_lengths = [[dist for i,dist in enumerate(row) if i in bonds_numeric[j]] for j,row in enumerate(tqdm(bond_dists))]
bond_lengths_mean = [ np.mean(x) for x in bond_lengths]
n_bonds = [len(x) for x in bonds_numeric]

bond_data = {'n_bonds':n_bonds, 'bond_lengths_mean': bond_lengths_mean }
bond_df = pd.DataFrame(bond_data)
structures = structures.join(bond_df)


# In[ ]:


structures.head()


# ### Dihedral angle
# 
# We are going to compute the dihedral angle. This angle is derived from the first 4 atoms in the molecule.
# 
# The calculations are done on the GPU by using cupy package.

# In[ ]:


# https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
def dihedral_angle(data): 
        
    vals = np.array(data[:, 3:6], dtype=np.float64)
    mol_names = np.array(data[:, 0], dtype=np.str)
 
    result = np.zeros((data.shape[0], 2), dtype=object)
    # use every 4 rows to compute the dihedral angle
    for idx in range(0, vals.shape[0] - 4, 4):

        a0 = vals[idx]
        a1 = vals[idx + 1]
        a2 = vals[idx + 2]
        a3 = vals[idx + 3]
        
        b0 = a0 - a1
        b1 = a2 - a1
        b2 = a3 - a2
        
        # normalize b1 so that it does not influence magnitude of vector
        # rejections that come next
        b1 /= np.linalg.norm(b1)
    
        # vector rejections
        # v = projection of b0 onto plane perpendicular to b1
        #   = b0 minus component that aligns with b1
        # w = projection of b2 onto plane perpendicular to b1
        #   = b2 minus component that aligns with b1

        v = b0 - np.dot(b0, b1) * b1
        w = b2 - np.dot(b2, b1) * b1

        # angle between v and w in a plane is the torsion angle
        # v and w may not be normalized but that's fine since tan is y/x
        x = np.dot(v, w)
        y = np.dot(np.cross(b1, v), w)
       
        # We want all 4 first rows for every molecule to have the same value
        # (in order to have the same length as the dataframe)
        result[idx:idx + 4] = [mol_names[idx], np.degrees(np.arctan2(y, x))]
        
    return result


# In[ ]:


from datetime import datetime
startTime = datetime.now()
dihedral = dihedral_angle(structures[structures.groupby('molecule_name')['atom_index'].transform('count').ge(4)].groupby('molecule_name').head(4).values)
print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - startTime))


# In[ ]:


themap = {k:v for k, v in dihedral if k}


# In[ ]:


structures['dihedral'] = structures['molecule_name'].map(themap)


# In[ ]:


structures.head()


# We can see that for every molecule we leave the same value for the dihedral angle.

# In[ ]:


def map_atom_info(df, atom_idx):
    #df = pd.merge(df, structures[['molecule_name', 'atom_index', 'x', 'y', 'z', 'EN', 'rad', 'n_bonds', 'bond_lengths_mean', 'dihedral']], how = 'left',
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


# In[ ]:


train = map_atom_info(df_train, 0)
train = map_atom_info(train, 1)


# In[ ]:


test = map_atom_info(df_test, 0)
test = map_atom_info(test, 1)


# In[ ]:


train.head()


# ### Compute some distances

# In[ ]:


# Euclidean Distance
def dist(a, b, ax=1):
    return cp.linalg.norm(a - b, axis=ax)


# In[ ]:


train_atom_0 = cp.asarray(train[['x_0', 'y_0', 'z_0']].values)
train_atom_1 = cp.asarray(train[['x_1', 'y_1', 'z_1']].values)

train['dist'] = dist(train_atom_1, train_atom_0).get()
train['dist_x'] = dist( cp.asarray(train[['x_0']].values),  cp.asarray(train[['x_1']].values)).get()
train['dist_y'] = dist( cp.asarray(train[['y_0']].values),  cp.asarray(train[['y_1']].values)).get()
train['dist_z'] = dist( cp.asarray(train[['z_0']].values),  cp.asarray(train[['z_1']].values)).get()


# In[ ]:


test_atom_0 = cp.asarray(test[['x_0', 'y_0', 'z_0']].values)
test_atom_1 = cp.asarray(test[['x_1', 'y_1', 'z_1']].values)

test['dist'] = dist(test_atom_1, test_atom_0).get()
test['dist_x'] = dist( cp.asarray(test[['x_0']].values),  cp.asarray(test[['x_1']].values)).get()
test['dist_y'] = dist( cp.asarray(test[['y_0']].values),  cp.asarray(test[['y_1']].values)).get()
test['dist_z'] = dist( cp.asarray(test[['z_0']].values),  cp.asarray(test[['z_1']].values)).get()


# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# ### Create some features
# 
# (https://www.kaggle.com/artgor/brute-force-feature-engineering)

# In[ ]:


'''def create_features(df):
    #df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
    #df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
    #df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
    #df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    #df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    #df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')
    
    df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']
    #df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    #df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']
    df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')
    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
    #df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']
   # df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']
    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
    df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']
    df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']
    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    #df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']
    #df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']
    df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
    df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']
    df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']
    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    #df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']
    #df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']
    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
    #df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']
    #df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']
    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    #df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']
    #df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']
    df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']
    df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']
    #df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')
    #df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')
    #df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']
    #df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']
    #df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')
    #df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']
    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']
    df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']
    #df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')
    #df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')
    df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')
    df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']

   
    df = reduce_mem_usage(df)
    return df'''
def create_features(df):
    #df['molecule_couples'] = df.groupby('molecule_name')['id'].transform('count')
    #df['molecule_dist_mean'] = df.groupby('molecule_name')['dist'].transform('mean')
    df['molecule_dist_min'] = df.groupby('molecule_name')['dist'].transform('min')
    df['molecule_dist_max'] = df.groupby('molecule_name')['dist'].transform('max')
    #df['atom_0_couples_count'] = df.groupby(['molecule_name', 'atom_index_0'])['id'].transform('count')
    #df['atom_1_couples_count'] = df.groupby(['molecule_name', 'atom_index_1'])['id'].transform('count')
    
    #df[f'molecule_atom_index_0_x_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['x_1'].transform('std')
    df[f'molecule_atom_index_0_y_1_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('mean')
    df[f'molecule_atom_index_0_y_1_mean_diff'] = df[f'molecule_atom_index_0_y_1_mean'] - df['y_1']
    #df[f'molecule_atom_index_0_y_1_mean_div'] = df[f'molecule_atom_index_0_y_1_mean'] / df['y_1']
    #df[f'molecule_atom_index_0_y_1_max'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('max')
    #df[f'molecule_atom_index_0_y_1_max_diff'] = df[f'molecule_atom_index_0_y_1_max'] - df['y_1']
    #df[f'molecule_atom_index_0_y_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['y_1'].transform('std')
    #df[f'molecule_atom_index_0_z_1_std'] = df.groupby(['molecule_name', 'atom_index_0'])['z_1'].transform('std')
    df[f'molecule_atom_index_0_dist_mean'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('mean')
    df[f'molecule_atom_index_0_dist_mean_diff'] = df[f'molecule_atom_index_0_dist_mean'] - df['dist']
    #df[f'molecule_atom_index_0_dist_mean_div'] = df[f'molecule_atom_index_0_dist_mean'] / df['dist']
    df[f'molecule_atom_index_0_dist_max'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('max')
    #df[f'molecule_atom_index_0_dist_max_diff'] = df[f'molecule_atom_index_0_dist_max'] - df['dist']
    #df[f'molecule_atom_index_0_dist_max_div'] = df[f'molecule_atom_index_0_dist_max'] / df['dist']
    df[f'molecule_atom_index_0_dist_min'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('min')
    #df[f'molecule_atom_index_0_dist_min_diff'] = df[f'molecule_atom_index_0_dist_min'] - df['dist']
    df[f'molecule_atom_index_0_dist_min_div'] = df[f'molecule_atom_index_0_dist_min'] / df['dist']
   # df[f'molecule_atom_index_0_dist_std'] = df.groupby(['molecule_name', 'atom_index_0'])['dist'].transform('std')
    #df[f'molecule_atom_index_0_dist_std_diff'] = df[f'molecule_atom_index_0_dist_std'] - df['dist']
    #df[f'molecule_atom_index_0_dist_std_div'] = df[f'molecule_atom_index_0_dist_std'] / df['dist']
    df[f'molecule_atom_index_1_dist_mean'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('mean')
    #df[f'molecule_atom_index_1_dist_mean_diff'] = df[f'molecule_atom_index_1_dist_mean'] - df['dist']
    df[f'molecule_atom_index_1_dist_mean_div'] = df[f'molecule_atom_index_1_dist_mean'] / df['dist']
    df[f'molecule_atom_index_1_dist_max'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('max')
    #df[f'molecule_atom_index_1_dist_max_diff'] = df[f'molecule_atom_index_1_dist_max'] - df['dist']
    #df[f'molecule_atom_index_1_dist_max_div'] = df[f'molecule_atom_index_1_dist_max'] / df['dist']
    df[f'molecule_atom_index_1_dist_min'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('min')
    #df[f'molecule_atom_index_1_dist_min_diff'] = df[f'molecule_atom_index_1_dist_min'] - df['dist']
    #df[f'molecule_atom_index_1_dist_min_div'] = df[f'molecule_atom_index_1_dist_min'] / df['dist']
    #df[f'molecule_atom_index_1_dist_std'] = df.groupby(['molecule_name', 'atom_index_1'])['dist'].transform('std')
    #df[f'molecule_atom_index_1_dist_std_diff'] = df[f'molecule_atom_index_1_dist_std'] - df['dist']
    #df[f'molecule_atom_index_1_dist_std_div'] = df[f'molecule_atom_index_1_dist_std'] / df['dist']
    df[f'molecule_atom_1_dist_mean'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('mean')
    df[f'molecule_atom_1_dist_min'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('min')
    df[f'molecule_atom_1_dist_min_diff'] = df[f'molecule_atom_1_dist_min'] - df['dist']
    df[f'molecule_atom_1_dist_min_div'] = df[f'molecule_atom_1_dist_min'] / df['dist']
    #df[f'molecule_atom_1_dist_std'] = df.groupby(['molecule_name', 'atom_1'])['dist'].transform('std')
    #df[f'molecule_atom_1_dist_std_diff'] = df[f'molecule_atom_1_dist_std'] - df['dist']
    df[f'molecule_type_dist_mean'] = df.groupby(['molecule_name', 'type'])['dist'].transform('mean')
    df[f'molecule_type_dist_mean_diff'] = df[f'molecule_type_dist_mean'] - df['dist']
    #df[f'molecule_type_dist_mean_div'] = df[f'molecule_type_dist_mean'] / df['dist']
    df[f'molecule_type_dist_max'] = df.groupby(['molecule_name', 'type'])['dist'].transform('max')
    df[f'molecule_type_dist_min'] = df.groupby(['molecule_name', 'type'])['dist'].transform('min')
    #df[f'molecule_type_dist_std'] = df.groupby(['molecule_name', 'type'])['dist'].transform('std')
    #df[f'molecule_type_dist_std_diff'] = df[f'molecule_type_dist_std'] - df['dist']

    df = reduce_mem_usage(df)
    return df


# In[ ]:


train = create_features(train)
test = create_features(test)


# In[ ]:


train.head()


# In[ ]:


#train = train.dropna()
#test = test.dropna()


# In[ ]:


#for i in ['atom_index_0', 'atom_index_1', 'type']:
for i in ['atom_0', 'atom_1', 'type']:
    class_le = LabelEncoder()   
    train[i] = class_le.fit_transform(train[i].values)
    test[i] = class_le.fit_transform(test[i].values)


# In[ ]:


'''cols = [

'id',
#'molecule_name',
'atom_index_0', 
'atom_index_1', 
'type', 
#'atom_0',
'EN_x',
'rad_x',
'n_bonds_x',
'bond_lengths_mean_x',
'x_0', 
'y_0', 
'z_0', 
'dihedral_x',
#'atom_1',
'EN_y',
'rad_y',
'n_bonds_y',
'bond_lengths_mean_y',
'x_1', 
'y_1', 
'z_1', 
'dihedral_y',
'dist',
'dist_x', 
'dist_y', 
'dist_z',
'molecule_atom_index_0_dist_min',
#'molecule_atom_index_0_dist_max',
'molecule_atom_index_1_dist_min',
'molecule_atom_index_0_dist_mean',
'molecule_atom_index_0_dist_std',
'molecule_atom_index_1_dist_std',
#'molecule_atom_index_1_dist_max',
'molecule_atom_index_1_dist_mean',
#'molecule_atom_index_0_dist_max_diff',
#'molecule_atom_index_0_dist_max_div',
'molecule_atom_index_0_dist_std_diff',
'molecule_atom_index_0_dist_std_div',
#'atom_0_couples_count',
#'molecule_atom_index_0_dist_min_div',
'molecule_atom_index_1_dist_std_diff',
#'molecule_at_mean',
#'molecule_atom_index_1_dist_max_diff',
'molecule_atom_index_0_y_1_std',
#'molecule_atom_index_1_dist_mean_diff',
'molecule_atom_index_1_dist_std_div',
#'molecule_atom_index_1_dist_mean_div',
#'molecule_atom_index_1_dist_min_diff',
#'molecule_atom_index_1_dist_min_div',
#'molecule_atom_index_1_dist_max_div',
'molecule_atom_index_0_z_1_std',
'molecule_type_dist_std_diff',
#'molecule_atom_1_dist_min_diff',
'molecule_atom_index_0_x_1_std',
#'molecule_dist_min',
#'molecule_atom_index_0_disom_index_0_dist_mean_div',
#'atom_1_couples_count',
#'molecule_atom_index_0_dist_mean_diff',
#'molecule_couples',
#'molecule_distt_min_diff',
'molecule_atom_index_0_y_1_mean_diff',
#'molecule_type_dist_min',
#'molecule_atom_1_dist_min_div',
#'molecule_dist_max',
#'molecule_atom_1_dist_std_diff',
#'molecule_type_dist_max',
#'molecule_atom_index_0_y_1_max_diff',
'molecule_type_dist_mean_diff',
#'molecule_atom_1_dist_mean',
'molecule_atom_index_0_y_1_mean_div',
'molecule_type_dist_mean_div'
]'''
cols = [

'id',
'atom_index_0', 
'atom_index_1', 
'type', 
'atom_0',
'EN_x',
'rad_x',
'n_bonds_x',
'bond_lengths_mean_x',
'x_0', 
'y_0', 
'z_0', 
'dihedral_x',
'atom_1',
'EN_y',
'rad_y',
'n_bonds_y',
'bond_lengths_mean_y',
'x_1', 
'y_1', 
'z_1', 
'dihedral_y',
'dist',
'dist_x', 
'dist_y', 
'dist_z',
'molecule_atom_index_0_dist_min',
'molecule_atom_index_0_dist_max',
'molecule_atom_index_1_dist_min',
'molecule_atom_index_0_dist_mean',
#'molecule_atom_index_0_dist_std',
#'molecule_atom_index_1_dist_std',
'molecule_atom_index_1_dist_max',
'molecule_atom_index_1_dist_mean',
#'molecule_atom_index_0_dist_max_diff',
#'molecule_atom_index_0_dist_max_div',
#'molecule_atom_index_0_dist_std_diff',
#'molecule_atom_index_0_dist_std_div',
#'atom_0_couples_count',
'molecule_atom_index_0_dist_min_div',
#'molecule_atom_index_1_dist_std_diff',
#'molecule_atom_index_0_dist_mean_div',
#'atom_1_couples_count',
'molecule_atom_index_0_dist_mean_diff',
#'molecule_couples',
#'molecule_dist_mean',
#'molecule_atom_index_1_dist_max_diff',
#'molecule_atom_index_0_y_1_std',
#'molecule_atom_index_1_dist_mean_diff',
#'molecule_atom_index_1_dist_std_div',
'molecule_atom_index_1_dist_mean_div',
#'molecule_atom_index_1_dist_min_diff',
#'molecule_atom_index_1_dist_min_div',
#'molecule_atom_index_1_dist_max_div',
#'molecule_atom_index_0_z_1_std',
#'molecule_type_dist_std_diff',
'molecule_atom_1_dist_min_diff',
#'molecule_atom_index_0_x_1_std',
'molecule_dist_min',
#'molecule_atom_index_0_dist_min_diff',
'molecule_atom_index_0_y_1_mean_diff',
'molecule_type_dist_min',
'molecule_atom_1_dist_min_div',
'molecule_dist_max',
#'molecule_atom_1_dist_std_diff',
'molecule_type_dist_max',
#'molecule_atom_index_0_y_1_max_diff',
'molecule_type_dist_mean_diff',
'molecule_atom_1_dist_mean',
#'molecule_atom_index_0_y_1_mean_div',
#'molecule_type_dist_mean_div'
]


# In[ ]:


def data(df):
    X_train, X_val, y_train, y_val  = train_test_split(df[cols].values,
                                                       df.loc[:, 'scalar_coupling_constant'].values,
                                                       test_size=0.2,
                                                       random_state=1340)
        
    return X_train, X_val, y_train, y_val


# In[ ]:


X_test = test[cols].values


# ### Train model

# In[ ]:


num_boost_round = 4000
early_stopping_rounds = 200
verbose_eval = 200

#X_train, X_val, y_train, y_val = data(train)


# In[ ]:


evals_result = {}

params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mae',
            'learning_rate': 0.2,
            'num_leaves': 900, 
            'reg_alpha': 0.5, 
            'reg_lambda': 0.5, 
            'max_bin': 63,
            'gpu_use_dp': 'false',
            'sparse_threshold': 1,
             #'nthread': 4, 
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'min_child_samples': 45
        }


# In[ ]:


#lgb_train = lgb.Dataset(X_train, y_train)
#lgb_val = lgb.Dataset(X_val, y_val)


# In[ ]:


#model = lgb.train(params,
#                  lgb_train,
#                  num_boost_round=num_boost_round,
#                  valid_sets=[lgb_val],
#                  early_stopping_rounds=early_stopping_rounds, 
#                  evals_result=evals_result, 
#                  verbose_eval=verbose_eval)


# In[ ]:


n_folds = 2
k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=2)
predictions = np.zeros(len(test))
oof = np.zeros(len(train))

for train_idx, val_idx in k_fold.split(train[cols].values, train.loc[:, 'scalar_coupling_constant'].values):
#for train_idx, val_idx in k_fold.split(X_train, y_train):

    lgb_train = lgb.Dataset(train[cols].values[train_idx], train.loc[:, 'scalar_coupling_constant'].values[train_idx])
    
    lgb_val = lgb.Dataset(train[cols].values[val_idx], train.loc[:, 'scalar_coupling_constant'].values[val_idx])  
    
    #lgb_train = lgb.Dataset(X_train[train_idx], y_train[train_idx])
    
    #lgb_val = lgb.Dataset(X_train[val_idx], y_train[val_idx])  
    
    model = lgb.train(params,
                      lgb_train,
                      num_boost_round=num_boost_round,
                      valid_sets=[lgb_val],
                      early_stopping_rounds=early_stopping_rounds, 
                      evals_result=evals_result, 
                      verbose_eval=verbose_eval)
    
    #model = lgb.LGBMRegressor(**params,
    #                          n_estimators=num_boost_round,
    #                          verbose_eval=verbose_eval)
    
    
    #model.fit(train[cols].values[train_idx],
    #          train.loc[:, 'scalar_coupling_constant'].values[train_idx], 
    #          eval_set=[(train[cols].values[train_idx],train.loc[:, 'scalar_coupling_constant'].values[train_idx]),
    #                    (train[cols].values[val_idx], train.loc[:, 'scalar_coupling_constant'].values[val_idx])],
    #          early_stopping_rounds=early_stopping_rounds)
    
    #oof[val_idx] = model.predict(train[cols].values[val_idx], num_iteration=model.best_iteration) 
    
    predictions += model.predict(X_test, num_iteration=model.best_iteration) / n_folds


# In[ ]:


#!pip install thundersvm


# In[ ]:


#!conda install cudatoolkit=9.0 -y


# In[ ]:


#idx_not_finite = np.where(np.isfinite(X_train[:, 40]) == False)#


# In[ ]:


#X_train_SVR = np.delete(X_train, idx_not_finite, axis=0)#
#y_train_SVR = np.delete(y_train, idx_not_finite)


# In[ ]:


#from thundersvm import SVR#

#model_SVR = SVR()
#model_SVR.fit(X_train_SVR ,y_train_SVR)


# In[ ]:


#model.save_model('model.txt', num_iteration=model.best_iteration)


# In[ ]:


#preds = model.predict(X_test, num_iteration=model.best_iteration)


# In[ ]:


def submit(predictions):
    submit = pd.read_csv('../input/sample_submission.csv')
    submit["scalar_coupling_constant"] = predictions
    submit.to_csv("submission.csv", index=False)


# In[ ]:


submit(predictions)

