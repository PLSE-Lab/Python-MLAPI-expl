#!/usr/bin/env python
# coding: utf-8

# # A big thanks to all kaggleer out there

# ## Core Idea
# 
# Despite a lot of creeping Physics and Chemistry knowledge introduced in the description, this competition is more about Geometry and pattern matching.
# 
# The hypothesis of this kernel is next:
# 1. If we have two similar sets of atoms with the same distances between them and the same types - the scalar coupling constant should be very close.
# 2. More closest atoms to the pair of atoms under prediction have higher influence on scalar coupling constant then those with higher distance
# 
# So, basically, this problem could be dealt with some kind of K-Nearest Neighbor algorithm or any tree-based - e.g. LightGBM, in case we can find some representation which would describe similar configurations with similar feature sets.
# 
# Each atom is described with 3 cartesian coordinates. This representation is not stable. Each coupling pair is located in a different point in space and two similar coupling sets would have very different X,Y,Z.
# 
# So, instead of using coordinates let's consider next system:
# 1. Take each pair of atoms as two first core atoms
# 2. Calculate the center between the pair
# 3. Find all n-nearest atoms to the center (excluding first two atoms)
# 4. Take two closest atoms from step 3 - they will be 3rd and 4th core atoms
# 5. Calculate the distances from 4 core atoms to the rest of the atoms and to the core atoms as well
# 
# Using this representation each atom position can be described by 4 distances from the core atoms. This representation is stable to rotation and translation. And it's suitable for pattern-matching. So, we can take a sequence of atoms, describe each by 4 distances + atom type(H,O,etc) and looking up for the same pattern we can find similar configurations and detect scalar coupling constant.
# 
# Here I used LightGBM, because sklearn KNN can't deal with the amount of data. My blind guess is that hand-crafted KNN can outperform LightGBM.
# 
# Let's code the solution!

# ## Load Everything

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np

import math
import gc
import copy

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor


# In[ ]:


DATA_PATH = '../input'
SUBMISSIONS_PATH = './'
# use atomic numbers to recode atomic names
ATOMIC_NUMBERS = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9
}


# In[ ]:


pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 120)
pd.set_option('display.max_columns', 120)


# ## Load Dataset

# By default all data is read as `float64` and `int64`. We can trade this uneeded precision for memory and higher prediction speed. So, let's read with Pandas all the data in the minimal representation: 

# In[ ]:


train_dtypes = {
    'molecule_name': 'category',
    'atom_index_0': 'int8',
    'atom_index_1': 'int8',
    'type': 'category',
    'scalar_coupling_constant': 'float32'
}
train_csv = pd.read_csv(f'{DATA_PATH}/train.csv', index_col='id', dtype=train_dtypes)
train_csv['molecule_index'] = train_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
train_csv = train_csv[['molecule_index', 'atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant']]
train_csv.head(10)


# In[ ]:


print('Shape: ', train_csv.shape)
print('Total: ', train_csv.memory_usage().sum())
train_csv.memory_usage()


# In[ ]:


submission_csv = pd.read_csv(f'{DATA_PATH}/sample_submission.csv', index_col='id')


# In[ ]:


test_csv = pd.read_csv(f'{DATA_PATH}/test.csv', index_col='id', dtype=train_dtypes)
test_csv['molecule_index'] = test_csv['molecule_name'].str.replace('dsgdb9nsd_', '').astype('int32')
test_csv = test_csv[['molecule_index', 'atom_index_0', 'atom_index_1', 'type']]
test_csv.head(10)


# In[ ]:


structures_dtypes = {
    'molecule_name': 'category',
    'atom_index': 'int8',
    'atom': 'category',
    'x': 'float32',
    'y': 'float32',
    'z': 'float32'
}
structures_csv = pd.read_csv(f'{DATA_PATH}/structures.csv', dtype=structures_dtypes)
structures_csv['molecule_index'] = structures_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
structures_csv = structures_csv[['molecule_index', 'atom_index', 'atom', 'x', 'y', 'z']]
structures_csv['atom'] = structures_csv['atom'].replace(ATOMIC_NUMBERS).astype('int8')
structures_csv.head(10)


# In[ ]:


print('Shape: ', structures_csv.shape)
print('Total: ', structures_csv.memory_usage().sum())
structures_csv.memory_usage()


# ## Build Distance Dataset

# In[ ]:


def build_type_dataframes(base, structures, coupling_type):
    base = base[base['type'] == coupling_type].drop('type', axis=1).copy()
    base = base.reset_index()
    base['id'] = base['id'].astype('int32')
    structures = structures[structures['molecule_index'].isin(base['molecule_index'])]
    return base, structures


# In[ ]:


def add_coordinates(base, structures, index):
    df = pd.merge(base, structures, how='inner',
                  left_on=['molecule_index', f'atom_index_{index}'],
                  right_on=['molecule_index', 'atom_index']).drop(['atom_index'], axis=1)
    df = df.rename(columns={
        'atom': f'atom_{index}',
        'x': f'x_{index}',
        'y': f'y_{index}',
        'z': f'z_{index}'
    })
    return df


# In[ ]:


def add_atoms(base, atoms):
    df = pd.merge(base, atoms, how='inner',
                  on=['molecule_index', 'atom_index_0', 'atom_index_1'])
    return df


# In[ ]:


def merge_all_atoms(base, structures):
    df = pd.merge(base, structures, how='left',
                  left_on=['molecule_index'],
                  right_on=['molecule_index'])
    df = df[(df.atom_index_0 != df.atom_index) & (df.atom_index_1 != df.atom_index)]
    return df


# In[ ]:


def add_center(df):
    df['x_c'] = ((df['x_1'] + df['x_0']) * np.float32(0.5))
    df['y_c'] = ((df['y_1'] + df['y_0']) * np.float32(0.5))
    df['z_c'] = ((df['z_1'] + df['z_0']) * np.float32(0.5))

def add_distance_to_center(df):
    df['d_c'] = ((
        (df['x_c'] - df['x'])**np.float32(2) +
        (df['y_c'] - df['y'])**np.float32(2) + 
        (df['z_c'] - df['z'])**np.float32(2)
    )**np.float32(0.5))

def add_distance_between(df, suffix1, suffix2):
    df[f'd_{suffix1}_{suffix2}'] = ((
        (df[f'x_{suffix1}'] - df[f'x_{suffix2}'])**np.float32(2) +
        (df[f'y_{suffix1}'] - df[f'y_{suffix2}'])**np.float32(2) + 
        (df[f'z_{suffix1}'] - df[f'z_{suffix2}'])**np.float32(2)
    )**np.float32(0.5))


# In[ ]:


def add_distances(df):
    n_atoms = 1 + max([int(c.split('_')[1]) for c in df.columns if c.startswith('x_')])
    
    for i in range(1, n_atoms):
        for vi in range(min(4, i)):
            add_distance_between(df, i, vi)


# In[ ]:


def add_n_atoms(base, structures):
    dfs = structures['molecule_index'].value_counts().rename('n_atoms').to_frame()
    return pd.merge(base, dfs, left_on='molecule_index', right_index=True)


# In[ ]:


def build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=10):
    base, structures = build_type_dataframes(some_csv, structures_csv, coupling_type)
    base = add_coordinates(base, structures, 0)
    base = add_coordinates(base, structures, 1)
    
    base = base.drop(['atom_0', 'atom_1'], axis=1)
    atoms = base.drop('id', axis=1).copy()
    if 'scalar_coupling_constant' in some_csv:
        atoms = atoms.drop(['scalar_coupling_constant'], axis=1)
        
    add_center(atoms)
    atoms = atoms.drop(['x_0', 'y_0', 'z_0', 'x_1', 'y_1', 'z_1'], axis=1)

    atoms = merge_all_atoms(atoms, structures)
    
    add_distance_to_center(atoms)
    
    atoms = atoms.drop(['x_c', 'y_c', 'z_c', 'atom_index'], axis=1)
    atoms.sort_values(['molecule_index', 'atom_index_0', 'atom_index_1', 'd_c'], inplace=True)
    atom_groups = atoms.groupby(['molecule_index', 'atom_index_0', 'atom_index_1'])
    atoms['num'] = atom_groups.cumcount() + 2
    atoms = atoms.drop(['d_c'], axis=1)
    atoms = atoms[atoms['num'] < n_atoms]

    atoms = atoms.set_index(['molecule_index', 'atom_index_0', 'atom_index_1', 'num']).unstack()
    atoms.columns = [f'{col[0]}_{col[1]}' for col in atoms.columns]
    atoms = atoms.reset_index()
    
    # downcast back to int8
    for col in atoms.columns:
        if col.startswith('atom_'):
            atoms[col] = atoms[col].fillna(0).astype('int8')
            
    atoms['molecule_index'] = atoms['molecule_index'].astype('int32')
    
    full = add_atoms(base, atoms)
    add_distances(full)
    
    full.sort_values('id', inplace=True)
    
    return full


# In[ ]:


def take_n_atoms(df, n_atoms, four_start=4):
    labels = []
    for i in range(2, n_atoms):
        label = f'atom_{i}'
        labels.append(label)

    for i in range(n_atoms):
        num = min(i, 4) if i < four_start else 4
        for j in range(num):
            labels.append(f'd_{i}_{j}')
    if 'scalar_coupling_constant' in df:
        labels.append('scalar_coupling_constant')
    return df[labels]


# ## Check LightGBM with the smallest type

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndef type_select(types = '1JHN'):\n    full = build_couple_dataframe(train_csv, structures_csv, types, n_atoms=10)\n    print(full.shape)\n\n\n    df = take_n_atoms(full, 7)\n    # LightGBM performs better with 0-s then with NaN-s\n    df = df.fillna(0)\n    # df.columns\n\n    X_data = df.drop(['scalar_coupling_constant'], axis=1).values.astype('float32')\n    y_data = df['scalar_coupling_constant'].values.astype('float32')\n\n    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=128)\n    print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)\n    \n    return X_train, X_val, y_train, y_val\n")


# We don't calculate distances for `d_0_x`, `d_1_1`, `d_2_2`, `d_2_3`, `d_3_3` because we already have them in later atoms(`d_0_1` == `d_1_0`) or they are equal to zeros(e.g. `d_1_1`, `d_2_2`).

# For experiments, full dataset can be built with higher number of atoms, and for building a training/validation sets we can trim them:

# In[ ]:


# LGB_PARAMS = {
#     'objective': 'regression',
#     'metric': 'mae',
#     'verbosity': -1,
#     'boosting_type': 'gbdt',
#     'learning_rate': 0.1455,
#     'num_leaves': 128,
#     'min_child_samples': 79,
#     'max_depth': 11,
#     'subsample_freq': 1,
#     'subsample': 0.88,
#     'bagging_seed': 14,
#     'reg_alpha': 0.1,
#     'reg_lambda': 0.3,
#     'colsample_bytree': 1.0
# }

# -1.0147180396368805

# model = LGBMRegressor(**LGB_PARAMS, n_estimators=3000, n_jobs = -1)
# model.fit(X_train, y_train, 
#         eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',
#         verbose=400, early_stopping_rounds=500)

# y_pred = model.predict(X_val)
# np.log(mean_absolute_error(y_val, y_pred))


# In[ ]:


# LGB_PARAMS = {
#     'objective': 'regression',
#     'metric': 'mae',
#     'verbosity': -1,
#     'boosting_type': 'gbdt',
#     'learning_rate': 0.1455,
#     'num_leaves': 129,
#     'min_child_samples': 79,
#     'max_depth': 13,
#     'subsample_freq': 1,
#     'subsample': 0.88,
#     'bagging_seed': 15,
#     'reg_alpha': 0.10108,
#     'reg_lambda': 0.30013,
#     'colsample_bytree': 1.0
# }
# #-1.0178287146032219 #r_a .101
# -1.0179811056684678 #'learning_rate': 0.1455, 'max_depth': 12, 'subsample': 0.88, 

# model = LGBMRegressor(**LGB_PARAMS, n_estimators=3000, n_jobs = -1)
# model.fit(X_train, y_train, 
#         eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',
#         verbose=400, early_stopping_rounds=500)

# y_pred = model.predict(X_val)
# np.log(mean_absolute_error(y_val, y_pred))


# In[ ]:


model_params = {
    '1JHN': 7,
    '1JHC': 10,
    '2JHH': 9,
    '2JHN': 9,
    '2JHC': 9,
    '3JHH': 9,
    '3JHC': 10,
    '3JHN': 10
}

model_params.keys()


# In[ ]:


X_train, X_val, y_train, y_val = type_select(types = '3JHN')


# In[ ]:


get_ipython().run_cell_magic('time', '', "LGB_PARAMS = {\n    'objective': 'regression',\n    'metric': 'mae',\n    'verbosity': -1,\n    'boosting_type': 'gbdt',\n    'learning_rate': 0.1455,\n    'num_leaves': 129,\n    'min_child_samples': 78,\n    'max_depth': 13,\n    'subsample_freq': 1,\n    'subsample': 0.88,\n    'bagging_seed': 15,\n    'reg_alpha': 0.10107001,\n    'reg_lambda': 0.300132,\n    'colsample_bytree': 1.0\n}\n# -1.0327442550828587\n\nLGB_PARAMS={'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.14848931924611134, 'max_depth': -1, 'min_child_samples': 80, 'categorical_feature':categorical_feature, 'num_leaves': 129, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'subsample': 0.89, 'subsample_freq': 1, 'verbosity': -1} #-2.181925527723545\n\n\nmodel = LGBMRegressor(**LGB_PARAMS, n_estimators=3000, n_jobs = -1)\nmodel.fit(X_train, y_train, \n        eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',\n        verbose=400, early_stopping_rounds=500)\n\ny_pred = model.predict(X_val)\nnp.log(mean_absolute_error(y_val, y_pred))")


# In[ ]:


categorical_feature=[0,1,2,3,4]


# In[ ]:


# LGB_PARAMS_3JHN={'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.15848931924611134, 'max_depth': 14, 'min_child_samples': 80,  'num_leaves': 129, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.23, 'subsample': 0.9, 'subsample_freq': 1, 'verbosity': -1}

# LGB_PARAMS_3JHN={'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.15848931924611134, 'max_depth': 14, 'min_child_samples': 80, 'num_leaves': 129, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'subsample': 0.89, 'subsample_freq': 1, 'verbosity': -1}  #-2.1795166793513037

# LGB_PARAMS_3JHN={'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.15848931924611134, 'max_depth': 14, 'min_child_samples': 80, 'categorical_feature':categorical_feature, 'num_leaves': 129, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'subsample': 0.89, 'subsample_freq': 1, 'verbosity': -1} #-2.1795305847801667

# LGB_PARAMS_3JHN={'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.14848931924611134, 'max_depth': 14, 'min_child_samples': 80, 'categorical_feature':categorical_feature, 'num_leaves': 129, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'subsample': 0.89, 'subsample_freq': 1, 'verbosity': -1} #-2.181925527723545


LGB_PARAMS_3JHN={'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.14548931924611134, 'max_depth': 14, 'min_child_samples': 80,  'num_leaves': 129, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'subsample': 0.89, 'subsample_freq': 1, 'verbosity': -1} #-2.181925527723545


# In[ ]:


# from sklearn.preprocessing import CategoricalEncoder

# cat_encoder = CategoricalEncoder()

# pd.DataFrame(X_train).columns[:4]

# housing_cat_reshaped = pd.DataFrame(X_train).values.reshape(-1, 1)
# housing_cat_1hot = cat_encoder.fit_trasform(housing_cat_reshaped)
# housing_cat_1hot


# In[ ]:


# %%time

# model = LGBMRegressor(**LGB_PARAMS_3JHN, n_estimators=3000, n_jobs = -1)
# model.fit(X_train, y_train, 
#         eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',
#         verbose=400, early_stopping_rounds=500, **categorical_feature)

# y_pred = model.predict(X_val)
# np.log(mean_absolute_error(y_val, y_pred))


# In[ ]:





# 

# In[ ]:


def build_x_y_data(some_csv, coupling_type, n_atoms):
    full = build_couple_dataframe(some_csv, structures_csv, coupling_type, n_atoms=n_atoms)
    
    df = take_n_atoms(full, n_atoms)
    df = df.fillna(0)
    print(df.columns)
    
    if 'scalar_coupling_constant' in df:
        X_data = df.drop(['scalar_coupling_constant'], axis=1).values.astype('float32')
        y_data = df['scalar_coupling_constant'].values.astype('float32')
    else:
        X_data = df.values.astype('float32')
        y_data = None
    
    return X_data, y_data


# 

# In[ ]:


def train_and_predict_for_one_coupling_type(coupling_type, submission, n_atoms, n_folds=4, n_splits=4, random_state=128):
    print(f'*** Training Model for {coupling_type} ***')
    
    X_data, y_data = build_x_y_data(train_csv, coupling_type, n_atoms)
    X_test, _ = build_x_y_data(test_csv, coupling_type, n_atoms)
    y_pred = np.zeros(X_test.shape[0], dtype='float32')

    cv_score = 0
    
    if n_folds > n_splits:
        n_splits = n_folds
    
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    print('fitting {} with {} LGB_PARAMS'.format(coupling_type,LGB_PARAMS))
    for fold, (train_index, val_index) in enumerate(kfold.split(X_data, y_data)):
        if fold >= n_folds:
            break

        X_train, X_val = X_data[train_index], X_data[val_index]
        y_train, y_val = y_data[train_index], y_data[val_index]

        model = LGBMRegressor(**LGB_PARAMS, n_estimators= 9000, n_jobs = -1)
        model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='mae',
            verbose=1000, early_stopping_rounds=500)

        y_val_pred = model.predict(X_val)
        val_score = np.log(mean_absolute_error(y_val, y_val_pred))
        print(f'{coupling_type} Fold {fold}, logMAE: {val_score}')
        
        cv_score += val_score / n_folds
        y_pred += model.predict(X_test) / n_folds
        
    plt.scatter(y_val, y_val_pred)   
    plt.show()
    submission.loc[test_csv['type'] == coupling_type, 'scalar_coupling_constant'] = y_pred
    return cv_score


# ## Train Model

# Let's build a separate model for each type of coupling. Dataset is split into 5 pieces and in this kernel we will use only 3 folds for speed up.
# 
# Main tuning parameter is the number of atoms. I took good numbers, but accuracy can be improved a bit by tuning them for each type.

# In[ ]:


model_params = {
    '1JHN': 7,
    '1JHC': 10,
    '2JHH': 9,
    '2JHN': 9,
    '2JHC': 9,
    '3JHH': 9,
    '3JHC': 10,
    '3JHN': 10
}



cat_code = {'1JHN': 5,'1JHC': 7,'2JHH': 7,
             '2JHN': 7,'2JHC': 7,'3JHH': 7,
            '3JHC': 8,'3JHN': 7}

categorical_feature = [0,1,2,3,4,5,6]



LGB_PARAMS_2JHC={'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.14548931824400012, 'max_depth': -1, 'min_child_samples': 81,'categorical_feature':categorical_feature, 'num_leaves': 129, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.30, 'subsample': 0.89, 'subsample_freq': 1, 'verbosity': -1} #-1.10519978867864
LGB_PARAMS_3JHN={'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.1454893092, 'max_depth': -1, 'min_child_samples': 81,  'num_leaves': 129, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'subsample': 0.89, 'subsample_freq': 1, 'verbosity': -1} #-2.181925527723545
LGB_PARAMS_2JHN={'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.1454893072, 'max_depth': -1, 'min_child_samples': 82,  'num_leaves': 130, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'subsample': 0.89, 'subsample_freq': 1, 'verbosity': -1} #-2.181925527723545


N_FOLDS = 5
submission = submission_csv.copy()

cv_scores = {}
for coupling_type in model_params.keys():
    if( coupling_type == '2JHC'):
        categorical_feature = [0,1,2,3,4,5,6]
        LGB_PARAMS = {'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.14548931824400012, 'max_depth': -1, 'min_child_samples': 81,'categorical_feature':categorical_feature, 'num_leaves': 129, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.30, 'subsample': 0.89, 'subsample_freq': 1, 'verbosity': -1}
    elif(coupling_type == '3JHN'):
        categorical_feature = [0,1,2,3,4,5,6]
        LGB_PARAMS = {'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.1454893092, 'max_depth': -1, 'min_child_samples': 81,  'num_leaves': 129, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'subsample': 0.89, 'subsample_freq': 1, 'verbosity': -1} 
    elif(coupling_type == '2JHN'):
        categorical_feature = [0,1,2,3,4,5,6]
        LGB_PARAMS = {'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.1454893072, 'max_depth': -1, 'min_child_samples': 82,  'num_leaves': 130, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.3, 'subsample': 0.89, 'subsample_freq': 1, 'verbosity': -1}
    elif(coupling_type == '3JHC'):
        categorical_feature = [0,1,2,3,4,5,6,7]
        LGB_PARAMS={'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.14548931824400012, 'max_depth': -1, 'min_child_samples': 81,'categorical_feature':categorical_feature, 'num_leaves': 129, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.30, 'subsample': 0.89, 'subsample_freq': 1, 'verbosity': -1} #-1.10519978867864
    elif(coupling_type == '1JHN'):
        categorical_feature = [0,1,2,3,4]
        LGB_PARAMS={'bagging_seed': 14, 'colsample_bytree': 1.0, 'learning_rate': 0.1454893092, 'max_depth': -1, 'min_child_samples': 81,'categorical_feature':categorical_feature, 'num_leaves': 129, 'random_state': 42, 'reg_alpha': 0.1, 'reg_lambda': 0.30, 'subsample': 0.89, 'subsample_freq': 1, 'verbosity': -1} #-1.10519978867864

    else:
        LGB_PARAMS = LGB_PARAMS_2JHN
    cv_score = train_and_predict_for_one_coupling_type(
        coupling_type, submission, n_atoms=model_params[coupling_type], n_folds=N_FOLDS)
    cv_scores[coupling_type] = cv_score
    


# Checking cross-validation scores for each type:

# In[ ]:


np.mean(list(cv_scores.values()))


# Sanity check for all cells to be filled with predictions:

# In[ ]:


submission.head(10)


# ## Submission Model

# In[ ]:


submission.to_csv(f'{SUBMISSIONS_PATH}/submission.csv')


# ## Room for improvement

# There are many steps, how to improve the score for this kernel:
# * Tune LGB hyperparameters - I did nothing for this   **checked***
# * Tune number of atoms for each type        **checked**
# * Try to add other features
# * Play with categorical features for atom types (one-hot-encoding, CatBoost?)   **checked***
# * Try other tree libraries
# 
# Also, this representation fails badly on `*JHC` coupling types. The main reason for this is that 3rd and 4th atoms are usually located on the same distance and representation starts "jittering" randomly picking one of them. So, two similar configurations will have different representation due to usage of 3/4 of 4/3 distances.
# 
# The biggest challenge would be to implement handcrafted KNN with some compiled language(Rust, C++, C).
# 
# Would be cool to see this kernel forked and addressed some of the issues with higher LB score.
