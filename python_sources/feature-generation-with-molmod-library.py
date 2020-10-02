#!/usr/bin/env python
# coding: utf-8

# # Basic feature generation with molmod library
# ## Inroduction / prerequisites
# 
# The [molmod](http://molmod.github.io/molmod/index.html) library is 
# > > a Python library with many compoments that are useful to write molecular modeling programs.
# 
# The library provides fairly easy way to extract data from `*.xyz` files, starting from basic features like bonds and distances between atoms up to advanced features like dihedral angles and advanced pattern matching using graphs.
# 
# Here I present an example how to extract some basic features from [CHAMPS competition](https://www.kaggle.com/c/champs-scalar-coupling).
# 
# In order to use run code from this kernel You need molmod library installed.
# 
# ## Preparations
# 
# Let's load libraries

# In[ ]:


import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import lightgbm as lgb
import gc
import seaborn as sns
import molmod
import warnings
import multiprocessing
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
print(os.listdir('../input'))


# Check wehere our competition data is stored.

# In[ ]:


data_dir = '../input/champs-scalar-coupling/' if 'champs-scalar-coupling' in os.listdir('../input') else '../input/'


# Evaluation function for predictions.

# In[ ]:


def eval_fn(y_pred, y_test, c_type):
    diff = np.full((y_pred.shape[0], 2), 1e-9)
    diff[:,1] = np.abs(y_pred - y_test)
    step_1 = pd.DataFrame(
        {'diff': np.amax(diff, 1), 'type': c_type}
    ).groupby(
        'type'
    ).mean()
    return np.sum(np.log(step_1['diff'])) / step_1.shape[0]


# ## Load and preprocess data

# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv(data_dir + 'train.csv')\ntest = pd.read_csv(data_dir + 'test.csv')")


# Add basic features - atomic number for atom_1 (atom_0 is always hydrogen) and number of bonds between target atoms.

# In[ ]:


atoms = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
def map_atom(atom):
    return atoms[atom]
train['atom_1'] = train['type'].astype(str).str[3].apply(map_atom)
test['atom_1'] = test['type'].astype(str).str[3].apply(map_atom)
train['n_bonds'] = train['type'].astype(str).str[0].astype(np.int)
test['n_bonds'] = test['type'].astype(str).str[0].astype(np.int)


# Some helper functions

# In[ ]:


def pad_1d_array(a, length, value = np.nan):
    '''This function will right pad a numpy array `a` to `length` with `value`'''
    if a.shape[0] >= length:
        return a
    return np.pad(a, (0, length - a.shape[0]), 'constant', constant_values = value)


# In[ ]:


def get_neighbor_indices(molecule, index, limit = 4):
    '''Get neighbor indices from molmod.molecules.Molecules object for `index`.
    As sometimes the max bond limit is exceeded add limiting value'''
    neighbor_indices = list(molecule.graph.neighbors[index])
    while len(neighbor_indices) > limit:
        distances_tmp = molecule.distance_matrix[index, neighbor_indices]
        neighbor_indices.remove(neighbor_indices[np.argmax(distances_tmp)])
    return neighbor_indices


# ## Feature generation
# 
# Main feature generating function

# In[ ]:


def gen_features(row):
    '''Generate features for train/test entry, using molmod.molecules.Molecule class''' 
    # This is some hashing in order not to repeat loading molecule data from file
    # We expect the following indices in `row` 0: molecule name; 1: atom index 0; 2: atom index 1
    if row[0] != gen_features.molecule_name:
        # Load molecule in hashed var
        gen_features.molecule = molmod.molecules.Molecule.from_file(f'{data_dir}/structures/{row[0]}.xyz')
        # Remember loaded molecule name
        gen_features.molecule_name = row[0]
        # Generate graph
        gen_features.molecule.set_default_graph()
    # Distance between target atoms
    distance = gen_features.molecule.distance_matrix[row[1], row[2]]
    # Neighbor indices for both target atoms (up to 4 - default limit)
    neighbors_idxs_0 = get_neighbor_indices(gen_features.molecule, row[1], 1)
    neighbors_idxs_1 = get_neighbor_indices(gen_features.molecule, row[2], 4)
    # Get atomic numbers for neighbor atoms
    # Atom 0 is always hydrogen and has no more than 1 neighbor
    neighbor_atoms_0 = pad_1d_array(gen_features.molecule.numbers[neighbors_idxs_0].astype(np.float), 1)
    # Atom 1 may be anyone and may have up to 4 neighbors
    neighbor_atoms_1 = pad_1d_array(gen_features.molecule.numbers[neighbors_idxs_1].astype(np.float), 4)
    # Get distances to neighboring atoms
    neighbor_distances_0 = pad_1d_array(
        gen_features.molecule.distance_matrix[row[1], neighbors_idxs_0]
        , 1
    )
    neighbor_distances_1 = pad_1d_array(
        gen_features.molecule.distance_matrix[row[2], neighbors_idxs_1]
        , 4
    )
    # Get normalized graph to first order neighborhood, may be used directly as factor or to match some patterns
    neighborhood = gen_features.molecule.graph.get_subgraph(
        list(set([*neighbors_idxs_0, *neighbors_idxs_1])) + [row[1], row[2]]
        , True
    ).blob
    # Put all together in a single array
    return np.hstack((
        distance, neighbor_atoms_0, neighbor_atoms_1, neighbor_distances_0, neighbor_distances_1, neighborhood
    ))

# Variables for hashing
gen_features.molecule = gen_features.molecule_name = None

# Names for newly generated features
# new_columns = ['distance'] + [
#     f'{feature}_{i}_{j}' for feature in ['n_atom', 'n_distance'] for i in range(2) for j in range(4)
# ] + ['neighborhood']
new_columns = [
    'distance', 'n_atom_0_0', 'n_atom_1_0', 'n_atom_1_1', 'n_atom_1_2', 'n_atom_1_3'
    , 'n_distance_0_0', 'n_distance_1_0', 'n_distance_1_1', 'n_distance_1_2', 'n_distance_1_3', 'neighborhood']


# Add new features to train and test sets. We will not use `pandas.DataFrame.apply()` for our non-trivial `gen_features()` as people on the internet [say](https://ys-l.github.io/posts/2015/08/28/how-not-to-use-pandas-apply/) it is not memory and performance friendly. We split our data frame in chunks and generate features for each chunk separately. To speed things up we use some parallel processing.

# In[ ]:


def index_marks(nrows, chunk_size):
    '''Get indices where to split df'''
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)

def split_df(df, chunk_size):
    '''Split df into chunks not larger than `chunk_size`'''
    indices = index_marks(df.shape[0], chunk_size)
    return np.split(df, indices)


# At first run on small part of data to check that everything is OK

# In[ ]:


train_head = train.head(13).copy()
train_parts = split_df(train_head, 2)
for part in train_parts:
    with multiprocessing.Pool(4) as pool:
        part[new_columns] = pd.DataFrame( pool.map(gen_features, np.array(part[['molecule_name', 'atom_index_0', 'atom_index_1']])), index = part.index)
    gc.collect()
train_head = pd.concat(train_parts)
del train_parts
gc.collect()
train_head


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_parts = split_df(train, 100000)\nfor part in train_parts:\n    with multiprocessing.Pool(4) as pool:\n        part[new_columns] = pd.DataFrame( pool.map(gen_features, np.array(part[['molecule_name', 'atom_index_0', 'atom_index_1']])), index = part.index)\n    gc.collect()\ntrain = pd.concat(train_parts)\ndel train_parts\ngc.collect()")


# In[ ]:


train.head(20)


# In[ ]:


get_ipython().run_cell_magic('time', '', "test_parts = split_df(test, 100000)\nnew_vals = []\nfor part in test_parts:\n    with multiprocessing.Pool(4) as pool:\n        part[new_columns] = pd.DataFrame( pool.map(gen_features, np.array(part[['molecule_name', 'atom_index_0', 'atom_index_1']])), index = part.index)\n    gc.collect()\ntest = pd.concat(test_parts)\ndel test_parts\ngc.collect()")


# In[ ]:


for col in [col for col in new_columns if not re.search('(neighborhood|type)', col)]:
    train[col] = train[col].astype(np.float)
    test[col] = test[col].astype(np.float)


# ### Missing value imputation

# In[ ]:


atom_cols = [c for c in train.columns if re.search('_atom', c)]
distance_cols = [c for c in train.columns if re.search('distance', c)]

def summary(series):
    return {
        'mean': np.mean(series)
        , 'sd': np.std(series)
        , 'min': np.amin(series)
        , 'max': np.amax(series)
        , 'NAs': np.sum(np.isnan(series)) / series.shape[0]
    }

for col in atom_cols:
    print(f'{col}: {summary(train[col])}')

for col in distance_cols:
    print(f'{col}: {summary(train[col])}')


# Seems that `-10` should be fine for missing values.
# 
# Set all atomic numbers to integers.

# In[ ]:


for col in atom_cols + distance_cols:
    train[col] = train[col].fillna(-10)

for col in atom_cols:
    train[col] = train[col].astype(np.int)


# In[ ]:


for f in [c for c in train.columns if re.search('(neighborhood|type)', c)]:
    lbl = LabelEncoder()
    lbl.fit(list(train[f].values) + list(test[f].values))
    train[f] = lbl.transform(list(train[f].values))
    test[f] = lbl.transform(list(test[f].values))


# Select columns for model.

# In[ ]:


model_cols = [c for c in train.columns if not re.search('(^id$|molecule_name|_index|coupling_constant)', c)]
model_cols


# Split training set for modelling.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    train[model_cols]
    , train['scalar_coupling_constant']
    , test_size = 0.15, random_state = 0
)
print(X_train.shape)
print(X_test.shape)


# Do some default LightGBM modeling.

# In[ ]:


params = {'boosting': 'gbdt', 'colsample_bytree': 1, 
          'learning_rate': 0.1, 'max_depth': 200, 'metric': 'mae',
          'min_child_samples': 50, 'num_leaves': 500, 
          'objective': 'regression', 'reg_alpha': 0.5, 
          'reg_lambda': 0.8, 'subsample': 0.5,
          'n_jobs': 4
     }

lgtrain = lgb.Dataset(X_train, label = y_train)
lgval = lgb.Dataset(X_test, label = y_test)

model_lgb = lgb.train(params, lgtrain, 1000, valid_sets = [lgtrain, lgval], early_stopping_rounds = 250, verbose_eval = 500)

def eval_fn(y_pred, y_test, c_type):
    diff = np.full((y_pred.shape[0], 2), 1e-9)
    diff[:,1] = np.abs(y_pred - y_test)
    step_1 = pd.DataFrame(
        {'diff': np.amax(diff, 1), 'type': c_type}
    ).groupby(
        'type'
    ).mean()
    return np.sum(np.log(step_1['diff'])) / step_1.shape[0]

y_pred = model_lgb.predict(X_test)
score = eval_fn(y_pred, y_test, X_test['atom_1'])
print(f'Evaluation score: {score}')


# Feature importance.

# In[ ]:


def plotImp(model, X , num = 20):
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importance(),X.columns)), columns = ['Value','Feature'])
    plt.figure(figsize = (40, 20))
    sns.set(font_scale = 5)
    sns.barplot(x = "Value", y = "Feature", data=feature_imp.sort_values(by = "Value", ascending = False)[0:num])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.xscale('log')
    plt.show()

plotImp(model_lgb, X_train, 20)


# Predict and save.

# In[ ]:


predictions = model_lgb.predict(test[model_cols])
result = pd.DataFrame({'id': test['id'], 'scalar_coupling_constant': predictions})
print(result.shape)
result.to_csv('prediction.csv', index = False)


# ## Conclusions
# 
# Here we extracted just some very basic features with molmod library and achieved pretty good result. The library is certainly not limited to this, it provides vast opurtunities to generate geometric and molecular pattern features.
# 
# The feature extraction is quite slow as it is done for each row indivually. However it could be speeded up by parallel processing and/or by extracting only the required atom indices with molmod and then calculating distances and angles in vectorized way.
# 
# While using it for a particular `type` group I am able to get close to **-2** scores and I'm not quite done yet.
