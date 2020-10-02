#!/usr/bin/env python
# coding: utf-8

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

from xgboost import XGBRegressor
import os
from sklearn.decomposition import PCA


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
train_csv = pd.read_csv('../input/champs-scalar-coupling/train.csv', index_col='id', dtype=train_dtypes)
train_csv['molecule_index'] = train_csv.molecule_name.str.replace('dsgdb9nsd_', '').astype('int32')
train_csv = train_csv[['molecule_index', 'atom_index_0', 'atom_index_1', 'type', 'scalar_coupling_constant']]
train_csv.head(10)


# In[ ]:


print('Shape: ', train_csv.shape)
print('Total: ', train_csv.memory_usage().sum())
train_csv.memory_usage()


# In[ ]:


submission_csv = pd.read_csv('../input/champs-scalar-coupling/sample_submission.csv', index_col='id')


# In[ ]:


test_csv = pd.read_csv('../input/champs-scalar-coupling/test.csv', index_col='id', dtype=train_dtypes)
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
structures_csv = pd.read_csv('../input/champs-scalar-coupling/structures.csv', dtype=structures_dtypes)
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


# ## Check XGB with the smallest type

# In[ ]:


test_csv_ = test_csv.copy()
test_csv_["scalar_coupling_constant"] = "unknown"
concated = pd.concat([train_csv, test_csv_])


# In[ ]:


get_ipython().run_cell_magic('time', '', "full = build_couple_dataframe(concated, structures_csv, '1JHC', n_atoms=15)\nprint(full.shape)")


# We don't calculate distances for `d_0_x`, `d_1_1`, `d_2_2`, `d_2_3`, `d_3_3` because we already have them in later atoms(`d_0_1` == `d_1_0`) or they are equal to zeros(e.g. `d_1_1`, `d_2_2`).

# In[ ]:


full


# In[ ]:


full.columns


# For experiments, full dataset can be built with higher number of atoms, and for building a training/validation sets we can trim them:

# In[ ]:


df = take_n_atoms(full, 7)
# LightGBM performs better with 0-s then with NaN-s
df = df.fillna(0)
df.columns


# In[ ]:


df.head()


# In[ ]:


# X_data = df.drop(['scalar_coupling_constant'], axis=1).values.astype('float32')
# y_data = df['scalar_coupling_constant'].values.astype('float32')
df["std"] = df.drop(['scalar_coupling_constant'], axis=1).values.std(axis=1)
df["std"] = df["std"].apply(lambda x: round(x, 1))
X_data = df.sample(frac=1, random_state=43).drop(['scalar_coupling_constant'], axis=1).values

y_data = df['scalar_coupling_constant'].values

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=128)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


# **First 4 item in train have std < 1. They are useless**

# In[ ]:


df = df.iloc[4:]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.manifold import TSNE\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport plotly.graph_objs as go\nimport plotly.offline as py\npy.init_notebook_mode(connected=True)\n\nX_embedded = TSNE(n_components=2, perplexity=25, random_state=50).fit_transform(X_data[:100000])')


# # Plot by scalar_coupling_constant

# In[ ]:


tsne_data = pd.DataFrame(data={"x_axis": X_embedded[:,0], "y_axis": X_embedded[:,1],
                              "scalar_coupling_constant": y_data[:X_embedded.shape[0]]})


# In[ ]:


tsne_data["scalar_coupling_constant"] = (tsne_data["scalar_coupling_constant"]/10).astype(int)


# In[ ]:


def plot_2d(df, x, y):
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=x, y=y,
        hue='scalar_coupling_constant',
        palette=sns.color_palette('bright', tsne_data["scalar_coupling_constant"].nunique()),
        data=df,
        legend='full',
        alpha=0.9
    )
    plt.show()
    

def plot_3d(df, x, y, z):
    trace1 = go.Scatter3d(x=df[x].values, y=df[y].values, z=df[z].values,
        mode='markers',
        marker=dict(
            color=df['scalar_coupling_constant'].values,
            colorscale = "Jet",
            opacity=0.,
            size=2
        )
    )

    figure_data = [trace1]
    layout = go.Layout(
        scene = dict(
            xaxis = dict(title=x),
            yaxis = dict(title=y),
            zaxis = dict(title=z),
        ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        showlegend=True
    )

    fig = go.Figure(data=figure_data, layout=layout)
    py.iplot(fig, filename='3d_scatter')


# In[ ]:


plot_2d(tsne_data, x="x_axis", y="y_axis")


# # Plot by std

# In[ ]:


tsne_data = pd.DataFrame(data={"x_axis": X_embedded[:,0], "y_axis": X_embedded[:,1],
                              "std": df["std"].values[:X_embedded.shape[0]]})

def plot_2d(df, x, y):
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x=x, y=y,
        hue='std',
        palette=sns.color_palette('bright', tsne_data["std"].nunique()),
        data=df,
        legend='full',
        alpha=0.9
    )
    plt.show()
    

def plot_3d(df, x, y, z):
    trace1 = go.Scatter3d(x=df[x].values, y=df[y].values, z=df[z].values,
        mode='markers',
        marker=dict(
            color=df['std'].values,
            colorscale = "Jet",
            opacity=0.,
            size=2
        )
    )

    figure_data = [trace1]
    layout = go.Layout(
        scene = dict(
            xaxis = dict(title=x),
            yaxis = dict(title=y),
            zaxis = dict(title=z),
        ),
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        ),
        showlegend=True
    )

    fig = go.Figure(data=figure_data, layout=layout)
    py.iplot(fig, filename='3d_scatter')
    
plot_2d(tsne_data, x="x_axis", y="y_axis")

