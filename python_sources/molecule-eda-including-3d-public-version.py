#!/usr/bin/env python
# coding: utf-8

# EDA for Molecule in this kernel, including 3D analysis, and also raised a problem about the train and test dataset, looking for someone's help, or team up.
# Lack of chemical knowledge.
# My first public kernel, if it helps, please upvote it, thanks.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


structures = pd.read_csv('../input/structures.csv')
structures.head()


# # For all

# In[ ]:


print(train.shape, test.shape, structures.shape)


# In[ ]:


print('There are {} unique molecules in trainset'.format(train['molecule_name'].nunique()))
print('There are {} unique molecules in testset'.format(test['molecule_name'].nunique()))
print('There are {} unique molecules in structures'.format(structures['molecule_name'].nunique()))


# In[ ]:


print('There are {} atom_index_0 & {} atom_index_1 & {} types in trainset'.format(train['atom_index_0'].nunique(), train['atom_index_1'].nunique(), train['type'].nunique()))
print('There are {} atom_index_0 & {} atom_index_1 & {} types in testset'.format(test['atom_index_0'].nunique(), test['atom_index_1'].nunique(), test['type'].nunique()))
print('There are {} atom_index & {} atom in structures'.format(structures['atom_index'].nunique(), structures['atom'].nunique()))


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

plt.subplots(figsize=(24, 6))
plt.subplot(1, 3, 1)
plt.title("train['atom_index_0']")
plt.hist(train['atom_index_0'], bins=29)
plt.subplot(1, 3, 2)
plt.title("train['atom_index_1']")
plt.hist(train['atom_index_1'], bins=29)
plt.subplot(1, 3, 3)
plt.title("train['scalar_coupling_constant']")
plt.hist(train['scalar_coupling_constant'], bins=100)
plt.show()

plt.subplots(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.title("test['atom_index_0']")
plt.hist(test['atom_index_0'], bins=29)
plt.subplot(1, 2, 2)
plt.title("test['atom_index_1']")
plt.hist(test['atom_index_1'], bins=29)
plt.show()

plt.subplots(figsize=(24, 6))
plt.subplot(1, 3, 1)
plt.title("structures['x']")
plt.hist(structures['x'], bins=100)
plt.subplot(1, 3, 2)
plt.title("structures['y']")
plt.hist(structures['y'], bins=100)
plt.subplot(1, 3, 3)
plt.title("structures['z']")
plt.hist(structures['z'], bins=100)
plt.show()


# In[ ]:


train_atom = pd.concat([train['atom_index_0'], train['atom_index_1']])
test_atom = pd.concat([test['atom_index_0'], test['atom_index_1']])
plt.subplots(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.title("train_atom")
plt.hist(train_atom, bins=29)
plt.subplot(1, 2, 2)
plt.title("test_atom")
plt.hist(test_atom, bins=29)
plt.show()


# # For each type

# In[ ]:


import seaborn as sns
fig, axes = plt.subplots(2, 4, figsize = (18, 8))
axes = axes.flatten()
for i, type_ in enumerate(np.unique(train['type'])):
    ix = train['type'] == type_
    _ = sns.distplot(train['scalar_coupling_constant'][ix], ax=axes[i])
    axes[i].set(title=f'{type_}. Rows: {ix.sum()}')
fig.tight_layout()


# In[ ]:


train['scalar_coupling_constant'].describe()


# In[ ]:


train_1JHC = train[train['type'] == '1JHC']
train_1JHN = train[train['type'] == '1JHN']
train_2JHC = train[train['type'] == '2JHC']
train_2JHH = train[train['type'] == '2JHH']
train_2JHN = train[train['type'] == '2JHN']
train_3JHC = train[train['type'] == '3JHC']
train_3JHH = train[train['type'] == '3JHH']
train_3JHN = train[train['type'] == '2JHN']

print('train_1JHC atom unique', pd.concat([train_1JHC['atom_index_0'], train_1JHC['atom_index_1']]).nunique())
print('train_1JHN atom unique', pd.concat([train_1JHN['atom_index_0'], train_1JHN['atom_index_1']]).nunique())
print('train_2JHC atom unique', pd.concat([train_2JHC['atom_index_0'], train_2JHC['atom_index_1']]).nunique())
print('train_2JHH atom unique', pd.concat([train_2JHH['atom_index_0'], train_2JHH['atom_index_1']]).nunique())
print('train_2JHN atom unique', pd.concat([train_2JHN['atom_index_0'], train_2JHN['atom_index_1']]).nunique())
print('train_3JHC atom unique', pd.concat([train_3JHC['atom_index_0'], train_3JHC['atom_index_1']]).nunique())
print('train_3JHH atom unique', pd.concat([train_3JHH['atom_index_0'], train_3JHH['atom_index_1']]).nunique())
print('train_3JHN atom unique', pd.concat([train_3JHN['atom_index_0'], train_3JHN['atom_index_1']]).nunique())


# In[ ]:


test_1JHC = test[test['type'] == '1JHC']
test_1JHN = test[test['type'] == '1JHN']
test_2JHC = test[test['type'] == '2JHC']
test_2JHH = test[test['type'] == '2JHH']
test_2JHN = test[test['type'] == '2JHN']
test_3JHC = test[test['type'] == '3JHC']
test_3JHH = test[test['type'] == '3JHH']
test_3JHN = test[test['type'] == '2JHN']

print('test_1JHC atom unique', pd.concat([test_1JHC['atom_index_0'], test_1JHC['atom_index_1']]).nunique())
print('test_1JHN atom unique', pd.concat([test_1JHN['atom_index_0'], test_1JHN['atom_index_1']]).nunique())
print('test_2JHC atom unique', pd.concat([test_2JHC['atom_index_0'], test_2JHC['atom_index_1']]).nunique())
print('test_2JHH atom unique', pd.concat([test_2JHH['atom_index_0'], test_2JHH['atom_index_1']]).nunique())
print('test_2JHN atom unique', pd.concat([test_2JHN['atom_index_0'], test_2JHN['atom_index_1']]).nunique())
print('test_3JHC atom unique', pd.concat([test_3JHC['atom_index_0'], test_3JHC['atom_index_1']]).nunique())
print('test_3JHH atom unique', pd.concat([test_3JHH['atom_index_0'], test_3JHH['atom_index_1']]).nunique())
print('test_3JHN atom unique', pd.concat([test_3JHN['atom_index_0'], test_3JHN['atom_index_1']]).nunique())


# In[ ]:


plt.subplots(8, 2, figsize=(12, 24))
plt.subplot(8, 2, 1)
plt.hist(pd.concat([train_1JHC['atom_index_0'], train_1JHC['atom_index_1']]), bins=29)
plt.subplot(8, 2, 2)
plt.hist(pd.concat([test_1JHC['atom_index_0'], test_1JHC['atom_index_1']]), bins=29)
plt.subplot(8, 2, 3)
plt.hist(pd.concat([train_1JHN['atom_index_0'], train_1JHN['atom_index_1']]), bins=29)
plt.subplot(8, 2, 4)
plt.hist(pd.concat([test_1JHN['atom_index_0'], test_1JHN['atom_index_1']]), bins=29)
plt.subplot(8, 2, 5)
plt.hist(pd.concat([train_2JHC['atom_index_0'], train_2JHC['atom_index_1']]), bins=29)
plt.subplot(8, 2, 6)
plt.hist(pd.concat([test_2JHC['atom_index_0'], test_2JHC['atom_index_1']]), bins=29)
plt.subplot(8, 2, 7)
plt.hist(pd.concat([train_2JHH['atom_index_0'], train_2JHH['atom_index_1']]), bins=29)
plt.subplot(8, 2, 8)
plt.hist(pd.concat([test_2JHH['atom_index_0'], test_2JHH['atom_index_1']]), bins=29)
plt.subplot(8, 2, 9)
plt.hist(pd.concat([train_2JHN['atom_index_0'], train_2JHN['atom_index_1']]), bins=29)
plt.subplot(8, 2, 10)
plt.hist(pd.concat([test_2JHN['atom_index_0'], test_2JHN['atom_index_1']]), bins=29)
plt.subplot(8, 2, 11)
plt.hist(pd.concat([train_3JHC['atom_index_0'], train_3JHC['atom_index_1']]), bins=29)
plt.subplot(8, 2, 12)
plt.hist(pd.concat([test_3JHC['atom_index_0'], test_3JHC['atom_index_1']]), bins=29)
plt.subplot(8, 2, 13)
plt.hist(pd.concat([train_3JHH['atom_index_0'], train_3JHH['atom_index_1']]), bins=29)
plt.subplot(8, 2, 14)
plt.hist(pd.concat([test_3JHH['atom_index_0'], test_3JHH['atom_index_1']]), bins=29)
plt.subplot(8, 2, 15)
plt.hist(pd.concat([train_3JHN['atom_index_0'], train_3JHN['atom_index_1']]), bins=29)
plt.subplot(8, 2, 16)
plt.hist(pd.concat([test_3JHN['atom_index_0'], test_3JHN['atom_index_1']]), bins=29)
plt.show()


# <font size=5>Problem here.</font>

# <font size=3>Here is the problem, unbalanced data for 3JHH between train and test.
# If anywhere of my code is wrong? Or is it the truth?
# How to deal with that?</font>

# # For structure

# Reference: https://www.kaggle.com/chechir/molecular-eda-3d-plots

# In[ ]:


train = train.merge(structures, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'], how='left')
train = train.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], how='left', suffixes=('_a', '_b'))
train.drop(['atom_index_0', 'atom_index_1'], axis=1, inplace=True)

test['scalar_coupling_constant'] = np.nan
test = test.merge(structures, left_on=['molecule_name', 'atom_index_0'], right_on=['molecule_name', 'atom_index'], how='left')
test = test.merge(structures, left_on=['molecule_name', 'atom_index_1'], right_on=['molecule_name', 'atom_index'], how='left', suffixes=('_a', '_b'))
test.drop(['atom_index_0', 'atom_index_1'], axis=1, inplace=True)


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print('train set shape: {}, test set shape: {}'.format(train.shape, test.shape))


# In[ ]:


print('train atom_0 types: {}, train atom_1 types: {}'.format(train['atom_a'].nunique(), train['atom_b'].nunique()))
print('test atom_0 types: {}, test atom_1 types: {}'.format(test['atom_a'].nunique(), test['atom_b'].nunique()))


# # 3D analysis

# Reference: https://www.kaggle.com/pestipeti/interactive-3d-molecule-structure

# In[ ]:


import plotly.offline as py
import plotly.graph_objs as go

from plotly import tools

color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')
py.init_notebook_mode(connected=True)

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999

def show_molecule(mdata, mstruct):
    mdata = mdata.merge(right=mstruct, how='left',
                        left_on=['molecule_name', 'atom_index_0'],
                        right_on=['molecule_name', 'atom_index'])
    mdata.rename(index=str, columns={"x": "x0", "y": "y0", "z": "z0", "atom": "atom0"}, inplace=True)
    mdata.drop(['atom_index'], axis=1, inplace=True)

    mdata = mdata.merge(right=mstruct, how='left',
                  left_on=['molecule_name', 'atom_index_1'],
                  right_on=['molecule_name', 'atom_index']
                 )
    mdata.rename(index=str, columns={"x": "x1", "y": "y1", "z": "z1", "atom": "atom1"}, inplace=True)
    mdata.drop(['atom_index'], axis=1, inplace=True)    
    
    data = []
  
    atoms = mstruct['atom'].unique()
    types = mdata['type'].unique()
    
    atom_cfg = {
        'H': {"name": "Hydrogen", "color": "#757575", "size": 4},
        'C': {"name": "Carbon", "color": "#f44336", "size": 12},
        'O': {"name": "Oxygen", "color": "#03a9f4", "size": 12},
        'N': {"name": "Nitrogen", "color": "#ff9800", "size": 12},
        'F': {"name": "Fluorine", "color": "#673ab7", "size": 12},
    }
    
    type_cfg = {
        '2JHH': {"color": "#757575", "width": 2},
        '3JHH': {"color": "#757575", "width": 3},

        '1JHC': {"color": "#f44336", "width": 1},
        '2JHC': {"color": "#f44336", "width": 2},
        '3JHC': {"color": "#f44336", "width": 3},

        '1JHN': {"color": "#ff9800", "width": 2},
        '2JHN': {"color": "#ff9800", "width": 2},
        '3JHN': {"color": "#ff9800", "width": 3},
    }

    for atom, config in atom_cfg.items(): 
        if atom in atoms:
            data.append(
                go.Scatter3d(
                    x=mstruct[mstruct['atom'] == atom]['x'].values,
                    y=mstruct[mstruct['atom'] == atom]['y'].values,
                    z=mstruct[mstruct['atom'] == atom]['z'].values,
                    mode='markers',
                    marker=dict(
                        color=config['color'],
                        size=config['size'],
                        opacity=0.8
                    ),
                    name=config['name']
                )
            )

    for ctype, config in type_cfg.items():
        if ctype in types:
            eX = []; eY = []; eZ = []
            for row in mdata[mdata['type'] == ctype].iterrows():
                rd = row[1]
                eX += [rd['x0'], rd['x1']]
                eY += [rd['y0'], rd['y1']]
                eZ += [rd['z0'], rd['z1']]            
            
            data.append(
                go.Scatter3d(
                    x=eX,
                    y=eY,
                    z=eZ,
                    mode='lines',
                    line=dict(color=config['color'], width=config['width']),
                    name=ctype
                )
            )            

    axis=dict(showbackground=True, showline=False, zeroline=False, showgrid=True, showticklabels=False, title='')
    layout = go.Layout(
        margin=dict(l=50, r=50, b=50, t=50),
        width=720,
        height=640,
        showlegend=True,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='molecule')


# In[ ]:


structures = pd.read_csv('../input/structures.csv')
def show_train(name, train):
    print(name)
#     molecule = name
#     print(train[train['molecule_name'] == molecule])
    show_molecule(train[train['molecule_name'] == name], structures[structures['molecule_name'] == name])


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test['scalar_coupling_constant'] = np.nan
molecules = pd.concat([train, test])


# In[ ]:


for i in range(1, 5):
    show_train('dsgdb9nsd_00000'+str(i), molecules)
    show_train('dsgdb9nsd_00001'+str(i), molecules)
    show_train('dsgdb9nsd_00011'+str(i), molecules)
    show_train('dsgdb9nsd_10000'+str(i), molecules)


# In[ ]:


molecule = 'dsgdb9nsd_128739'
show_molecule(molecules[molecules['molecule_name'] == molecule], structures[structures['molecule_name'] == molecule])

