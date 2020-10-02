#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import random
import ase
from ase import Atoms

import ase.visualize
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression

import seaborn as sns
sns.set()
import os
print(os.listdir("../input"))
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
plt.rcParams["patch.force_edgecolor"] = True
from IPython.display import display
import seaborn as sns
sns.set(style="whitegrid", palette="muted", color_codes=True)
sns.set_style("whitegrid", {'grid.linestyle': '--'})
red = sns.xkcd_rgb["light red"]
green = sns.xkcd_rgb["medium green"]
blue = sns.xkcd_rgb["denim blue"]


# In[ ]:


structures = pd.read_csv('../input/structures.csv')
train = pd.read_csv('../input/train.csv')
train["target_exp"] = np.log1p(np.exp(train.scalar_coupling_constant))/10
test = pd.read_csv("../input/test.csv")
scalar_coupling_contributions = pd.read_csv('../input/scalar_coupling_contributions.csv')
sample_submission= pd.read_csv("../input/sample_submission.csv")


# In[ ]:


structures.head()


# In[ ]:


random_molecule = random.choice(structures['molecule_name'].unique())
molecule = structures[structures['molecule_name'] == random_molecule]
display(molecule)


# In[ ]:


atoms = molecule.iloc[:, 3:].values
print(atoms)


# In[ ]:


symbols = molecule.iloc[:, 2].values
print(symbols)


# In[ ]:


def view(molecule):
    # Select a molecule
    mol = structures[structures['molecule_name'] == molecule]
    
    # Get atomic coordinates
    xcart = mol.iloc[:, 3:].values
    
    # Get atomic symbols
    symbols = mol.iloc[:, 2].values
    
    # Display molecule
    system = Atoms(positions=xcart, symbols=symbols)
    print('Molecule Name: %s.' %molecule)
    return ase.visualize.view(system, viewer="x3d")

random_molecule = random.choice(structures['molecule_name'].unique())
view(random_molecule)


# In[ ]:


train.head()


# In[ ]:


pe=pd.read_csv("../input/potential_energy.csv")
pe.head()


# In[ ]:


pe['potential_energy'].plot(kind='hist',
                              figsize=(15, 5),
                              bins=200,
                              title='Distribution of Potential Energy',
                              color='b')
plt.show()


# In[ ]:


dipoles=pd.read_csv("../input/dipole_moments.csv")
dipoles.head()
dipoles["total"] = np.abs(dipoles.X) + np.abs(dipoles.Y) + np.abs(dipoles.Z)


# In[ ]:


plt.figure(figsize=(20,5))
sns.distplot(dipoles.X, label="X")
sns.distplot(dipoles.Y, label="Y")
sns.distplot(dipoles.Z, label="Z")

plt.legend();


# In[ ]:


mulliken=pd.read_csv("../input/mulliken_charges.csv")
mulliken.head()


# In[ ]:


plt.figure(figsize=(5,5))
sns.distplot(mulliken.atom_index,label="X")
sns.distplot(mulliken.mulliken_charge,label="Y")
plt.legend();


# In[ ]:


plt.scatter(mulliken.atom_index, mulliken.mulliken_charge, alpha=0.5)
plt.show()


# In[ ]:


mst = pd.read_csv("../input/magnetic_shielding_tensors.csv")
mst.head()


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(mst.XZ, label="Z")
sns.distplot(mst.XY, label="Y")
sns.distplot(mst.XX, label="X")


plt.legend();


# In[ ]:


train['type_0'] = train['type'].apply(lambda x: x[0]).astype(int)
train['type_1'] = train['type'].apply(lambda x: x[2:3])
train['type_2'] = train['type'].apply(lambda x: x[3:4])
train.head()


# In[ ]:


sns.violinplot(x="type_0", y="atom_index_1",
               split=True, data=train)


# In[ ]:


sns.violinplot(x="type_1", y="atom_index_1",
               split=True, data=train)


# In[ ]:


sns.violinplot(x="type_2", y="atom_index_1",
               split=True, data=train)


# In[ ]:


scc = pd.read_csv('../input/scalar_coupling_contributions.csv')
scc.head()
scc.groupby('type').count()['molecule_name'].sort_values().plot(kind='barh',
                                                                color='blue',
                                                               figsize=(15, 5)
                                                               )


# Using Nim J model
# "https://www.kaggle.com/namesj/beating-benchmark-with-linear-regression"

# In[ ]:


train = train[train.atom_index_0!=0] 

train = pd.merge(train, structures, left_on  = ['molecule_name', 'atom_index_0'],
                  right_on = ['molecule_name',  'atom_index'], how='left')
train = pd.merge(train, structures, left_on  = ['molecule_name', 'atom_index_1'],
                  right_on = ['molecule_name',  'atom_index'], how='left')
test = pd.merge(test, structures, left_on  = ['molecule_name', 'atom_index_0'],
                  right_on = ['molecule_name',  'atom_index'], how='left')
test = pd.merge(test, structures, left_on  = ['molecule_name', 'atom_index_1'],
                  right_on = ['molecule_name',  'atom_index'], how='left')

train = pd.merge(train, scalar_coupling_contributions, how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])
test = pd.merge(test, scalar_coupling_contributions, how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])

lbl = LabelEncoder()
for i in range(4):
    train['type'+str(i)] = lbl.fit_transform(train['type'].map(lambda x: str(x)[i]))
    test['type'+str(i)] = lbl.transform(test['type'].map(lambda x: str(x)[i]))

y = train.scalar_coupling_constant.values
train.drop(['id', 'molecule_name', 'atom_index_0','atom_index_1', 'atom_index_x', 'atom_index_y', 'scalar_coupling_constant', 'type'], axis=1, inplace=True)
test.drop(['id', 'molecule_name', 'atom_index_0','atom_index_1', 'atom_index_x', 'atom_index_y', 'type'], axis=1, inplace=True)

def get_dummies(train, test):
    encoded = pd.get_dummies(pd.concat([train,test], axis=0))
    train_rows = train.shape[0]
    train = encoded.iloc[:train_rows, :]
    test = encoded.iloc[train_rows:, :] 
    return train,test

train, test = get_dummies(train, test)

X = train.loc[:,['x_x', 'y_x', 'z_x', 'x_y', 'y_y', 'z_y', 'type0', 'type1',
       'type2', 'type3', 'atom_x_H', 'atom_y_C', 'atom_y_H', 'atom_y_N']]
X_test = test.loc[:,['x_x', 'y_x', 'z_x', 'x_y', 'y_y', 'z_y', 'type0', 'type1',
       'type2', 'type3', 'atom_x_H', 'atom_y_C', 'atom_y_H', 'atom_y_N']]

dt = KNeighborsRegressor()
test['fc'] = dt.fit(X, train.fc.values).predict(X_test)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train.fc.values.reshape(-1, 1), y)
sample_submission['scalar_coupling_constant'] = lr.predict(test.fc.values.reshape(-1, 1))
sample_submission.head()
sample_submission.to_csv('submission.csv', index = False)

