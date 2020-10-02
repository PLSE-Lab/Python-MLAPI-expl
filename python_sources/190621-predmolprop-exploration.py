#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# import data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
structures = pd.read_csv("../input/structures.csv")


# It's annoying to print the head of a Dataframe in stdout because it will often truncate the columns, so let's write a function to force it to print all columns.

# In[ ]:


def PrintDataframe(df):
    print(df.to_string())

PrintDataframe(train.head())


# In[ ]:


train.head()


# Let's find the number of molecule types, and how many times they appear. Interestingly, the value counts appear normally distributed. In retrospect, this information may not be so useful, since the molecule name will not be used a feature. According to the competition data description, "...no molecule in the training data is found in the test data."

# In[ ]:


# analyze molecule types
mol_name_col = train.molecule_name
print("number of unique molecules =",len(mol_name_col.unique()))
plt.figure(figsize=(3,3))
plt.ylabel('counts')
plt.xlabel('number of times a molecule appears')
plt.title('Train')
sns.distplot(mol_name_col.value_counts().values,kde=False,bins=range(120))
# cleanup
del mol_name_col


# In[ ]:


# analyze molecule types
mol_name_col = test.molecule_name
print("number of unique molecules =",len(mol_name_col.unique()))
plt.figure(figsize=(3,3))
plt.ylabel('counts')
plt.xlabel('number of times a molecule appears')
plt.title('Test')
sns.distplot(mol_name_col.value_counts().values,kde=False,bins=range(120))
# cleanup
del mol_name_col


# Next consider the coupling types. Notice that they are very unbalanced, but train and test are unbalanced in the same way.

# In[ ]:


# look at couping types
def VisualizeCounts(*cols):
    """
    takes a dataframe column (i.e. a Series)
    prints the counts and graphs 
    """
    fig, axs = plt.subplots(ncols=len(cols), figsize=(5*len(cols), 4))
    titles = ['train','test']
    for i,col in enumerate(cols):
        print(col.value_counts(),'\n')
        sns.countplot(col,ax = axs[i], order = sorted(col.unique()))
        axs[i].title.set_text(titles[i])

VisualizeCounts(train.type,test.type)


# If we consider the scalar coupling values, we see that they are basically clustered around 0, with a peak at around 80.

# In[ ]:


# looking at scalar_coupling values
sns.distplot(train.scalar_coupling_constant)


# Let's see how well the coupling type corresponds to the coupling constant.
# 
# Note that the coupling type can be easily interpreted. The number prefix tells you how many bonds there are between the two atoms; the last two letters are the two elements involved. For example, 1JHC, means that the coupling is between one hydrogen and one carbon atom, and there is only one bond between them. 
# 
# Since the coupling strength is inversely proportional to the number of bonds, 1JHN and 1JHC have the strongest coupling. 
# 
# All of the coupling constants are positive except for 2JHH, which is negative. 
# 
# Note that since the maximum number of bonds is three, there are at most two atoms in between the atoms of interest. 

# In[ ]:


g = sns.FacetGrid(train, col="type", col_order = sorted(train.type.unique()),sharex=False,sharey=False)
g.map(sns.distplot, "scalar_coupling_constant");


# We can extract the number of bonds and the atom types just using the coupling type. Note that the first atom in the coupling type is always H. You can confirm by yourself (using some code further down that extracts the atom type from the structures data) that atom_0 is always H. Therefore, we need only extract atom_type_1.

# In[ ]:


def DecomposeType(df):
    df['num_bonds'] = df.type.map(lambda s: int(s[0]))
    df['atom_1_type'] = df.type.map(lambda s: s[-1])
    df.drop(columns='type',inplace=True)


DecomposeType(train)
DecomposeType(test)

PrintDataframe(train.head())


# Let's see what we can get out of structures

# In[ ]:


PrintDataframe(structures.head())


# The most obvious feature to extract is the distance between atom 0 and atom 1. I copied the method in this [kernel](https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark) by Chanran Kim

# In[ ]:


def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop(['atom_index','atom'], axis=1)
    df = df.rename(columns={'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)


# In[ ]:


train_r0 = train[['x_0', 'y_0', 'z_0']].values
train_r1 = train[['x_1', 'y_1', 'z_1']].values
test_r0 = test[['x_0', 'y_0', 'z_0']].values
test_r1 = test[['x_1', 'y_1', 'z_1']].values

# calculate through space distance
train['space_dr'] = np.linalg.norm(train_r0 - train_r1, axis=1)
test['space_dr'] = np.linalg.norm(test_r0 - test_r1, axis=1)

# drop coordinates
train.drop(columns = ['x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1'], inplace = True)
test.drop(columns = ['x_0', 'y_0', 'z_0','x_1', 'y_1', 'z_1'], inplace = True)

PrintDataframe(train.head())


# It might also be useful to have an idea of what other atoms are in the molecule. 

# In[ ]:


# Group by molecule_name and atom, then use size() to count how many of each atom
# Then use unstack to make the atom types, which groupby made into indices, into columns
# Finally, because not all molecule contain all five atoms, fill NAs with 0
structure_atoms=structures.groupby(['molecule_name','atom'],sort=False).size().unstack('atom').fillna(0)
# calculate the total number of atoms
structure_atoms['total_atoms']=structure_atoms.sum(axis=1)
# remove index names (optional)
structure_atoms.columns = structure_atoms.columns.set_names(None)
structure_atoms.index=structure_atoms.index.set_names(None)
# rename columns
structure_atoms = structure_atoms.rename(columns={'C':'num_C','H':'num_H','N':'num_N',
                                                  'O':'num_O','F':'num_F'})
plt.figure(figsize=(3,3))
#plt.hist(structure_atoms.total_atoms,)
sns.distplot(structure_atoms.total_atoms,kde=False,bins=range(5,30))
plt.ylabel('counts')
plt.xlabel('number of atoms')

# extend train and test
train=pd.merge(train, structure_atoms, how = 'left', 
                     left_on  = ['molecule_name'], right_index = True)

test=pd.merge(test, structure_atoms, how = 'left', 
                     left_on  = ['molecule_name'], right_index = True)

PrintDataframe(train.head())


# In[ ]:


structure_atoms.describe()


# We can then use this information to confirm that the number of times a molecule appears in either the training set or test set is an indication of the molecule size.

# In[ ]:


# analyze molecule types
mol_name_col = train.molecule_name
mol_name_counts=mol_name_col.value_counts()
merged=pd.merge(mol_name_counts, structure_atoms, how = 'left', left_index = True, right_index = True)
plt.figure(figsize=(3,3))
sns.scatterplot(x="total_atoms",y="molecule_name", data = merged)
plt.ylabel('number appearances in train')
plt.xlabel('number of atoms')


# In[ ]:


# analyze molecule types
mol_name_col = test.molecule_name
mol_name_counts=mol_name_col.value_counts()
merged=pd.merge(mol_name_counts, structure_atoms, how = 'left', left_index = True, right_index = True)
plt.figure(figsize=(3,3))
sns.scatterplot(x="total_atoms",y="molecule_name", data = merged)
plt.ylabel('number appearances in test')
plt.xlabel('number of atoms')

