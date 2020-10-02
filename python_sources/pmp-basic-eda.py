#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[ ]:


import os
import numpy as np
import pandas as pd
from pathlib import Path

print(os.listdir("../input"))


# ### Reading and inspecting data

# In[ ]:


get_ipython().run_cell_magic('time', '', "path=Path('../input')\ntrain=pd.read_csv(path/'train.csv')\ntest=pd.read_csv(path/'test.csv')\nstruct=pd.read_csv(path/'structures.csv')")


# In[ ]:


print(f'The shape of train is {train.shape}')
print(f'The shape of test is {test.shape}')
print(f'The shape of struct is {struct.shape}')
print(f"\nThe number of NA's in train is {train.isna().sum().sum()}.")
print(f"The number of NA's in test is {test.isna().sum().sum()}.")
print(f"The number of NA's in struct is {struct.isna().sum().sum()}.")
print(f"\nThe column names of train are \n{train.columns}.")
print(f"\nThe column names of test are \n{test.columns}.")
print(f"\nThe column names of struct are \n{struct.columns}.")


# In[ ]:


train.head(20)


# In[ ]:


test.head(20)


# In[ ]:


struct.head(20)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ncols=test.columns\n\ndata=pd.concat([train[cols], test])')


# In[ ]:


data.describe()


# ### Counting values

# In[ ]:


print(f"The number of molecules in the train set is {train['molecule_name'].nunique()}.")
print(f"The number of molecules in the test set is {test['molecule_name'].nunique()}")

print(f"\nThe number of interaction types in the train set is {train['type'].nunique()}.")
print(f"The number of interaction types in the test set is {test['type'].nunique()}")
print(f"The number of interaction types in the train and test sets combined is {data['type'].nunique()}")

print(f"\nThe number of atomic types in struct is {struct['atom'].nunique()}")


# In[ ]:


types_count = data['type'].value_counts()
types_order = types_count.index.values
types_count


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns

width = 22
height = 7
fs = '24'

plt.figure(figsize=(width, height))

sns.set(font_scale=1.6)

plt.subplot(1, 3, 1)
sns.countplot(data['type'], order=types_order)
plt.title('train+test', fontsize=fs)
plt.ylabel('Counts')

plt.subplot(1, 3, 2)
sns.countplot(train['type'], order=types_order)
plt.title('train', fontsize=fs)
plt.ylabel('Counts')

plt.subplot(1, 3, 3)
sns.countplot(test['type'], order=types_order)
plt.title('test', fontsize=fs)
plt.ylabel('Counts')

plt.tight_layout()


# The distributions of the interaction types are very similar in train and test.

# In[ ]:


struct['atom'].value_counts()


# In[ ]:


width = 6
height = 6
fs = '20'
fs_label = '17'

plt.figure(figsize=(width, height))

sns.set(font_scale=1.2)

sns.countplot(x='atom', data=struct, order = struct['atom'].value_counts().index)
plt.title("Atoms present in the 'struct' file", fontsize=fs)
plt.xlabel('Atoms', fontsize=fs_label)
plt.ylabel('Counts', fontsize=fs_label)

plt.tight_layout()


# In[ ]:


print(f"The minimum values of the 'atom_index_1' and 'atom_index_2' are {data.atom_index_0.min()}"       f" and {data.atom_index_1.min()}, respectively.")

print(f"The maximum values of the 'atom_index_1' and 'atom_index_2' are {data.atom_index_0.max()}"       f" and {data.atom_index_1.max()}, respectively.")


# Thus, the largest number of atoms in a molecule is 29.

# Let's take a little bit closer look at the number of atoms per molecule.

# In[ ]:


atom_counts = data.groupby('molecule_name').size().reset_index(name='count')
atom_counts.head()


# In[ ]:


atom_counts.tail()


# In[ ]:


print(f"The total number of molecules in train and test is {len(atom_counts)}.")
print(f"The minimum number of couplings per molecule is {np.min(atom_counts['count'].values)}.")
print(f"The maximum number of couplings per moluecule is {np.max(atom_counts['count'].values)}.")


# ## Scalar Coupling Constants
# 
# Let's take a look at the numerical values of the scalar coupling counstants in the train set. We will be groupping our observations by the interaction type.
# 
# The list of unique coupling types:

# In[ ]:


coupling_types = train['type'].unique()

coupling_types


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nvsize = 4\nhsize = 2\n\nplt.figure()\nfig, ax = plt.subplots(vsize,hsize,figsize=(18,20))\n\nfor (i, ct) in enumerate(coupling_types):\n    i += 1\n    plt.subplot(vsize, hsize, i)\n\n    sns.distplot(train.loc[train[\'type\'] == ct, \'scalar_coupling_constant\'], color=\'blue\', bins=60, label=ct)\n    \n    plt.title("Scalar Coupling Type "+ct, fontsize=\'20\')\n    plt.xlabel(\'Scalar Coupling Constant\', fontsize=\'16\')\n    plt.ylabel(\'Density\', fontsize=\'16\')\n    locs, labels = plt.xticks()\n    plt.tick_params(axis=\'x\', which=\'major\', labelsize=16)#, pad=-40)\n    plt.tick_params(axis=\'y\', which=\'major\', labelsize=16)\n    #plt.legend(loc=\'best\', fontsize=\'16\')\n    \nplt.tight_layout()    \nplt.show()')


# In[ ]:




