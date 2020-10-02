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
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Reading data files

# In[ ]:


train_data=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
ml_ch=pd.read_csv("../input/mulliken_charges.csv")
dp_mts=pd.read_csv("../input/dipole_moments.csv")
structures=pd.read_csv("../input/structures.csv")
mngt_she_tensors=pd.read_csv("../input/magnetic_shielding_tensors.csv")
pot_ener=pd.read_csv("../input/potential_energy.csv")
scc=pd.read_csv("../input/scalar_coupling_contributions.csv")


# training data

# 

# In[ ]:


print(train_data.shape)
print(train_data.molecule_name.nunique())
train_data.head()


# about 'dsgdb9nsd_000001' molecule (CH4) 

# In[ ]:


train_data[train_data['molecule_name']=='dsgdb9nsd_000001']


# unique molecules list in train

# In[ ]:


#train_data.atom_index_0.value_counts()


# In[ ]:


mols_train=train_data['molecule_name'].unique()


# In[ ]:


structures.head()


# In[ ]:


structures.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# dipolements

# In[ ]:


print(dp_mts.shape)
print(dp_mts.molecule_name.nunique())
dp_mts.head()


# dipolemoments represents complete molecule

# In[ ]:





# Mullikan charges

# In[ ]:


print(ml_ch.shape)
print(ml_ch.molecule_name.nunique())
ml_ch.head()


# structures

# In[ ]:


print(structures.shape)
print(structures.molecule_name.nunique())
structures.head()


# about 'dsgdb9nsd_000001' molecule (CH4)  structure

# In[ ]:


structures[structures['molecule_name']=='dsgdb9nsd_000001']


# test data

# In[ ]:


print(test_data.shape)
print(test_data.molecule_name.nunique())
test_data.head()


# lets check atom  'dsgdb9nsd_000004' in test data

# In[ ]:


test_data.atom_index_0.value_counts()


# In[ ]:


test_data[test_data['molecule_name']=='dsgdb9nsd_000004']


# In[ ]:


85003+45772


# In[ ]:


mols_test=test_data['molecule_name'].unique()


# In[ ]:


mols_test


# total molecules in train and test is equal to 130775, which are given in total molecules in 'structure' data 

# In[ ]:





# magnetic sheild tensors

# In[ ]:


print(mngt_she_tensors.shape)
print(mngt_she_tensors.molecule_name.nunique())
mngt_she_tensors.head(10)


# about 'dsgdb9nsd_000001' molecule (CH4)  magnetic sheild tensors

# In[ ]:


mngt_she_tensors[mngt_she_tensors['molecule_name']=='dsgdb9nsd_000001']


# In[ ]:





# potential energies

# In[ ]:


print(pot_ener.shape)
print(pot_ener.molecule_name.nunique())
pot_ener.head()


# potemtial energy represent about complete molecule

# In[ ]:





# scalar_coupling_contributions

# In[ ]:


print(scc.shape)
print(scc.molecule_name.nunique())
scc.head()


# In[ ]:


scc[scc['molecule_name']=='dsgdb9nsd_000001']


# scc represents individual atom-atom and direction

# 

# In[ ]:


merged1_train=pd.merge(train_data,scc,on=['molecule_name','atom_index_0','atom_index_1'])


# In[ ]:


merged1_train


# In[ ]:


structures


# In[ ]:


pd.merge(test_data,scc,on=['molecule_name','atom_index_0','atom_index_1'])


# In[ ]:


test_data


# In[ ]:





# In[ ]:


structures.groupby(['molecule_name','atom'])['atom'].size()


# creating dummy variable for train

# In[ ]:





# In[ ]:


train_data.dtypes


# In[ ]:


train_data['atom_index_0']=train_data['atom_index_0'].astype('category')
train_data['atom_index_1']=train_data['atom_index_1'].astype('category')


# In[ ]:


train_data.dtypes


# In[ ]:


train_data[train_data['atom_index_0']==0]
train_data.drop([1600734,1600735,1600736],inplace=True)


# In[ ]:


dummies_type_train=pd.get_dummies(train_data['type'])
dummies_index0_train=pd.get_dummies(train_data['atom_index_0'],prefix='0_index')
dummies_index1_train=pd.get_dummies(train_data['atom_index_1'],prefix='1_index')


# In[ ]:


print(dummies_index0_train.shape)
print(dummies_index1_train.shape)


# In[ ]:





# In[ ]:


dummies_index0_train.columns


# creating dummy variable for test

# In[ ]:


dummies_type_test=pd.get_dummies(test_data['type'])
dummies_index0_test=pd.get_dummies(test_data['atom_index_0'],prefix='0_index')
dummies_index1_test=pd.get_dummies(test_data['atom_index_1'],prefix='1_index')


# In[ ]:


dummies_index0_test.columns


# In[ ]:


print(dummies_index0_test.shape)
print(dummies_index1_test.shape)


# In[ ]:


dummies_index0_train


# In[ ]:


train_data[train_data.atom_index_0=='0']


# In[ ]:


dummies_index0_train=dummies_index0_train.drop('0_index_0',axis=1)


# In[ ]:


X_train=pd.concat([dummies_type_train,dummies_index0_train,dummies_index1_train],axis=1).values


# In[ ]:


X_train.shape


# In[ ]:


X_test=pd.concat([dummies_type_test,dummies_index0_test,dummies_index1_test],axis=1).values


# In[ ]:


y_train=train_data['scalar_coupling_constant'].values


# In[ ]:


y_train.shape


# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


ridge = Ridge(alpha=0.1, normalize=True) 


# In[ ]:


ridge.fit(X_train,y_train)


# In[ ]:


y_pred=ridge.predict(X_test)


# In[ ]:


sample=pd.read_csv("../input/sample_submission.csv")


# In[ ]:


sample.columns


# In[ ]:


sample['scalar_coupling_constant']=y_pred


# In[ ]:


sample.to_csv('sample.csv',index=False)


# In[ ]:


#practice


# In[ ]:


train_data


# In[ ]:


distance=[]
for i in range(0,len(structures)):
    distance.append(np.sqrt((structures['x'][i])**2+(structures['y'][i])**2+(structures['z'][i])**2))


# In[ ]:


structures['distance']=distance


# In[ ]:


structures

