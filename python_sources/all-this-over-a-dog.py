#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import *

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')
print(train.shape, test.shape, sub.shape)

train['atom1'] = train['type'].map(lambda x: str(x)[2])
train['atom2'] = train['type'].map(lambda x: str(x)[3])
test['atom1'] = test['type'].map(lambda x: str(x)[2])
test['atom2'] = test['type'].map(lambda x: str(x)[3])

lbl = preprocessing.LabelEncoder()
for i in range(4):
    train['type'+str(i)] = lbl.fit_transform(train['type'].map(lambda x: str(x)[i]))
    test['type'+str(i)] = lbl.transform(test['type'].map(lambda x: str(x)[i]))

structures = pd.read_csv('../input/structures.csv').rename(columns={'atom_index':'atom_index_0', 'x':'x0', 'y':'y0', 'z':'z0', 'atom':'atom1'})
train = pd.merge(train, structures, how='left', on=['molecule_name', 'atom_index_0', 'atom1'])
test = pd.merge(test, structures, how='left', on=['molecule_name', 'atom_index_0', 'atom1'])
del structures

structures = pd.read_csv('../input/structures.csv').rename(columns={'atom_index':'atom_index_1', 'x':'x1', 'y':'y1', 'z':'z1', 'atom':'atom2'})
train = pd.merge(train, structures, how='left', on=['molecule_name', 'atom_index_1', 'atom2'])
test = pd.merge(test, structures, how='left', on=['molecule_name', 'atom_index_1', 'atom2'])
del structures

train['structure_pos_dif'] = train['atom_index_0'] - train['atom_index_1']
test['structure_pos_dif'] = test['atom_index_0'] - test['atom_index_1']

mc = pd.read_csv('../input/mulliken_charges.csv').rename(columns={'atom_index':'atom_index_0'})
train = pd.merge(train, mc, how='left', on=['molecule_name', 'atom_index_0'])
test = pd.merge(test, mc, how='left', on=['molecule_name', 'atom_index_0'])
del mc

scc = pd.read_csv('../input/scalar_coupling_contributions.csv')
train = pd.merge(train, scc, how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])
test = pd.merge(test, scc, how='left', on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])
del scc
print(train.shape, test.shape, sub.shape)


# In[ ]:


train.drop(columns=['id', 'molecule_name', 'atom1', 'atom2', 'atom_index_0', 'atom_index_1'], inplace=True)
test.drop(columns=['molecule_name', 'atom1', 'atom2', 'atom_index_0', 'atom_index_1'], inplace=True)
print(train.shape, test.shape, sub.shape)


# In[ ]:


#https://www.kaggle.com/artgor/molecular-properties-eda-and-models
def fdist(df):
    p0 = df[['x0', 'y0', 'z0']].values
    p1 = df[['x1', 'y1', 'z1']].values
    df['dist'] = np.linalg.norm(p0 - p1, axis=1)
    df['dist_to_type_mean'] = df['dist'] / df.groupby('type')['dist'].transform('mean')
    df['dist_x'] = np.square(df['x0'] - df['x1'])
    df['dist_x_to_type_mean'] = df['dist_x'] / df.groupby('type')['dist_x'].transform('mean')
    df['dist_y'] = np.square(df['y0'] - df['y1'])
    df['dist_y_to_type_mean'] = df['dist_y'] / df.groupby('type')['dist_y'].transform('mean')
    df['dist_z'] = np.square(df['z0'] - df['z1'])
    df['dist_z_to_type_mean'] = df['dist_z'] / df.groupby('type')['dist_z'].transform('mean')
    return df

train = fdist(train)
test = fdist(test)
print(train.shape, test.shape, sub.shape)


# In[ ]:


def features(df, df2):
    for c in ['mulliken_charge', 'fc', 'sd', 'pso', 'dso', 'dist', 'dist_to_type_mean']:
        for agg in ['min', 'max', 'sum', 'mean']:
            tmp = eval('df.groupby(["type"], as_index=False)[c].' + agg + '().rename(columns={"' + c + '":"' + agg + '"+ "_" +"' + c + '"})')
            df = pd.merge(df, tmp, how='left', on=['type'])
            df2 = pd.merge(df2, tmp, how='left', on=['type'])
    return df, df2

train, test = features(train, test)
print(train.shape, test.shape)


# In[ ]:


def features(df, df2):
    for c in ['mulliken_charge', 'fc', 'sd', 'pso', 'dso', 'dist']:
        print(c)
        for agg in ['min', 'max', 'sum', 'mean']:
            c_ = agg + '_' + c
            tmp = eval('df.groupby(["type"], as_index=False)[c].' + agg + '().rename(columns={"' + c + '":"' + c_ + '"})')
            df = pd.merge(df, tmp, how='left', on=['type'])
            df2 = pd.merge(df2, tmp, how='left', on=['type'])
        df.drop(columns=[c], inplace=True)
        df2.drop(columns=[c], inplace=True)
    return df, df2

train, test = features(train, test)
print(train.shape, test.shape)


# In[ ]:


def features(df):
    for c in ['0', '1']:
        col = [c1 + c  for c1 in ['x','y','z']]
        for agg in ['min', 'max', 'sum', 'mean', 'std', 'skew', 'kurtosis']:
            df[c+agg] = eval('df[col].' + agg + '(axis=1)')
            df[c+'a'+agg] = eval('df[col].abs().' + agg + '(axis=1)')
    return df

train = features(train).fillna(0)
test = features(test).fillna(0)
print(train.shape, test.shape)


# In[ ]:


col = [c for c in train.columns if c not in ['scalar_coupling_constant', 'type']]
reg = ensemble.ExtraTreesRegressor(n_jobs=-1, n_estimators=10, random_state=4)

reg.fit(train[col], train['scalar_coupling_constant'])
test['scalar_coupling_constant']  = reg.predict(test[col])
test[['id', 'scalar_coupling_constant']].to_csv('submission.csv', index=False)

