#!/usr/bin/env python
# coding: utf-8

# This kernel contains feature engineering using the molecule structures file and subsequent training of several lightgbm estimators and prediction.
# The training set is split because training all at once has led to out of memory.
# The most recent addition are the openbabel features.

# In[ ]:


get_ipython().system('conda install -c openbabel openbabel -y')


# In[ ]:


# Do the standard imports and have a look at the data 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import KFold, StratifiedKFold

print(os.listdir("../input"))

"""
Reduce Mem usage function taken from here:
https://www.kaggle.com/artgor/artgor-utils
"""
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
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
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max and c_prec == np.finfo(np.float16).precision:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

df_struct = pd.read_csv('../input/structures.csv')

# Any results you write to the current directory are saved as output.

df_struct = reduce_mem_usage(df_struct)

print('The largest molecule has {m} atoms'.format(m=str(df_struct['atom_index'].max())))


# In[ ]:


"""
The type 3JHC has 1.51 M records. This is important later on since we split the training by type
"""
# df = pd.read_csv('../input/train.csv')
# df = reduce_mem_usage(df)
# type_list = list(df['type'].unique())
# for typ in type_list:
#     print(typ + ' ' + str(df[df['type'] == typ].shape[0]))


# In[ ]:


df_struct['atom'].unique()


# In[ ]:


# df = pd.read_csv('../input/train.csv')
# df.shape


# In[ ]:


import openbabel
obConversion = openbabel.OBConversion()
obConversion.SetInFormat("xyz")
xyz_path = '../input/structures'


# In[ ]:


# !ls ../input/structures


# This kernel generates some Features and writes to files for further processing.
# - Distance between the atoms
# - OHE encoded connection type and and atom types
# - Distance to all other atoms in the molecule and the cosine of the angle w.r.t. the vector between the atoms
# 
# Training is done with lightgbm with custom loss and objective. The data split has to be split in three equal parts to prevent memory exhaustion. This drags on the result.
# 
# Finally Feature importance and a graphical representation of the results are shown.

# the feature engineering is done with matrix style operations. While it is not as flexible as going line by line it is far more performant.
# 
# 

# In[ ]:


import progressbar as pb

# there are 29 molecules in the largest molecule
n = 29
df_molecules = pd.DataFrame(df_struct['molecule_name'].unique())
df_molecules.columns = ['molecule_name']
# next part is to retrieve some features via the openbabel module

def enrich_row(row):
    mol = openbabel.OBMol()
    obConversion.ReadFile(mol, '{xyz_path}/{molecule_name}.xyz'.format(xyz_path=xyz_path, molecule_name=row['molecule_name']))
    row['num_atoms'] = mol.NumAtoms()
    row['num_bonds'] = mol.NumBonds()
    row['num_hvy'] = mol.NumHvyAtoms()
    row['num_residues'] = mol.NumResidues()
    row['num_rotors']  = mol.NumRotors()
    row['mol_energy'] = mol.GetEnergy()
    row['mol_wt'] = mol.GetMolWt()
    row['mol_mass'] = mol.GetExactMass()
    row['mol_charge'] = mol.GetTotalCharge()
    row['mol_spin'] = mol.GetTotalSpinMultiplicity()
    row['mol_dimension'] = mol.GetDimension()
    return row
    
df_molecules = df_molecules.apply(enrich_row, axis=1)

df_molecules.head()


# In[ ]:


for i in range(n):
    rsuff = '_{idx}'.format(idx=str(i))
    df_molecules = df_molecules.merge(df_struct[df_struct['atom_index'] == i], how='left',
                                     on='molecule_name',
                                     suffixes=('', rsuff))
    drop_cols = ['atom_index']
    df_molecules = df_molecules.drop(drop_cols, axis=1)


# rename the first columns for ease of use later on
df_molecules = df_molecules.rename(index=str, columns={'atom': 'atom_0', 'x': 'x_0', 'y': 'y_0', 'z': 'z_0'})
df_molecules.head()


# In[ ]:


print(df_molecules.shape)


# Next step is to do the actual feature engineering. The steps are as follows:
# - Join every line with its corresponding molecule structure
# - Rotate all the (x,y,z) cartesian coordinates such that atom 1 is at (0,0,0) and atom 2 is at (x,0,0)
# - OHE the type and all atom types

# In[ ]:


df_train = pd.read_csv('../input/train.csv')

print(df_train.shape)
atom_list = list(df_struct['atom'].unique())
type_list = list(df_train['type'].unique())

# save relative freq for later use as weights. This is meant to improve adherence to the weighted objective

weight_dict = dict()
for typ in type_list:
    weight_dict[typ] = float(df_train.shape[0]) / float(df_train[df_train['type'] == typ].shape[0])

n_train = df_train.shape[0]
def get_features(df_train, df_struct, atom_list, type_list, n=29):
    df_train = df_train.merge(df_struct, how='left',
                              left_on=['molecule_name', 'atom_index_0'],
                              right_on=['molecule_name', 'atom_index']).drop('atom_index', axis=1)
    df_train = df_train.rename(index=str, columns={'atom': 'atom_l', 'x': 'x_l', 'y': 'y_l', 'z': 'z_l'})
    df_train = df_train.merge(df_struct, how='left',
                              left_on=['molecule_name', 'atom_index_1'],
                              right_on=['molecule_name', 'atom_index']).drop('atom_index', axis=1)
    df_train = df_train.rename(index=str, columns={'atom': 'atom_r', 'x': 'x_r', 'y': 'y_r', 'z': 'z_r'})
    df_train = df_train.merge(df_molecules, how='left', on='molecule_name')
    # make OHE
    # for typ in type_list:
    #     colname = 'type_' + typ
    #     df_train[colname] = df_train['type'] == typ
    # df_train = df_train.drop('type', axis=1)
    cat_cols = ['atom_l', 'atom_r']
    for atom in atom_list:
        df_train['count_{atom}'.format(atom=atom)] = 0
    for i in range(n):
        cat_cols.append('atom_{idx}'.format(idx=str(i)))
    for cat_col in cat_cols:
        for atom in atom_list:
            colname = cat_col + '_' + atom
            df_train[colname] = df_train[cat_col] == atom
            df_train['count_{atom}'.format(atom=atom)] += df_train[cat_col] == atom
        df_train = df_train.drop(cat_col, axis=1)
    # make the atom_index_0 atom the (0,0,0) in the coordinate system
    for dimcol in ['x', 'y', 'z']:
        for lridx in ['l', 'r']:
            tgtcol = dimcol+'_'+lridx
            srccol = dimcol+'_l'
            df_train[tgtcol] = df_train[tgtcol] - df_train[srccol]
    # now we can drop the *_r columns again as they are all 0
    df_train = df_train.drop(['x_l', 'y_l', 'z_l'], axis=1)
    # now rotate everything such that atom_index_1 is at (x, 0, 0)
    # first rotate around (0, 0, 1)
    df_train['dist'] = np.sqrt(df_train['y_r']*df_train['y_r']+df_train['x_r']*df_train['x_r'])
    df_train['sintheta'] = -df_train['y_r']/df_train['dist']
    df_train['costheta'] = df_train['x_r']/df_train['dist']
    for i in range(n):
        x_colname = 'x_' + str(i)
        y_colname = 'y_' + str(i)
        df_train['x_tmp'] = df_train[x_colname]
        df_train[x_colname] = df_train[x_colname]*df_train['costheta']-df_train[y_colname]*df_train['sintheta']
        df_train[y_colname] = df_train['x_tmp']*df_train['sintheta']+df_train[y_colname]*df_train['costheta']
    x_colname = 'x_r'
    y_colname = 'y_r'
    df_train['x_tmp'] = df_train[x_colname]
    df_train[x_colname] = df_train[x_colname]*df_train['costheta']-df_train[y_colname]*df_train['sintheta']
    df_train[y_colname] = df_train['x_tmp']*df_train['sintheta']+df_train[y_colname]*df_train['costheta']
    # now rotate around (0, 1, 0)
    df_train['dist'] = np.sqrt(df_train['z_r']*df_train['z_r']+df_train['x_r']*df_train['x_r'])
    df_train['sintheta'] = -df_train['z_r']/df_train['dist']
    df_train['costheta'] = df_train['x_r']/df_train['dist']
    for i in range(n):
        x_colname = 'x_' + str(i)
        z_colname = 'z_' + str(i)
        df_train['x_tmp'] = df_train[x_colname]
        df_train[x_colname] = df_train[x_colname]*df_train['costheta']-df_train[z_colname]*df_train['sintheta']
        df_train[z_colname] = df_train['x_tmp']*df_train['sintheta']+df_train[z_colname]*df_train['costheta']
    x_cols = []
    y_cols = []
    z_cols = []
    for i in range(n):
        x_cols.append('x_{num}'.format(num=str(i)))
        y_cols.append('y_{num}'.format(num=str(i)))
        z_cols.append('z_{num}'.format(num=str(i)))        
    df_train['x_min'] = df_train[x_cols].min(axis=1)
    df_train['x_max'] = df_train[x_cols].max(axis=1)
    df_train['x_std'] = df_train[x_cols].std(axis=1)
    df_train['y_min'] = df_train[y_cols].min(axis=1)
    df_train['y_max'] = df_train[y_cols].max(axis=1)
    df_train['y_std'] = df_train[y_cols].std(axis=1)
    df_train['z_min'] = df_train[z_cols].min(axis=1)
    df_train['z_max'] = df_train[z_cols].max(axis=1)
    df_train['z_std'] = df_train[z_cols].std(axis=1)
    
    for i in range(n):
        dist_colname = 'distance_' + str(i)
        x_colname = 'x_' + str(i)
        y_colname = 'y_' + str(i)
        z_colname = 'z_' + str(i)
        df_train[dist_colname] = df_train[x_colname]*df_train[x_colname]+df_train[y_colname]*df_train[y_colname]+df_train[z_colname]*df_train[z_colname]
        for spatial_colname in [x_colname, y_colname, z_colname]:
            df_train[spatial_colname] = df_train[spatial_colname]/df_train[dist_colname]
    x_colname = 'x_r'
    z_colname = 'z_r'
    df_train['x_tmp'] = df_train[x_colname]
    df_train[x_colname] = df_train[x_colname]*df_train['costheta']-df_train[z_colname]*df_train['sintheta']
    df_train[z_colname] = df_train['x_tmp']*df_train['sintheta']+df_train[z_colname]*df_train['costheta']
    # lets drop everything that is useless and lets give it a try
    df_train = df_train.drop(['x_tmp',
                              'y_r',
                              'z_r',
                              'sintheta',
                              'costheta',
                              'dist',
                              'molecule_name',
                              'id',
                              'atom_index_0',
                              'atom_index_1'], axis=1)
    return df_train

# df_train = get_features(df_train, df_struct, atom_list, type_list)
# print(df_train.shape)
# df_train.head()


# The ensemble class helps training multiple estimators on different parts of the data. This is a workaround to prevent memory exhaust.

# In[ ]:



del df_train

import lightgbm as lgb
from sklearn.model_selection import train_test_split


class lgb_ensemble(object):
    def __init__(self):
        self.estimators = []
        self.maxlen = 800_000 # maximum allowed chunk size to prevent lgb memory overflow
        self.n_estimators = 15000

    def train(self, df_train, target, params, typ):
        chunk_start = 0
        chunk_end = self.maxlen
        kf = KFold(n_splits=4)
        while chunk_start < df_train.shape[0]:
            X = df_train.iloc[chunk_start:chunk_end, :]
            Y = target.iloc[chunk_start:chunk_end]
            for train_index, test_index in kf.split(X):
                # we consider the chunk df_train[chunk_start:chunk_end] in each iteration
                x_train, x_valid = X.iloc[train_index, :], X.iloc[test_index, :]
                y_train, y_valid = Y.iloc[train_index], Y.iloc[test_index]
                # x_train, x_valid, y_train, y_valid = train_test_split(df_train, target, test_size=0.15)
                d_train = lgb.Dataset(x_train, label=y_train)
                d_valid = lgb.Dataset(x_valid, label=y_valid)
                # The type_list is defined outside if this class, using it here again can be considered sloppy. However i dont care, call the police if you need to
                # cust_loss = lambda y_hat, d_train: self.custom_loss(y_hat, d_train, type_list, x_valid)
                # cust_objective = lambda y_hat, d_train: self.custom_objective(y_hat, d_train, type_list, x_train)
                watchlist = [d_valid]
                estim = lgb.train(params,
                                    d_train,
                                    self.n_estimators,
                                    watchlist,
                                    verbose_eval=1000,
                                    early_stopping_rounds=1000)
                self.estimators.append({'estimator': estim, 'numdata': df_train.shape[0], 'typ': typ})
        return list(y_valid), list(estim.predict(x_valid))
    def predict(self, features):
        y_pred = np.zeros((features.shape[0], ), dtype=np.float32)
        norm = np.zeros((features.shape[0], ), dtype=np.float32)
        types = features['type']
        features = features.drop('type', axis=1)
        normalization = 0
        for estim in self.estimators:
            y_pred[types == estim['typ']] = estim['estimator'].predict(features[types == estim['typ']])
            norm[types == estim['typ']] += 1
        return y_pred / norm


# In[ ]:


# del df_train
params = {
                    # 'min_data_in_leaf': 20,
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'mae', # the actual target is log(mae) weighted per type
                    'max_depth': 15,
                    'num_leaves': 120,
                    'learning_rate': 0.15,
                    'feature_fraction': 0.9,
                    'verbose': 1
                    # 'bagging_fraction': 0.5,
                    # 'bagging_freq': 10,
                }

"""
The next step is to train one model for each type there is
"""

# chunksize = int(n_train/3)
# df_train_all = pd.read_csv('../input/train.csv', chunksize=chunksize)
estimator = lgb_ensemble()
y_valid_all = []
y_pred_all = []
# max_len = 1_200_000
for typ in type_list:
    df_train = pd.read_csv('../input/train.csv')
    df_train = df_train[df_train['type'] == typ]
    # if df_train.shape[0] > max_len:
    #     df_train = df_train.iloc[:max_len, :]
    df_train = reduce_mem_usage(df_train)
# for df_train in df_train_all:
    # weights = [weight_dict[row['type']] for i, row in df_train.iterrows()]
    df_train = get_features(df_train, df_struct, atom_list, type_list)
    target = df_train['scalar_coupling_constant']
    df_train = df_train.drop(['scalar_coupling_constant', 'type'], axis=1)
    df_train.head()
    y_valid, y_pred = estimator.train(df_train, target, params, typ)
    for i in range(len(y_valid)):
        y_valid_all.append(y_valid[i])
        y_pred_all.append(y_pred[i])

df_train.head()


# Now apply the model to test data

# In[ ]:


# apply the model to test data
del df_train
df_test_all = pd.read_csv('../input/test.csv', chunksize=10_000)
isfirst = True

for df_test in df_test_all:
    df_test = reduce_mem_usage(df_test)
    df_test_orig = df_test.copy()
    df_test = get_features(df_test, df_struct, atom_list, type_list)
    df_test_orig['scalar_coupling_constant'] = estimator.predict(df_test)
    if isfirst:
        df_test_orig.filter(['id', 'scalar_coupling_constant']).to_csv('prediction.csv', index=False, mode='w')
        isfirst = False
    else:
        df_test_orig.filter(['id', 'scalar_coupling_constant']).to_csv('prediction.csv', header=False, index=False, mode='a')
    
print('Written to disk')


# Do some visualizations of feature importance and model performance

# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
warnings.simplefilter(action='ignore', category=FutureWarning)

feature_importances = estimator.estimators[0]['estimator'].feature_importance(importance_type='gain')

for i in range(1, len(estimator.estimators)):
    feature_importances += estimator.estimators[i]['estimator'].feature_importance(importance_type='gain')

feature_imp = pd.DataFrame(sorted(zip(feature_importances,df_train.columns)), columns=['Value','Feature'])
feature_imp['Value'] = feature_imp['Value']/feature_imp['Value'].sum() # normalize to 1
sorted_values = feature_imp.sort_values(by="Value", ascending=False)
sorted_values['cum'] = np.log(sorted_values['Value'].cumsum()/sorted_values['Value'].sum())
sorted_values = sorted_values[:20]

# save for later use
important_features = sorted_values['Feature'].copy()

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=sorted_values)
plt.title('LightGBM Feature Importance')
plt.tight_layout()
plt.show()


# In[ ]:


# plot predicted vs actual valid y's

print(mpl.rcParams['agg.path.chunksize'])
mpl.rcParams['agg.path.chunksize'] = 10000
plt.figure(figsize=(20, 10))
plt.plot(y_valid_all, y_pred_all, '.')
plt.title('Prediction Plot')
plt.tight_layout()
plt.show()

