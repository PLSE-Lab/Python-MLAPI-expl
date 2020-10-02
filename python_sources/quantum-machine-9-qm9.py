#!/usr/bin/env python
# coding: utf-8

# Quantum Machine 9 (QM9) dataset is now uploaded to Kaggle, [find here](https://www.kaggle.com/zaharch/quantum-machine-9-aka-qm9). 
# 
# Note that QM9 contains extra information both for the train and the test datasets of the competition. **This kernel extracts the features from the dataset and saves them into file data.covs.pickle for convenience of use**. I think you can even add this kernel to your pipeline of kernels to experiment with the features, or just download the output file (it takes a few hours to create it). The kernel also includes a simple LightGBM model on the extracted features with feature importance graph. Spoiler: mulliken partial charges for the two atoms are on top.
# 
# Note: I am not sure how to extract the information from the list of frequencies correctly, currently I have just taken min/max and mean values of the list.
# 
# Disclaimer: **the dataset is not allowed to use for your final submissions in this competition**. But we can still learn from it.

# **Does QM9 contain the information from extra files given in the competition?**
# 
# 1. dipole_moments.csv contains X,Y,Z values per molecule and I found that sqrt(X^2+Y^2+Z^2)=mu where mu is given in QM9
# 2. mulliken_charges.csv matches the mulliken charges from QM9
# 3. scalar_coupling_contributions.csv I can't find this info in QM9
# 4. magnetic_shielding_tensors.csv I can't find this info in QM9

# 
# QM9 contains the structure information, and additionally the following information both for the train and the test:
# 
# 1. Mulliken partial charge for each atom
# 2. Frequencies for degrees of freedom
# 3. SMILES from GDB9 and for relaxed geometry
# 4. InChI for GDB9 and for relaxed geometry
# 
# and also the following 17 properties per molecule:
# 
# `
# I. Property  Unit         Description
#  1  tag       -            gdb9; string constant to ease extraction via grep
#  2  index     -            Consecutive, 1-based integer identifier of molecule
#  3  A         GHz          Rotational constant A
#  4  B         GHz          Rotational constant B
#  5  C         GHz          Rotational constant C
#  6  mu        Debye        Dipole moment
#  7  alpha     Bohr^3       Isotropic polarizability
#  8  homo      Hartree      Energy of Highest occupied molecular orbital (HOMO)
#  9  lumo      Hartree      Energy of Lowest occupied molecular orbital (LUMO)
# 10  gap       Hartree      Gap, difference between LUMO and HOMO
# 11  r2        Bohr^2       Electronic spatial extent
# 12  zpve      Hartree      Zero point vibrational energy
# 13  U0        Hartree      Internal energy at 0 K
# 14  U         Hartree      Internal energy at 298.15 K
# 15  H         Hartree      Enthalpy at 298.15 K
# 16  G         Hartree      Free energy at 298.15 K
# 17  Cv        cal/(mol K)  Heat capacity at 298.15 K
# `

# Example of QM9 data format (for dsgdb9nsd_000001.xyz):
# 
# `
# 5
# gdb 1	157.7118	157.70997	157.70699	0.	13.21	-0.3877	0.1171	0.5048	35.3641	0.044749	-40.47893	-40.476062	-40.475117	-40.498597	6.469	
# C	-0.0126981359	 1.0858041578	 0.0080009958	-0.535689
# H	 0.002150416	-0.0060313176	 0.0019761204	 0.133921
# H	 1.0117308433	 1.4637511618	 0.0002765748	 0.133922
# H	-0.540815069	 1.4475266138	-0.8766437152	 0.133923
# H	-0.5238136345	 1.4379326443	 0.9063972942	 0.133923
# 1341.307	1341.3284	1341.365	1562.6731	1562.7453	3038.3205	3151.6034	3151.6788	3151.7078
# C	C	
# InChI=1S/CH4/h1H4	InChI=1S/CH4/h1H4
# `

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from pathlib import Path
import csv
import pickle
from joblib import Parallel, delayed
from tqdm import tqdm_notebook as tqdm
import pdb
import lightgbm as lgb
import xgboost as xgb
import random
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold
import seaborn as sn

PATH_QM9 = Path('../input/quantum-machine-9-aka-qm9/dsgdb9nsd.xyz')
PATH_BASE = Path('../input/champs-scalar-coupling')
PATH_WORKING = Path('../working')


# In[ ]:


train = pd.read_csv(PATH_BASE/'train.csv')
test = pd.read_csv(PATH_BASE/'test.csv')

both = pd.concat([train, test], axis=0, sort=False)
both = both.set_index('molecule_name',drop=False)

both.sort_index(inplace=True)


# In[ ]:


def processQM9_file(filename):
    path = PATH_QM9/filename
    molecule_name = filename[:-4]
    
    row_count = sum(1 for row in csv.reader(open(path)))
    na = row_count-5
    freqs = pd.read_csv(path,sep=' |\t',engine='python',skiprows=row_count-3,nrows=1,header=None)
    sz = freqs.shape[1]
    is_linear = np.nan
    if 3*na - 5 == sz:
        is_linear = False
    elif 3*na - 6 == sz:
        is_linear = True
    
    stats = pd.read_csv(path,sep=' |\t',engine='python',skiprows=1,nrows=1,header=None)
    stats = stats.loc[:,2:]
    stats.columns = ['rc_A','rc_B','rc_C','mu','alpha','homo','lumo','gap','r2','zpve','U0','U','H','G','Cv']
    
    stats['freqs_min'] = freqs.values[0].min()
    stats['freqs_max'] = freqs.values[0].max()
    stats['freqs_mean'] = freqs.values[0].mean()
    stats['linear'] = is_linear
    
    mm = pd.read_csv(path,sep='\t',engine='python', skiprows=2, skipfooter=3, names=range(5))[4]
    if mm.dtype == 'O':
        mm = mm.str.replace('*^','e',regex=False).astype(float)
    stats['mulliken_min'] = mm.min()
    stats['mulliken_max'] = mm.max()
    stats['mulliken_mean'] = mm.mean()
    
    stats['molecule_name'] = molecule_name
    
    data = pd.merge(both.loc[[molecule_name],:].reset_index(drop=True), stats, how='left', on='molecule_name')
    data['mulliken_atom_0'] = mm[data['atom_index_0'].values].values
    data['mulliken_atom_1'] = mm[data['atom_index_1'].values].values
    
    return data

def processQM9_list(files):
    df = pd.DataFrame()
    for i,filename in enumerate(files):
        stats = processQM9_file(filename)
        df = pd.concat([df, stats], axis = 0)
    return df


# In[ ]:


all_files = os.listdir(PATH_BASE/'structures')

get_ipython().run_line_magic('time', 'result = Parallel(n_jobs=4, temp_folder=PATH_WORKING)     (delayed(processQM9_list)(all_files[100*idx:min(100*(idx+1), len(all_files))]) for idx in tqdm(range(int(np.ceil(len(all_files)/100)))))')

data = pd.concat(result)
data = data.reset_index(drop=True)
data.to_pickle(PATH_WORKING/'data.covs.pickle')


# In[ ]:


data = pd.read_pickle(PATH_WORKING/'data.covs.pickle')


# # Model

# In[ ]:


def setSeeds(seed = 1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# In[ ]:


cat_names = ['type','linear']
for i in cat_names:
    data[i] = pd.factorize(data[i])[0]


# In[ ]:


data_train = data.loc[~data['scalar_coupling_constant'].isnull()].reset_index(drop=True)
data_test = data.loc[data['scalar_coupling_constant'].isnull()].reset_index(drop=True)


# In[ ]:


params_lgb = {'application': 'regression_l1',
              'metric': 'l1',
              'num_leaves': 90,
              'max_depth': 7,
              'learning_rate': 1,
              'bagging_freq' : 1,
              'bagging_fraction': 0.9,
              'feature_fraction': 0.9,
              'min_split_gain': 0.02,
              'min_child_samples': 50,
              'min_child_weight': 0.01,
              'lambda_l2': 0.05,
              'lambda_l1': 0.01,
              'verbosity': -1,
              'data_random_seed': 17}

FOLDS_VALID = 5
meta_cols = ['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'scalar_coupling_constant']
target = 'scalar_coupling_constant'

fi = pd.DataFrame()
models = []

setSeeds(1234)

data_filt = data_train.drop(meta_cols, axis=1, errors='ignore')
predictions = np.zeros(len(data_filt))

kf = KFold(n_splits=FOLDS_VALID, random_state=1234, shuffle=True)
fold_splits = kf.split(data_filt)

for i, (dev_index, val_index) in enumerate(fold_splits):

    print('Fold', i)

    Xt, Xv = data_filt.loc[dev_index, :], data_filt.loc[val_index, :]
    yt, yv = data_train.loc[dev_index, target], data_train.loc[val_index, target]

    d_train = lgb.Dataset(Xt, yt)
    d_valid = lgb.Dataset(Xv, yv)

    watchlist = [d_train, d_valid]
    model = lgb.train(params_lgb,
                      train_set=d_train,
                      num_boost_round=500,
                      valid_sets=watchlist,
                      verbose_eval=100,
                      early_stopping_rounds=100)

    predictions[val_index] = model.predict(Xv, num_iteration = model.best_iteration)

    fold_importance = pd.DataFrame()
    fold_importance["feature"] = Xt.columns
    fold_importance["importance"] = model.feature_importance()
    fold_importance["fold"] = i
    fi = pd.concat([fi, fold_importance], axis=0)

    models.append(model)

print('MAE', (predictions - data_train[target]).abs().mean())


# In[ ]:


data_test_filt = data_test.drop(meta_cols, axis=1, errors='ignore')

preds = np.zeros((FOLDS_VALID, len(data_test)))
for i in range(FOLDS_VALID):
    print('Fold', i)
    preds[i,:] = models[i].predict(data_test_filt, num_iteration=models[i].best_iteration)


# In[ ]:


sub = pd.read_csv(PATH_BASE/'sample_submission.csv')
sub['id'] = data_test['id']
sub['scalar_coupling_constant'] = preds.mean(0)
sub.to_csv(PATH_WORKING/'submission.csv', index=False)


# In[ ]:


fi['score'] = fi[["feature", "importance"]].groupby('feature').transform('mean')

plt.figure(figsize=(16, 12))
sn.barplot(x="importance", y="feature", data=fi.sort_values(by="score", ascending=False))
plt.title('LGB Features (avg over folds)')

