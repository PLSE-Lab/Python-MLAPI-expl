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
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm
import gc
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from functools import partial

atomic_radius = {'H': 0.38, 'C': 0.77, 'N': 0.75, 'O': 0.73, 'F': 0.71}
atomic_protons = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}
atomic_mass = {'H': 1.0079, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994, 'F': 18.9984}
electronegativity = {'H': 2.2, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98}

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sub = pd.read_csv('../input/sample_submission.csv')

# get xyz data for each atom
structures = pd.read_csv('../input/structures.csv')

# map atom_index_{0|1} to x_0, ..., z_1
def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how='left',
                  left_on=['molecule_name', f'atom_index_{atom_idx}'],
                  right_on=['molecule_name', 'atom_index'])
    #
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df


train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)

# molecule properties: atom_counts
atom_cnt = structures['molecule_name'].value_counts().reset_index(level=0)
atom_cnt.rename({'index': 'molecule_name', 'molecule_name': 'atom_count'}, axis=1, inplace=True)
train = pd.merge(train, atom_cnt, how='left', on='molecule_name')
test = pd.merge(test, atom_cnt, how='left', on='molecule_name')
del atom_cnt

H_cnt = structures['molecule_name'][structures['atom']=='H'].value_counts().reset_index(level=0)
H_cnt.rename({'index': 'molecule_name', 'molecule_name': 'H_count'}, axis=1, inplace=True)
train = pd.merge(train, H_cnt, how='left', on='molecule_name')
test = pd.merge(test, H_cnt, how='left', on='molecule_name')
del H_cnt

C_cnt = structures['molecule_name'][structures['atom']=='C'].value_counts().reset_index(level=0)
C_cnt.rename({'index': 'molecule_name', 'molecule_name': 'C_count'}, axis=1, inplace=True)
train = pd.merge(train, C_cnt, how='left', on='molecule_name')
test = pd.merge(test, C_cnt, how='left', on='molecule_name')
del C_cnt

O_cnt = structures['molecule_name'][structures['atom']=='O'].value_counts().reset_index(level=0)
O_cnt.rename({'index': 'molecule_name', 'molecule_name': 'O_count'}, axis=1, inplace=True)
train = pd.merge(train, O_cnt, how='left', on='molecule_name')
test = pd.merge(test, O_cnt, how='left', on='molecule_name')
del O_cnt

N_cnt = structures['molecule_name'][structures['atom']=='N'].value_counts().reset_index(level=0)
N_cnt.rename({'index': 'molecule_name', 'molecule_name': 'N_count'}, axis=1, inplace=True)
train = pd.merge(train, N_cnt, how='left', on='molecule_name')
test = pd.merge(test, N_cnt, how='left', on='molecule_name')
del N_cnt

F_cnt = structures['molecule_name'][structures['atom']=='F'].value_counts().reset_index(level=0)
F_cnt.rename({'index': 'molecule_name', 'molecule_name': 'F_count'}, axis=1, inplace=True)
train = pd.merge(train, F_cnt, how='left', on='molecule_name')
test = pd.merge(test, F_cnt, how='left', on='molecule_name')
del F_cnt
del structures

# atom_1 properties
def atom_props(df):
    df['atomic_radius'] = df['atom_1'].apply(lambda x: atomic_radius[x])
    df['atomic_protons'] = df['atom_1'].apply(lambda x: atomic_protons[x])
    df['atomic_mass'] = df['atom_1'].apply(lambda x: atomic_mass[x])
    df['electronegativity'] = df['atom_1'].apply(lambda x: electronegativity[x])
    return df


train = atom_props(train)
test = atom_props(test)

# split type
train['type_0'] = train['type'].apply(lambda x: x[0])
train['type_1'] = train['type'].apply(lambda x: x[1:])
test['type_0'] = test['type'].apply(lambda x: x[0])
test['type_1'] = test['type'].apply(lambda x: x[1:])

# distances
train_p_0 = train[['x_0', 'y_0', 'z_0']].values
train_p_1 = train[['x_1', 'y_1', 'z_1']].values
test_p_0 = test[['x_0', 'y_0', 'z_0']].values
test_p_1 = test[['x_1', 'y_1', 'z_1']].values

train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)

train['dist_x'] = (train['x_0'] - train['x_1']) ** 2
test['dist_x'] = (test['x_0'] - test['x_1']) ** 2
train['dist_y'] = (train['y_0'] - train['y_1']) ** 2
test['dist_y'] = (test['y_0'] - test['y_1']) ** 2
train['dist_z'] = (train['z_0'] - train['z_1']) ** 2
test['dist_z'] = (test['z_0'] - test['z_1']) ** 2

train['dist_to_type_mean'] = train['dist'] / train.groupby('type')['dist'].transform('mean')
test['dist_to_type_mean'] = test['dist'] / test.groupby('type')['dist'].transform('mean')

train['dist_to_type_0_mean'] = train['dist'] / train.groupby('type_0')['dist'].transform('mean')
test['dist_to_type_0_mean'] = test['dist'] / test.groupby('type_0')['dist'].transform('mean')

train['dist_to_type_1_mean'] = train['dist'] / train.groupby('type_1')['dist'].transform('mean')
test['dist_to_type_1_mean'] = test['dist'] / test.groupby('type_1')['dist'].transform('mean')

# dipole moments of molecules
dipole_moments = pd.read_csv('../input/dipole_moments.csv')
train = pd.merge(train, dipole_moments, how='left',
              left_on='molecule_name',
              right_on='molecule_name')
del dipole_moments

# mulliken charges of each atom
mulliken_charges = pd.read_csv('../input/mulliken_charges.csv')
train = pd.merge(train, mulliken_charges, how='left',
              left_on=['molecule_name', 'atom_index_0'],
              right_on=['molecule_name', 'atom_index'])

train.rename(columns = { "mulliken_charge": "mc_0"}, inplace=True)

train = pd.merge(train, mulliken_charges, how='left',
              left_on=['molecule_name', 'atom_index_1'],
              right_on=['molecule_name', 'atom_index'])

train.rename(columns={"mulliken_charge": "mc_1"}, inplace=True)
del mulliken_charges

# scc
scc = pd.read_csv('../input/scalar_coupling_contributions.csv')
train = pd.merge(train, scc, how='left',
                 on=['molecule_name', 'atom_index_0', 'atom_index_1', 'type'])
del scc

# magnetic_shielding_tensors
magnetic_shielding_tensors = pd.read_csv('../input/magnetic_shielding_tensors.csv')
train = pd.merge(train, magnetic_shielding_tensors, how='left',
              left_on=['molecule_name', 'atom_index_0'],
              right_on=['molecule_name', 'atom_index'])

train.rename(columns = { "XX": "XX_0", "YX": "YX_0", "ZX": "ZX_0",
                         "XY": "XY_0", "YY": "YY_0", "ZY": "ZY_0",
                         "XZ": "XZ_0", "YZ": "YZ_0", "ZZ": "ZZ_0" }, inplace=True)

train = pd.merge(train, magnetic_shielding_tensors, how='left',
              left_on=['molecule_name', 'atom_index_1'],
              right_on=['molecule_name', 'atom_index'])

train.rename(columns = { "XX": "XX_1", "YX": "YX_1", "ZX": "ZX_1",
                         "XY": "XY_1", "YY": "YY_1", "ZY": "ZY_1",
                         "XZ": "XZ_1", "YZ": "YZ_1", "ZZ": "ZZ_1" }, inplace=True)
del magnetic_shielding_tensors

# potential energy of molecules
potential_energy = pd.read_csv('../input/potential_energy.csv')
train = pd.merge(train, potential_energy, how='left',
              left_on='molecule_name',
              right_on='molecule_name')
del potential_energy

gc.collect()

train = train.fillna(0)
test = test.fillna(0)

# Define searched space for LightGBM
hyper_space = {'objective': 'regression',
               'metric':'mae',
               'boosting':'gbdt',
               #'n_estimators': hp.choice('n_estimators', [25, 40, 50, 75, 100, 250, 500]),
               'max_depth':  hp.choice('max_depth', [5, 8, 10, 12, 15]),
               'num_leaves': hp.choice('num_leaves', [100, 250, 500, 650, 750, 1000,1300]),
               'subsample': hp.choice('subsample', [.3, .5, .7, .8, 1]),
               'colsample_bytree': hp.choice('colsample_bytree', [ .6, .7, .8, .9, 1]),
               'learning_rate': hp.choice('learning_rate', [.1, .2, .3]),
               'reg_alpha': hp.choice('reg_alpha', [.1, .2, .3, .4, .5, .6]),
               'reg_lambda':  hp.choice('reg_lambda', [.1, .2, .3, .4, .5, .6]),
               'min_child_samples': hp.choice('min_child_samples', [20, 45, 70, 100]),
               'verbose': -1}

# evaluation metric
def metric(df, preds):
    df["prediction"] = preds
    maes = []
    for t in df.type.unique():
        y_true = df[df.type==t].iloc[:,1].values # column 1 is the target whatever the y_var
        y_pred = df[df.type==t].prediction.values
        mae = np.log(metrics.mean_absolute_error(y_true, y_pred))
        maes.append(mae)
    return np.mean(maes)

# encode categorical features
cat_feats = ['type', 'type_0', 'type_1']
for f in cat_feats:
    lbl = LabelEncoder()
    lbl.fit(list(train[f].values) + list(train[f].values))
    train[f] = lbl.transform(list(train[f].values))
    test[f] = lbl.transform(list(test[f].values))

# model dp, mc, scc, etc.
y_vars = ['X', 'Y', 'Z'] + ['mc_0', 'mc_1'] + ['fc', 'sd', 'pso', 'dso'] +\
            ['XX_0', 'YX_0', 'ZX_0', 'XY_0', 'YY_0', 'ZY_0',
             'XZ_0', 'YZ_0', 'ZZ_0', 'XX_1', 'YX_1', 'ZX_1', 'XY_1',
             'YY_1', 'ZY_1', 'XZ_1', 'YZ_1', 'ZZ_1'] + ['potential_energy']

# pred vars
pred_vars = [v for v in train.columns if v not in y_vars +\
             ['id', 'molecule_name', 'scalar_coupling_constant', 'atom_0', 'atom_1', 'atom_index_x',
              'atom_index_y', 'atom_index_x', 'atom_index_y']]

# train-val split
X_train, X_val, y_train, y_val = train_test_split(train[pred_vars], train[y_vars], test_size=0.10, random_state=42)

for i in np.arange(1,len(y_vars)):
    df_val = pd.DataFrame({"type": X_val["type"]})
    df_val[y_vars[i]] = y_val.iloc[:,i]
    #
    lgtrain = lightgbm.Dataset(X_train, label=y_train.iloc[:,i])
    lgval = lightgbm.Dataset(X_val, label=y_val.iloc[:,i])
    #
    def evaluate_metric(params):
        model_lgb = lightgbm.train(params, lgtrain, 100,
                                   valid_sets=[lgtrain, lgval], early_stopping_rounds=2,
                                   verbose_eval=10)
        pred = model_lgb.predict(X_val)
        score = metric(df_val, pred)
        print(score)
        return {
            'loss': score,
            'status': STATUS_OK,
            'stats_running': STATUS_RUNNING
        }
    #
    #
    # Trial DB, algorithm, max_evals
    trials = Trials()
    algo = partial(tpe.suggest, n_startup_jobs=-1)
    MAX_EVALS = 15
    #
    # Fit Tree Parzen Estimator
    best_vals = fmin(evaluate_metric, space=hyper_space, verbose=1,
                     algo=algo, max_evals=MAX_EVALS, trials=trials)
    #
    # Best parameters
    best_params = space_eval(hyper_space, best_vals)
    #
    # Prediction
    model_lgb = lightgbm.train(best_params, lgtrain, 400,
                          valid_sets=[lgtrain, lgval], early_stopping_rounds=3,
                          verbose_eval=50)
    #
    train[y_vars[i]] = model_lgb.predict(train[pred_vars])
    test = pd.concat([test, pd.DataFrame(model_lgb.predict(test[pred_vars]))], axis=1)
    test.rename(columns={0: y_vars[i]}, inplace=True)

